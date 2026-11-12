//! Storage health: whether the redundancy under the filesystems is intact.
//!
//! A mirror that has lost a disk keeps serving every read and write, so
//! nothing on the dashboard changes — until the second disk goes. The weeks
//! between the two failures are the whole point of the mirror, and they are
//! only useful if somebody hears about the first one. The kernel says so in
//! three places, all readable without root:
//!
//! - md RAID, `/proc/mdstat`: `[2/1] [U_]` is a two-disk array with one
//!   member missing, `(F)` marks a failed member, and a `recovery` line is a
//!   rebuild in progress — redundancy still lost until it finishes.
//! - ZFS, `/proc/spl/kstat/zfs/<pool>/state`: `ONLINE`, `DEGRADED`,
//!   `FAULTED`, `SUSPENDED`, the same word `zpool status` prints.
//! - btrfs, `/sys/fs/btrfs/<uuid>/devinfo/<id>/`: `error_stats` (read, write,
//!   flush, corruption and generation errors, kept on disk across mounts) and
//!   `missing`.
//!
//! SMART needs root and a vendor-specific ioctl, so it is out of reach for a
//! monitor that asks for no privileges; block-layer IO errors are already in
//! the kernel-events panel.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArrayState {
    #[default]
    Healthy,
    /// Rebuilding onto a replacement: still without redundancy until done.
    Rebuilding,
    /// Running without full redundancy, or reporting device errors.
    Degraded,
    /// Not serving at all.
    Failed,
}

impl ArrayState {
    pub fn label(self) -> &'static str {
        match self {
            ArrayState::Healthy => "ok",
            ArrayState::Rebuilding => "rebuilding",
            ArrayState::Degraded => "degraded",
            ArrayState::Failed => "failed",
        }
    }

    /// Anything but healthy: a rebuilding array has no redundancy either.
    pub fn is_unhealthy(self) -> bool {
        self != ArrayState::Healthy
    }
}

/// One redundant volume: an md array, a ZFS pool or a btrfs filesystem.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageArray {
    /// `md`, `zfs` or `btrfs`.
    pub kind: String,
    /// `md0`, the pool name, or the btrfs label (its uuid when unlabelled).
    pub name: String,
    /// `raid1`, `raid5`... for md; empty where the source does not say.
    pub level: String,
    pub state: ArrayState,
    /// The source's own words: `[2/1] [U_]`, `DEGRADED`, `3 errors on sdb`.
    pub detail: String,
    /// Percent done of a resync, rebuild, reshape or check, while one runs.
    pub progress: Option<f64>,
    /// What is running, alongside `progress`: `recovery`, `resync`, `check`.
    pub activity: Option<String>,
    /// btrfs device errors summed across devices, since the counters were
    /// last reset.
    pub errors: Option<u64>,
}

/// Every array `/proc/mdstat` lists.
pub fn parse_mdstat(content: &str) -> Vec<StorageArray> {
    let mut out: Vec<StorageArray> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        // `md0 : active raid1 sdb1[1] sda1[0](F)`
        if let Some((name, rest)) = line.split_once(" : ")
            && name.starts_with("md")
            && !line.starts_with(' ')
        {
            let mut words = rest.split_whitespace();
            let activity = words.next().unwrap_or("");
            let mut level = String::new();
            let mut failed = 0;
            for w in words {
                if w.starts_with("raid") || w == "linear" {
                    level = w.to_string();
                } else if w.ends_with("(F)") {
                    failed += 1;
                }
            }
            let state = match (activity, failed) {
                ("inactive", _) => ArrayState::Failed,
                (_, 0) => ArrayState::Healthy,
                _ => ArrayState::Degraded,
            };
            out.push(StorageArray {
                kind: "md".into(),
                name: name.trim().to_string(),
                level,
                state,
                detail: if activity == "inactive" { "inactive".into() } else { String::new() },
                ..Default::default()
            });
            continue;
        }
        let Some(array) = out.last_mut() else { continue };
        // `976630464 blocks super 1.2 [2/1] [U_]`
        if let Some(status) =
            trimmed.rsplit(' ').next().filter(|s| s.starts_with('[') && s.ends_with(']'))
            && status[1..status.len() - 1].chars().all(|c| c == 'U' || c == '_')
            && !status[1..status.len() - 1].is_empty()
        {
            let counts = trimmed
                .split_whitespace()
                .find(|w| w.starts_with('[') && w.contains('/'))
                .unwrap_or("");
            array.detail = format!("{counts} {status}").trim().to_string();
            if status.contains('_') && array.state == ArrayState::Healthy {
                array.state = ArrayState::Degraded;
            }
            continue;
        }
        // `[==>.......]  recovery = 12.6% (123/976) finish=80.1min speed=...`
        for activity in ["recovery", "resync", "reshape", "check", "repair"] {
            let Some(after) = trimmed.split_once(&format!("{activity} =")).map(|(_, a)| a) else {
                continue;
            };
            array.progress = after.split('%').next().and_then(|p| p.trim().parse().ok());
            array.activity = Some(activity.to_string());
            // Rebuilding onto a replacement is still running without
            // redundancy; a resync or check of a complete array is upkeep.
            if matches!(activity, "recovery" | "reshape") && array.state != ArrayState::Failed {
                array.state = ArrayState::Rebuilding;
            }
            break;
        }
    }
    out
}

/// A ZFS pool's state word, as `zpool status` prints it.
pub fn zfs_state(name: &str, word: &str) -> StorageArray {
    let word = word.trim();
    let state = match word {
        "ONLINE" => ArrayState::Healthy,
        "FAULTED" | "UNAVAIL" | "SUSPENDED" => ArrayState::Failed,
        _ => ArrayState::Degraded,
    };
    StorageArray {
        kind: "zfs".into(),
        name: name.to_string(),
        state,
        detail: word.to_string(),
        ..Default::default()
    }
}

/// The sum of one btrfs device's `error_stats`.
pub fn parse_btrfs_errors(content: &str) -> u64 {
    content.lines().filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).sum()
}

fn read_trim(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn read_zfs(root: &Path) -> Vec<StorageArray> {
    let Ok(entries) = fs::read_dir(root) else { return Vec::new() };
    let mut out: Vec<StorageArray> = entries
        .flatten()
        .filter_map(|e| {
            // Pools are the directories with a `state` file; the rest of the
            // directory is module-wide statistics.
            let word = read_trim(&e.path().join("state"))?;
            Some(zfs_state(&e.file_name().to_string_lossy(), &word))
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn read_btrfs(root: &Path) -> Vec<StorageArray> {
    let Ok(entries) = fs::read_dir(root) else { return Vec::new() };
    let mut out = Vec::new();
    for fs_dir in entries.flatten() {
        let devinfo = fs_dir.path().join("devinfo");
        let Ok(devices) = fs::read_dir(&devinfo) else { continue };
        let uuid = fs_dir.file_name().to_string_lossy().to_string();
        let name = read_trim(&fs_dir.path().join("label"))
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| uuid.chars().take(8).collect());
        let (mut errors, mut missing, mut devices_seen, mut readable) = (0u64, 0u32, 0u32, false);
        for dev in devices.flatten() {
            devices_seen += 1;
            if read_trim(&dev.path().join("missing")).as_deref() == Some("1") {
                missing += 1;
            }
            // `error_stats` is 5.14+; older kernels show nothing rather
            // than a count of zero they cannot vouch for.
            if let Ok(stats) = fs::read_to_string(dev.path().join("error_stats")) {
                readable = true;
                errors += parse_btrfs_errors(&stats);
            }
        }
        let state = match (missing, errors) {
            (0, 0) => ArrayState::Healthy,
            _ => ArrayState::Degraded,
        };
        let detail = match (missing, errors) {
            (0, 0) => format!("{devices_seen} dev"),
            (0, e) => format!("{e} errors"),
            (m, 0) => format!("{m} of {devices_seen} missing"),
            (m, e) => format!("{m} of {devices_seen} missing, {e} errors"),
        };
        out.push(StorageArray {
            kind: "btrfs".into(),
            name,
            state,
            detail,
            errors: readable.then_some(errors),
            ..Default::default()
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Every redundant volume this host has, worst first. Empty on the ordinary
/// single-disk machine, which is what keeps the panel quiet there.
pub fn read() -> Vec<StorageArray> {
    if !cfg!(target_os = "linux") {
        return Vec::new();
    }
    let mut out = fs::read_to_string("/proc/mdstat").map(|c| parse_mdstat(&c)).unwrap_or_default();
    out.extend(read_zfs(Path::new("/proc/spl/kstat/zfs")));
    out.extend(read_btrfs(Path::new("/sys/fs/btrfs")));
    // Worst first, so a degraded array is the row the panel cannot cut off.
    out.sort_by_key(|a| std::cmp::Reverse(a.state as u8));
    out
}

/// The figure a `storage` rule measures: how many arrays are unhealthy, of
/// all of them or of the one `target` names.
pub fn unhealthy_count(arrays: &[StorageArray], target: &str) -> usize {
    arrays
        .iter()
        .filter(|a| target.is_empty() || a.name == target)
        .filter(|a| a.state.is_unhealthy())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real mdstat shape: a degraded mirror rebuilding onto a new disk, a
    // healthy RAID5 being checked, and an inactive array.
    const MDSTAT: &str = "Personalities : [raid1] [raid6] [raid5] [raid4]
md0 : active raid1 sdc1[2] sda1[0] sdb1[1](F)
      976630464 blocks super 1.2 [2/1] [U_]
      [==>..................]  recovery = 12.6% (123456789/976630464) finish=80.1min speed=176543K/sec
      bitmap: 1/8 pages [4KB], 65536KB chunk

md1 : active raid5 sdf1[3] sde1[1] sdd1[0]
      1953260544 blocks super 1.2 level 5, 512k chunk, algorithm 2 [3/3] [UUU]
      [=====>...............]  check = 27.3% (266669056/976630272) finish=60.2min speed=196443K/sec

md2 : inactive sdg1[0](S)
      976630464 blocks super 1.2

unused devices: <none>
";

    #[test]
    fn md_arrays_report_missing_members_rebuilds_and_checks() {
        let a = parse_mdstat(MDSTAT);
        assert_eq!(a.len(), 3);
        assert_eq!((a[0].name.as_str(), a[0].level.as_str()), ("md0", "raid1"));
        assert_eq!(a[0].state, ArrayState::Rebuilding, "a failed member being replaced");
        assert_eq!(a[0].detail, "[2/1] [U_]");
        assert_eq!(a[0].progress, Some(12.6));
        assert_eq!(a[0].activity.as_deref(), Some("recovery"));
        // A check of a complete array is upkeep, not trouble.
        assert_eq!(a[1].state, ArrayState::Healthy);
        assert_eq!(a[1].detail, "[3/3] [UUU]");
        assert_eq!(a[1].activity.as_deref(), Some("check"));
        assert_eq!(a[2].state, ArrayState::Failed);
    }

    #[test]
    fn a_degraded_array_with_no_rebuild_says_degraded() {
        let a = parse_mdstat("md5 : active raid1 sda1[0]\n      1000 blocks [2/1] [U_]\n");
        assert_eq!(a[0].state, ArrayState::Degraded);
        assert!(parse_mdstat("Personalities : [raid1]\nunused devices: <none>\n").is_empty());
    }

    #[test]
    fn zfs_states_map_onto_three_levels_of_worry() {
        assert_eq!(zfs_state("tank", "ONLINE\n").state, ArrayState::Healthy);
        assert_eq!(zfs_state("tank", "DEGRADED").state, ArrayState::Degraded);
        assert_eq!(zfs_state("tank", "SUSPENDED").state, ArrayState::Failed);
        // An unfamiliar word is worth a look, not silence.
        assert_eq!(zfs_state("tank", "REMOVED").state, ArrayState::Degraded);
    }

    #[test]
    fn btrfs_errors_are_summed_and_a_missing_device_is_degraded() {
        let stats =
            "write_errs 0\nread_errs 3\nflush_errs 0\ncorruption_errs 1\ngeneration_errs 0\n";
        assert_eq!(parse_btrfs_errors(stats), 4);

        let root = std::env::temp_dir().join(format!("crabmon-btrfs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let fs1 = root.join("1111aaaa-0000");
        for (dev, errs, missing) in [("1", stats, "0"), ("2", "read_errs 0\n", "1")] {
            let d = fs1.join("devinfo").join(dev);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("error_stats"), errs).unwrap();
            fs::write(d.join("missing"), missing).unwrap();
        }
        fs::write(fs1.join("label"), "data\n").unwrap();
        let healthy = root.join("2222bbbb-0000").join("devinfo").join("1");
        fs::create_dir_all(&healthy).unwrap();
        fs::write(healthy.join("error_stats"), "read_errs 0\n").unwrap();

        let got = read_btrfs(&root);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "2222bbbb", "an unlabelled filesystem goes by its uuid");
        assert_eq!(got[0].state, ArrayState::Healthy);
        assert_eq!(got[1].name, "data");
        assert_eq!(got[1].state, ArrayState::Degraded);
        assert_eq!(got[1].errors, Some(4));
        assert_eq!(got[1].detail, "1 of 2 missing, 4 errors");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn zfs_pools_are_the_directories_with_a_state_file() {
        let root = std::env::temp_dir().join(format!("crabmon-zfs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("tank")).unwrap();
        fs::write(root.join("tank").join("state"), "DEGRADED\n").unwrap();
        fs::write(root.join("arcstats"), "not a pool").unwrap();
        let got = read_zfs(&root);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].name.as_str(), got[0].state), ("tank", ArrayState::Degraded));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_rule_counts_unhealthy_arrays_overall_or_by_name() {
        let a = parse_mdstat(MDSTAT);
        assert_eq!(unhealthy_count(&a, ""), 2);
        assert_eq!(unhealthy_count(&a, "md1"), 0);
        assert_eq!(unhealthy_count(&a, "md0"), 1);
    }
}
