//! Comparing two snapshots.
//!
//! "It got slow after the deploy" is a question about a difference, and a pair
//! of 2000-row process tables is not a readable answer. `--diff` reduces two
//! snapshots — two `--once` exports, or the two ends of one recording — to what
//! actually changed between them.

use serde::Serialize;

use crate::format::{compact_bytes, human_duration};
use crate::metrics::{ProcRow, Snapshot};

/// A process present in both snapshots, with how it moved.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProcDelta {
    pub pid: u32,
    pub name: String,
    pub cpu_before: f32,
    pub cpu_after: f32,
    pub mem_before: u64,
    pub mem_after: u64,
}

impl ProcDelta {
    pub fn mem_delta(&self) -> i64 {
        self.mem_after as i64 - self.mem_before as i64
    }
    pub fn cpu_delta(&self) -> f32 {
        self.cpu_after - self.cpu_before
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiskDelta {
    pub mount: String,
    pub used_before: u64,
    pub used_after: u64,
}

impl DiskDelta {
    pub fn delta(&self) -> i64 {
        self.used_after as i64 - self.used_before as i64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Diff {
    pub host_before: String,
    pub host_after: String,
    pub seconds_apart: u64,
    pub cpu: (f64, f64),
    pub load1: (f64, f64),
    pub mem_used: (u64, u64),
    pub swap_used: (u64, u64),
    pub procs: (usize, usize),
    pub appeared: Vec<ProcRow>,
    pub vanished: Vec<ProcRow>,
    pub movers: Vec<ProcDelta>,
    pub disks: Vec<DiskDelta>,
}

/// How many rows of each list a report shows. A diff that prints 900 vanished
/// kernel threads is as unreadable as the two tables it replaced.
pub const LIST_LIMIT: usize = 12;

/// A process's identity across two snapshots.
///
/// PID alone is not it: over the minutes a useful diff spans, a PID can be
/// recycled, and the report would then claim a process grew by 4 GB when it is
/// simply a different program.
fn identity(p: &ProcRow) -> (u32, u64) {
    (p.pid, p.start_time_unix)
}

pub fn compare(before: &Snapshot, after: &Snapshot) -> Diff {
    let before_by_id: std::collections::HashMap<(u32, u64), &ProcRow> =
        before.procs.iter().map(|p| (identity(p), p)).collect();
    let after_by_id: std::collections::HashMap<(u32, u64), &ProcRow> =
        after.procs.iter().map(|p| (identity(p), p)).collect();

    let mut appeared: Vec<ProcRow> =
        after.procs.iter().filter(|p| !before_by_id.contains_key(&identity(p))).cloned().collect();
    appeared.sort_by_key(|a| std::cmp::Reverse(a.mem));

    let mut vanished: Vec<ProcRow> =
        before.procs.iter().filter(|p| !after_by_id.contains_key(&identity(p))).cloned().collect();
    vanished.sort_by_key(|a| std::cmp::Reverse(a.mem));

    let mut movers: Vec<ProcDelta> = after
        .procs
        .iter()
        .filter_map(|p| {
            before_by_id.get(&identity(p)).map(|was| ProcDelta {
                pid: p.pid,
                name: p.name.clone(),
                cpu_before: was.cpu,
                cpu_after: p.cpu,
                mem_before: was.mem,
                mem_after: p.mem,
            })
        })
        .filter(|d| d.mem_delta() != 0 || d.cpu_delta().abs() > 0.5)
        .collect();
    // Biggest absolute memory move first: growth is what a diff is usually for,
    // but a process that released a gigabyte is just as informative.
    movers.sort_by_key(|a| std::cmp::Reverse(a.mem_delta().abs()));

    let mut disks: Vec<DiskDelta> = after
        .disks
        .iter()
        .filter_map(|d| {
            before.disks.iter().find(|b| b.mount == d.mount).map(|was| DiskDelta {
                mount: d.mount.clone(),
                used_before: was.used,
                used_after: d.used,
            })
        })
        .filter(|d| d.delta() != 0)
        .collect();
    disks.sort_by_key(|a| std::cmp::Reverse(a.delta().abs()));

    Diff {
        host_before: before.host.hostname.clone(),
        host_after: after.host.hostname.clone(),
        seconds_apart: after.taken_at_unix.saturating_sub(before.taken_at_unix),
        cpu: (before.cpu.avg(), after.cpu.avg()),
        load1: (before.host.load[0], after.host.load[0]),
        mem_used: (before.mem.used, after.mem.used),
        swap_used: (before.mem.swap_used, after.mem.swap_used),
        procs: (before.procs.len(), after.procs.len()),
        appeared,
        vanished,
        movers,
        disks,
    }
}

fn signed_bytes(delta: i64) -> String {
    let sign = if delta < 0 { "-" } else { "+" };
    format!("{sign}{}", compact_bytes(delta.unsigned_abs()))
}

impl Diff {
    /// True when the two snapshots are not from the same machine, which is
    /// worth saying out loud before anyone reads the numbers as a change over
    /// time.
    pub fn hosts_differ(&self) -> bool {
        self.host_before != self.host_after
            && !self.host_before.is_empty()
            && !self.host_after.is_empty()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        if self.hosts_differ() {
            out.push_str(&format!(
                "warning: different hosts ({} → {})\n\n",
                self.host_before, self.host_after
            ));
        }
        out.push_str(&format!(
            "{} → {}   {} apart\n\n",
            self.host_before,
            self.host_after,
            human_duration(self.seconds_apart)
        ));
        out.push_str(&format!(
            "  cpu    {:>8.1}% → {:>8.1}%   ({:+.1})\n",
            self.cpu.0,
            self.cpu.1,
            self.cpu.1 - self.cpu.0
        ));
        out.push_str(&format!(
            "  load   {:>9.2} → {:>9.2}   ({:+.2})\n",
            self.load1.0,
            self.load1.1,
            self.load1.1 - self.load1.0
        ));
        out.push_str(&format!(
            "  mem    {:>9} → {:>9}   ({})\n",
            compact_bytes(self.mem_used.0),
            compact_bytes(self.mem_used.1),
            signed_bytes(self.mem_used.1 as i64 - self.mem_used.0 as i64)
        ));
        out.push_str(&format!(
            "  swap   {:>9} → {:>9}   ({})\n",
            compact_bytes(self.swap_used.0),
            compact_bytes(self.swap_used.1),
            signed_bytes(self.swap_used.1 as i64 - self.swap_used.0 as i64)
        ));
        out.push_str(&format!(
            "  procs  {:>9} → {:>9}   ({:+})\n",
            self.procs.0,
            self.procs.1,
            self.procs.1 as i64 - self.procs.0 as i64
        ));

        if !self.disks.is_empty() {
            out.push_str("\nfilesystems\n");
            for d in self.disks.iter().take(LIST_LIMIT) {
                out.push_str(&format!("  {:<24} {}\n", d.mount, signed_bytes(d.delta())));
            }
        }

        section(&mut out, "grew / shrank", self.movers.len(), |out| {
            for m in self.movers.iter().take(LIST_LIMIT) {
                out.push_str(&format!(
                    "  {:>7} {:<22} {:>9}  cpu {:>5.1} → {:>5.1}\n",
                    m.pid,
                    crate::format::truncate_fit(&m.name, 22),
                    signed_bytes(m.mem_delta()),
                    m.cpu_before,
                    m.cpu_after
                ));
            }
        });
        section(&mut out, "started", self.appeared.len(), |out| {
            for p in self.appeared.iter().take(LIST_LIMIT) {
                out.push_str(&format!(
                    "  {:>7} {:<22} {:>9}\n",
                    p.pid,
                    crate::format::truncate_fit(&p.name, 22),
                    compact_bytes(p.mem)
                ));
            }
        });
        section(&mut out, "exited", self.vanished.len(), |out| {
            for p in self.vanished.iter().take(LIST_LIMIT) {
                out.push_str(&format!(
                    "  {:>7} {:<22} {:>9}\n",
                    p.pid,
                    crate::format::truncate_fit(&p.name, 22),
                    compact_bytes(p.mem)
                ));
            }
        });
        out
    }
}

fn section(out: &mut String, title: &str, total: usize, body: impl FnOnce(&mut String)) {
    if total == 0 {
        return;
    }
    out.push_str(&format!("\n{title} ({total})\n"));
    body(out);
    if total > LIST_LIMIT {
        out.push_str(&format!("  … and {} more\n", total - LIST_LIMIT));
    }
}

/// Read a snapshot from a `--once` export or a recording.
///
/// A recording yields its last frame, which is what "compare against this
/// capture" means when it is used as one side of a pair.
pub fn load(path: &std::path::Path) -> Result<Snapshot, String> {
    let (first, last) = load_ends(path)?;
    let _ = first;
    Ok(last)
}

/// Both ends of a file: for a single-frame export they are the same snapshot,
/// for a recording they are its first and last frames.
pub fn load_ends(path: &std::path::Path) -> Result<(Snapshot, Snapshot), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("{}: empty", path.display()));
    }
    // A single JSON document (what `--once` writes, pretty-printed over many
    // lines) parses as a whole; anything else is treated as JSON Lines.
    if let Ok(snap) = serde_json::from_str::<Snapshot>(trimmed) {
        return Ok((snap.clone(), snap));
    }
    let frames =
        crate::record::parse_jsonl(trimmed).map_err(|e| format!("{}: {e}", path.display()))?;
    let first = frames.first().cloned().unwrap_or_default();
    let last = frames.last().cloned().unwrap_or_default();
    Ok((first, last))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, DiskRow, HostInfo, MemSample};

    fn row(pid: u32, start: u64, name: &str, cpu: f32, mem: u64) -> ProcRow {
        ProcRow { pid, start_time_unix: start, name: name.into(), cpu, mem, ..Default::default() }
    }

    fn snap(at: u64, procs: Vec<ProcRow>) -> Snapshot {
        Snapshot {
            host: HostInfo { hostname: "box".into(), load: [1.0, 0.0, 0.0], ..Default::default() },
            cpu: CpuSample { per_core: vec![10.0], freq_mhz: vec![] },
            mem: MemSample { total: 1000, used: 100, ..Default::default() },
            procs,
            taken_at_unix: at,
            ..Default::default()
        }
    }

    #[test]
    fn processes_are_sorted_into_started_exited_and_moved() {
        let before = snap(100, vec![row(1, 1, "stays", 1.0, 100), row(2, 2, "goes", 0.0, 50)]);
        let after = snap(160, vec![row(1, 1, "stays", 9.0, 900), row(3, 3, "new", 0.0, 70)]);

        let d = compare(&before, &after);
        assert_eq!(d.seconds_apart, 60);
        assert_eq!(d.appeared.iter().map(|p| p.pid).collect::<Vec<_>>(), vec![3]);
        assert_eq!(d.vanished.iter().map(|p| p.pid).collect::<Vec<_>>(), vec![2]);
        assert_eq!(d.movers.len(), 1);
        assert_eq!(d.movers[0].mem_delta(), 800);
        assert_eq!(d.movers[0].cpu_delta(), 8.0);
    }

    #[test]
    fn a_recycled_pid_is_two_processes_not_one_that_grew() {
        // Same PID, different start time: reporting this as "+4 GB" would be a
        // fabrication, and over the minutes a diff spans it does happen.
        let before = snap(100, vec![row(42, 1, "old", 0.0, 1000)]);
        let after = snap(200, vec![row(42, 2, "new", 0.0, 5_000_000_000)]);

        let d = compare(&before, &after);
        assert!(d.movers.is_empty(), "{:?}", d.movers);
        assert_eq!(d.appeared.len(), 1);
        assert_eq!(d.vanished.len(), 1);
    }

    #[test]
    fn unchanged_processes_are_left_out_of_the_report() {
        let p = vec![row(1, 1, "idle", 0.0, 100)];
        let d = compare(&snap(100, p.clone()), &snap(200, p));
        assert!(d.movers.is_empty());
        assert!(d.appeared.is_empty());
        assert!(d.vanished.is_empty());
        let text = d.to_text();
        assert!(!text.contains("started"), "{text}");
    }

    #[test]
    fn filesystem_growth_is_reported_with_a_sign() {
        let mut before = snap(100, vec![]);
        let mut after = snap(200, vec![]);
        before.disks = vec![DiskRow { mount: "/".into(), used: 1000, ..Default::default() }];
        after.disks = vec![DiskRow { mount: "/".into(), used: 500, ..Default::default() }];
        let d = compare(&before, &after);
        assert_eq!(d.disks[0].delta(), -500);
        assert!(d.to_text().contains("-500B"), "{}", d.to_text());
    }

    #[test]
    fn comparing_two_machines_says_so_before_the_numbers() {
        let mut before = snap(100, vec![]);
        before.host.hostname = "web-1".into();
        let mut after = snap(200, vec![]);
        after.host.hostname = "web-2".into();
        let d = compare(&before, &after);
        assert!(d.hosts_differ());
        assert!(d.to_text().starts_with("warning: different hosts"), "{}", d.to_text());
    }

    #[test]
    fn long_lists_are_capped_and_say_how_many_were_left_out() {
        let before = snap(100, vec![]);
        let after = snap(200, (0..40).map(|i| row(i, i as u64, "x", 0.0, 1)).collect());
        let text = compare(&before, &after).to_text();
        assert!(text.contains("started (40)"), "{text}");
        assert!(text.contains("… and 28 more"), "{text}");
    }

    #[test]
    fn both_a_single_export_and_a_recording_load() {
        let dir = std::env::temp_dir();
        let one = dir.join(format!("crabmon-diff-one-{}.json", std::process::id()));
        let many = dir.join(format!("crabmon-diff-many-{}.jsonl", std::process::id()));

        // `--once` writes a pretty-printed document with an extra version field.
        let export = crate::export::to_json(&snap(100, vec![row(1, 1, "a", 0.0, 5)]));
        std::fs::write(&one, &export).unwrap();
        let (first, last) = load_ends(&one).unwrap();
        assert_eq!(first.taken_at_unix, 100);
        assert_eq!(last.procs[0].name, "a");

        let lines: String =
            (1..=3).map(|n| serde_json::to_string(&snap(n * 10, vec![])).unwrap() + "\n").collect();
        std::fs::write(&many, lines).unwrap();
        let (first, last) = load_ends(&many).unwrap();
        assert_eq!((first.taken_at_unix, last.taken_at_unix), (10, 30));
        assert_eq!(load(&many).unwrap().taken_at_unix, 30, "a recording diffs from its end");

        let _ = std::fs::remove_file(&one);
        let _ = std::fs::remove_file(&many);
        assert!(load_ends(std::path::Path::new("/nonexistent.json")).is_err());
    }

    #[test]
    fn json_output_round_trips_as_json() {
        let d = compare(&snap(100, vec![row(1, 1, "a", 0.0, 1)]), &snap(200, vec![]));
        let v: serde_json::Value = serde_json::from_str(&d.to_json()).expect("valid json");
        assert_eq!(v["procs"][0], 1);
        assert_eq!(v["vanished"][0]["name"], "a");
    }
}
