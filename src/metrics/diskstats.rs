//! `/proc/diskstats` reading and — the part that was broken — mapping a
//! filesystem's device name onto the kernel block device that actually has
//! counters. `/dev/mapper/root_crypt` is a symlink to `/dev/dm-0`, and only
//! `dm-0` appears in diskstats.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Cumulative (read_bytes, written_bytes) per kernel device name.
pub type DiskStats = BTreeMap<String, (u64, u64)>;

/// `/proc/diskstats` reports in 512-byte sectors regardless of the device's
/// logical block size.
const SECTOR_BYTES: u64 = 512;

pub fn parse_diskstats(content: &str) -> DiskStats {
    let mut out = BTreeMap::new();
    for line in content.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        // major minor name reads merged sectors_read ms writes merged sectors_written ...
        if f.len() < 10 {
            continue;
        }
        let (Ok(rd), Ok(wr)) = (f[5].parse::<u64>(), f[9].parse::<u64>()) else {
            continue;
        };
        out.insert(
            f[2].to_string(),
            (rd.saturating_mul(SECTOR_BYTES), wr.saturating_mul(SECTOR_BYTES)),
        );
    }
    out
}

/// Parse `iostat -x` output, which is how FreeBSD exposes per-device IO
/// without linking against libdevstat.
///
/// The columns are `device r/s w/s kr/s kw/s ...`; the KB/s rates are already
/// per-second averages since boot, so they are converted to a pseudo-counter by
/// the caller's delta logic being bypassed — see `read_diskstats_freebsd`.
pub fn parse_iostat(content: &str) -> BTreeMap<String, (f64, f64)> {
    let mut out = BTreeMap::new();
    let mut columns: Option<(usize, usize)> = None;
    for line in content.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        // Header row: locate the kr/s and kw/s columns rather than assuming.
        if f[0] == "device" || f[0] == "extended" {
            let kr = f.iter().position(|c| *c == "kr/s");
            let kw = f.iter().position(|c| *c == "kw/s");
            if let (Some(kr), Some(kw)) = (kr, kw) {
                columns = Some((kr, kw));
            }
            continue;
        }
        let Some((kr, kw)) = columns else { continue };
        if f.len() <= kr.max(kw) {
            continue;
        }
        let (Ok(read_kb), Ok(write_kb)) = (f[kr].parse::<f64>(), f[kw].parse::<f64>()) else {
            continue;
        };
        out.insert(f[0].to_string(), (read_kb * 1024.0, write_kb * 1024.0));
    }
    out
}

/// Per-device byte rates, already per-second. Only FreeBSD uses this path;
/// Linux derives rates from the cumulative counters instead.
pub fn read_rates_freebsd() -> BTreeMap<String, (f64, f64)> {
    #[cfg(target_os = "freebsd")]
    {
        use std::process::Command;
        // `-x` gives per-device extended statistics; `-w 1 -c 2` would block for
        // a second, so the single-shot since-boot average is used instead.
        if let Ok(out) = Command::new("iostat").args(["-x"]).output() {
            if out.status.success() {
                return parse_iostat(&String::from_utf8_lossy(&out.stdout));
            }
        }
    }
    BTreeMap::new()
}

pub fn read_diskstats() -> DiskStats {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/diskstats") {
            return parse_diskstats(&content);
        }
    }
    BTreeMap::new()
}

/// Resolve a filesystem device path to the kernel name diskstats knows.
///
/// `resolve` is the symlink resolver (`fs::canonicalize` in production, a stub
/// in tests). Returns the basename of the resolved path, which is `dm-0` for
/// device-mapper volumes and `nvme0n1p3` for plain partitions.
pub fn resolve_device_key<F>(device: &str, stats: &DiskStats, resolve: F) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    let candidates = [resolve(device), Some(device.to_string())];
    for cand in candidates.into_iter().flatten() {
        let base = Path::new(&cand)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or(cand.clone());
        if stats.contains_key(&base) {
            return Some(base);
        }
    }
    None
}

/// Production resolver: follow symlinks under `/dev`.
pub fn canonicalize_dev(path: &str) -> Option<String> {
    fs::canonicalize(path).ok().map(|p| p.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed to the columns crabmon reads; real lines have 20 fields.
    const SAMPLE: &str = "\
 259       0 nvme0n1 141920 30281 9825922 24010 305512 264429 22550432 152441 0 0 0
 259       3 nvme0n1p3 210 0 8642 25 12 0 24 3 0 0 0
 253       0 dm-0 138404 0 9772946 26268 522906 0 22550376 335880 0 0 0
 short line
";

    #[test]
    fn sectors_are_converted_to_bytes() {
        let stats = parse_diskstats(SAMPLE);
        assert_eq!(stats.len(), 3, "the malformed line is skipped");
        assert_eq!(stats["dm-0"], (9_772_946 * 512, 22_550_376 * 512));
        assert_eq!(stats["nvme0n1p3"].0, 8_642 * 512);
    }

    #[test]
    fn non_numeric_counters_are_skipped_not_zeroed() {
        let stats = parse_diskstats(" 8 0 sda x 0 y 0 0 0 z 0 0 0\n");
        assert!(stats.is_empty());
    }

    #[test]
    fn device_mapper_volumes_resolve_to_their_dm_device() {
        // The exact shape of the bug: sysinfo reports the mapper path, and only
        // `dm-0` has counters.
        let stats = parse_diskstats(SAMPLE);
        let resolver =
            |p: &str| (p == "/dev/mapper/nvme0n1p4_crypt").then(|| "/dev/dm-0".to_string());
        assert_eq!(
            resolve_device_key("/dev/mapper/nvme0n1p4_crypt", &stats, resolver),
            Some("dm-0".to_string())
        );
    }

    #[test]
    fn plain_partitions_resolve_without_a_symlink() {
        let stats = parse_diskstats(SAMPLE);
        assert_eq!(
            resolve_device_key("/dev/nvme0n1p3", &stats, |_| None),
            Some("nvme0n1p3".to_string())
        );
    }

    // Verbatim from `iostat -x` on FreeBSD 14. Note the leading "extended
    // device statistics" banner and the column order.
    const IOSTAT: &str = "extended device statistics
device       r/s     w/s     kr/s     kw/s  ms/r  ms/w  ms/o  ms/t qlen  %b
ada0        12.3     4.5    512.0    128.5   0.4   0.9   0.0   0.5    0   2
nvd0         0.0     1.2      0.0     64.0   0.0   0.3   0.0   0.3    0   0
";

    #[test]
    fn freebsd_iostat_columns_are_located_by_name_not_position() {
        let rates = parse_iostat(IOSTAT);
        assert_eq!(rates.len(), 2);
        assert_eq!(rates["ada0"], (512.0 * 1024.0, 128.5 * 1024.0));
        assert_eq!(rates["nvd0"], (0.0, 64.0 * 1024.0));
    }

    #[test]
    fn iostat_output_without_a_header_yields_nothing() {
        // Rather than misreading whatever columns happen to be there.
        assert!(parse_iostat(
            "ada0 1 2 3 4
"
        )
        .is_empty());
        assert!(parse_iostat("").is_empty());
    }

    #[test]
    fn malformed_iostat_rows_are_skipped() {
        let content = "device       r/s     w/s     kr/s     kw/s
ada0  1 2 x y
ada1 1 2 3 4
";
        let rates = parse_iostat(content);
        assert_eq!(rates.len(), 1);
        assert!(rates.contains_key("ada1"));
    }

    #[test]
    fn unknown_devices_resolve_to_nothing_rather_than_a_wrong_match() {
        let stats = parse_diskstats(SAMPLE);
        assert_eq!(resolve_device_key("/dev/sdz1", &stats, |_| None), None);
        assert_eq!(resolve_device_key("tmpfs", &stats, |_| None), None);
    }

    /// The resolver the other tests stand in for. `resolve_device_key` hands it
    /// whatever the mount table says the device is, which on a machine with no
    /// `/dev` at all — a minimal container, or Windows — is not a path.
    #[test]
    fn the_production_symlink_resolver_answers_none_instead_of_failing() {
        assert_eq!(canonicalize_dev("/dev/does-not-exist-crabmon"), None);
        assert_eq!(canonicalize_dev("tmpfs"), None);
        assert_eq!(canonicalize_dev(""), None);

        // A path that does exist comes back absolute, which is what the lookup
        // against /proc/diskstats needs in order to take the last component.
        let real = canonicalize_dev(".").expect("the working directory resolves");
        assert!(real.starts_with('/') || cfg!(windows), "{real} is not absolute");
    }
}
