//! `/proc/diskstats` reading and — the part that was broken — mapping a
//! filesystem's device name onto the kernel block device that actually has
//! counters. `/dev/mapper/root_crypt` is a symlink to `/dev/dm-0`, and only
//! `dm-0` appears in diskstats.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// The cumulative counters one kernel block device publishes.
///
/// Bytes alone cannot say whether a device is *saturated*: an NVMe serving
/// 8 MB/s of 4K random reads is pinned, and a throughput gauge draws it as
/// nearly idle. `io_ticks_ms` and the service times are what answer that, and
/// they arrive on the same line already being parsed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiskCounters {
    pub read_bytes: u64,
    pub write_bytes: u64,
    /// Reads plus writes completed, the denominator of the service time.
    pub ios: u64,
    /// Milliseconds spent servicing reads and writes, summed over requests.
    /// Concurrent requests each contribute, so this can outrun wall time.
    pub io_ms: u64,
    /// Milliseconds during which the queue was non-empty. Bounded by wall
    /// time, which is what makes it a utilisation figure.
    pub io_ticks_ms: u64,
    pub in_flight: u64,
}

/// Cumulative counters per kernel device name.
pub type DiskStats = BTreeMap<String, DiskCounters>;

/// `/proc/diskstats` reports in 512-byte sectors regardless of the device's
/// logical block size.
const SECTOR_BYTES: u64 = 512;

/// What one device was doing over an interval, however the platform measured it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DeviceRates {
    pub read_bps: f64,
    pub write_bps: f64,
    /// Fraction of the interval the device had IO in flight, 0..=1. `None`
    /// where the platform does not report it rather than 0, which would draw
    /// an idle device.
    pub util: Option<f64>,
    /// Mean time from queueing to completion, in milliseconds. `None` when no
    /// request completed in the interval — there is no average of nothing.
    pub await_ms: Option<f64>,
    pub in_flight: u64,
}

impl DiskCounters {
    /// Rates between two readings of the same device.
    ///
    /// `util` is `Δio_ticks / Δwall`, clamped: the kernel samples the queue on
    /// its own timer, so a busy device routinely reports a few milliseconds
    /// more than the interval actually held.
    pub fn rates(prev: &DiskCounters, cur: &DiskCounters, secs: f64) -> DeviceRates {
        let ios = cur.ios.saturating_sub(prev.ios);
        let busy_ms = super::rate(prev.io_ticks_ms, cur.io_ticks_ms, secs) * secs;
        let service_ms = super::rate(prev.io_ms, cur.io_ms, secs) * secs;
        DeviceRates {
            read_bps: super::rate(prev.read_bytes, cur.read_bytes, secs),
            write_bps: super::rate(prev.write_bytes, cur.write_bytes, secs),
            util: (secs > 0.0).then(|| (busy_ms / (secs * 1000.0)).clamp(0.0, 1.0)),
            // A counter reset shows up as zero completions, not as a division
            // by a negative delta.
            await_ms: (ios > 0 && cur.ios >= prev.ios).then(|| service_ms / ios as f64),
            in_flight: cur.in_flight,
        }
    }
}

pub fn parse_diskstats(content: &str) -> DiskStats {
    let mut out = BTreeMap::new();
    for line in content.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        // major minor name reads merged sectors_read ms_reading writes merged
        // sectors_written ms_writing in_flight io_ticks weighted_io_ticks
        if f.len() < 10 {
            continue;
        }
        let (Ok(rd), Ok(wr)) = (f[5].parse::<u64>(), f[9].parse::<u64>()) else {
            continue;
        };
        // Fields past sectors_written have been there since Linux 2.6, but a
        // short line must still yield the bytes rather than nothing: that is
        // all the pre-0.7 parser ever read, and the tests pin trimmed samples.
        let num = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        out.insert(
            f[2].to_string(),
            DiskCounters {
                read_bytes: rd.saturating_mul(SECTOR_BYTES),
                write_bytes: wr.saturating_mul(SECTOR_BYTES),
                ios: num(3).saturating_add(num(7)),
                io_ms: num(6).saturating_add(num(10)),
                io_ticks_ms: num(12),
                in_flight: num(11),
            },
        );
    }
    out
}

/// Parse `iostat -x` output, which is how FreeBSD exposes per-device IO
/// without linking against libdevstat.
///
/// The columns are `device r/s w/s kr/s kw/s ... ms/t qlen %b`; the rates are
/// already per-second averages since boot, and `%b` is the utilisation figure
/// Linux derives from `io_ticks`.
pub fn parse_iostat(content: &str) -> BTreeMap<String, DeviceRates> {
    let mut out = BTreeMap::new();
    let mut columns: Option<(usize, usize)> = None;
    let mut extra: (Option<usize>, Option<usize>, Option<usize>) = (None, None, None);
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
                extra = (
                    f.iter().position(|c| *c == "%b"),
                    f.iter().position(|c| *c == "ms/t"),
                    f.iter().position(|c| *c == "qlen"),
                );
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
        let at = |i: Option<usize>| i.and_then(|i| f.get(i)).and_then(|v| v.parse::<f64>().ok());
        out.insert(
            f[0].to_string(),
            DeviceRates {
                read_bps: read_kb * 1024.0,
                write_bps: write_kb * 1024.0,
                util: at(extra.0).map(|b| (b / 100.0).clamp(0.0, 1.0)),
                await_ms: at(extra.1),
                in_flight: at(extra.2).unwrap_or(0.0).max(0.0) as u64,
            },
        );
    }
    out
}

/// Per-device rates, already per-second. Only FreeBSD uses this path; Linux
/// derives rates from the cumulative counters instead.
pub fn read_rates_freebsd() -> BTreeMap<String, DeviceRates> {
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

    // Trimmed to the columns crabmon reads; real lines have 20 fields. The
    // last three are in_flight, io_ticks and weighted io_ticks.
    const SAMPLE: &str = "\
 259       0 nvme0n1 141920 30281 9825922 24010 305512 264429 22550432 152441 0 88000 0
 259       3 nvme0n1p3 210 0 8642 25 12 0 24 3 0 40 0
 253       0 dm-0 138404 0 9772946 26268 522906 0 22550376 335880 2 90000 0
 short line
";

    #[test]
    fn sectors_are_converted_to_bytes() {
        let stats = parse_diskstats(SAMPLE);
        assert_eq!(stats.len(), 3, "the malformed line is skipped");
        assert_eq!(stats["dm-0"].read_bytes, 9_772_946 * 512);
        assert_eq!(stats["dm-0"].write_bytes, 22_550_376 * 512);
        assert_eq!(stats["nvme0n1p3"].read_bytes, 8_642 * 512);
    }

    #[test]
    fn the_saturation_counters_are_read_from_the_same_line() {
        let stats = parse_diskstats(SAMPLE);
        let dm = stats["dm-0"];
        assert_eq!(dm.ios, 138_404 + 522_906, "reads and writes completed");
        assert_eq!(dm.io_ms, 26_268 + 335_880, "milliseconds servicing both directions");
        assert_eq!(dm.io_ticks_ms, 90_000, "milliseconds with the queue non-empty");
        assert_eq!(dm.in_flight, 2);
    }

    #[test]
    fn a_device_pinned_by_small_reads_reads_as_busy_even_though_bytes_are_low() {
        // The failure a throughput gauge hides: 4 MB/s, and the queue never
        // empties. Bytes say idle; io_ticks says the disk is the bottleneck.
        let prev =
            DiskCounters { read_bytes: 0, ios: 0, io_ms: 0, io_ticks_ms: 0, ..Default::default() };
        let cur = DiskCounters {
            read_bytes: 4_000_000,
            ios: 1_000,
            io_ms: 9_500,
            io_ticks_ms: 990,
            in_flight: 12,
            ..Default::default()
        };
        let r = DiskCounters::rates(&prev, &cur, 1.0);
        assert_eq!(r.read_bps, 4_000_000.0);
        assert_eq!(r.util, Some(0.99));
        assert_eq!(r.await_ms, Some(9.5), "ten requests deep at ~9.5 ms each");
        assert_eq!(r.in_flight, 12);
    }

    #[test]
    fn utilisation_cannot_exceed_the_interval_it_was_measured_over() {
        // The kernel samples the queue on its own timer, so a busy device
        // routinely reports a few more milliseconds than the interval held.
        let prev = DiskCounters::default();
        let cur = DiskCounters { io_ticks_ms: 1_040, ios: 1, io_ms: 1, ..Default::default() };
        assert_eq!(DiskCounters::rates(&prev, &cur, 1.0).util, Some(1.0));
    }

    #[test]
    fn an_idle_interval_has_no_average_service_time_rather_than_zero() {
        // Nothing completed, so there is no average to report. Zero would draw
        // an instant disk, which is the opposite of "no data".
        let c = DiskCounters { ios: 5, io_ms: 50, io_ticks_ms: 10, ..Default::default() };
        let r = DiskCounters::rates(&c, &c, 1.0);
        assert_eq!(r.await_ms, None);
        assert_eq!(r.util, Some(0.0));
    }

    #[test]
    fn a_counter_reset_produces_a_gap_rather_than_a_negative_latency() {
        // Device removed and re-added: every counter restarts at zero.
        let prev =
            DiskCounters { ios: 9_000, io_ms: 90_000, io_ticks_ms: 8_000, ..Default::default() };
        let cur = DiskCounters { ios: 3, io_ms: 4, io_ticks_ms: 2, ..Default::default() };
        let r = DiskCounters::rates(&prev, &cur, 1.0);
        assert_eq!(r.await_ms, None);
        assert_eq!(r.util, Some(0.0));
        assert_eq!(r.read_bps, 0.0);
    }

    #[test]
    fn a_short_diskstats_line_still_yields_its_byte_counters() {
        // Everything past sectors_written is optional: that is all crabmon read
        // before 0.7, and a kernel that stops short must not lose the bytes.
        let stats = parse_diskstats(" 8 0 sda 1 2 100 4 5 6 200\n");
        assert_eq!(stats["sda"].read_bytes, 100 * 512);
        assert_eq!(stats["sda"].write_bytes, 200 * 512);
        assert_eq!(stats["sda"].io_ticks_ms, 0);
    }

    #[test]
    fn non_numeric_counters_are_skipped_not_zeroed() {
        let stats = parse_diskstats(" 8 0 sda x 0 y 0 0 0 z 0 0 0\n");
        assert!(stats.is_empty());
    }

    #[test]
    fn a_garbled_trailing_field_costs_that_counter_not_the_device() {
        // The bytes parsed; refusing the whole row over an unreadable
        // io_ticks would drop a real device from the panel.
        let stats = parse_diskstats(" 8 0 sda 1 2 100 4 5 6 200 8 9 nonsense\n");
        assert_eq!(stats["sda"].read_bytes, 100 * 512);
        assert_eq!(stats["sda"].io_ticks_ms, 0);
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
        assert_eq!(rates["ada0"].read_bps, 512.0 * 1024.0);
        assert_eq!(rates["ada0"].write_bps, 128.5 * 1024.0);
        assert_eq!(rates["nvd0"].read_bps, 0.0);
        assert_eq!(rates["nvd0"].write_bps, 64.0 * 1024.0);
    }

    #[test]
    fn freebsd_reports_utilisation_directly_as_percent_busy() {
        // `%b` is what `io_ticks` is derived into on Linux, already a
        // percentage, so it needs scaling rather than differencing.
        let rates = parse_iostat(IOSTAT);
        assert_eq!(rates["ada0"].util, Some(0.02));
        assert_eq!(rates["ada0"].await_ms, Some(0.5), "the ms/t column");
        assert_eq!(rates["nvd0"].util, Some(0.0));
    }

    #[test]
    fn iostat_without_the_extended_columns_reports_no_utilisation() {
        // `iostat` without `-x` has the rates but neither %b nor ms/t. Absent
        // must read as unknown, not as an idle disk.
        let rates = parse_iostat("device r/s w/s kr/s kw/s\nada0 1 2 8 4\n");
        assert_eq!(rates["ada0"].read_bps, 8.0 * 1024.0);
        assert_eq!(rates["ada0"].util, None);
        assert_eq!(rates["ada0"].await_ms, None);
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
