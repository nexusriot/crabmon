//! The metric layer: plain data types plus a `MetricSource` trait so the whole
//! app can be driven from a fixture in tests instead of the live host.

pub mod cgroup;
pub mod diskstats;
pub mod gpu;
pub mod meminfo;
pub mod netclass;
pub mod power;
pub mod procgroup;
pub mod psi;
pub mod sockets;
pub mod sysinfo_source;

use serde::{Deserialize, Serialize};

pub use cgroup::CgroupInfo;
pub use gpu::GpuInfo;
pub use meminfo::MemDetail;
pub use power::PowerSample;
pub use procgroup::GroupBy;
pub use psi::PsiSample;
pub use sysinfo_source::SysinfoSource;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HostInfo {
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub arch: String,
    pub cpu_brand: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub uptime_secs: u64,
    pub boot_time_unix: u64,
    pub load: [f64; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuSample {
    /// Per-core usage, 0..100.
    pub per_core: Vec<f32>,
    /// Per-core frequency in MHz (empty when the platform does not report it).
    pub freq_mhz: Vec<u64>,
}

impl CpuSample {
    pub fn avg(&self) -> f64 {
        if self.per_core.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.per_core.iter().sum();
        (sum as f64 / self.per_core.len() as f64).clamp(0.0, 100.0)
    }

    pub fn max_freq(&self) -> Option<u64> {
        self.freq_mhz.iter().copied().max().filter(|f| *f > 0)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MemSample {
    pub total: u64,
    pub used: u64,
    pub available: u64,
    pub swap_total: u64,
    pub swap_used: u64,
    /// The buffers/cache breakdown that a used/total ratio hides.
    pub detail: MemDetail,
}

impl MemSample {
    pub fn ratio(&self) -> f64 {
        ratio(self.used, self.total)
    }
    pub fn swap_ratio(&self) -> f64 {
        ratio(self.swap_used, self.swap_total)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcRow {
    pub pid: u32,
    pub ppid: Option<u32>,
    pub name: String,
    pub cpu: f32,
    pub mem: u64,
    pub virt: u64,
    /// Single-letter process state: R, S, D, Z, T, I, ?
    pub state: char,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<u32>,
    /// Seconds since the process started.
    pub run_time: u64,
    pub start_time_unix: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    pub read_bps: f64,
    pub write_bps: f64,
    /// Scheduling priority, where the platform exposes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nice: Option<i32>,
    /// systemd unit owning the process, e.g. `docker.service`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service: Option<String>,
    /// Short container id, when the process lives in one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub cmd: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub exe: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub cwd: String,
}

impl ProcRow {
    pub fn io_bps(&self) -> f64 {
        self.read_bps + self.write_bps
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NetIface {
    pub name: String,
    pub rx_bps: f64,
    pub tx_bps: f64,
    pub rx_total: u64,
    pub tx_total: u64,
    pub mac: String,
    pub errors_rx: u64,
    pub errors_tx: u64,
    /// True for loopback, bridges, veth pairs, tunnels — excluded from the
    /// aggregate by default so VPN traffic is not counted twice.
    pub virtual_iface: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DiskRow {
    pub mount: String,
    pub device: String,
    /// Kernel block-device name (`dm-0`, `nvme0n1p3`) once resolved.
    pub kernel_name: String,
    pub fs: String,
    pub kind: String,
    pub removable: bool,
    pub total: u64,
    pub used: u64,
    pub read_bps: f64,
    pub write_bps: f64,
    /// Inodes can run out long before bytes do.
    pub inodes_total: u64,
    pub inodes_used: u64,
}

impl DiskRow {
    pub fn ratio(&self) -> f64 {
        ratio(self.used, self.total)
    }

    pub fn inode_ratio(&self) -> f64 {
        ratio(self.inodes_used, self.inodes_total)
    }

    /// Inode exhaustion is only worth showing when it is worse than the bytes.
    pub fn inodes_are_the_problem(&self) -> bool {
        self.inodes_total > 0 && self.inode_ratio() > self.ratio() + 0.1
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Sensor {
    pub label: String,
    pub temp: f32,
    pub critical: Option<f32>,
}

impl Sensor {
    /// Grouping key: the driver prefix, e.g. `coretemp Core 8` → `coretemp`.
    pub fn group(&self) -> &str {
        self.label.split_whitespace().next().unwrap_or(&self.label)
    }
}

/// One row of the sensor panel once per-core readings are collapsed.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorGroup {
    pub label: String,
    pub max_temp: f32,
    pub avg_temp: f32,
    pub critical: Option<f32>,
    pub count: usize,
}

/// Collapse `coretemp Core 0..N` style floods into one row per driver.
pub fn group_sensors(sensors: &[Sensor]) -> Vec<SensorGroup> {
    let mut order: Vec<String> = Vec::new();
    let mut acc: std::collections::HashMap<String, Vec<&Sensor>> = std::collections::HashMap::new();
    for s in sensors {
        let key = s.group().to_string();
        if !acc.contains_key(&key) {
            order.push(key.clone());
        }
        acc.entry(key).or_default().push(s);
    }
    let mut out: Vec<SensorGroup> = order
        .into_iter()
        .map(|key| {
            let members = &acc[&key];
            let max_temp = members.iter().map(|s| s.temp).fold(f32::MIN, f32::max);
            let sum: f32 = members.iter().map(|s| s.temp).sum();
            let critical = members.iter().filter_map(|s| s.critical).fold(None, |acc, c| {
                Some(match acc {
                    Some(a) => f32::min(a, c),
                    None => c,
                })
            });
            SensorGroup {
                label: key,
                max_temp,
                avg_temp: sum / members.len() as f32,
                critical,
                count: members.len(),
            }
        })
        .collect();
    out.sort_by(|a, b| b.max_temp.partial_cmp(&a.max_temp).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// One complete poll of the host. The UI renders only from this.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Snapshot {
    pub host: HostInfo,
    pub cpu: CpuSample,
    pub mem: MemSample,
    pub cgroup: Option<CgroupInfo>,
    pub procs: Vec<ProcRow>,
    pub nets: Vec<NetIface>,
    pub disks: Vec<DiskRow>,
    pub sensors: Vec<Sensor>,
    pub gpus: Vec<GpuInfo>,
    pub psi: PsiSample,
    pub power: PowerSample,
    pub taken_at_unix: u64,
}

impl Snapshot {
    /// Aggregate network rate, optionally counting virtual interfaces.
    pub fn net_totals(&self, include_virtual: bool) -> (f64, f64) {
        self.nets
            .iter()
            .filter(|n| include_virtual || !n.virtual_iface)
            .fold((0.0, 0.0), |(rx, tx), n| (rx + n.rx_bps, tx + n.tx_bps))
    }

    pub fn disk_io_totals(&self) -> (f64, f64) {
        self.disks.iter().fold((0.0, 0.0), |(r, w), d| (r + d.read_bps, w + d.write_bps))
    }
}

/// Anything that can produce a `Snapshot`. Implemented by `SysinfoSource` for
/// the real host, `ReplaySource` for a recording, `RemoteSource` for another
/// machine, and by fixtures in the test-suite.
pub trait MetricSource {
    fn snapshot(&mut self, dt: std::time::Duration) -> Snapshot;

    /// `(position, length)` for sources backed by a finite timeline. Live
    /// sources return `None`, which is what makes the scrub keys inert for them.
    fn timeline(&self) -> Option<(usize, usize)> {
        None
    }

    /// Jump to a position on the timeline. A no-op for live sources.
    fn seek(&mut self, _position: usize) {}

    /// Read the current frame *without* advancing. Scrubbing needs this:
    /// calling `snapshot` after a `seek` would step past the frame just sought.
    /// Live sources have nothing to peek at.
    fn peek(&self) -> Option<Snapshot> {
        None
    }

    /// A one-line description for the status bar, e.g. the recording's path.
    fn label(&self) -> Option<String> {
        None
    }

    /// Identifies the frame `snapshot` last returned, for sources that sample
    /// asynchronously and hand the same frame out more than once.
    ///
    /// `None` — the default, and what every in-process source returns — means
    /// every call produces a genuinely new sample. `App` uses this to avoid
    /// recording one sample into the history several times while a slow source
    /// is still working on the next.
    fn frame_id(&self) -> Option<u64> {
        None
    }
}

pub fn ratio(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64).clamp(0.0, 1.0)
    }
}

/// Rate of a monotonic counter, guarding against counter resets and dt == 0.
pub fn rate(prev: u64, cur: u64, secs: f64) -> f64 {
    if secs <= 0.0 || cur < prev {
        return 0.0;
    }
    (cur - prev) as f64 / secs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_and_ratios_survive_empty_inputs() {
        assert_eq!(CpuSample::default().avg(), 0.0);
        assert_eq!(MemSample::default().ratio(), 0.0);
        assert_eq!(ratio(5, 0), 0.0);
        assert_eq!(ratio(50, 100), 0.5);
        // Used > total (cgroup edge case) must not overflow the gauge.
        assert_eq!(ratio(200, 100), 1.0);
    }

    #[test]
    fn rate_ignores_counter_resets_and_zero_intervals() {
        assert_eq!(rate(100, 200, 1.0), 100.0);
        assert_eq!(rate(200, 100, 1.0), 0.0, "counter reset must not go negative");
        assert_eq!(rate(0, 100, 0.0), 0.0);
    }

    #[test]
    fn sensor_grouping_collapses_per_core_floods() {
        let sensors: Vec<Sensor> = (0..20)
            .map(|i| Sensor {
                label: format!("coretemp Core {i}"),
                temp: 50.0 + i as f32,
                critical: Some(110.0),
            })
            .chain([Sensor {
                label: "nvme Composite x temp1".into(),
                temp: 39.0,
                critical: Some(84.0),
            }])
            .chain([Sensor { label: "acpitz temp1".into(), temp: 66.0, critical: None }])
            .collect();

        let groups = group_sensors(&sensors);
        assert_eq!(groups.len(), 3, "one row per driver");
        assert_eq!(groups[0].label, "coretemp");
        assert_eq!(groups[0].count, 20);
        assert_eq!(groups[0].max_temp, 69.0);
        // acpitz (66) must still be visible above nvme (39) — the old flat list
        // pushed both off the panel entirely.
        assert_eq!(groups[1].label, "acpitz");
        assert_eq!(groups[2].label, "nvme");
        assert_eq!(groups[2].critical, Some(84.0));
    }

    #[test]
    fn aggregate_network_excludes_virtual_interfaces_by_default() {
        let snap = Snapshot {
            nets: vec![
                NetIface {
                    name: "wlp67s0".into(),
                    rx_bps: 1000.0,
                    tx_bps: 100.0,
                    ..Default::default()
                },
                NetIface {
                    name: "lo".into(),
                    rx_bps: 9000.0,
                    tx_bps: 9000.0,
                    virtual_iface: true,
                    ..Default::default()
                },
                NetIface {
                    name: "wg0".into(),
                    rx_bps: 900.0,
                    tx_bps: 90.0,
                    virtual_iface: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(snap.net_totals(false), (1000.0, 100.0));
        assert_eq!(snap.net_totals(true), (10900.0, 9190.0));
    }
}
