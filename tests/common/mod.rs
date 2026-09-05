//! Shared fixtures: a `MetricSource` that replays canned snapshots, so the
//! whole application can be driven without touching the host.

use std::time::Duration;

use crabmon::metrics::meminfo::MemDetail;
use crabmon::metrics::power::{Battery, PowerSample};
use crabmon::metrics::psi::{Pressure, PressureLine, PsiSample};
use crabmon::metrics::{
    CgroupInfo, CpuSample, DiskRow, GpuInfo, HostInfo, MemSample, MetricSource, NetIface, ProcRow,
    Sensor, Snapshot,
};

pub struct FakeSource {
    pub frames: Vec<Snapshot>,
    pub calls: usize,
}

impl FakeSource {
    pub fn new(frames: Vec<Snapshot>) -> Self {
        Self { frames, calls: 0 }
    }

    pub fn single(snap: Snapshot) -> Self {
        Self::new(vec![snap])
    }
}

impl MetricSource for FakeSource {
    fn snapshot(&mut self, _dt: Duration) -> Snapshot {
        let snap = self.frames[self.calls.min(self.frames.len() - 1)].clone();
        self.calls += 1;
        snap
    }
}

pub fn proc_row(pid: u32, ppid: Option<u32>, name: &str, cpu: f32, mem: u64) -> ProcRow {
    ProcRow {
        pid,
        ppid,
        name: name.into(),
        cpu,
        mem,
        virt: mem * 3,
        state: 'S',
        user: Some("vlad".into()),
        uid: Some(1000),
        run_time: 100 + pid as u64,
        start_time_unix: 1_700_000_000 + pid as u64,
        threads: Some(1),
        read_bps: 0.0,
        write_bps: 0.0,
        nice: Some(0),
        service: Some(format!("{name}.service")),
        container: None,
        cmd: format!("/usr/bin/{name} --flag"),
        exe: format!("/usr/bin/{name}"),
        cwd: "/home/vlad".into(),
    }
}

/// A representative host: 4 cores, 16 GiB, a few processes, one of each panel.
pub fn snapshot() -> Snapshot {
    Snapshot {
        host: HostInfo {
            hostname: "testbox".into(),
            os: "Debian 12".into(),
            kernel: "6.1.0".into(),
            arch: "x86_64".into(),
            cpu_brand: "Test CPU".into(),
            logical_cores: 4,
            physical_cores: Some(2),
            uptime_secs: 16_020,
            boot_time_unix: 1_700_000_000,
            load: [0.86, 1.2, 0.9],
        },
        cpu: CpuSample {
            per_core: vec![10.0, 55.0, 95.0, 2.0],
            freq_mhz: vec![2400, 2400, 3800, 800],
        },
        mem: MemSample {
            total: 16_035_467_264,
            used: 5_866_434_560,
            available: 9_000_000_000,
            swap_total: 4_294_967_296,
            swap_used: 1_073_741_824,
            detail: MemDetail {
                buffers: 1_038_405_632,
                cached: 4_862_152_704,
                sreclaimable: 517_660_672,
                shmem: 352_206_848,
                dirty: 1_425_408,
                writeback: 0,
                swap_cached: 0,
            },
        },
        cgroup: Some(CgroupInfo {
            path: "/docker/abc".into(),
            containerized: true,
            mem_current: Some(536_870_912),
            mem_max: Some(2_147_483_648),
            cpu_quota_cores: Some(1.5),
        }),
        procs: vec![
            proc_row(1, None, "systemd", 0.1, 12_000_000),
            proc_row(100, Some(1), "sshd", 0.0, 8_000_000),
            proc_row(200, Some(100), "bash", 0.5, 4_000_000),
            proc_row(300, Some(200), "firefox", 87.5, 475_815_936),
            proc_row(400, Some(1), "kworker/3:1", 1.5, 0),
            // A thread of firefox: the sampler reports no task count for these.
            ProcRow { threads: None, ..proc_row(301, Some(300), "StreamTrans", 0.2, 0) },
        ],
        nets: vec![
            NetIface {
                name: "wlp67s0".into(),
                rx_bps: 125_000.0,
                tx_bps: 25_000.0,
                rx_total: 461_468_749,
                tx_total: 12_345_678,
                mac: "70:08:94:87:de:ef".into(),
                errors_rx: 0,
                errors_tx: 0,
                virtual_iface: false,
            },
            NetIface {
                name: "lo".into(),
                rx_bps: 900_000.0,
                tx_bps: 900_000.0,
                rx_total: 13_959_104,
                tx_total: 13_959_104,
                mac: "00:00:00:00:00:00".into(),
                errors_rx: 0,
                errors_tx: 0,
                virtual_iface: true,
            },
        ],
        disks: vec![
            DiskRow {
                mount: "/".into(),
                device: "/dev/mapper/root_crypt".into(),
                kernel_name: "dm-0".into(),
                fs: "ext4".into(),
                kind: "SSD".into(),
                removable: false,
                total: 500_000_000_000,
                used: 250_000_000_000,
                read_bps: 2048.0,
                write_bps: 1_048_576.0,
                inodes_total: 31_121_408,
                inodes_used: 3_286_829,
            },
            DiskRow {
                mount: "/boot".into(),
                device: "/dev/nvme0n1p3".into(),
                kernel_name: "nvme0n1p3".into(),
                fs: "ext4".into(),
                kind: "SSD".into(),
                removable: false,
                total: 1_000_000_000,
                used: 980_000_000,
                read_bps: 0.0,
                write_bps: 0.0,
                inodes_total: 65_536,
                inodes_used: 64_512,
            },
            // Plenty of space, almost no inodes left: the failure mode a
            // bytes-only gauge hides completely.
            DiskRow {
                mount: "/var".into(),
                device: "/dev/mapper/var".into(),
                kernel_name: "dm-1".into(),
                fs: "ext4".into(),
                kind: "SSD".into(),
                removable: false,
                total: 100_000_000_000,
                used: 12_000_000_000,
                read_bps: 0.0,
                write_bps: 0.0,
                inodes_total: 65_536,
                inodes_used: 64_900,
            },
        ],
        sensors: vec![
            Sensor { label: "coretemp Core 0".into(), temp: 54.0, critical: Some(110.0) },
            Sensor { label: "coretemp Core 1".into(), temp: 65.0, critical: Some(110.0) },
            Sensor { label: "nvme Composite temp1".into(), temp: 39.8, critical: Some(84.8) },
            Sensor { label: "acpitz temp1".into(), temp: 66.0, critical: None },
        ],
        gpus: vec![GpuInfo {
            name: "Intel i915".into(),
            vendor: "Intel".into(),
            busy_percent: None,
            vram_used: None,
            vram_total: None,
            temp_c: None,
            freq_mhz: Some(600),
            max_freq_mhz: Some(2400),
        }],
        psi: PsiSample {
            cpu: Some(Pressure {
                some: PressureLine { avg10: 0.0, avg60: 0.01, avg300: 0.0, total: 15_945_564 },
                full: Some(PressureLine::default()),
            }),
            memory: None,
            io: Some(Pressure {
                some: PressureLine { avg10: 2.55, avg60: 2.70, avg300: 2.10, total: 308_957_030 },
                full: Some(PressureLine {
                    avg10: 2.55,
                    avg60: 2.65,
                    avg300: 2.06,
                    total: 304_405_703,
                }),
            }),
        },
        power: PowerSample {
            batteries: vec![Battery {
                name: "BAT0".into(),
                percent: 98.0,
                status: "Discharging".into(),
                power_w: Some(-12.0),
                energy_wh: Some(57.4),
                energy_full_wh: Some(58.75),
                time_remaining_secs: Some(17_220),
                health_percent: Some(97.9),
            }],
            ac_online: Some(false),
            rapl_watts: None,
        },
        taken_at_unix: 1_788_297_407,
    }
}

/// A config with everything deterministic for rendering assertions.
/// No alert rules by default, so tests that assert on the status line are not
/// competing with the shipped defaults.
pub fn config() -> crabmon::Config {
    crabmon::Config { theme: "mono".into(), alerts: vec![], ..Default::default() }
}

pub fn app() -> crabmon::App {
    crabmon::App::new(config(), Box::new(FakeSource::single(snapshot())))
}
