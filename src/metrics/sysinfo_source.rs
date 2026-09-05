//! The live host implementation of `MetricSource`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sysinfo::{
    Components, CpuRefreshKind, Disks, MemoryRefreshKind, Networks, ProcessRefreshKind,
    ProcessStatus, RefreshKind, System, UpdateKind, Users,
};

use super::diskstats::{self, DiskStats};
use super::procgroup::CgroupPaths;
use super::{
    cgroup, gpu, meminfo, netclass, power, procgroup, psi, rate, CpuSample, DiskRow, GpuInfo,
    HostInfo, MemSample, MetricSource, NetIface, ProcRow, Sensor, Snapshot,
};

/// Device lists (mounted filesystems, interfaces, sensors) are rescanned on this
/// cadence so hot-plugged disks and freshly created containers show up without
/// a restart. `refresh()` alone only updates entries already in the list.
const RELIST_EVERY: Duration = Duration::from_secs(5);

/// sysinfo needs at least this long between CPU samples for the numbers to mean
/// anything; the UI refuses to go faster.
pub const MIN_REFRESH: Duration = Duration::from_millis(200);

pub struct SysinfoSource {
    sys: System,
    networks: Networks,
    disks: Disks,
    components: Components,
    users: Users,
    last_relist: Instant,

    prev_net: BTreeMap<String, (u64, u64)>,
    prev_diskstats: DiskStats,
    prev_proc_io: HashMap<u32, (u64, u64)>,
    /// device path → kernel name, cached because it hits the filesystem.
    dev_key_cache: HashMap<String, Option<String>>,
    /// pid → cgroup path, cached because reading it for every process on every
    /// tick would mean thousands of extra file reads a second.
    cgroup_paths: CgroupPaths,
    /// pid → (start time, nice). A `getpriority` per process per tick costs a
    /// couple of percent of a core on a busy machine for a value that almost
    /// never changes, so it is refreshed on the relist cadence instead.
    nice_cache: HashMap<u32, (u64, Option<i32>)>,
    prev_rapl_uj: Option<u64>,

    virtual_prefixes: Vec<String>,
    use_nvidia_smi: bool,
    host: HostInfo,
}

fn process_refresh_kind() -> ProcessRefreshKind {
    // `OnlyIfNotSet` keeps the per-tick cost near the old cpu+memory-only refresh
    // while still populating the columns the detail pane needs for new processes.
    ProcessRefreshKind::new()
        .with_cpu()
        .with_memory()
        .with_disk_usage()
        .with_user(UpdateKind::OnlyIfNotSet)
        .with_cmd(UpdateKind::OnlyIfNotSet)
        .with_exe(UpdateKind::OnlyIfNotSet)
        .with_cwd(UpdateKind::OnlyIfNotSet)
}

fn refresh_kind() -> RefreshKind {
    RefreshKind::new()
        .with_cpu(CpuRefreshKind::new().with_cpu_usage().with_frequency())
        .with_memory(MemoryRefreshKind::new().with_ram().with_swap())
        .with_processes(process_refresh_kind())
}

pub fn status_char(status: ProcessStatus) -> char {
    match status {
        ProcessStatus::Run => 'R',
        ProcessStatus::Sleep => 'S',
        ProcessStatus::Idle => 'I',
        ProcessStatus::Zombie => 'Z',
        ProcessStatus::Stop => 'T',
        ProcessStatus::Tracing => 't',
        ProcessStatus::Dead => 'X',
        ProcessStatus::Wakekill => 'K',
        ProcessStatus::Waking => 'W',
        ProcessStatus::Parked => 'P',
        ProcessStatus::LockBlocked => 'L',
        ProcessStatus::UninterruptibleDiskSleep => 'D',
        _ => '?',
    }
}

/// Cached `getpriority`, keyed by PID and start time so a recycled PID cannot
/// inherit the previous process's nice value.
fn nice_of(cache: &mut HashMap<u32, (u64, Option<i32>)>, pid: u32, start_time: u64) -> Option<i32> {
    match cache.get(&pid) {
        Some((seen, nice)) if *seen == start_time => *nice,
        _ => {
            let nice = get_priority(pid);
            cache.insert(pid, (start_time, nice));
            nice
        }
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl SysinfoSource {
    pub fn new(virtual_prefixes: Vec<String>, use_nvidia_smi: bool) -> Self {
        let mut sys = System::new_with_specifics(refresh_kind());
        sys.refresh_specifics(refresh_kind());

        let networks = Networks::new_with_refreshed_list();
        let disks = Disks::new_with_refreshed_list();
        let components = Components::new_with_refreshed_list();
        let users = Users::new_with_refreshed_list();

        let prev_net = networks
            .iter()
            .map(|(n, d)| (n.to_string(), (d.total_received(), d.total_transmitted())))
            .collect();
        let prev_proc_io = sys
            .processes()
            .iter()
            .map(|(pid, p)| {
                let du = p.disk_usage();
                (pid.as_u32(), (du.total_read_bytes, du.total_written_bytes))
            })
            .collect();

        let host = HostInfo {
            hostname: System::host_name().unwrap_or_else(|| "unknown".into()),
            os: System::long_os_version().unwrap_or_else(|| System::name().unwrap_or_default()),
            kernel: System::kernel_version().unwrap_or_default(),
            arch: System::cpu_arch().unwrap_or_default(),
            cpu_brand: sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
            logical_cores: sys.cpus().len(),
            physical_cores: sys.physical_core_count(),
            uptime_secs: System::uptime(),
            boot_time_unix: System::boot_time(),
            load: [0.0; 3],
        };

        Self {
            sys,
            networks,
            disks,
            components,
            users,
            last_relist: Instant::now(),
            prev_net,
            prev_diskstats: diskstats::read_diskstats(),
            prev_proc_io,
            dev_key_cache: HashMap::new(),
            cgroup_paths: CgroupPaths::default(),
            nice_cache: HashMap::new(),
            prev_rapl_uj: None,
            virtual_prefixes,
            use_nvidia_smi,
            host,
        }
    }

    pub fn host(&self) -> &HostInfo {
        &self.host
    }

    fn relist_if_due(&mut self) {
        if self.last_relist.elapsed() < RELIST_EVERY {
            return;
        }
        self.last_relist = Instant::now();
        // Picks up hot-plugged disks, new veth/tun interfaces and new sensors.
        self.networks.refresh_list();
        self.disks.refresh_list();
        self.components.refresh_list();
        self.users.refresh_list();
        self.dev_key_cache.clear();
        // Picks up nice values changed by anything other than crabmon.
        self.nice_cache.clear();
    }

    fn device_key(&mut self, device: &str, stats: &DiskStats) -> Option<String> {
        if let Some(cached) = self.dev_key_cache.get(device) {
            return cached.clone();
        }
        let resolved = diskstats::resolve_device_key(device, stats, diskstats::canonicalize_dev);
        self.dev_key_cache.insert(device.to_string(), resolved.clone());
        resolved
    }

    fn collect_procs(&mut self, secs: f64) -> Vec<ProcRow> {
        let mut seen: HashSet<u32> = HashSet::with_capacity(self.sys.processes().len());
        let mut rows = Vec::with_capacity(self.sys.processes().len());

        for (pid, p) in self.sys.processes() {
            let pid_u = pid.as_u32();
            seen.insert(pid_u);
            let du = p.disk_usage();
            let (read_bps, write_bps) = match self.prev_proc_io.get(&pid_u) {
                Some(&(pr, pw)) => {
                    (rate(pr, du.total_read_bytes, secs), rate(pw, du.total_written_bytes, secs))
                }
                None => (0.0, 0.0),
            };
            self.prev_proc_io.insert(pid_u, (du.total_read_bytes, du.total_written_bytes));

            let user = p
                .user_id()
                .and_then(|uid| self.users.get_user_by_id(uid))
                .map(|u| u.name().to_string());

            let start_time = p.start_time();
            let cgroup_path = self.cgroup_paths.get(pid_u, start_time).map(str::to_string);
            let service = cgroup_path.as_deref().and_then(procgroup::service_of);
            let container = cgroup_path.as_deref().and_then(procgroup::container_of);

            rows.push(ProcRow {
                pid: pid_u,
                ppid: p.parent().map(|p| p.as_u32()),
                name: p.name().to_string(),
                cpu: p.cpu_usage(),
                mem: p.memory(),
                virt: p.virtual_memory(),
                state: status_char(p.status()),
                uid: p.user_id().map(|u| **u),
                user,
                run_time: p.run_time(),
                start_time_unix: p.start_time(),
                threads: p.tasks().map(|t| t.len()),
                read_bps,
                write_bps,
                nice: nice_of(&mut self.nice_cache, pid_u, start_time),
                service,
                container,
                cmd: p.cmd().join(" "),
                exe: p.exe().map(|e| e.to_string_lossy().to_string()).unwrap_or_default(),
                cwd: p.cwd().map(|c| c.to_string_lossy().to_string()).unwrap_or_default(),
            });
        }

        // Drop bookkeeping for processes that have exited so the map does not
        // grow without bound on a long-running session.
        self.prev_proc_io.retain(|pid, _| seen.contains(pid));
        self.cgroup_paths.retain_live(&|pid| seen.contains(&pid));
        self.nice_cache.retain(|pid, _| seen.contains(pid));
        rows
    }

    fn collect_nets(&mut self, secs: f64) -> Vec<NetIface> {
        let mut out = Vec::new();
        let mut next = BTreeMap::new();
        for (name, d) in self.networks.iter() {
            let (rx, tx) = (d.total_received(), d.total_transmitted());
            let (rx_bps, tx_bps) = match self.prev_net.get(name) {
                Some(&(prx, ptx)) => (rate(prx, rx, secs), rate(ptx, tx, secs)),
                None => (0.0, 0.0),
            };
            next.insert(name.to_string(), (rx, tx));
            out.push(NetIface {
                name: name.to_string(),
                rx_bps,
                tx_bps,
                rx_total: rx,
                tx_total: tx,
                mac: d.mac_address().to_string(),
                errors_rx: d.total_errors_on_received(),
                errors_tx: d.total_errors_on_transmitted(),
                virtual_iface: netclass::is_virtual(name, &self.virtual_prefixes),
            });
        }
        self.prev_net = next;
        out.sort_by(|a, b| {
            (b.rx_bps + b.tx_bps)
                .partial_cmp(&(a.rx_bps + a.tx_bps))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (b.rx_total + b.tx_total).cmp(&(a.rx_total + a.tx_total)))
        });
        out
    }

    fn collect_disks(&mut self, secs: f64) -> Vec<DiskRow> {
        let stats = diskstats::read_diskstats();
        let mut io: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        for (dev, &(r, w)) in stats.iter() {
            if let Some(&(pr, pw)) = self.prev_diskstats.get(dev) {
                io.insert(dev.clone(), (rate(pr, r, secs), rate(pw, w, secs)));
            }
        }
        // FreeBSD has no /proc/diskstats; iostat already reports rates, so they
        // are used directly rather than differenced.
        if io.is_empty() {
            io.extend(diskstats::read_rates_freebsd());
        }

        // `disks` is borrowed immutably below, so resolve device keys first.
        let listed: Vec<(String, String, String, String, bool, u64, u64)> = self
            .disks
            .iter()
            .map(|d| {
                (
                    d.mount_point().to_string_lossy().to_string(),
                    d.name().to_string_lossy().to_string(),
                    d.file_system().to_string_lossy().to_string(),
                    format!("{:?}", d.kind()),
                    d.is_removable(),
                    d.total_space(),
                    d.available_space(),
                )
            })
            .collect();

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (mount, device, fs, kind, removable, total, avail) in listed {
            if !seen.insert(mount.clone()) {
                continue;
            }
            let kernel_name = self.device_key(&device, &stats).unwrap_or_default();
            let (read_bps, write_bps) = io.get(&kernel_name).copied().unwrap_or((0.0, 0.0));
            let (inodes_total, inodes_used) = inode_usage(&mount);
            out.push(DiskRow {
                mount,
                device,
                kernel_name,
                fs,
                kind,
                removable,
                total,
                used: total.saturating_sub(avail),
                read_bps,
                write_bps,
                inodes_total,
                inodes_used,
            });
        }
        self.prev_diskstats = stats;
        out
    }

    fn collect_sensors(&self) -> Vec<Sensor> {
        let mut out: Vec<Sensor> = self
            .components
            .iter()
            .map(|c| Sensor {
                label: c.label().to_string(),
                temp: c.temperature(),
                critical: c.critical().filter(|c| *c > 0.0),
            })
            .filter(|s| s.temp.is_finite() && s.temp > 0.0)
            .collect();
        out.sort_by(|a, b| b.temp.partial_cmp(&a.temp).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn collect_gpus(&self) -> Vec<GpuInfo> {
        gpu::read_all(self.use_nvidia_smi)
    }
}

impl MetricSource for SysinfoSource {
    fn snapshot(&mut self, dt: Duration) -> Snapshot {
        let secs = dt.as_secs_f64();
        self.relist_if_due();

        self.sys.refresh_specifics(refresh_kind());
        self.networks.refresh();
        self.disks.refresh();
        self.components.refresh();

        let load = System::load_average();
        let mut host = self.host.clone();
        host.uptime_secs = System::uptime();
        host.load = [load.one, load.five, load.fifteen];
        host.logical_cores = self.sys.cpus().len();

        let cpu = CpuSample {
            per_core: self.sys.cpus().iter().map(|c| c.cpu_usage().clamp(0.0, 100.0)).collect(),
            freq_mhz: self.sys.cpus().iter().map(|c| c.frequency()).collect(),
        };

        let mem = MemSample {
            // sysinfo 0.30 reports bytes, not KiB.
            total: self.sys.total_memory(),
            used: self.sys.used_memory(),
            available: self.sys.available_memory(),
            swap_total: self.sys.total_swap(),
            swap_used: self.sys.used_swap(),
            detail: meminfo::read(),
        };

        Snapshot {
            host,
            cpu,
            mem,
            cgroup: cgroup::read().filter(|c| c.is_interesting()),
            procs: self.collect_procs(secs),
            nets: self.collect_nets(secs),
            disks: self.collect_disks(secs),
            sensors: self.collect_sensors(),
            gpus: self.collect_gpus(),
            psi: psi::read(),
            power: power::read(&mut self.prev_rapl_uj, secs),
            taken_at_unix: now_unix(),
        }
    }
}

/// Send a signal to a process. Unix only; a no-op error elsewhere.
#[cfg(unix)]
pub fn send_signal(pid: u32, sig: nix::sys::signal::Signal) -> std::io::Result<()> {
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), sig)
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
}

#[cfg(not(unix))]
pub fn send_signal(_pid: u32, _sig: ()) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "signals are Unix-only"))
}

/// Change a process's nice value. Unix only.
#[cfg(unix)]
pub fn set_priority(pid: u32, nice: i32) -> std::io::Result<()> {
    // SAFETY: setpriority takes plain integers and reports failure via errno.
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, pid, nice) };
    if rc == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn set_priority(_pid: u32, _nice: i32) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "renice is Unix-only"))
}

/// Read a process's nice value.
///
/// `getpriority` returns -1 both for a genuine nice value of -1 and for an
/// error, so errno must be cleared beforehand to tell them apart. The way to
/// clear it is libc-specific, so only Linux gets the exact answer; elsewhere a
/// process niced to exactly -1 reads as unknown, which costs nothing but the
/// prompt's prefill.
#[cfg(target_os = "linux")]
pub fn get_priority(pid: u32) -> Option<i32> {
    // SAFETY: clearing errno and calling getpriority; neither touches Rust state.
    let v = unsafe {
        *libc::__errno_location() = 0;
        libc::getpriority(libc::PRIO_PROCESS, pid)
    };
    if v == -1 && std::io::Error::last_os_error().raw_os_error().unwrap_or(0) != 0 {
        None
    } else {
        Some(v)
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn get_priority(pid: u32) -> Option<i32> {
    // SAFETY: getpriority takes plain integers and touches no Rust state.
    let v = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid) };
    (v != -1).then_some(v)
}

#[cfg(not(unix))]
pub fn get_priority(_pid: u32) -> Option<i32> {
    None
}

/// Pin a process to a set of CPUs. Linux only.
#[cfg(target_os = "linux")]
pub fn set_affinity(pid: u32, cpus: &[usize]) -> std::io::Result<()> {
    let mut set = nix::sched::CpuSet::new();
    for c in cpus {
        set.set(*c).map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
    }
    nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(pid as i32), &set)
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
}

#[cfg(not(target_os = "linux"))]
pub fn set_affinity(_pid: u32, _cpus: &[usize]) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "affinity is Linux-only"))
}

/// Inode totals for a mount point. Linux/Unix only; zeroes elsewhere.
#[cfg(unix)]
fn inode_usage(mount: &str) -> (u64, u64) {
    use std::ffi::CString;
    let Ok(path) = CString::new(mount) else { return (0, 0) };
    // SAFETY: `statvfs` fills the struct or returns non-zero; the path is a
    // valid NUL-terminated C string for the duration of the call.
    let mut buf: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(path.as_ptr(), &mut buf) };
    if rc != 0 || buf.f_files == 0 {
        return (0, 0);
    }
    let total = buf.f_files as u64;
    (total, total.saturating_sub(buf.f_ffree as u64))
}

#[cfg(not(unix))]
fn inode_usage(_mount: &str) -> (u64, u64) {
    (0, 0)
}

/// Parse a CPU list like `0-3,8,10-11` into indices, rejecting out-of-range CPUs.
pub fn parse_cpu_list(s: &str, max_cpu: usize) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().map_err(|_| format!("bad range: {part}"))?;
                let b: usize = b.trim().parse().map_err(|_| format!("bad range: {part}"))?;
                if a > b {
                    return Err(format!("reversed range: {part}"));
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().map_err(|_| format!("bad cpu: {part}"))?),
        }
    }
    if out.is_empty() {
        return Err("no CPUs selected".into());
    }
    if let Some(bad) = out.iter().find(|c| **c >= max_cpu) {
        return Err(format!("cpu {bad} does not exist (0-{})", max_cpu.saturating_sub(1)));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_accept_ranges_and_singletons() {
        assert_eq!(parse_cpu_list("0-3,8", 16).unwrap(), vec![0, 1, 2, 3, 8]);
        assert_eq!(parse_cpu_list("2", 16).unwrap(), vec![2]);
        assert_eq!(parse_cpu_list(" 1 , 1 , 0 ", 16).unwrap(), vec![0, 1]);
    }

    #[test]
    fn cpu_lists_reject_nonsense_before_it_reaches_the_kernel() {
        assert!(parse_cpu_list("", 16).is_err());
        assert!(parse_cpu_list("3-1", 16).is_err());
        assert!(parse_cpu_list("x", 16).is_err());
        assert!(parse_cpu_list("0-99", 16).is_err(), "must not pin to CPUs that do not exist");
    }

    #[test]
    fn status_letters_match_the_ps_convention() {
        assert_eq!(status_char(ProcessStatus::Run), 'R');
        assert_eq!(status_char(ProcessStatus::Sleep), 'S');
        assert_eq!(status_char(ProcessStatus::Zombie), 'Z');
        assert_eq!(status_char(ProcessStatus::UninterruptibleDiskSleep), 'D');
    }

    #[test]
    fn the_refresh_floor_matches_what_sysinfo_requires() {
        assert!(MIN_REFRESH >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    }
}
