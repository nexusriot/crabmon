//! The live host implementation of `MetricSource`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sysinfo::{
    Components, CpuRefreshKind, Disks, MemoryRefreshKind, Networks, ProcessRefreshKind,
    ProcessStatus, RefreshKind, System, UpdateKind, Users,
};

use super::diskstats::{self, DeviceRates, DiskCounters, DiskStats};
use super::fds::LimitCache;
use super::procgroup::CgroupPaths;
use super::{
    CpuSample, DiskRow, GpuInfo, HostInfo, MemSample, MetricSource, NetIface, ProcRow, Sensor,
    Snapshot, cgroup, containers, fds, gpu, identity, kmsg, kstat, meminfo, netclass, netns,
    netstat, power, procgroup, procnet, psi, rate, smaps, smart, storage, units,
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
    /// `None` where the platform cannot list them; everything else still works.
    networks: Option<Networks>,
    disks: Option<Disks>,
    components: Option<Components>,
    users: Option<Users>,
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
    /// pid → RLIMIT_NOFILE, on the same "read once per process" terms as the
    /// cgroup path: a limit is set at exec and almost never changed after.
    fd_limits: LimitCache,
    /// pid → what it exec'd into, for the processes whose name the sampler is
    /// still reporting from before the hand-off.
    execs: identity::ExecCache,
    /// pid → network namespace, cached on the same "read once per process"
    /// terms as the cgroup path: a process is put in its namespace at start
    /// and practically never moves.
    netns_of: HashMap<u32, (u64, Option<u64>)>,
    /// Cumulative per-namespace byte counters, for the rates.
    netns: netns::NetnsRates,
    /// The namespace crabmon itself is in, so the host's own traffic is not
    /// reported as another container's.
    host_netns: Option<u64>,
    prev_rapl_uj: Option<u64>,
    /// `/proc/stat` and `/proc/vmstat`, kept for the next sample's deltas.
    kstat: kstat::KstatReader,
    /// `/proc/net/snmp` and friends, kept for the next sample's deltas.
    tcp: netstat::TcpReader,
    /// Follows `/dev/kmsg`. Opened once and drained per sample: the device
    /// replays the whole ring on open, which is what makes an OOM kill from
    /// twenty minutes ago visible the moment crabmon starts.
    kmsg: kmsg::KmsgReader,
    /// Last `usage_usec` per cgroup, so its CPU reads as a rate rather than
    /// as a total since the container started.
    prev_cgroup_usage: HashMap<String, cgroup::CgroupCounters>,
    /// Container id → name, from the runtimes' own files.
    container_names: containers::ContainerNames,

    virtual_prefixes: Vec<String>,
    nvidia_smi: Option<gpu::NvidiaSmi>,
    /// `systemctl show`, on its own thread, when `[units]` is on.
    units: Option<units::UnitWatcher>,
    smart: Option<smart::SmartWatcher>,
    /// Count each process's open descriptors. Off by default: it is a
    /// `getdents` sweep per process per sample.
    count_fds: bool,
    /// Read `/proc/<pid>/smaps_rollup` per process. Opt-in for the same
    /// reason the descriptor walk is: one more read per process per sample.
    read_pss: bool,
    /// Read per-process TCP byte rates from `ss`. Opt-in: one `ss` run per
    /// sample.
    read_net: bool,
    net: procnet::NetTracker,
    /// When the counters were last read, so the interval a rate is divided
    /// by can be the real one; see `effective_interval`.
    last_sampled: Option<Instant>,
    host: HostInfo,
    sample_ticks: HashMap<u32, u64>,
    /// Each process's major-fault counter at the last sample, with the start
    /// time it belonged to, so a recycled PID starts a fresh rate.
    prev_majflt: HashMap<u32, (u64, u64)>,
}

/// A container's root filesystem, mounted on the host.
///
/// Docker and containerd mount an `overlay` for every running container under
/// their state directory, and each one reports the size and usage of whatever
/// filesystem holds its upper layer — `/`, again, on most hosts. Listed as a
/// disk each, they pushed real mounts out of the panel and repeated `/`'s
/// figures once per container, in the panel, the diff and the exporter.
/// An overlay mounted *as* `/` is kept: that is crabmon running inside a
/// container, or a live system, and it is the only root there is.
fn is_container_overlay(mount: &str, fs: &str) -> bool {
    fs == "overlay" && mount != "/"
}

fn process_refresh_kind() -> ProcessRefreshKind {
    // `OnlyIfNotSet` keeps the per-tick cost near the old cpu+memory-only refresh
    // while still populating the columns the detail pane needs for new processes.
    ProcessRefreshKind::nothing()
        .with_cpu()
        .with_memory()
        .with_disk_usage()
        // Refreshing the task list became opt-in in sysinfo 0.33; without it
        // `Process::tasks()` goes stale, which is not just a wrong THR
        // column — `threads_are_distinguishable` decides from it whether
        // this platform can tell a thread from a process at all, and so
        // whether `H` has anything to hide.
        .with_tasks()
        .with_user(UpdateKind::OnlyIfNotSet)
        .with_cmd(UpdateKind::OnlyIfNotSet)
        .with_exe(UpdateKind::OnlyIfNotSet)
        .with_cwd(UpdateKind::OnlyIfNotSet)
}

fn refresh_kind() -> RefreshKind {
    RefreshKind::nothing()
        .with_cpu(CpuRefreshKind::nothing().with_usage().with_frequency())
        .with_memory(MemoryRefreshKind::nothing().with_ram().with_swap())
        .with_processes(process_refresh_kind())
}

/// The numeric user id, where the platform has one.
///
/// Unix identifies a user by a number; Windows identifies one by a SID, which
/// is not a number and does not fit the column. `None` there is the honest
/// answer — the user *name* is reported on both — rather than a fabricated 0,
/// which on Unix would read as root.
#[cfg(unix)]
fn uid_of(p: &sysinfo::Process) -> Option<u32> {
    p.user_id().map(|u| **u)
}

#[cfg(not(unix))]
fn uid_of(_p: &sysinfo::Process) -> Option<u32> {
    None
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
    /// Fails only where sysinfo cannot read this platform at all, which since
    /// 0.40 it reports as an error rather than a `System` full of zeros.
    pub fn new(virtual_prefixes: Vec<String>, use_nvidia_smi: bool) -> anyhow::Result<Self> {
        use anyhow::Context;
        let unsupported = "this platform cannot be monitored";
        let _quiet = super::bsd::MuteStderr::new();
        let mut sys = System::new_with_specifics(refresh_kind()).context(unsupported)?;
        sys.refresh_specifics(refresh_kind());

        // The device lists are optional extras: a platform without networks
        // or sensors (sysinfo marks iOS networks unsupported, for one) still
        // has a process table worth showing.
        let networks = Networks::new_with_refreshed_list().ok();
        let disks = Disks::new_with_refreshed_list().ok();
        let components = Components::new_with_refreshed_list().ok();
        let users = Users::new_with_refreshed_list().ok();

        let prev_net = networks
            .iter()
            .flat_map(|n| n.iter())
            .map(|(n, d)| {
                (n.to_string_lossy().to_string(), (d.total_received(), d.total_transmitted()))
            })
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
            hostname: System::host_name().unwrap_or_else(|_| "unknown".into()),
            os: super::bsd::host().os.clone().unwrap_or_else(|| {
                // `uname -v` style strings end in a newline on some systems.
                System::long_os_version()
                    .or_else(|_| System::name())
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            }),
            kernel: super::bsd::host()
                .kernel
                .clone()
                .unwrap_or_else(|| System::kernel_version().unwrap_or_default()),
            arch: System::cpu_arch(),
            cpu_brand: sys
                .cpus()
                .first()
                .map(|c| c.brand().trim().to_string())
                .filter(|b| !b.is_empty())
                .or_else(|| super::bsd::host().cpu_model.clone())
                .unwrap_or_default(),
            logical_cores: sys.cpus().len(),
            physical_cores: System::physical_core_count().ok(),
            uptime_secs: System::uptime().unwrap_or(0),
            boot_time_unix: System::boot_time().unwrap_or(0),
            load: [0.0; 3],
        };

        Ok(Self {
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
            fd_limits: LimitCache::default(),
            execs: identity::ExecCache::default(),
            netns_of: HashMap::new(),
            netns: netns::NetnsRates::default(),
            host_netns: netns::id_of(std::process::id()),
            prev_rapl_uj: None,
            kstat: kstat::KstatReader::new(),
            tcp: netstat::TcpReader::new(),
            kmsg: kmsg::KmsgReader::open(),
            prev_cgroup_usage: HashMap::new(),
            container_names: containers::ContainerNames::default(),
            virtual_prefixes,
            nvidia_smi: use_nvidia_smi.then(gpu::NvidiaSmi::spawn),
            units: None,
            smart: None,
            count_fds: false,
            read_pss: false,
            read_net: false,
            net: procnet::NetTracker::default(),
            last_sampled: None,
            host,
            sample_ticks: HashMap::new(),
            prev_majflt: HashMap::new(),
        })
    }

    /// Ask for the FD column's data. A builder rather than a constructor
    /// argument so every existing caller keeps the cheap behaviour.
    /// Watch systemd units for failures and restart loops. Linux only: there
    /// is no systemd anywhere else.
    pub fn with_units(mut self, settings: Option<units::UnitSettings>) -> SysinfoSource {
        if cfg!(target_os = "linux") {
            self.units = settings.map(units::UnitWatcher::spawn);
        }
        self
    }

    /// Read SMART on a thread of its own; see `[smart]`.
    pub fn with_smart(mut self, settings: Option<smart::SmartSettings>) -> SysinfoSource {
        self.smart = settings.map(smart::SmartWatcher::spawn);
        self
    }

    pub fn with_pss(mut self, read_pss: bool) -> SysinfoSource {
        self.read_pss = read_pss;
        self
    }

    /// Measure each process's TCP traffic; see `[procs] net`.
    pub fn with_net(mut self, read_net: bool) -> SysinfoSource {
        self.read_net = read_net;
        self
    }

    pub fn with_fds(mut self, count_fds: bool) -> SysinfoSource {
        self.count_fds = count_fds;
        self
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
        self.refresh_lists(true);
        if let Some(u) = &mut self.users {
            u.refresh();
        }
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

        // One `ss` run for the whole table, before the loop. `None` where it
        // cannot be run, which leaves every row's figure unmeasured rather
        // than claiming the machine is idle.
        let net_rates = if self.read_net {
            procnet::read().map(|socks| self.net.update(&socks, secs))
        } else {
            None
        };

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
                .and_then(|uid| self.users.as_ref()?.get_user_by_id(uid))
                .map(|u| u.name().to_string())
                .map(|n| super::bsd::user_name(uid_of(p), n));

            let start_time = p.start_time();
            // Both are `None` when the sweep is off, which the UI renders as
            // "not asked for" rather than as a process holding nothing.
            let (fd_count, fd_limit) = if self.count_fds {
                (fds::count(pid_u), self.fd_limits.get(pid_u, start_time))
            } else {
                (None, None)
            };
            // `None` when it was not asked for, so an absent figure is
            // never confused with a process that shares nothing.
            //
            // Threads are skipped: every thread of a process shares one
            // address space, so `smaps_rollup` reports the *process's*
            // figures for each of them. Showing that on a thread row would
            // invite exactly the double-counting PSS exists to remove — and
            // on a machine with thousands of threads it is thousands of
            // reads for numbers already shown one row up.
            let is_process = p.tasks().is_some();
            let rollup = (self.read_pss && is_process).then(|| smaps::read(pid_u)).flatten();
            let netns = match self.netns_of.get(&pid_u) {
                Some((seen, ns)) if *seen == start_time => *ns,
                _ => {
                    let ns = netns::id_of(pid_u);
                    self.netns_of.insert(pid_u, (start_time, ns));
                    ns
                }
            };
            let cgroup_path = self.cgroup_paths.get(pid_u, start_time).map(str::to_string);
            let service = cgroup_path.as_deref().and_then(procgroup::service_of);
            let container = cgroup_path.as_deref().and_then(procgroup::container_of);

            // One read of /proc/<pid>/stat answers both "is this still the
            // process we sampled?" and "how many major faults has it taken?".
            // Threads are skipped: their counters are the process's again.
            let now_stat = super::control::stat_counters(pid_u);
            let major_faults_ps = match now_stat.filter(|_| is_process) {
                Some((ticks, faults)) => match self.prev_majflt.insert(pid_u, (ticks, faults)) {
                    Some((seen, prev)) if seen == ticks => Some(rate(prev, faults, secs)),
                    _ => None,
                },
                None => None,
            };

            // What the process is running *now*. `exec` keeps the PID and the
            // start time, and sysinfo reads a process's name only when it
            // first sees the PID and treats an unchanged start time as
            // "nothing to re-read" — so a wrapper that execs the real program
            // kept the wrapper's name, argv, exe and cwd for as long as it
            // ran. `/proc/<pid>/comm` is one tiny file and always current, so
            // it is checked for every process on every tick; the heavier
            // re-reads happen only when it disagrees, and are remembered.
            let identity::Identity { name, cmd, exe, cwd } = self.execs.resolve(
                pid_u,
                start_time,
                identity::Identity {
                    name: p.name().to_string_lossy().to_string(),
                    cmd: p.cmd().iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" "),
                    exe: p.exe().map(|e| e.to_string_lossy().to_string()).unwrap_or_default(),
                    cwd: p.cwd().map(|c| c.to_string_lossy().to_string()).unwrap_or_default(),
                },
            );

            let name = super::bsd::tidy_name(name, &exe);
            // Only FreeBSD reports an empty argument vector for a retitled
            // process; a Linux kernel thread's empty command line is real.
            let cmd = if cfg!(target_os = "freebsd") {
                super::bsd::fallback_cmd(cmd, &name, &exe)
            } else {
                cmd
            };
            rows.push(ProcRow {
                pid: pid_u,
                ppid: p.parent().map(|p| p.as_u32()),
                name,
                cpu: p.cpu_usage(),
                mem: p.memory(),
                pss: rollup.map(|r| r.pss),
                uss: rollup.map(|r| r.uss),
                swap: rollup.map(|r| r.swap),
                // A process with no sockets is a process moving nothing, which
                // is a measurement; only a failed `ss` leaves it blank.
                net_rx_bps: net_rates
                    .as_ref()
                    .filter(|_| is_process)
                    .map(|m| m.get(&pid_u).map_or(0.0, |r| r.rx_bps)),
                net_tx_bps: net_rates
                    .as_ref()
                    .filter(|_| is_process)
                    .map(|m| m.get(&pid_u).map_or(0.0, |r| r.tx_bps)),
                virt: p.virtual_memory(),
                state: status_char(p.status()),
                uid: uid_of(p),
                user,
                run_time: p.run_time(),
                start_time_unix: p.start_time(),
                // Only attach a precise identity if the incarnation survived
                // the whole refresh. A replacement during collection must not
                // acquire the old row's name and a new process's control token.
                start_time_ticks: self
                    .sample_ticks
                    .get(&pid_u)
                    .copied()
                    .filter(|ticks| now_stat.map(|(t, _)| t) == Some(*ticks)),
                major_faults_ps,
                // `+ 1` for the main thread: sysinfo's `tasks()` is the set
                // of *sub*tasks, so a single-threaded process reports an
                // empty set and every other one is short by itself. Without
                // this the column reads 0 for a live process, which is not a
                // number any process can have. (sysinfo 0.30 included the
                // main task; 0.33 onwards does not.)
                threads: p.tasks().map(|t| t.len() + 1),
                read_bps,
                write_bps,
                nice: nice_of(&mut self.nice_cache, pid_u, start_time),
                fds: fd_count,
                fd_limit,
                service,
                container,
                container_name: None,
                netns,
                cmd,
                exe,
                cwd,
            });
        }

        // Drop bookkeeping for processes that have exited so the map does not
        // grow without bound on a long-running session.
        self.prev_proc_io.retain(|pid, _| seen.contains(pid));
        self.cgroup_paths.retain_live(&|pid| seen.contains(&pid));
        self.nice_cache.retain(|pid, _| seen.contains(pid));
        self.fd_limits.retain_live(&|pid| seen.contains(&pid));
        self.execs.retain_live(&|pid| seen.contains(&pid));
        self.netns_of.retain(|pid, _| seen.contains(pid));
        self.prev_majflt.retain(|pid, _| seen.contains(pid));
        rows
    }

    /// `(name, cmd, exe, cwd)` for one process, correcting the sampler's
    /// cached copies when the process has exec'd since they were taken.
    ///
    /// Split out so the exec check is one place rather than four, and so the
    /// ordinary path — the name is what it always was — is visibly the cheap
    /// one: a single small read, and nothing else.
    /// Per-namespace throughput, read once per namespace rather than once per
    /// process: the counters belong to the namespace, and every process in a
    /// container reports the same ones.
    fn collect_netns(&mut self, procs: &[ProcRow], secs: f64) -> Vec<netns::NetNamespace> {
        let members: Vec<netns::Member<'_>> = procs
            .iter()
            .map(|p| netns::Member {
                pid: p.pid,
                netns: p.netns,
                container: p.container.as_deref(),
            })
            .collect();
        self.netns.collect(&members, self.host_netns, secs, netns::counters_of)
    }

    /// Attach a name to every container id the runtimes' files can name.
    fn name_containers(&mut self, procs: &mut [ProcRow], cgroups: &mut [cgroup::CgroupStat]) {
        let live: HashSet<String> = procs.iter().filter_map(|p| p.container.clone()).collect();
        if live.is_empty() {
            return;
        }
        self.container_names.resolve(&live);
        for p in procs.iter_mut() {
            p.container_name = p
                .container
                .as_deref()
                .and_then(|id| self.container_names.get(id))
                .map(String::from);
        }
        for c in cgroups.iter_mut().filter(|c| c.container) {
            c.container_name = self.container_names.get(&c.name).map(String::from);
        }
    }

    /// Refresh whichever device lists this platform has. `relist` also
    /// discovers new entries, which costs a directory walk each.
    fn refresh_lists(&mut self, relist: bool) {
        if let Some(n) = &mut self.networks {
            n.refresh(relist);
        }
        if let Some(d) = &mut self.disks {
            d.refresh(relist);
        }
        if let Some(c) = &mut self.components {
            c.refresh(relist);
        }
    }

    fn collect_nets(&mut self, secs: f64) -> Vec<NetIface> {
        let mut out = Vec::new();
        let mut next = BTreeMap::new();
        for (name, d) in self.networks.iter().flat_map(|n| n.iter()) {
            // Interface names are `OsString` since sysinfo 0.40; Linux ones
            // are ASCII in practice, and a lossy one still keys consistently.
            let name = name.to_string_lossy();
            let name = name.as_ref();
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
                drops_rx: netclass::sysfs_stat(name, "rx_dropped").unwrap_or(0),
                drops_tx: netclass::sysfs_stat(name, "tx_dropped").unwrap_or(0),
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
        let mut io: BTreeMap<String, DeviceRates> = BTreeMap::new();
        for (dev, cur) in stats.iter() {
            if let Some(prev) = self.prev_diskstats.get(dev) {
                io.insert(dev.clone(), DiskCounters::rates(prev, cur, secs));
            }
        }
        // Where the counters could not be read at all (an `iostat` too old for
        // `-I`), fall back to FreeBSD's since-boot averages: rough, but not blank.
        if stats.is_empty() {
            io.extend(diskstats::read_rates_freebsd());
        }
        let mounted_from = diskstats::read_mount_devices();

        // `disks` is borrowed immutably below, so resolve device keys first.
        let listed: Vec<(String, String, String, String, bool, u64, u64)> = self
            .disks
            .iter()
            .flat_map(|d| d.iter())
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
            // `sysinfo` calls a FreeBSD root filesystem `root`; the mount table
            // knows the device.
            let device = match mounted_from.get(&mount) {
                Some(real) if !device.starts_with("/dev/") && real.starts_with("/dev/") => {
                    real.clone()
                }
                _ => device,
            };
            if is_container_overlay(&mount, &fs) || !seen.insert(mount.clone()) {
                continue;
            }
            let kernel_name = self.device_key(&device, &stats).unwrap_or_default();
            let rates = io.get(&kernel_name).copied().unwrap_or_default();
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
                read_bps: rates.read_bps,
                write_bps: rates.write_bps,
                util: rates.util,
                await_ms: rates.await_ms,
                in_flight: rates.in_flight,
                inodes_total,
                inodes_used,
            });
        }
        self.prev_diskstats = stats;
        out
    }

    /// Read each container's and unit's own cgroup accounting.
    ///
    /// The paths come from the processes just collected, so nothing is
    /// enumerated that is not actually running, and the set is capped: a host
    /// with hundreds of containers must not turn a panel into hundreds of
    /// file reads per refresh.
    fn collect_cgroups(&mut self, secs: f64) -> Vec<cgroup::CgroupStat> {
        let mut paths: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for pid in self.sys.processes().keys() {
            let Some(path) = self.cgroup_paths.peek(pid.as_u32()) else { continue };
            // Only groups that are a container or a unit: the rest are slices
            // and scopes nobody is asking about.
            if procgroup::container_of(path).is_none() && procgroup::service_of(path).is_none() {
                continue;
            }
            if seen.insert(path) {
                paths.push(path.to_string());
                if paths.len() >= cgroup::MAX_CGROUPS {
                    break;
                }
            }
        }

        let mut out = Vec::with_capacity(paths.len());
        let mut usage: HashMap<String, cgroup::CgroupCounters> =
            HashMap::with_capacity(paths.len());
        for path in paths {
            let prev = self.prev_cgroup_usage.get(&path).copied();
            let Some((stat, now)) = cgroup::read_stat(&path, prev, secs) else { continue };
            usage.insert(path.clone(), now);
            if stat.is_interesting() {
                out.push(stat);
            }
        }
        // Dropped groups take their previous usage with them, so a container
        // id reused later starts a fresh rate rather than a huge delta.
        self.prev_cgroup_usage = usage;
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn collect_sensors(&self) -> Vec<Sensor> {
        let mut out: Vec<Sensor> = self
            .components
            .iter()
            .flat_map(|c| c.iter())
            .filter_map(|c| {
                Some(Sensor {
                    label: c.label().to_string(),
                    // `None` is a sensor that could not be read, which is not
                    // a sensor reading zero.
                    temp: c.temperature()?,
                    critical: c.critical().filter(|c| *c > 0.0),
                })
            })
            .filter(|s| s.temp.is_finite() && s.temp > 0.0)
            .collect();
        out.sort_by(|a, b| b.temp.partial_cmp(&a.temp).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn collect_gpus(&self) -> Vec<GpuInfo> {
        gpu::read_all(self.nvidia_smi.as_ref())
    }
}

/// The interval to divide a counter's growth by: the time that really passed
/// between the two reads, never less than the caller's own figure.
///
/// The caller's `dt` is a promise about the gap between samples, and the
/// sampler thread kept it from the wrong clock — measured from when it
/// started, not from when the first sample was read — so the second sample of
/// every TUI run arrived with a gap of a few microseconds for a second's worth
/// of traffic. The result was a first network rate of 305 GiB/s on a 20 GiB
/// interface, which stayed in the chart's history and set its scale until it
/// scrolled out four minutes later. The source knows when it last read the
/// counters, so it checks the figure against that. Larger `dt` is kept — a
/// caller that measured a longer gap is not contradicted — and the first
/// sample, which has nothing to compare, keeps `dt`, which is zero and means
/// "learn the baseline, report no rates".
pub fn effective_interval(last: Option<Instant>, now: Instant, dt: Duration) -> Duration {
    match last {
        Some(prev) if !dt.is_zero() => dt.max(now.saturating_duration_since(prev)),
        _ => dt,
    }
}

impl MetricSource for SysinfoSource {
    fn snapshot(&mut self, dt: Duration) -> Snapshot {
        self.relist_if_due();
        self.kmsg.drain();

        self.sample_ticks = self
            .sys
            .processes()
            .keys()
            .filter_map(|pid| {
                super::control::start_ticks(pid.as_u32()).map(|ticks| (pid.as_u32(), ticks))
            })
            .collect();
        let quiet = super::bsd::MuteStderr::new();
        self.sys.refresh_specifics(refresh_kind());
        drop(quiet);
        self.refresh_lists(false);

        // Stamped here, after the slow part, because this is where the counters
        // below are read. Taken at the top instead, a refresh that stalled for
        // two seconds under heavy IO left the next interval at its nominal
        // quarter-second while the counters had moved for two: a 2.5 GB write
        // showed as 10 GB/s on a disk that does 1.4, and the frame after it as
        // zero. Found on FreeBSD under `dd`; the same arithmetic ran everywhere.
        let started = Instant::now();
        let secs = effective_interval(self.last_sampled, started, dt).as_secs_f64();
        self.last_sampled = Some(started);

        let mut host = self.host.clone();
        host.uptime_secs = System::uptime().unwrap_or(host.uptime_secs);
        // Windows has no load average and says so since sysinfo 0.40.
        host.load = System::load_average().map_or([0.0; 3], |l| [l.one, l.five, l.fifteen]);
        host.logical_cores = self.sys.cpus().len();

        let cpu = CpuSample {
            per_core: self.sys.cpus().iter().map(|c| c.usage().clamp(0.0, 100.0)).collect(),
            // sysinfo has no clock on FreeBSD; the nominal one stands in.
            freq_mhz: self
                .sys
                .cpus()
                .iter()
                .map(|c| match c.frequency() {
                    0 => super::bsd::host().mhz.unwrap_or(0),
                    f => f,
                })
                .collect(),
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

        let (cpu_time, vm) = self.kstat.sample(secs);
        let tcp = self.tcp.sample(secs);
        let mut procs = self.collect_procs(secs);
        let netns = self.collect_netns(&procs, secs);
        let mut cgroups = self.collect_cgroups(secs);
        self.name_containers(&mut procs, &mut cgroups);

        // Not watched is said as such, so a `unit` rule stays quiet and the
        // exporter leaves the series out rather than reporting zero failures
        // about units nobody looked at.
        let units_now = match &self.units {
            Some(w) => w.latest(),
            None if cfg!(target_os = "linux") => {
                (Vec::new(), Some("[units] enabled is off".into()))
            }
            None => (Vec::new(), Some("systemd is Linux-only".into())),
        };
        let mut storage = storage::read();
        let smart_unreadable = match &self.smart {
            Some(w) => {
                let (disks, why) = w.latest();
                storage.extend(disks);
                // Worst first across both, so a failing disk outranks a
                // healthy array in a short panel.
                storage.sort_by_key(|a| std::cmp::Reverse(a.state as u8));
                why
            }
            None => None,
        };
        Snapshot {
            host,
            cpu,
            mem,
            cgroup: cgroup::read().filter(|c| c.is_interesting()),
            cgroups,
            procs,
            nets: self.collect_nets(secs),
            disks: self.collect_disks(secs),
            sensors: self.collect_sensors(),
            gpus: self.collect_gpus(),
            psi: psi::read(),
            power: power::read(&mut self.prev_rapl_uj, secs),
            netns,
            kernel: self.kmsg.events().to_vec(),
            kernel_unreadable: self.kmsg.error.clone(),
            cpu_time,
            vm,
            tcp,
            storage,
            smart_unreadable,
            units: units_now.0,
            units_unreadable: units_now.1,
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
    // `pid as _`, not `pid as i32`: the `who` argument is `id_t` on Linux and
    // `int` on the BSDs, so naming either one breaks the build on the other.
    //
    // SAFETY: setpriority takes plain integers and reports failure via errno.
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, pid as _, nice) };
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
    let v = unsafe { libc::getpriority(libc::PRIO_PROCESS, pid as _) };
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
                // Bound-check *before* materialising: `0-4000000000` would
                // otherwise try to allocate 32 GB and abort the process, which
                // skips the panic hook and leaves the terminal in raw mode.
                if b >= max_cpu {
                    return Err(format!(
                        "cpu {b} does not exist (0-{})",
                        max_cpu.saturating_sub(1)
                    ));
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
    fn container_rootfs_overlays_are_not_disks_but_an_overlay_root_is() {
        let layer = "/var/lib/docker/rootfs/overlayfs/6d537c7115ab7d2b1148bf6afb2445fc";
        assert!(is_container_overlay(layer, "overlay"));
        assert!(!is_container_overlay("/", "overlay"), "inside a container, / is the overlay");
        assert!(!is_container_overlay("/home", "ext4"));
    }

    /// The thread count has to be the number of threads the process has.
    ///
    /// sysinfo's `tasks()` is the set of *sub*tasks and excludes the main
    /// thread — 0.30 included it, 0.33 onwards does not — so the adapter
    /// adds one. A silent off-by-one here reads as 0 threads for a live
    /// process, and it is the sort of thing that only an upstream bump
    /// introduces and only a live check catches.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_thread_count_matches_what_proc_says() {
        use crate::metrics::MetricSource;

        let mut source = SysinfoSource::new(Vec::new(), false).unwrap();
        let snap = source.snapshot(Duration::from_millis(1));

        let mut checked = 0;
        for row in &snap.procs {
            // `None` is a thread, which has no thread count of its own.
            let Some(reported) = row.threads else { continue };
            let Ok(dir) = std::fs::read_dir(format!("/proc/{}/task", row.pid)) else {
                continue; // exited between the sample and now
            };
            let actual = dir.count();
            // Only processes that did not change shape in between.
            if actual == 0 {
                continue;
            }
            assert!(reported > 0, "{} (pid {}) reports no threads at all", row.name, row.pid);
            if reported == actual {
                checked += 1;
            }
        }
        assert!(checked > 0, "no process could be cross-checked against /proc");
    }

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
    fn an_enormous_range_is_rejected_without_being_materialised() {
        // The affinity prompt accepts digits and `-`, so this is three
        // keystrokes away. Expanding it first meant a 32 GB allocation and an
        // abort that skips the panic hook, leaving the terminal in raw mode.
        let start = std::time::Instant::now();
        assert!(parse_cpu_list("0-4000000000", 16).is_err());
        assert!(parse_cpu_list("0-18446744073709551615", 16).is_err());
        assert!(start.elapsed() < Duration::from_secs(2), "the range was expanded anyway");
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

    #[test]
    fn a_tiny_gap_from_the_caller_is_raised_to_the_time_that_really_passed() {
        let t0 = Instant::now();
        let later = t0 + Duration::from_millis(900);
        // The sampler thread's second request: microseconds by its clock,
        // nearly a second by the counters'.
        assert_eq!(
            effective_interval(Some(t0), later, Duration::from_micros(5)),
            Duration::from_millis(900)
        );
        // A caller that measured a longer gap is not contradicted.
        assert_eq!(
            effective_interval(Some(t0), later, Duration::from_secs(2)),
            Duration::from_secs(2)
        );
    }

    #[test]
    fn the_first_sample_and_a_zero_gap_still_mean_no_rates() {
        let now = Instant::now();
        // Nothing to compare against: the caller's figure stands, and zero
        // makes every rate zero rather than the growth since boot.
        assert_eq!(effective_interval(None, now, Duration::ZERO), Duration::ZERO);
        assert_eq!(
            effective_interval(None, now, Duration::from_millis(5)),
            Duration::from_millis(5)
        );
        // Zero asked for explicitly stays zero even with a history.
        let t0 = now - Duration::from_secs(3);
        assert_eq!(effective_interval(Some(t0), now, Duration::ZERO), Duration::ZERO);
        // A clock that appears to run backwards does not underflow.
        assert_eq!(
            effective_interval(Some(now + Duration::from_secs(1)), now, Duration::from_millis(7)),
            Duration::from_millis(7)
        );
    }

    /// The whole path, against the real counters: a second sample taken after
    /// a real pause but announced as microseconds must report the traffic over
    /// the pause, not over the microseconds.
    #[test]
    fn a_second_sample_announced_as_instant_is_divided_by_the_real_gap() {
        let Ok(mut source) = SysinfoSource::new(vec![], false) else { return };
        let first = source.snapshot(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(400));
        let second = source.snapshot(Duration::from_micros(2));
        for n in &second.nets {
            let Some(before) = first.nets.iter().find(|b| b.name == n.name) else { continue };
            let grew = n.rx_total.saturating_sub(before.rx_total) as f64;
            // The rate can be no more than the growth over the pause (a
            // little under 400 ms of it, allowing for the clock). Divided by
            // two microseconds it was two hundred thousand times that.
            assert!(
                n.rx_bps <= grew / 0.3 + 1.0,
                "{}: {} B/s from {} bytes over a 400 ms pause",
                n.name,
                n.rx_bps,
                grew
            );
        }
    }
}
