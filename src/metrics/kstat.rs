//! Where the CPU time went, from `/proc/stat`, and how hard memory is paging,
//! from `/proc/vmstat`.
//!
//! `sysinfo` reports one busy percentage per core, which is the right headline
//! and the wrong diagnosis. On a VPS the question is often *steal* — time the
//! hypervisor gave to somebody else — and on a machine that feels frozen while
//! its CPU graph looks idle it is *iowait*. Neither is in a busy percentage.
//!
//! Likewise swap *usage* says nothing about swap *activity*: a box holding
//! 800 MB of pages nobody has touched since boot is fine, and one moving
//! 50 MB/s in and out of swap is thrashing, at exactly the same gauge reading.

use std::fs;

use serde::{Deserialize, Serialize};

/// The aggregate `cpu` line of `/proc/stat`, in clock ticks since boot.
///
/// `guest` and `guest_nice` are left out on purpose: the kernel already counts
/// them inside `user` and `nice`, so adding them would count that time twice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuTimes {
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
}

impl CpuTimes {
    fn total(&self) -> u64 {
        self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
    }
}

/// Share of all CPU time over the last interval, per mode, in percent. The
/// eight fields sum to 100.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuModes {
    pub user: f64,
    pub nice: f64,
    pub system: f64,
    pub idle: f64,
    pub iowait: f64,
    pub irq: f64,
    pub softirq: f64,
    pub steal: f64,
}

/// The names a `cputime` rule's target may use, in the order the panel and
/// the exporter list them.
pub const CPU_MODES: [&str; 8] =
    ["user", "nice", "system", "iowait", "irq", "softirq", "steal", "idle"];

impl CpuModes {
    /// The percentage spent between two readings, or `None` when no time
    /// passed or a counter went backwards (a reset, or two different hosts).
    pub fn between(prev: &CpuTimes, cur: &CpuTimes) -> Option<CpuModes> {
        let d = |a: u64, b: u64| b.checked_sub(a);
        let parts = [
            d(prev.user, cur.user)?,
            d(prev.nice, cur.nice)?,
            d(prev.system, cur.system)?,
            d(prev.idle, cur.idle)?,
            d(prev.iowait, cur.iowait)?,
            d(prev.irq, cur.irq)?,
            d(prev.softirq, cur.softirq)?,
            d(prev.steal, cur.steal)?,
        ];
        let total = d(prev.total(), cur.total())?;
        if total == 0 {
            return None;
        }
        let pct = |v: u64| v as f64 * 100.0 / total as f64;
        Some(CpuModes {
            user: pct(parts[0]),
            nice: pct(parts[1]),
            system: pct(parts[2]),
            idle: pct(parts[3]),
            iowait: pct(parts[4]),
            irq: pct(parts[5]),
            softirq: pct(parts[6]),
            steal: pct(parts[7]),
        })
    }

    /// The share a mode name refers to. An empty name is everything that is
    /// not idle — iowait included, since a CPU waiting on a disk is not one
    /// free to do other work — which is what `cputime>90` should mean.
    pub fn get(&self, mode: &str) -> Option<f64> {
        Some(match mode.trim().to_ascii_lowercase().as_str() {
            "" | "busy" => 100.0 - self.idle,
            "user" | "usr" => self.user,
            "nice" => self.nice,
            "system" | "sys" => self.system,
            "idle" => self.idle,
            "iowait" | "wait" => self.iowait,
            "irq" => self.irq,
            "softirq" => self.softirq,
            "steal" => self.steal,
            _ => return None,
        })
    }
}

/// Parse the aggregate `cpu ` line. Kernels before 2.6.11 have no `steal`
/// column; a missing trailing column reads as zero rather than failing.
pub fn parse_cpu_times(stat: &str) -> Option<CpuTimes> {
    let line = stat.lines().find(|l| l.starts_with("cpu "))?;
    let mut it = line.split_whitespace().skip(1).map(|v| v.parse::<u64>());
    let mut next = || it.next().unwrap_or(Ok(0)).ok();
    Some(CpuTimes {
        user: next()?,
        nice: next()?,
        system: next()?,
        idle: next()?,
        iowait: next()?,
        irq: next()?,
        softirq: next()?,
        steal: next()?,
    })
}

/// The `/proc/vmstat` counters crabmon reads, in pages or events since boot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VmCounters {
    pub pswpin: u64,
    pub pswpout: u64,
    pub pgmajfault: u64,
    /// Since 4.13. `None` before then, which is not the same as zero.
    pub oom_kill: Option<u64>,
}

pub fn parse_vmstat(content: &str) -> Option<VmCounters> {
    let mut out = VmCounters::default();
    let mut seen = 0;
    for line in content.lines() {
        let Some((key, value)) = line.split_once(' ') else { continue };
        let Ok(v) = value.trim().parse::<u64>() else { continue };
        match key {
            "pswpin" => out.pswpin = v,
            "pswpout" => out.pswpout = v,
            "pgmajfault" => out.pgmajfault = v,
            "oom_kill" => out.oom_kill = Some(v),
            _ => continue,
        }
        seen += 1;
    }
    (seen > 0).then_some(out)
}

/// Paging over the last interval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VmActivity {
    /// Bytes read back in from swap per second. The figure that means the
    /// machine is short of memory *now*, not that it once was.
    pub swap_in_bps: f64,
    /// Bytes written out to swap per second.
    pub swap_out_bps: f64,
    /// Major page faults per second: pages that had to come from disk, swap
    /// or file-backed, before a process could continue.
    pub major_faults_ps: f64,
    /// OOM kills since boot, from the kernel's own counter. Readable where
    /// `/dev/kmsg` is not, so it is the one OOM signal an unprivileged
    /// crabmon always has.
    pub oom_kills: Option<u64>,
}

/// The names a `vm` rule's target may use. An empty target is swap traffic
/// in both directions.
pub const VM_SERIES: [&str; 4] = ["swap", "swapin", "swapout", "majfault"];

impl VmActivity {
    pub fn between(
        prev: &VmCounters,
        cur: &VmCounters,
        secs: f64,
        page_size: u64,
    ) -> Option<VmActivity> {
        if secs <= 0.0 {
            return None;
        }
        let r = |a: u64, b: u64| super::rate(a, b, secs);
        Some(VmActivity {
            swap_in_bps: r(prev.pswpin, cur.pswpin) * page_size as f64,
            swap_out_bps: r(prev.pswpout, cur.pswpout) * page_size as f64,
            major_faults_ps: r(prev.pgmajfault, cur.pgmajfault),
            oom_kills: cur.oom_kill,
        })
    }

    /// The figure a `vm` rule's target names, with its unit.
    pub fn get(&self, series: &str) -> Option<(f64, &'static str)> {
        Some(match series.trim().to_ascii_lowercase().as_str() {
            "" | "swap" => (self.swap_in_bps + self.swap_out_bps, " B/s"),
            "swapin" => (self.swap_in_bps, " B/s"),
            "swapout" => (self.swap_out_bps, " B/s"),
            "majfault" => (self.major_faults_ps, "/s"),
            _ => return None,
        })
    }
}

/// Whether `target` names something a `vm` rule can measure.
pub fn vm_target_is_known(target: &str) -> bool {
    VmActivity::default().get(target).is_some()
}

/// Whether `target` names a CPU mode a `cputime` rule can measure.
pub fn cpu_mode_is_known(target: &str) -> bool {
    CpuModes::default().get(target).is_some()
}

/// Keeps the previous reading of both files so each sample is a rate.
#[derive(Debug)]
pub struct KstatReader {
    prev_cpu: Option<CpuTimes>,
    prev_vm: Option<VmCounters>,
    page_size: u64,
}

impl Default for KstatReader {
    fn default() -> Self {
        Self::new()
    }
}

impl KstatReader {
    /// Primed with a first reading, so the first sample is already a rate.
    pub fn new() -> Self {
        Self { prev_cpu: read_cpu(), prev_vm: read_vm(), page_size: page_size() }
    }

    pub fn sample(&mut self, secs: f64) -> (Option<CpuModes>, Option<VmActivity>) {
        let cpu = read_cpu();
        let vm = read_vm();
        let modes = match (&self.prev_cpu, &cpu) {
            (Some(p), Some(c)) => CpuModes::between(p, c),
            _ => None,
        };
        let activity = match (&self.prev_vm, &vm) {
            (Some(p), Some(c)) => VmActivity::between(p, c, secs, self.page_size),
            _ => None,
        };
        // Keep the old reading when a read fails, so one bad read costs one
        // sample rather than turning the next one into a rate since boot.
        if cpu.is_some() {
            self.prev_cpu = cpu;
        }
        if vm.is_some() {
            self.prev_vm = vm;
        }
        (modes, activity)
    }
}

fn read_cpu() -> Option<CpuTimes> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    parse_cpu_times(&fs::read_to_string("/proc/stat").ok()?)
}

fn read_vm() -> Option<VmCounters> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    parse_vmstat(&fs::read_to_string("/proc/vmstat").ok()?)
}

/// `pswpin`/`pswpout` count pages, and a page is 64 KiB on some arm64 and
/// ppc64 kernels, so 4096 would understate their swap traffic sixteenfold.
fn page_size() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sysconf takes no pointers and has no preconditions.
        let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if v > 0 {
            return v as u64;
        }
    }
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    // The first lines of /proc/stat on the development machine.
    const STAT: &str = "cpu  4705 150 1120 16250 520 0 45 210 0 0
cpu0 1393 28 323 4093 120 0 30 50 0 0
intr 114930548 113199788 3 0 5 263 0 4 [...]
";

    #[test]
    fn the_aggregate_line_is_parsed_and_per_core_lines_ignored() {
        let t = parse_cpu_times(STAT).unwrap();
        assert_eq!(t.user, 4705);
        assert_eq!(t.idle, 16250);
        assert_eq!(t.iowait, 520);
        assert_eq!(t.steal, 210);
    }

    #[test]
    fn an_old_kernel_without_steal_reads_it_as_zero() {
        let t = parse_cpu_times("cpu  10 0 5 100 2 0 1\n").unwrap();
        assert_eq!(t.softirq, 1);
        assert_eq!(t.steal, 0);
    }

    #[test]
    fn modes_are_shares_of_the_interval_and_sum_to_a_hundred() {
        let a = CpuTimes { user: 100, idle: 800, iowait: 50, steal: 50, ..Default::default() };
        let b = CpuTimes { user: 140, idle: 840, iowait: 60, steal: 60, ..Default::default() };
        let m = CpuModes::between(&a, &b).unwrap();
        assert_eq!(m.user, 40.0);
        assert_eq!(m.idle, 40.0);
        assert_eq!(m.iowait, 10.0);
        assert_eq!(m.steal, 10.0);
        let sum = m.user + m.nice + m.system + m.idle + m.iowait + m.irq + m.softirq + m.steal;
        assert!((sum - 100.0).abs() < 1e-9);
        assert_eq!(m.get(""), Some(60.0));
        assert_eq!(m.get("steal"), Some(10.0));
        assert_eq!(m.get("bogus"), None);
    }

    #[test]
    fn a_counter_going_backwards_is_no_reading_rather_than_a_huge_one() {
        let a = CpuTimes { user: 100, idle: 100, ..Default::default() };
        let b = CpuTimes { user: 50, idle: 200, ..Default::default() };
        assert_eq!(CpuModes::between(&a, &b), None);
        assert_eq!(CpuModes::between(&a, &a), None);
    }

    const VMSTAT: &str = "nr_free_pages 123
pgpgin 999
pswpin 1000
pswpout 2000
pgmajfault 300
oom_kill 2
";

    #[test]
    fn swap_traffic_is_reported_in_bytes_not_pages() {
        let a = parse_vmstat(VMSTAT).unwrap();
        assert_eq!(a.oom_kill, Some(2));
        let b = VmCounters { pswpin: 1010, pswpout: 2000, pgmajfault: 340, oom_kill: Some(3) };
        let v = VmActivity::between(&a, &b, 2.0, 4096).unwrap();
        assert_eq!(v.swap_in_bps, 5.0 * 4096.0);
        assert_eq!(v.swap_out_bps, 0.0);
        assert_eq!(v.major_faults_ps, 20.0);
        assert_eq!(v.oom_kills, Some(3));
        assert_eq!(v.get("").unwrap().0, 5.0 * 4096.0);
        assert_eq!(v.get("majfault"), Some((20.0, "/s")));
    }

    #[test]
    fn a_kernel_without_the_oom_counter_says_so_rather_than_zero() {
        let v = parse_vmstat("pswpin 1\npswpout 1\npgmajfault 1\n").unwrap();
        assert_eq!(v.oom_kill, None);
        assert_eq!(parse_vmstat("nr_free_pages 5\n"), None);
    }

    #[test]
    fn rule_targets_are_validated_against_what_can_be_measured() {
        for m in CPU_MODES {
            assert!(cpu_mode_is_known(m), "{m}");
        }
        for s in VM_SERIES {
            assert!(vm_target_is_known(s), "{s}");
        }
        assert!(cpu_mode_is_known(""));
        assert!(!cpu_mode_is_known("stael"));
        assert!(!vm_target_is_known("swpin"));
    }
}
