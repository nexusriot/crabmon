//! cgroup v2/v1 awareness. Inside a container the host's totals say nothing
//! about what this process may use, so when a limit exists crabmon shows the
//! cgroup's own usage and quota in a panel of their own.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CgroupInfo {
    /// The cgroup path from `/proc/self/cgroup`.
    pub path: String,
    pub containerized: bool,
    pub mem_current: Option<u64>,
    /// `None` when the cgroup is unlimited (`max`).
    pub mem_max: Option<u64>,
    /// Effective CPU allowance in cores, e.g. `1.5` for `150000 100000`.
    pub cpu_quota_cores: Option<f64>,
}

impl CgroupInfo {
    pub fn mem_ratio(&self) -> Option<f64> {
        match (self.mem_current, self.mem_max) {
            (Some(c), Some(m)) if m > 0 => Some((c as f64 / m as f64).clamp(0.0, 1.0)),
            _ => None,
        }
    }

    /// Worth showing a panel for? A limitless root cgroup is just noise.
    pub fn is_interesting(&self) -> bool {
        self.containerized || self.mem_max.is_some() || self.cpu_quota_cores.is_some()
    }
}

/// Parse the unified (`0::/path`) or legacy (`N:ctrl:/path`) cgroup file.
pub fn parse_proc_cgroup(content: &str) -> Option<String> {
    let mut legacy = None;
    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() != 3 {
            continue;
        }
        if parts[0] == "0" && parts[1].is_empty() {
            return Some(parts[2].to_string());
        }
        if parts[1].split(',').any(|c| c == "memory") {
            legacy = Some(parts[2].to_string());
        }
    }
    legacy
}

/// `max` means unlimited; anything else is a byte count.
pub fn parse_mem_max(content: &str) -> Option<u64> {
    let t = content.trim();
    if t == "max" {
        return None;
    }
    let v = t.parse::<u64>().ok()?;
    // cgroup v1 signals "unlimited" with a huge sentinel rather than `max`.
    if v >= u64::MAX / 4096 { None } else { Some(v) }
}

/// `"150000 100000"` → 1.5 cores; `"max 100000"` → unlimited.
pub fn parse_cpu_max(content: &str) -> Option<f64> {
    let mut it = content.split_whitespace();
    let quota = it.next()?;
    let period: f64 = it.next().unwrap_or("100000").parse().ok()?;
    if quota == "max" || period <= 0.0 {
        return None;
    }
    let quota: f64 = quota.parse().ok()?;
    Some(quota / period)
}

pub fn looks_containerized(cgroup_path: &str, dockerenv_exists: bool) -> bool {
    dockerenv_exists
        || ["/docker/", "/docker-", "kubepods", "/lxc/", "containerd", "/podman", "libpod"]
            .iter()
            .any(|m| cgroup_path.contains(m))
}

/// One cgroup's own accounting, as the kernel keeps it.
///
/// Summing per-process figures is the only way crabmon could describe a
/// container before this, and it is wrong in both directions: it misses
/// kernel memory, page cache and anything charged to the group rather than
/// to a task, it misses processes that came and went between samples, and —
/// the part that matters most — it cannot see the *limit*, so "this
/// container is at 94% of what it is allowed" was not expressible at all.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CgroupStat {
    /// The cgroup path, which is also its identity across samples.
    pub path: String,
    /// What to call it: a container id or a systemd unit.
    pub name: String,
    /// True when the name came from a container id rather than a unit.
    pub container: bool,
    pub mem_current: Option<u64>,
    /// `None` when unlimited.
    pub mem_max: Option<u64>,
    /// CPU used over the sampling interval, in cores.
    pub cpu_cores: Option<f64>,
    /// `None` when unlimited.
    pub cpu_quota_cores: Option<f64>,
    /// What the container is called — `postgres`, `web/nginx` — when the
    /// runtime's own files say. `name` stays the id, which is the identity.
    pub container_name: Option<String>,
    /// Share of CPU-quota periods in the last interval in which the group ran
    /// out of quota and was made to wait, 0..=1. `None` without a quota, or
    /// when no period elapsed.
    ///
    /// The figure average usage hides: a container using 40% of its quota on
    /// average can still be throttled in a third of its periods, because the
    /// quota is enforced per 100 ms, and every throttled period is latency.
    pub throttled_ratio: Option<f64>,
    /// OOM kills inside the group over its lifetime, from `memory.events`.
    /// The host can be fine while a container is killed at its own limit.
    pub oom_kills: Option<u64>,
    /// The group's own pressure stall, from its `*.pressure` files.
    pub psi: crate::metrics::psi::PsiSample,
}

/// The cumulative counters a group's rates are computed from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CgroupCounters {
    pub usage_usec: Option<u64>,
    pub nr_periods: Option<u64>,
    pub nr_throttled: Option<u64>,
}

impl CgroupStat {
    /// What to call the group on screen: the container's name where it is
    /// known, otherwise the id or unit.
    pub fn label(&self) -> &str {
        self.container_name.as_deref().unwrap_or(&self.name)
    }

    /// Whether `name` refers to this group, by id, unit or container name.
    pub fn is_named(&self, name: &str) -> bool {
        self.name == name || self.container_name.as_deref() == Some(name)
    }

    /// How close the group is to its own memory limit, 0..=1. `None` when it
    /// has no limit, which is not the same as having room.
    pub fn mem_ratio(&self) -> Option<f64> {
        match (self.mem_current, self.mem_max) {
            (Some(c), Some(m)) if m > 0 => Some((c as f64 / m as f64).clamp(0.0, 1.0)),
            _ => None,
        }
    }

    /// How close it is to its CPU quota, 0..=1.
    pub fn cpu_ratio(&self) -> Option<f64> {
        match (self.cpu_cores, self.cpu_quota_cores) {
            (Some(c), Some(q)) if q > 0.0 => Some((c / q).clamp(0.0, 1.0)),
            _ => None,
        }
    }

    /// Worth a row: something is limited, or something is being used.
    pub fn is_interesting(&self) -> bool {
        self.mem_max.is_some() || self.cpu_quota_cores.is_some() || self.mem_current.is_some()
    }
}

/// One `key value` line out of a flat-keyed cgroup file such as `cpu.stat`
/// or `memory.events`.
pub fn flat_key(content: &str, key: &str) -> Option<u64> {
    content.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next()? == key).then(|| it.next()?.parse().ok())?
    })
}

/// `usage_usec 1234567` out of `cpu.stat`.
pub fn parse_cpu_usage_usec(content: &str) -> Option<u64> {
    flat_key(content, "usage_usec")
}

/// The counters `cpu.stat` carries. `nr_periods` and `nr_throttled` are only
/// there when the cpu controller is enabled for the group.
pub fn parse_cpu_counters(content: &str) -> CgroupCounters {
    CgroupCounters {
        usage_usec: flat_key(content, "usage_usec"),
        nr_periods: flat_key(content, "nr_periods"),
        nr_throttled: flat_key(content, "nr_throttled"),
    }
}

/// Throttled periods as a share of the periods that elapsed between two
/// readings. `None` when no period elapsed — an idle group, or one with no
/// quota, whose period counter never moves — rather than a meaningless 0/0.
pub fn throttled_ratio(prev: &CgroupCounters, cur: &CgroupCounters) -> Option<f64> {
    let periods = cur.nr_periods?.checked_sub(prev.nr_periods?)?;
    let throttled = cur.nr_throttled?.checked_sub(prev.nr_throttled?)?;
    (periods > 0).then(|| (throttled as f64 / periods as f64).clamp(0.0, 1.0))
}

/// Cgroups read per sample. A machine running hundreds of containers would
/// otherwise turn a panel into a few hundred file reads every refresh.
pub const MAX_CGROUPS: usize = 128;

/// Read one cgroup's files. `prev_usage` carries the last `usage_usec` so a
/// CPU figure can be a rate rather than a total since boot.
pub fn read_stat(
    path: &str,
    prev: Option<CgroupCounters>,
    secs: f64,
) -> Option<(CgroupStat, CgroupCounters)> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let base: PathBuf = ["/sys/fs/cgroup", path.trim_start_matches('/')].iter().collect();
    if !base.exists() {
        return None;
    }
    let read = |name: &str| fs::read_to_string(base.join(name)).ok();

    let counters = read("cpu.stat").as_deref().map(parse_cpu_counters).unwrap_or_default();
    let usage = counters.usage_usec;
    let cpu_cores = match (prev.and_then(|p| p.usage_usec), usage) {
        // Microseconds of CPU per second of wall clock is cores.
        (Some(prev), Some(now)) if secs > 0.0 && now >= prev => {
            Some((now - prev) as f64 / 1_000_000.0 / secs)
        }
        _ => None,
    };

    let container = crate::metrics::procgroup::container_of(path);
    let name = container
        .clone()
        .or_else(|| crate::metrics::procgroup::service_of(path))
        .unwrap_or_else(|| path.to_string());

    let stat = CgroupStat {
        path: path.to_string(),
        name,
        container: container.is_some(),
        mem_current: read("memory.current").and_then(|s| s.trim().parse().ok()),
        mem_max: read("memory.max").and_then(|s| parse_mem_max(&s)),
        cpu_cores,
        cpu_quota_cores: read("cpu.max").and_then(|s| parse_cpu_max(&s)),
        container_name: None,
        throttled_ratio: prev.and_then(|p| throttled_ratio(&p, &counters)),
        oom_kills: read("memory.events").and_then(|s| flat_key(&s, "oom_kill")),
        psi: crate::metrics::psi::PsiSample {
            cpu: read("cpu.pressure").and_then(|s| crate::metrics::psi::parse_pressure(&s)),
            memory: read("memory.pressure").and_then(|s| crate::metrics::psi::parse_pressure(&s)),
            io: read("io.pressure").and_then(|s| crate::metrics::psi::parse_pressure(&s)),
        },
    };
    Some((stat, counters))
}

/// Read the calling process's cgroup. Linux only; `None` elsewhere.
pub fn read() -> Option<CgroupInfo> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let path = parse_proc_cgroup(&fs::read_to_string("/proc/self/cgroup").ok()?)?;
    let containerized = looks_containerized(&path, Path::new("/.dockerenv").exists());

    let base: PathBuf = ["/sys/fs/cgroup", path.trim_start_matches('/')].iter().collect();
    let read_u64 = |p: PathBuf| fs::read_to_string(p).ok();

    let mem_current = read_u64(base.join("memory.current"))
        .and_then(|s| s.trim().parse::<u64>().ok())
        .or_else(|| {
            read_u64(PathBuf::from("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                .and_then(|s| s.trim().parse::<u64>().ok())
        });
    let mem_max = read_u64(base.join("memory.max")).and_then(|s| parse_mem_max(&s)).or_else(|| {
        read_u64(PathBuf::from("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
            .and_then(|s| parse_mem_max(&s))
    });
    let cpu_quota_cores = read_u64(base.join("cpu.max")).and_then(|s| parse_cpu_max(&s));

    Some(CgroupInfo { path, containerized, mem_current, mem_max, cpu_quota_cores })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_usage_is_read_out_of_cpu_stat() {
        let stat = "usage_usec 12345678\nuser_usec 10000000\nsystem_usec 2345678\n";
        assert_eq!(parse_cpu_usage_usec(stat), Some(12_345_678));
        assert_eq!(parse_cpu_usage_usec("nr_periods 0\n"), None);
        assert_eq!(parse_cpu_usage_usec(""), None);
        assert_eq!(parse_cpu_usage_usec("usage_usec notanumber"), None);
    }

    /// The question per-process sums cannot answer.
    // cpu.stat of a throttled container, cgroup v2.
    const CPU_STAT: &str = "usage_usec 12345678
user_usec 10000000
system_usec 2345678
nr_periods 1000
nr_throttled 120
throttled_usec 4567890
";

    #[test]
    fn throttling_is_a_share_of_the_periods_in_the_interval() {
        let a = parse_cpu_counters(CPU_STAT);
        assert_eq!((a.nr_periods, a.nr_throttled), (Some(1000), Some(120)));
        let b =
            CgroupCounters { usage_usec: Some(0), nr_periods: Some(1010), nr_throttled: Some(124) };
        assert_eq!(throttled_ratio(&a, &b), Some(0.4));
        // No period elapsed: idle, or no quota. Not "never throttled".
        assert_eq!(throttled_ratio(&a, &a), None);
        // No cpu controller: the counters are absent, and so is the ratio.
        let bare = parse_cpu_counters("usage_usec 5\n");
        assert_eq!(throttled_ratio(&bare, &bare), None);
    }

    #[test]
    fn oom_kills_are_read_out_of_memory_events() {
        let events = "low 0\nhigh 14\nmax 3\noom 2\noom_kill 2\noom_group_kill 0\n";
        assert_eq!(flat_key(events, "oom_kill"), Some(2));
        // `oom` must not match `oom_kill`'s line or the other way round.
        assert_eq!(flat_key("oom_kill 7\n", "oom"), None);
    }

    #[test]
    fn a_container_is_found_by_its_id_or_its_name() {
        let s = CgroupStat {
            name: "3f9a1c2b4d5e".into(),
            container_name: Some("postgres".into()),
            ..Default::default()
        };
        assert!(s.is_named("3f9a1c2b4d5e") && s.is_named("postgres"));
        assert!(!s.is_named("postgre"));
        assert_eq!(s.label(), "postgres");
        assert_eq!(CgroupStat { name: "sshd".into(), ..Default::default() }.label(), "sshd");
    }

    #[test]
    fn a_group_reports_how_close_it_is_to_its_own_limits() {
        let s = CgroupStat {
            mem_current: Some(940),
            mem_max: Some(1000),
            cpu_cores: Some(1.8),
            cpu_quota_cores: Some(2.0),
            ..Default::default()
        };
        assert_eq!(s.mem_ratio(), Some(0.94));
        assert_eq!(s.cpu_ratio(), Some(0.9));

        // Unlimited is not the same as having room.
        let unlimited = CgroupStat { mem_current: Some(940), ..Default::default() };
        assert_eq!(unlimited.mem_ratio(), None);
        assert_eq!(unlimited.cpu_ratio(), None);
    }

    #[test]
    fn a_ratio_is_clamped_rather_than_exceeding_one() {
        let over = CgroupStat {
            mem_current: Some(2000),
            mem_max: Some(1000),
            cpu_cores: Some(9.0),
            cpu_quota_cores: Some(1.0),
            ..Default::default()
        };
        assert_eq!(over.mem_ratio(), Some(1.0));
        assert_eq!(over.cpu_ratio(), Some(1.0));
    }

    #[test]
    fn unified_hierarchy_wins_over_legacy_lines() {
        let v2 = "0::/user.slice/user-1000.slice/app.slice/app-Claude-6991.scope\n";
        assert_eq!(
            parse_proc_cgroup(v2).as_deref(),
            Some("/user.slice/user-1000.slice/app.slice/app-Claude-6991.scope")
        );
    }

    #[test]
    fn legacy_hierarchy_falls_back_to_the_memory_controller() {
        let v1 = "11:cpuset:/\n5:memory,cpu:/docker/abc123\n1:name=systemd:/init.scope\n";
        assert_eq!(parse_proc_cgroup(v1).as_deref(), Some("/docker/abc123"));
        assert_eq!(parse_proc_cgroup("garbage"), None);
    }

    #[test]
    fn unlimited_memory_reads_as_none_in_both_cgroup_versions() {
        assert_eq!(parse_mem_max("max\n"), None);
        assert_eq!(parse_mem_max("9223372036854771712\n"), None, "v1 sentinel");
        assert_eq!(parse_mem_max("536870912"), Some(536_870_912));
    }

    #[test]
    fn cpu_quota_converts_to_fractional_cores() {
        assert_eq!(parse_cpu_max("150000 100000"), Some(1.5));
        assert_eq!(parse_cpu_max("max 100000"), None);
        assert_eq!(parse_cpu_max("50000"), Some(0.5), "period defaults to 100000");
        assert_eq!(parse_cpu_max(""), None);
    }

    #[test]
    fn container_detection_uses_both_the_path_and_dockerenv() {
        assert!(looks_containerized("/docker/abc", false));
        assert!(looks_containerized("/kubepods/burstable/podxyz", false));
        assert!(looks_containerized("/user.slice", true));
        assert!(!looks_containerized("/user.slice/user-1000.slice", false));
    }

    #[test]
    fn a_limitless_root_cgroup_is_not_worth_a_panel() {
        let plain = CgroupInfo { path: "/user.slice".into(), ..Default::default() };
        assert!(!plain.is_interesting());
        assert_eq!(plain.mem_ratio(), None);

        let limited = CgroupInfo { mem_current: Some(512), mem_max: Some(2048), ..plain };
        assert!(limited.is_interesting());
        assert_eq!(limited.mem_ratio(), Some(0.25));
    }
}
