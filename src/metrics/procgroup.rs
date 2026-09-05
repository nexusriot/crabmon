//! Grouping processes by the cgroup that owns them, which is how systemd and
//! every container runtime already organise the machine.
//!
//! Reading `/proc/<pid>/cgroup` for a few thousand processes on every tick
//! would be wasteful, so paths are cached per PID and only read for processes
//! that are new since the last sample.

use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroupBy {
    #[default]
    None,
    /// systemd unit: `docker.service`, `app-foo.scope`, ...
    Service,
    /// Docker/podman/containerd container id.
    Container,
    User,
}

pub const ALL_GROUPINGS: [GroupBy; 4] =
    [GroupBy::None, GroupBy::Service, GroupBy::Container, GroupBy::User];

impl GroupBy {
    pub fn label(self) -> &'static str {
        match self {
            GroupBy::None => "none",
            GroupBy::Service => "service",
            GroupBy::Container => "container",
            GroupBy::User => "user",
        }
    }

    pub fn next(self) -> GroupBy {
        let i = ALL_GROUPINGS.iter().position(|g| *g == self).unwrap_or(0);
        ALL_GROUPINGS[(i + 1) % ALL_GROUPINGS.len()]
    }

    pub fn parse(s: &str) -> Option<GroupBy> {
        ALL_GROUPINGS.iter().copied().find(|g| g.label() == s.trim().to_ascii_lowercase())
    }
}

/// The systemd unit a cgroup path belongs to: the last `.service`, `.scope`,
/// `.socket`, `.mount` or `.slice` component.
pub fn service_of(cgroup_path: &str) -> Option<String> {
    cgroup_path
        .rsplit('/')
        .find(|c| {
            c.ends_with(".service")
                || c.ends_with(".scope")
                || c.ends_with(".socket")
                || c.ends_with(".mount")
        })
        .map(|c| c.to_string())
        // A bare slice is still more informative than nothing.
        .or_else(|| cgroup_path.rsplit('/').find(|c| c.ends_with(".slice")).map(String::from))
}

fn is_hex_id(s: &str) -> bool {
    s.len() >= 12 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// The container a cgroup path belongs to, as a short id.
///
/// Handles the layouts used by Docker (cgroup v1 and v2), containerd, podman
/// and kubepods.
pub fn container_of(cgroup_path: &str) -> Option<String> {
    for part in cgroup_path.split('/') {
        // v2 systemd driver: `docker-<id>.scope`, `crio-<id>.scope`, ...
        let stem = part
            .strip_suffix(".scope")
            .unwrap_or(part)
            .rsplit_once('-')
            .map(|(_, id)| id)
            .unwrap_or(part.strip_suffix(".scope").unwrap_or(part));
        if is_hex_id(stem) {
            return Some(stem[..12].to_string());
        }
        // v1 / cgroupfs driver: a bare id path component.
        if is_hex_id(part) {
            return Some(part[..12].to_string());
        }
    }
    None
}

/// Caches `/proc/<pid>/cgroup` per process, keyed by PID *and* start time so a
/// recycled PID cannot inherit the previous process's group.
#[derive(Debug, Default)]
pub struct CgroupPaths {
    cache: HashMap<u32, (u64, String)>,
}

impl CgroupPaths {
    /// Look up (and cache) the cgroup path for a process.
    pub fn get(&mut self, pid: u32, start_time_unix: u64) -> Option<&str> {
        let fresh = !matches!(self.cache.get(&pid), Some((seen, _)) if *seen == start_time_unix);
        if fresh {
            let path = read_proc_cgroup(pid)?;
            self.cache.insert(pid, (start_time_unix, path));
        }
        self.cache.get(&pid).map(|(_, p)| p.as_str())
    }

    /// Drop entries for processes that no longer exist.
    pub fn retain_live(&mut self, live: &dyn Fn(u32) -> bool) {
        self.cache.retain(|pid, _| live(*pid));
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

fn read_proc_cgroup(pid: u32) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let content = fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
        super::cgroup::parse_proc_cgroup(&content)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, fs::metadata("/proc"));
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_units_are_extracted_from_real_cgroup_paths() {
        // All four taken from the development machine.
        assert_eq!(service_of("/system.slice/docker.service").as_deref(), Some("docker.service"));
        assert_eq!(service_of("/init.scope").as_deref(), Some("init.scope"));
        assert_eq!(
            service_of("/user.slice/user-1000.slice/user@1000.service/app.slice/app-com.anthropic.Claude-7717.scope")
                .as_deref(),
            Some("app-com.anthropic.Claude-7717.scope"),
            "the innermost unit wins, not the outer user@1000.service"
        );
        assert_eq!(service_of("/user.slice").as_deref(), Some("user.slice"));
        assert_eq!(service_of("/"), None);
    }

    #[test]
    fn container_ids_are_found_in_every_common_layout() {
        let id = "3fa8c9d2e1b7a4f60c5d8e9a1b2c3d4e5f60718293a4b5c6d7e8f9012345678";
        // cgroup v2, systemd driver
        assert_eq!(
            container_of(&format!("/system.slice/docker-{id}.scope")).as_deref(),
            Some(&id[..12])
        );
        // cgroup v1, cgroupfs driver
        assert_eq!(container_of(&format!("/docker/{id}")).as_deref(), Some(&id[..12]));
        // kubernetes
        assert_eq!(
            container_of(&format!("/kubepods/besteffort/pod123/{id}")).as_deref(),
            Some(&id[..12])
        );
        // podman
        assert_eq!(
            container_of(&format!("/machine.slice/libpod-{id}.scope")).as_deref(),
            Some(&id[..12])
        );
    }

    #[test]
    fn ordinary_processes_belong_to_no_container() {
        assert_eq!(container_of("/system.slice/docker.service"), None);
        assert_eq!(
            container_of("/user.slice/user-1000.slice/user@1000.service/app.slice/app-foo.scope"),
            None
        );
        assert_eq!(container_of("/"), None);
    }

    #[test]
    fn short_hex_strings_are_not_mistaken_for_container_ids() {
        assert!(!is_hex_id("abc"));
        assert!(!is_hex_id("app-foo.scope"));
        assert!(is_hex_id("3fa8c9d2e1b7"));
    }

    #[test]
    fn grouping_modes_cycle_and_round_trip() {
        let mut g = GroupBy::None;
        for _ in 0..ALL_GROUPINGS.len() {
            g = g.next();
        }
        assert_eq!(g, GroupBy::None, "a full cycle returns to the start");
        for g in ALL_GROUPINGS {
            assert_eq!(GroupBy::parse(g.label()), Some(g));
        }
        assert_eq!(GroupBy::parse("nope"), None);
    }

    #[test]
    fn the_cache_is_invalidated_when_a_pid_is_recycled() {
        let mut paths = CgroupPaths::default();
        paths.cache.insert(42, (1000, "/system.slice/old.service".into()));
        assert_eq!(paths.get(42, 1000), Some("/system.slice/old.service"));

        // Same PID, different start time: the cached path must not be reused.
        // (`get` re-reads /proc and finds nothing for a fake pid, so it fails
        // closed rather than returning the stale group.)
        assert_ne!(paths.get(42, 2000), Some("/system.slice/old.service"));
    }

    #[test]
    fn dead_processes_are_evicted_so_the_cache_stays_bounded() {
        let mut paths = CgroupPaths::default();
        for pid in 0..100u32 {
            paths.cache.insert(pid, (0, "/x.slice".into()));
        }
        assert_eq!(paths.len(), 100);
        paths.retain_live(&|pid| pid < 10);
        assert_eq!(paths.len(), 10);
        paths.retain_live(&|_| false);
        assert!(paths.is_empty());
    }
}
