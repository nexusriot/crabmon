//! Open file descriptors per process, and the limit they run out against.
//!
//! File-descriptor exhaustion is to a process what inode exhaustion is to a
//! filesystem: the resource that runs out while the obvious gauge still shows
//! headroom. A daemon that has reached `RLIMIT_NOFILE` fails every `accept`
//! and `open` while its CPU, memory and disk figures all look ordinary.
//!
//! Counting is one `getdents` sweep per process, so it is opt-in (`[procs]
//! fds`). The *limit* barely changes over a process's life, so it is read once
//! per process and cached, the same trick `procgroup` uses for cgroup paths.

use std::collections::HashMap;

/// How many entries of one fd table are counted before giving up.
///
/// A process holding a million descriptors would otherwise cost a million
/// directory entries per sample, in the sampler. Past the cap the count is
/// reported as the cap: the panel's job is to say "this process is holding an
/// enormous number of files", and the exact figure past 65536 does not change
/// that answer.
pub const MAX_FDS_COUNTED: usize = 65_536;

/// Descriptors held by one process, or `None` if its fd table cannot be read —
/// another user's process, or one that exited mid-sweep. `Some(0)` would claim
/// a process holds nothing, which is never true of a live process.
pub fn count(pid: u32) -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let entries = std::fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
        // Counted, never collected: only the number is wanted, and a process
        // with 200k sockets would otherwise materialise 200k PathBufs.
        Some(entries.take(MAX_FDS_COUNTED).count() as u32)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// Entries listed in the detail pane's open-files section.
///
/// Bounded because the pane can only show so many, and a proxy holding 200k
/// descriptors should not cost 200k `readlink`s to open a popup with twenty
/// visible rows.
pub const MAX_FILES_LISTED: usize = 256;

/// One open descriptor and what it points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenFile {
    pub fd: u32,
    /// The resolved link target: a path, or a `pipe:[…]`/`anon_inode:…` label
    /// for the descriptors that are not files.
    pub target: String,
}

impl OpenFile {
    /// Whether this is a real path rather than one of the kernel's synthetic
    /// targets. Those are worth showing — a process wedged on an eventfd is
    /// still holding it — but they are not what "open files" means to anyone
    /// looking for the log that filled a disk.
    pub fn is_path(&self) -> bool {
        self.target.starts_with('/')
    }
}

/// The process's open descriptors, in numeric order.
///
/// Sockets are left out: the detail pane already lists them with their
/// addresses and states, and `socket:[4026531992]` next to that is noise.
/// Everything else is kept, including the pipes and anonymous inodes, because
/// "what is this process actually holding" is the question being asked.
///
/// `None` when the fd table cannot be read — another user's process, or one
/// that exited while the pane was opening — which the pane renders as "not
/// readable" rather than as a process holding nothing.
pub fn open_files(pid: u32) -> Option<Vec<OpenFile>> {
    #[cfg(target_os = "linux")]
    {
        let dir = std::fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
        let mut out: Vec<OpenFile> = dir
            .flatten()
            .take(MAX_FILES_LISTED)
            .filter_map(|e| {
                let fd: u32 = e.file_name().to_string_lossy().parse().ok()?;
                let target = std::fs::read_link(e.path()).ok()?.to_string_lossy().to_string();
                // A descriptor can be closed between the readdir and the
                // readlink; that is a process doing its job, not an error.
                (!target.starts_with("socket:")).then_some(OpenFile { fd, target })
            })
            .collect();
        out.sort_by_key(|f| f.fd);
        Some(out)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// The soft `RLIMIT_NOFILE` from `/proc/<pid>/limits`.
pub fn soft_limit(pid: u32) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_limits(&std::fs::read_to_string(format!("/proc/{pid}/limits")).ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// Pull the soft open-files limit out of `/proc/<pid>/limits`.
///
/// The file is column-aligned with spaces inside the limit names, so the value
/// is taken from the end of the line rather than by splitting on whitespace
/// and indexing — "Max open files" is three words, and the hard limit and the
/// units column sit after the soft one.
pub fn parse_limits(content: &str) -> Option<u64> {
    let line = content.lines().find(|l| l.starts_with("Max open files"))?;
    let rest = line["Max open files".len()..].trim();
    let soft = rest.split_whitespace().next()?;
    // "unlimited" is a real value for root-owned daemons, and no ratio against
    // it means anything.
    soft.parse().ok()
}

/// Soft limits, cached per PID *and* start time so a recycled PID cannot
/// inherit the previous process's limit.
#[derive(Debug, Default)]
pub struct LimitCache {
    seen: HashMap<u32, (u64, Option<u64>)>,
}

impl LimitCache {
    pub fn get(&mut self, pid: u32, start_time: u64) -> Option<u64> {
        match self.seen.get(&pid) {
            Some((at, limit)) if *at == start_time => *limit,
            _ => {
                let limit = soft_limit(pid);
                self.seen.insert(pid, (start_time, limit));
                limit
            }
        }
    }

    /// Drop entries for processes that have exited.
    pub fn retain_live(&mut self, live: &impl Fn(u32) -> bool) {
        self.seen.retain(|pid, _| live(*pid));
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim from /proc/self/limits on Linux 6.8.
    const LIMITS: &str = "\
Limit                     Soft Limit           Hard Limit           Units
Max cpu time              unlimited            unlimited            seconds
Max file size             unlimited            unlimited            bytes
Max processes             62841                62841                processes
Max open files            1024                 1048576              files
Max locked memory         8388608              8388608              bytes
Max pending signals       62841                62841                signals
";

    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_can_list_its_own_open_files() {
        // Hold a file open so there is something unambiguous to find.
        let path = std::env::temp_dir().join(format!("crabmon-fd-{}.tmp", std::process::id()));
        let file = std::fs::File::create(&path).expect("create");

        let files = open_files(std::process::id()).expect("our own fd table is readable");
        assert!(files.iter().any(|f| f.target == path.to_string_lossy()), "{files:?}");
        // stdin/stdout/stderr are always there, so the list is never empty.
        assert!(files.len() >= 3, "{files:?}");
        // In numeric order, so the list reads like `ls /proc/<pid>/fd`.
        let fds: Vec<u32> = files.iter().map(|f| f.fd).collect();
        let mut sorted = fds.clone();
        sorted.sort_unstable();
        assert_eq!(fds, sorted);

        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sockets_are_left_out_because_the_pane_already_lists_them_properly() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let files = open_files(std::process::id()).expect("readable");
        assert!(
            !files.iter().any(|f| f.target.starts_with("socket:")),
            "a bare socket inode is noise next to the socket table: {files:?}"
        );
        drop(listener);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_that_is_gone_reports_nothing_rather_than_an_empty_list() {
        // An empty list would read as "this process holds no files", which is
        // never true of a live process — the same distinction `count` makes.
        assert_eq!(open_files(u32::MAX), None);
    }

    #[test]
    fn a_path_is_told_apart_from_the_kernel_synthetic_targets() {
        assert!(OpenFile { fd: 3, target: "/var/log/app.log".into() }.is_path());
        assert!(!OpenFile { fd: 4, target: "pipe:[12345]".into() }.is_path());
        assert!(!OpenFile { fd: 5, target: "anon_inode:[eventfd]".into() }.is_path());
    }

    #[test]
    fn the_soft_limit_is_read_and_not_the_hard_one() {
        // Taking the hard limit would report a daemon at 4000/1048576 — 0.4%,
        // comfortable — when it is actually four times over the 1024 it will
        // actually fail at.
        assert_eq!(parse_limits(LIMITS), Some(1024));
    }

    #[test]
    fn a_limit_name_with_spaces_does_not_shift_the_column() {
        // "Max open files" is three words. Splitting the line on whitespace and
        // taking field 3 reads "files" from one row and a number from another.
        let shifted = "Limit  Soft Limit  Hard Limit  Units\nMax open files  8  16  files\n";
        assert_eq!(parse_limits(shifted), Some(8));
    }

    #[test]
    fn an_unlimited_soft_limit_has_no_ratio_rather_than_a_huge_one() {
        let unlimited =
            "Max open files            unlimited            unlimited            files\n";
        assert_eq!(parse_limits(unlimited), None);
    }

    #[test]
    fn a_file_without_the_row_yields_nothing() {
        assert_eq!(parse_limits(""), None);
        assert_eq!(parse_limits("Max processes 10 20 processes\n"), None);
    }

    #[test]
    fn the_limit_cache_re_reads_when_a_pid_is_recycled() {
        let mut cache = LimitCache::default();
        // Whatever this process's real limit is, asking twice must not grow
        // the cache; asking about the same PID with a new start time must.
        let mine = std::process::id();
        let first = cache.get(mine, 100);
        assert_eq!(cache.get(mine, 100), first, "cached");
        assert_eq!(cache.len(), 1);
        cache.get(mine, 200);
        assert_eq!(cache.len(), 1, "the entry is replaced, not appended");
    }

    #[test]
    fn exited_processes_are_evicted_so_the_cache_cannot_grow_without_bound() {
        let mut cache = LimitCache::default();
        cache.get(1, 1);
        cache.get(2, 1);
        assert_eq!(cache.len(), 2);
        cache.retain_live(&|pid| pid == 1);
        assert_eq!(cache.len(), 1);
        cache.retain_live(&|_| false);
        assert!(cache.is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn this_process_holds_at_least_its_own_standard_streams() {
        let mine = std::process::id();
        let n = count(mine).expect("a process can always read its own fd table");
        assert!(n >= 3, "stdin, stdout and stderr at minimum, got {n}");
        assert!(n as usize <= MAX_FDS_COUNTED);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_pid_that_does_not_exist_reads_as_unknown_not_as_zero() {
        // Zero would render as a process holding no files at all, which is
        // never true and which `fd>0` would then filter out.
        assert_eq!(count(u32::MAX), None);
        assert_eq!(soft_limit(u32::MAX), None);
    }
}
