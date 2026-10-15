//! Proportional and unique memory, from `/proc/<pid>/smaps_rollup`.
//!
//! Every memory figure crabmon shows is RSS, and RSS double-counts: a page
//! shared between eight Chromium processes is counted eight times, so adding
//! up a process tree's RSS gives a number larger than the machine has. That
//! makes "which program is using the memory" unanswerable precisely where it
//! is most often asked — a browser, a database with many workers, anything
//! that forks.
//!
//! - **PSS** divides each shared page among the processes mapping it, so the
//!   sum across processes is a real number and comparable between them.
//! - **USS** is what would actually be freed by killing the process, which is
//!   the number behind "will this help?".
//!
//! `smaps_rollup` is one read per process and the kernel does the summing;
//! the older `smaps` would mean parsing thousands of mapping lines per
//! process. It still costs a read per process per sample, so it is opt-in
//! (`[procs] pss`) exactly as the descriptor walk is.

/// What one process actually occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rollup {
    /// Proportional set size: private pages plus a share of each shared one.
    pub pss: u64,
    /// Unique set size: pages this process alone maps, i.e. what killing it
    /// would free.
    pub uss: u64,
    /// Anonymous pages that have been pushed to swap.
    pub swap: u64,
}

/// Parse the `smaps_rollup` format: `Key:   <n> kB` lines.
///
/// USS is not a field of its own; it is `Private_Clean + Private_Dirty`,
/// which is how every other tool derives it.
pub fn parse_rollup(content: &str) -> Option<Rollup> {
    let mut out = Rollup::default();
    let mut saw_pss = false;
    for line in content.lines() {
        let Some((key, rest)) = line.split_once(':') else { continue };
        let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        let bytes = kb.saturating_mul(1024);
        match key.trim() {
            "Pss" => {
                out.pss = bytes;
                saw_pss = true;
            }
            "Private_Clean" | "Private_Dirty" => out.uss = out.uss.saturating_add(bytes),
            "Swap" => out.swap = bytes,
            _ => {}
        }
    }
    // A rollup with no `Pss` line is not a rollup: kernels without
    // `CONFIG_PROC_PAGE_MONITOR` expose the file but not the field, and
    // reporting USS alone would be a different measurement under the same
    // column heading.
    saw_pss.then_some(out)
}

/// Read one process's rollup, or `None` where it cannot be read — another
/// user's process, a kernel thread (which maps nothing), or a kernel without
/// the file.
pub fn read(pid: u32) -> Option<Rollup> {
    #[cfg(target_os = "linux")]
    {
        parse_rollup(&std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// Whether this kernel exposes the file at all, so the UI can say "not
/// available here" rather than drawing a column of dashes.
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/proc/self/smaps_rollup").exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `smaps_rollup`, trimmed to the fields that are read.
    const ROLLUP: &str = "\
55d4c0000000-7ffd0e5ff000 ---p 00000000 00:00 0                          [rollup]
Rss:              123456 kB
Pss:               45678 kB
Pss_Dirty:         40000 kB
Shared_Clean:      60000 kB
Shared_Dirty:       1000 kB
Private_Clean:      2000 kB
Private_Dirty:     20000 kB
Referenced:       100000 kB
Anonymous:         21000 kB
Swap:                512 kB
Locked:                0 kB
";

    #[test]
    fn a_rollup_yields_proportional_and_unique_memory() {
        let r = parse_rollup(ROLLUP).expect("a rollup");
        assert_eq!(r.pss, 45_678 * 1024);
        // USS is the two private figures together.
        assert_eq!(r.uss, (2_000 + 20_000) * 1024);
        assert_eq!(r.swap, 512 * 1024);
    }

    /// PSS is the whole point; a file without it is a different measurement.
    #[test]
    fn a_file_without_pss_is_not_a_rollup() {
        assert!(parse_rollup("Rss: 100 kB\nPrivate_Clean: 4 kB\n").is_none());
        assert!(parse_rollup("").is_none());
        assert!(parse_rollup("nonsense").is_none());
    }

    #[test]
    fn malformed_lines_are_skipped_rather_than_guessed_at() {
        let r = parse_rollup("Pss: 10 kB\nPrivate_Clean: notanumber kB\nJunk\n").expect("a rollup");
        assert_eq!(r.pss, 10 * 1024);
        assert_eq!(r.uss, 0);
    }

    /// PSS is never larger than RSS and never smaller than USS; a parser that
    /// mixed the fields up would break one of those.
    #[test]
    fn the_figures_keep_their_ordering() {
        let r = parse_rollup(ROLLUP).unwrap();
        assert!(r.uss <= r.pss, "uss {} > pss {}", r.uss, r.pss);
        assert!(r.pss <= 123_456 * 1024, "pss exceeds rss");
    }

    /// Reading our own process must not panic, whatever the kernel answers.
    #[test]
    fn reading_this_process_is_safe() {
        let me = std::process::id();
        if let Some(r) = read(me) {
            assert!(r.pss > 0, "a live process maps something");
            assert!(r.uss <= r.pss);
        }
        // Either the file exists and parsed, or it does not exist here.
        assert_eq!(read(me).is_some(), is_available() || read(me).is_some());
    }
}
