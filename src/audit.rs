//! A local log of every action crabmon took on another process.
//!
//! Signals, renices and affinity changes are the only things crabmon does that
//! outlive the session, so they are the only things worth recording. The log is
//! append-only plain text, written best-effort: a full or read-only disk must
//! never stop the action itself.

use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Signal { name: String },
    Renice { nice: i32 },
    Affinity { cpus: String },
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Signal { name } => write!(f, "signal {name}"),
            Action::Renice { nice } => write!(f, "renice {nice}"),
            Action::Affinity { cpus } => write!(f, "affinity {cpus}"),
        }
    }
}

/// One log line. `unix_time` is passed in rather than read from the clock so
/// the formatting is testable.
pub fn format_entry(
    unix_time: u64,
    action: &Action,
    pid: u32,
    name: &str,
    result: Result<(), &str>,
) -> String {
    let outcome = match result {
        Ok(()) => "ok".to_string(),
        Err(e) => format!("failed: {}", e.replace('\n', " ")),
    };
    format!("{unix_time}\t{action}\tpid={pid}\tname={name}\t{outcome}")
}

pub fn default_path() -> PathBuf {
    let base =
        dirs::state_dir().or_else(dirs::data_local_dir).unwrap_or_else(|| PathBuf::from("."));
    base.join("crabmon").join("actions.log")
}

/// Append one entry. Errors are deliberately swallowed — the caller has already
/// performed the action, and failing to log it must not look like a failure.
pub fn append(path: &Path, line: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// The most recent entries, newest first, for the in-TUI viewer.
pub fn tail(path: &Path, limit: usize) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines: Vec<String> =
        content.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect();
    lines.reverse();
    lines.truncate(limit);
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("crabmon-audit-{tag}-{}.log", std::process::id()))
    }

    #[test]
    fn entries_are_tab_separated_and_name_the_action() {
        let line = format_entry(
            1_788_297_407,
            &Action::Signal { name: "SIGTERM".into() },
            4242,
            "firefox",
            Ok(()),
        );
        assert_eq!(line, "1788297407\tsignal SIGTERM\tpid=4242\tname=firefox\tok");
        assert_eq!(line.split('\t').count(), 5);
    }

    #[test]
    fn every_action_kind_formats() {
        assert!(format_entry(0, &Action::Renice { nice: -5 }, 1, "x", Ok(())).contains("renice -5"));
        assert!(format_entry(0, &Action::Affinity { cpus: "0-3".into() }, 1, "x", Ok(()))
            .contains("affinity 0-3"));
    }

    #[test]
    fn failures_are_recorded_with_their_reason_on_one_line() {
        let line = format_entry(
            1,
            &Action::Signal { name: "SIGKILL".into() },
            1,
            "init",
            Err("Operation not permitted\n"),
        );
        assert!(line.contains("failed: Operation not permitted"));
        assert_eq!(line.lines().count(), 1, "a multi-line error must not break the format");
    }

    #[test]
    fn appending_creates_the_file_and_keeps_previous_entries() {
        let path = tmp("append");
        let _ = std::fs::remove_file(&path);
        append(&path, "first");
        append(&path, "second");
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "first\nsecond\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unwritable_path_is_silently_tolerated() {
        // Logging must never be able to fail an action the user already took.
        append(Path::new("/proc/definitely/not/writable/actions.log"), "x");
    }

    #[test]
    fn the_tail_is_newest_first_and_bounded() {
        let path = tmp("tail");
        let _ = std::fs::remove_file(&path);
        for i in 0..10 {
            append(&path, &format!("entry {i}"));
        }
        let tail = tail(&path, 3);
        assert_eq!(tail, vec!["entry 9", "entry 8", "entry 7"]);
        assert!(super::tail(Path::new("/nonexistent"), 5).is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
