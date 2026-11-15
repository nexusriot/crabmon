//! A local log of every action crabmon took on another process.
//!
//! Signals, renices and affinity changes are the only things crabmon does that
//! outlive the session, so they are the only things worth recording. The log is
//! append-only plain text, written best-effort: a full or read-only disk must
//! never stop the action itself.

use std::fmt;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Signal {
        name: String,
    },
    Renice {
        nice: i32,
    },
    Affinity {
        cpus: String,
    },
    /// IO scheduling class and level, as `ionice(1)` spells it.
    IoPriority {
        value: String,
    },
    /// A `systemctl` verb on a unit: `restart` or `reset-failed`.
    Unit {
        verb: String,
    },
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Signal { name } => write!(f, "signal {name}"),
            Action::Renice { nice } => write!(f, "renice {nice}"),
            Action::Affinity { cpus } => write!(f, "affinity {cpus}"),
            Action::IoPriority { value } => write!(f, "ionice {value}"),
            Action::Unit { verb } => write!(f, "systemctl {verb}"),
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

/// Where a log that reached its size limit is moved to: `actions.log.1`.
/// One generation, replaced at each rotation — the log answers "what did I
/// just do to that process", and the viewer only ever shows the last 200.
pub fn rotated_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".1");
    path.with_file_name(name)
}

/// Append one entry, first moving the log aside to `rotated_path` if it has
/// reached `max_bytes` (0 never rotates). Errors are deliberately swallowed —
/// the caller has already performed the action, and failing to log it must
/// not look like a failure.
pub fn append(path: &Path, line: &str, max_bytes: u64) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if max_bytes > 0 && std::fs::metadata(path).is_ok_and(|m| m.len() >= max_bytes) {
        let _ = std::fs::rename(path, rotated_path(path));
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// The most recent entries, newest first, for the in-TUI viewer. Continues
/// into the rotated log when the current one has fewer than `limit`, so the
/// viewer does not go nearly blank the moment the log rotates.
pub fn tail(path: &Path, limit: usize) -> Vec<String> {
    let mut lines = tail_of(path, limit);
    if lines.len() < limit {
        lines.extend(tail_of(&rotated_path(path), limit - lines.len()));
    }
    lines
}

/// The last `limit` non-empty lines of one file, newest first, read from the
/// end in growing chunks rather than reading the whole file.
fn tail_of(path: &Path, limit: usize) -> Vec<String> {
    const CHUNK: u64 = 16 * 1024;
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let Ok(len) = f.seek(SeekFrom::End(0)) else { return Vec::new() };
    if limit == 0 {
        return Vec::new();
    }
    let mut start = len;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let step = CHUNK.min(start);
        start -= step;
        let mut chunk = vec![0u8; step as usize];
        if f.seek(SeekFrom::Start(start)).is_err() || f.read_exact(&mut chunk).is_err() {
            return Vec::new();
        }
        chunk.extend_from_slice(&buf);
        buf = chunk;
        // A line is complete once a newline precedes it, or the start of
        // the file does; the first, partial line of a chunk is not counted.
        let complete = buf.iter().filter(|&&b| b == b'\n').count();
        if start == 0 || complete > limit {
            break;
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines = text.lines();
    if start > 0 {
        lines.next(); // the partial line the read began in the middle of
    }
    let mut out: Vec<String> = lines.filter(|l| !l.trim().is_empty()).map(String::from).collect();
    out.reverse();
    out.truncate(limit);
    out
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
        assert!(
            format_entry(0, &Action::Renice { nice: -5 }, 1, "x", Ok(())).contains("renice -5")
        );
        assert!(
            format_entry(0, &Action::Affinity { cpus: "0-3".into() }, 1, "x", Ok(()))
                .contains("affinity 0-3")
        );
        assert!(
            format_entry(0, &Action::IoPriority { value: "idle".into() }, 1, "x", Ok(()))
                .contains("ionice idle")
        );
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
        append(&path, "first", 0);
        append(&path, "second", 0);
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "first\nsecond\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unwritable_path_is_silently_tolerated() {
        // Logging must never be able to fail an action the user already took.
        append(Path::new("/proc/definitely/not/writable/actions.log"), "x", 1);
    }

    #[test]
    fn the_tail_is_newest_first_and_bounded() {
        let path = tmp("tail");
        let _ = std::fs::remove_file(&path);
        for i in 0..10 {
            append(&path, &format!("entry {i}"), 0);
        }
        let tail = tail(&path, 3);
        assert_eq!(tail, vec!["entry 9", "entry 8", "entry 7"]);
        assert!(super::tail(Path::new("/nonexistent"), 5).is_empty());
        let _ = std::fs::remove_file(&path);
    }

    /// The log is bounded: at `max_bytes` it moves to `.1` (replacing the
    /// previous one) and starts again, and the viewer reads across the two.
    #[test]
    fn the_log_rotates_at_its_limit_and_the_tail_spans_the_rotation() {
        let path = tmp("rotate");
        let old = rotated_path(&path);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&old);
        // "entry N\n" is 8 bytes: a 20-byte limit holds three before rotating.
        for i in 0..7 {
            append(&path, &format!("entry {i}"), 20);
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "entry 6\n");
        assert_eq!(std::fs::read_to_string(&old).unwrap(), "entry 3\nentry 4\nentry 5\n");
        assert_eq!(tail(&path, 3), ["entry 6", "entry 5", "entry 4"]);
        assert_eq!(tail(&path, 10).len(), 4, "only one generation is kept");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&old);
    }

    /// A log far bigger than one read chunk still tails correctly, including
    /// across chunk boundaries that land mid-line.
    #[test]
    fn the_tail_of_a_large_log_is_read_from_the_end() {
        let path = tmp("bigtail");
        let _ = std::fs::remove_file(&path);
        let body: String =
            (0..20_000).map(|i| format!("entry {i:05} {}\n", "x".repeat(i % 37))).collect();
        std::fs::write(&path, body).unwrap();
        let got = tail(&path, 200);
        assert_eq!(got.len(), 200);
        assert!(got[0].starts_with("entry 19999 "), "{}", got[0]);
        assert!(got[199].starts_with("entry 19800 "), "{}", got[199]);
        assert_eq!(tail(&path, 0), Vec::<String>::new());
        let _ = std::fs::remove_file(&path);
    }

    /// Where the log goes when `[audit] path` is left empty. It must be a
    /// per-user location and not the working directory, or running crabmon from
    /// a checkout scatters `actions.log` through whatever tree you happened to
    /// be in.
    #[test]
    fn the_default_log_path_is_namespaced_under_a_state_directory() {
        let path = default_path();
        assert_eq!(path.file_name().unwrap(), "actions.log");
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "crabmon");
        assert!(path.parent().unwrap().parent().is_some(), "no directory above crabmon/");
    }
}
