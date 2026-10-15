//! A human-readable summary written beside each flight recording.
//!
//! A JSONL dump is the evidence, and it is evidence nobody can read without
//! crabmon in front of it. "Send me a recording of the slowdown" lands as a
//! bug report somebody will actually open when the recording arrives with a
//! page that says what fired, what the machine looked like at the time, and
//! what had changed in the minutes before it.
//!
//! Markdown, because the destination is a ticket or a chat message.

use crate::alerts::ActiveAlert;
use crate::diff;
use crate::format::{human_bps, human_bytes, human_duration};
use crate::metrics::Snapshot;

/// Processes listed in each table. Enough to see the shape, short enough to
/// paste into a ticket.
pub const TOP_N: usize = 8;

/// Build the summary for an incident.
///
/// `first` is the oldest frame in the recording and `at` the frame that
/// tripped the rule, so the report can say what *changed* rather than only
/// what things were like once it was already too late.
pub fn summary(rule: &ActiveAlert, first: &Snapshot, at: &Snapshot, dump: &str) -> String {
    let mut out = String::new();
    let host = match at.host.hostname.is_empty() {
        true => "this host".to_string(),
        false => at.host.hostname.clone(),
    };

    out.push_str(&format!("# {} on {host}\n\n", rule.name));
    out.push_str(&format!("- **Rule**: {}\n", rule.message));
    out.push_str(&format!(
        "- **Measured**: {:.1} against a threshold of {:.1}\n",
        rule.value, rule.threshold
    ));
    out.push_str(&format!("- **At**: {} (unix {})\n", stamp(at.taken_at_unix), at.taken_at_unix));
    if !at.host.os.is_empty() || !at.host.kernel.is_empty() {
        out.push_str(&format!("- **System**: {} {}\n", at.host.os, at.host.kernel));
    }
    out.push_str(&format!("- **Uptime**: {}\n", human_duration(at.host.uptime_secs)));
    out.push_str(&format!("- **Recording**: `{dump}`\n"));

    out.push_str("\n## The machine at that moment\n\n");
    out.push_str(&format!("- CPU **{:.1}%**, load {:.2}\n", at.cpu.avg(), at.host.load[0]));
    out.push_str(&format!(
        "- Memory **{}** of {} used ({} of it reclaimable cache)\n",
        human_bytes(at.mem.used),
        human_bytes(at.mem.total),
        human_bytes(at.mem.detail.buff_cache())
    ));
    if at.mem.swap_total > 0 {
        out.push_str(&format!(
            "- Swap **{}** of {} used\n",
            human_bytes(at.mem.swap_used),
            human_bytes(at.mem.swap_total)
        ));
    }
    if at.psi.is_available() {
        out.push_str(&format!("- Pressure (worst 10 s `some`) **{:.1}%**\n", at.psi.worst_avg10()));
    }
    if let Some(util) = at.disk_util_max() {
        out.push_str(&format!("- Busiest device **{:.0}% utilised**\n", util * 100.0));
    }
    let (rx, tx) = at.net_totals(false);
    out.push_str(&format!("- Network ↓ {} ↑ {}\n", human_bps(rx), human_bps(tx)));

    // The kernel's own account, which is the part no gauge can reconstruct
    // after the fact.
    if !at.kernel.is_empty() {
        out.push_str("\n## What the kernel said\n\n");
        for e in at.kernel.iter().rev().take(TOP_N) {
            let who = match (&e.process, e.pid) {
                (Some(name), Some(pid)) => format!("{name} ({pid}): "),
                _ => String::new(),
            };
            out.push_str(&format!(
                "- `{}` {who}{} — {} before this frame\n",
                e.kind.key_name(),
                e.text,
                human_duration(e.age_secs(at.host.uptime_secs))
            ));
        }
    }

    out.push_str("\n## Busiest processes\n\n");
    out.push_str(&table(at, TOP_N));

    // What changed over the recording. `compare` is the same code `--diff`
    // runs, so the report and the command agree about what "moved" means.
    if first.taken_at_unix != at.taken_at_unix {
        let d = diff::compare(first, at);
        out.push_str(&format!(
            "\n## What changed over the {} before it\n\n",
            human_duration(d.seconds_apart)
        ));
        out.push_str(&format!(
            "- CPU {:.1}% → **{:.1}%**, load {:.2} → **{:.2}**\n",
            d.cpu.0, d.cpu.1, d.load1.0, d.load1.1
        ));
        out.push_str(&format!(
            "- Memory {} → **{}**\n",
            human_bytes(d.mem_used.0),
            human_bytes(d.mem_used.1)
        ));
        out.push_str(&format!("- Processes {} → **{}**\n", d.procs.0, d.procs.1));
        if !d.movers.is_empty() {
            out.push_str("\n### Processes that moved\n\n");
            out.push_str("| process | pid | ΔCPU | ΔRSS |\n|---|---:|---:|---:|\n");
            for m in d.movers.iter().take(TOP_N) {
                out.push_str(&format!(
                    "| {} | {} | {:+.1}% | {} |\n",
                    escape(&m.name),
                    m.pid,
                    m.cpu_delta(),
                    signed_bytes(m.mem_delta())
                ));
            }
        }
        if !d.appeared.is_empty() {
            out.push_str("\n### Appeared\n\n");
            for p in d.appeared.iter().take(TOP_N) {
                out.push_str(&format!("- {} ({})\n", escape(&p.name), p.pid));
            }
        }
        if !d.vanished.is_empty() {
            out.push_str("\n### Vanished\n\n");
            for p in d.vanished.iter().take(TOP_N) {
                out.push_str(&format!("- {} ({})\n", escape(&p.name), p.pid));
            }
        }
    }

    out.push_str("\nReplay it with:\n\n```sh\n");
    out.push_str(&format!("crabmon --replay {dump}\n"));
    out.push_str("```\n");
    out
}

/// A table of the busiest processes, by CPU.
fn table(snap: &Snapshot, limit: usize) -> String {
    let mut procs: Vec<_> = snap.procs.iter().collect();
    procs.sort_by(|a, b| b.cpu.partial_cmp(&a.cpu).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = String::from("| process | pid | user | CPU | RSS |\n|---|---:|---|---:|---:|\n");
    for p in procs.into_iter().take(limit) {
        out.push_str(&format!(
            "| {} | {} | {} | {:.1}% | {} |\n",
            escape(&p.name),
            p.pid,
            escape(p.user.as_deref().unwrap_or("-")),
            p.cpu,
            human_bytes(p.mem)
        ));
    }
    out
}

/// `+1.2 GiB` / `-400 MiB`, since a delta's sign is the whole message.
fn signed_bytes(delta: i64) -> String {
    match delta < 0 {
        true => format!("-{}", human_bytes(delta.unsigned_abs())),
        false => format!("+{}", human_bytes(delta as u64)),
    }
}

/// A process name is argv, and argv can contain anything — including the `|`
/// that would otherwise split a Markdown table cell in two.
fn escape(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// A unix time as something readable, without pulling in a date library.
///
/// UTC, spelled out, because an incident report that says "14:05" without
/// saying where is a report two people will read differently.
pub fn stamp(unix: u64) -> String {
    let days = unix / 86_400;
    let secs = unix % 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Howard Hinnant's `civil_from_days`, which is the whole of a date library
/// that this needs.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The summary's filename for a dump at `dump_path`.
pub fn path_for(dump_path: &std::path::Path) -> std::path::PathBuf {
    dump_path.with_extension("md")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, MemSample, ProcRow};

    fn alert() -> ActiveAlert {
        ActiveAlert {
            name: "mem-pressure".into(),
            message: "mem-pressure 96.0% ≥ 90.0%".into(),
            value: 96.0,
            threshold: 90.0,
        }
    }

    fn snap(cpu: f32, used: u64, procs: Vec<ProcRow>) -> Snapshot {
        Snapshot {
            cpu: CpuSample { per_core: vec![cpu], freq_mhz: vec![] },
            mem: MemSample { total: 16 << 30, used, ..Default::default() },
            procs,
            taken_at_unix: 1_788_297_407,
            ..Default::default()
        }
    }

    fn proc(pid: u32, name: &str, cpu: f32, mem: u64) -> ProcRow {
        ProcRow { pid, name: name.into(), cpu, mem, start_time_unix: 1, ..Default::default() }
    }

    #[test]
    fn the_report_says_what_fired_and_what_it_measured() {
        let s = snap(90.0, 15 << 30, vec![proc(1, "postgres", 90.0, 8 << 30)]);
        let out = summary(&alert(), &s, &s, "dump.jsonl");
        assert!(out.contains("# mem-pressure"), "{out}");
        assert!(out.contains("96.0"), "the measurement is the point: {out}");
        assert!(out.contains("threshold of 90.0"), "{out}");
        assert!(out.contains("postgres"), "{out}");
        assert!(out.contains("crabmon --replay dump.jsonl"), "{out}");
    }

    /// The run-up is the half a single frame cannot show.
    #[test]
    fn the_report_says_what_changed_over_the_recording() {
        let mut before = snap(5.0, 4 << 30, vec![proc(1, "postgres", 5.0, 1 << 30)]);
        before.taken_at_unix -= 600;
        let after = snap(90.0, 15 << 30, vec![proc(1, "postgres", 90.0, 9 << 30)]);

        let out = summary(&alert(), &before, &after, "dump.jsonl");
        assert!(out.contains("What changed"), "{out}");
        assert!(out.contains("Processes that moved"), "{out}");
        assert!(out.contains("postgres"), "{out}");
        assert!(out.contains("+85.0%"), "the delta's sign is the message: {out}");
    }

    #[test]
    fn a_single_frame_recording_has_no_change_section() {
        let s = snap(90.0, 15 << 30, vec![]);
        let out = summary(&alert(), &s, &s, "dump.jsonl");
        assert!(!out.contains("What changed"), "{out}");
    }

    /// An argv can contain anything, including the character that splits a
    /// Markdown cell in two.
    #[test]
    fn a_process_name_cannot_break_the_table() {
        let s = snap(1.0, 1 << 30, vec![proc(1, "sh -c 'a | b'", 1.0, 1024)]);
        let out = summary(&alert(), &s, &s, "d.jsonl");
        let row = out.lines().find(|l| l.contains("sh -c")).expect("the row");
        assert_eq!(row.matches(" | ").count(), 4, "an argv split the row: {row}");
    }

    #[test]
    fn the_kernels_own_account_is_included_when_there_is_one() {
        let mut s = snap(10.0, 1 << 30, vec![]);
        s.host.uptime_secs = 2000;
        s.kernel = vec![crate::metrics::kmsg::parse_record(
            "4,1,1000000000,-;Out of memory: Killed process 991 (postgres)",
        )
        .unwrap()];
        let out = summary(&alert(), &s, &s, "d.jsonl");
        assert!(out.contains("What the kernel said"), "{out}");
        assert!(out.contains("postgres (991)"), "{out}");
    }

    #[test]
    fn a_timestamp_reads_as_a_date_and_says_which_zone() {
        // 2026-09-12T12:36:47Z
        assert_eq!(stamp(1_789_216_607), "2026-09-12 12:36:47 UTC");
        assert_eq!(stamp(0), "1970-01-01 00:00:00 UTC");
        // A leap day, because the arithmetic is the sort that gets it wrong.
        assert_eq!(stamp(1_709_164_800), "2024-02-29 00:00:00 UTC");
    }

    #[test]
    fn the_summary_sits_beside_its_dump() {
        let p = path_for(std::path::Path::new("/var/log/crabmon-hot-1700000000.jsonl"));
        assert_eq!(p.to_str().unwrap(), "/var/log/crabmon-hot-1700000000.md");
    }

    #[test]
    fn a_negative_delta_keeps_its_sign() {
        assert!(signed_bytes(-1024).starts_with('-'));
        assert!(signed_bytes(1024).starts_with('+'));
    }
}
