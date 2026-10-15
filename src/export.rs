//! Snapshot export: `P` in the TUI, or `--once --format json|csv|text` headless.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::metrics::{ProcRow, Snapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
    /// An aligned table for a person to read, the `top -b -n1` shape.
    ///
    /// Without it the only way to get the numbers out of crabmon was JSON or
    /// CSV, so "what is this box doing right now" over SSH meant either
    /// opening a full-screen interface or piping through `jq`.
    Text,
}

/// Every format a snapshot can be written in, so the docs and `.gitignore`
/// can be checked against the code rather than against a hand-kept list —
/// the same reason `ALL_SORTS` and `ALL_ALERT_KINDS` exist. A format added
/// without a matching ignore rule is an exported snapshot committed by
/// accident, which is how `.txt` slipped through.
pub const ALL_EXPORT_FORMATS: [ExportFormat; 3] =
    [ExportFormat::Json, ExportFormat::Csv, ExportFormat::Text];

impl ExportFormat {
    pub fn parse(s: &str) -> Option<ExportFormat> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Some(ExportFormat::Json),
            "csv" => Some(ExportFormat::Csv),
            "text" | "txt" | "plain" => Some(ExportFormat::Text),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
            ExportFormat::Text => "txt",
        }
    }
}

/// What lands in the file: the snapshot with the process list already trimmed.
#[derive(Serialize)]
struct Export<'a> {
    crabmon_version: &'static str,
    #[serde(flatten)]
    snapshot: &'a Snapshot,
}

/// `top_n == 0` keeps every process.
pub fn trim_procs(snap: &Snapshot, top_n: usize) -> Snapshot {
    let mut out = snap.clone();
    if top_n > 0 && out.procs.len() > top_n {
        out.procs.truncate(top_n);
    }
    out
}

pub fn to_json(snap: &Snapshot) -> String {
    let export = Export { crabmon_version: env!("CARGO_PKG_VERSION"), snapshot: snap };
    serde_json::to_string_pretty(&export).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

/// Escape a field for RFC 4180 CSV.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Column order of the CSV export. Every `ProcRow` field the JSON carries is
/// here too: a CSV that silently drops the cgroup, nice and path columns sends
/// people back to the JSON for exactly the questions CSV is good at.
pub const CSV_COLUMNS: [&str; 24] = [
    "pid",
    "ppid",
    "name",
    "cpu_percent",
    "mem_bytes",
    "pss_bytes",
    "uss_bytes",
    "virt_bytes",
    "state",
    "user",
    "uid",
    "nice",
    "run_time_secs",
    "start_time_unix",
    "threads",
    "fds",
    "fd_limit",
    "read_bps",
    "write_bps",
    "service",
    "container",
    "exe",
    "cwd",
    "cmd",
];

pub fn procs_to_csv(procs: &[ProcRow]) -> String {
    let mut out = CSV_COLUMNS.join(",");
    out.push('\n');
    for p in procs {
        let opt = |v: &Option<String>| csv_field(v.as_deref().unwrap_or(""));
        out.push_str(&format!(
            "{},{},{},{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.0},{:.0},{},{},{},{},{}\n",
            p.pid,
            p.ppid.map(|v| v.to_string()).unwrap_or_default(),
            csv_field(&p.name),
            p.cpu,
            p.mem,
            p.pss.map(|v| v.to_string()).unwrap_or_default(),
            p.uss.map(|v| v.to_string()).unwrap_or_default(),
            p.virt,
            p.state,
            opt(&p.user),
            p.uid.map(|v| v.to_string()).unwrap_or_default(),
            p.nice.map(|v| v.to_string()).unwrap_or_default(),
            p.run_time,
            p.start_time_unix,
            p.threads.map(|t| t.to_string()).unwrap_or_default(),
            p.fds.map(|v| v.to_string()).unwrap_or_default(),
            p.fd_limit.map(|v| v.to_string()).unwrap_or_default(),
            p.read_bps,
            p.write_bps,
            opt(&p.service),
            opt(&p.container),
            csv_field(&p.exe),
            csv_field(&p.cwd),
            csv_field(&p.cmd),
        ));
    }
    out
}

pub fn render(snap: &Snapshot, format: ExportFormat) -> String {
    match format {
        ExportFormat::Json => to_json(snap),
        ExportFormat::Csv => procs_to_csv(&snap.procs),
        ExportFormat::Text => to_text(snap),
    }
}

/// Width the text table assumes when nothing says otherwise. Wide enough for
/// the columns worth having, narrow enough to paste into a ticket.
pub const TEXT_WIDTH: u16 = 100;

/// A readable summary: a header line of machine-wide figures, then the
/// process table as the TUI would lay it out at `TEXT_WIDTH`.
///
/// The column planner does the laying out, so the text dump and the
/// interface agree about what is worth showing and in what order.
pub fn to_text(snap: &Snapshot) -> String {
    to_text_with(snap, &[], TEXT_WIDTH)
}

/// As `to_text`, honouring a configured column list and a width.
///
/// A `[procs] columns` list is a statement about which columns are worth
/// having, and it would be odd for that to apply to the interface and not to
/// the dump of the same table.
pub fn to_text_with(snap: &Snapshot, columns: &[String], width: u16) -> String {
    use crate::format::{human_bps, human_bytes, human_duration};

    let mut out = String::new();
    let host = match snap.host.hostname.is_empty() {
        true => "localhost".to_string(),
        false => snap.host.hostname.clone(),
    };
    out.push_str(&format!(
        "{host}  {} {}  up {}  load {:.2} {:.2} {:.2}\n",
        snap.host.os,
        snap.host.kernel,
        human_duration(snap.host.uptime_secs),
        snap.host.load[0],
        snap.host.load[1],
        snap.host.load[2],
    ));
    out.push_str(&format!(
        "cpu {:>5.1}%   mem {} / {} ({} cache)",
        snap.cpu.avg(),
        human_bytes(snap.mem.used),
        human_bytes(snap.mem.total),
        human_bytes(snap.mem.detail.buff_cache()),
    ));
    if snap.mem.swap_total > 0 {
        out.push_str(&format!(
            "   swap {} / {}",
            human_bytes(snap.mem.swap_used),
            human_bytes(snap.mem.swap_total)
        ));
    }
    let (rx, tx) = snap.net_totals(false);
    if rx > 0.0 || tx > 0.0 {
        out.push_str(&format!("   net ↓ {} ↑ {}", human_bps(rx), human_bps(tx)));
    }
    out.push_str("\n\n");
    out.push_str(&crate::ui::procs::text_table_with(&snap.procs, columns, width));
    out
}

/// `crabmon-<unix-timestamp>.<ext>` inside `dir`.
pub fn export_path(dir: &Path, unix_secs: u64, format: ExportFormat) -> PathBuf {
    dir.join(format!("crabmon-{unix_secs}.{}", format.extension()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, MemSample};

    fn snap() -> Snapshot {
        Snapshot {
            cpu: CpuSample { per_core: vec![10.0, 20.0], freq_mhz: vec![800, 900] },
            mem: MemSample { total: 16_035_467_264, used: 5_866_434_560, ..Default::default() },
            procs: vec![
                ProcRow { pid: 1, name: "init".into(), cpu: 0.5, mem: 1024, ..Default::default() },
                ProcRow {
                    pid: 2,
                    name: "we, \"ird\"".into(),
                    cmd: "a,b\nc".into(),
                    ..Default::default()
                },
            ],
            taken_at_unix: 1_788_297_407,
            ..Default::default()
        }
    }

    #[test]
    fn json_is_valid_and_carries_the_real_byte_counts() {
        let json = to_json(&snap());
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(v["mem"]["total"], 16_035_467_264u64);
        assert_eq!(v["procs"][0]["name"], "init");
        assert_eq!(v["crabmon_version"], env!("CARGO_PKG_VERSION"));
    }

    /// Every variant must round-trip through the spelling the config file
    /// and `--format` use, or a format is unreachable from either.
    #[test]
    fn every_format_parses_from_its_own_name_and_has_an_extension() {
        for format in ALL_EXPORT_FORMATS {
            let name = match format {
                ExportFormat::Json => "json",
                ExportFormat::Csv => "csv",
                ExportFormat::Text => "text",
            };
            assert_eq!(ExportFormat::parse(name), Some(format), "{name}");
            assert!(!format.extension().is_empty(), "{name} has no extension");
        }
    }

    #[test]
    fn csv_quotes_commas_quotes_and_newlines() {
        let csv = procs_to_csv(&snap().procs);
        let header = csv.lines().next().unwrap();
        assert_eq!(header.split(',').count(), CSV_COLUMNS.len());
        assert!(csv.contains("\"we, \"\"ird\"\"\""), "{csv}");
        assert!(csv.contains("\"a,b\nc\""), "{csv}");
    }

    #[test]
    fn csv_carries_the_columns_the_json_does() {
        let p = ProcRow {
            pid: 7,
            uid: Some(1000),
            nice: Some(-5),
            service: Some("docker.service".into()),
            container: Some("abc123def456".into()),
            exe: "/usr/bin/dockerd".into(),
            cwd: "/".into(),
            start_time_unix: 1_700_000_042,
            ..Default::default()
        };
        let csv = procs_to_csv(&[p]);
        let row = csv.lines().nth(1).unwrap();
        for expected in
            ["1000", "-5", "docker.service", "abc123def456", "/usr/bin/dockerd", "1700000042"]
        {
            assert!(row.contains(expected), "{expected} missing from {row}");
        }
        assert_eq!(row.split(',').count(), CSV_COLUMNS.len());
    }

    #[test]
    fn csv_of_an_empty_process_list_is_still_a_valid_header() {
        let csv = procs_to_csv(&[]);
        assert_eq!(csv.lines().count(), 1);
    }

    #[test]
    fn trimming_keeps_the_first_n_processes_and_zero_means_all() {
        let s = snap();
        assert_eq!(trim_procs(&s, 1).procs.len(), 1);
        assert_eq!(trim_procs(&s, 0).procs.len(), 2);
        assert_eq!(trim_procs(&s, 99).procs.len(), 2);
    }

    #[test]
    fn formats_parse_and_name_their_files() {
        assert_eq!(ExportFormat::parse("JSON"), Some(ExportFormat::Json));
        assert_eq!(ExportFormat::parse("csv"), Some(ExportFormat::Csv));
        assert_eq!(ExportFormat::parse("yaml"), None);
        assert_eq!(
            export_path(Path::new("/tmp"), 1234, ExportFormat::Csv),
            PathBuf::from("/tmp/crabmon-1234.csv")
        );
    }
}
