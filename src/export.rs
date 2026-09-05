//! Snapshot export: `P` in the TUI, or `--once --format json|csv` headless.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::metrics::{ProcRow, Snapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
}

impl ExportFormat {
    pub fn parse(s: &str) -> Option<ExportFormat> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Some(ExportFormat::Json),
            "csv" => Some(ExportFormat::Csv),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
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

pub fn procs_to_csv(procs: &[ProcRow]) -> String {
    let mut out = String::from(
        "pid,ppid,name,cpu_percent,mem_bytes,virt_bytes,state,user,run_time_secs,threads,read_bps,write_bps,cmd\n",
    );
    for p in procs {
        out.push_str(&format!(
            "{},{},{},{:.1},{},{},{},{},{},{},{:.0},{:.0},{}\n",
            p.pid,
            p.ppid.map(|v| v.to_string()).unwrap_or_default(),
            csv_field(&p.name),
            p.cpu,
            p.mem,
            p.virt,
            p.state,
            csv_field(p.user.as_deref().unwrap_or("")),
            p.run_time,
            p.threads.map(|t| t.to_string()).unwrap_or_default(),
            p.read_bps,
            p.write_bps,
            csv_field(&p.cmd),
        ));
    }
    out
}

pub fn render(snap: &Snapshot, format: ExportFormat) -> String {
    match format {
        ExportFormat::Json => to_json(snap),
        ExportFormat::Csv => procs_to_csv(&snap.procs),
    }
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

    #[test]
    fn csv_quotes_commas_quotes_and_newlines() {
        let csv = procs_to_csv(&snap().procs);
        let header = csv.lines().next().unwrap();
        assert_eq!(header.split(',').count(), 13);
        assert!(csv.contains("\"we, \"\"ird\"\"\""), "{csv}");
        assert!(csv.contains("\"a,b\nc\""), "{csv}");
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
