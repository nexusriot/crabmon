//! Persistent configuration. Every field has a default, and an unparseable or
//! partially-written file degrades to defaults rather than refusing to start.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::alerts::AlertRule;
use crate::metrics::netclass;
use crate::metrics::procgroup::GroupBy;
use crate::sort::SortBy;
use crate::ui::Layout;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    pub warn: f64,
    pub crit: f64,
    pub temp_warn: f64,
    pub temp_crit: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { warn: 0.7, crit: 0.9, temp_warn: 0.75, temp_crit: 0.9 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Panels {
    pub header: bool,
    pub cpu: bool,
    pub mem: bool,
    pub procs: bool,
    pub net: bool,
    pub sensors: bool,
    pub disks: bool,
    pub gpu: bool,
    pub cgroup: bool,
    pub psi: bool,
    pub power: bool,
}

impl Default for Panels {
    fn default() -> Self {
        Self {
            header: true,
            cpu: true,
            mem: true,
            procs: true,
            net: true,
            sensors: true,
            disks: true,
            gpu: true,
            cgroup: true,
            psi: true,
            power: true,
        }
    }
}

/// A filter query bound to a number key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedFilter {
    pub name: String,
    pub query: String,
}

fn default_saved_filters() -> Vec<SavedFilter> {
    vec![
        SavedFilter { name: "busy".into(), query: "cpu>5".into() },
        SavedFilter { name: "hungry".into(), query: "mem>500M".into() },
        SavedFilter { name: "mine".into(), query: "user:$USER".into() },
        SavedFilter { name: "io".into(), query: "io>100K".into() },
        SavedFilter { name: "stuck".into(), query: "state:D".into() },
    ]
}

/// Where crabmon records the actions it took on other processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    pub enabled: bool,
    /// Empty means the platform state directory.
    pub path: String,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self { enabled: true, path: String::new() }
    }
}

/// Settings for `--record`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordConfig {
    /// Processes kept per frame, busiest first. A full process list is roughly
    /// 1.5 MB per frame on a busy machine, which is gigabytes an hour; 0 keeps
    /// everything anyway if that is what you want.
    pub top_n: usize,
    /// Drop command lines, executable paths and working directories entirely.
    pub omit_paths: bool,
    /// Truncate command lines to this many characters. Electron and Chromium
    /// processes carry multi-kilobyte argv, which is most of a frame otherwise.
    /// 0 disables truncation.
    pub max_cmd_len: usize,
}

impl Default for RecordConfig {
    fn default() -> Self {
        Self { top_n: 100, omit_paths: false, max_cmd_len: 200 }
    }
}

/// Settings for `--serve`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServeConfig {
    /// Per-process series to export. 0 disables them.
    pub top_procs: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self { top_procs: 20 }
    }
}

/// Settings for the process table itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcsConfig {
    /// Explicit column order. Empty means fit as many as the terminal is wide
    /// enough for, which is the behaviour crabmon has always had.
    pub columns: Vec<String>,
    /// Look up listening ports for the rows currently on screen. Off by
    /// default: it walks one fd table per visible row.
    pub ports: bool,
    /// Rows to keep pinned across restarts is meaningless — PIDs are recycled —
    /// but the marker column can be switched off.
    pub pin_marker: bool,
}

impl Default for ProcsConfig {
    fn default() -> Self {
        Self { columns: Vec::new(), ports: false, pin_marker: true }
    }
}

/// Settings for `--remote`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RemoteConfig {
    /// Hold one SSH session open and read a stream of frames, instead of
    /// paying a connection and a process start for every sample. Falls back
    /// automatically when the far end is too old to understand `--stream`.
    pub stream: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self { stream: true }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    pub enabled: bool,
    /// Shell out to `nvidia-smi` each refresh. Off by default: it costs ~50 ms.
    pub nvidia_smi: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self { enabled: true, nvidia_smi: false }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportConfig {
    /// Empty means the current working directory.
    pub dir: String,
    pub format: String,
    /// 0 keeps every process.
    pub top_n: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self { dir: String::new(), format: "json".into(), top_n: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub refresh_ms: u64,
    pub sort_by: SortBy,
    pub sort_desc: bool,
    pub filter: String,
    pub tree: bool,
    /// Show individual threads alongside processes. Off by default: on Linux
    /// the sampler reports every task, which is ~4x more rows than processes.
    pub show_threads: bool,
    /// Aggregate the process table by systemd unit, container or user.
    pub group_by: GroupBy,
    pub layout: Layout,
    pub theme: String,
    pub group_sensors: bool,
    pub show_virtual_ifaces: bool,
    pub virtual_iface_prefixes: Vec<String>,
    /// Samples kept per chart.
    pub history_len: usize,
    pub thresholds: Thresholds,
    pub panels: Panels,
    /// Draw order of the right-hand column.
    pub panel_order: Vec<String>,
    pub gpu: GpuConfig,
    pub procs: ProcsConfig,
    pub export: ExportConfig,
    pub audit: AuditConfig,
    pub serve: ServeConfig,
    pub record: RecordConfig,
    pub remote: RemoteConfig,
    pub colors: BTreeMap<String, String>,
    /// Queries bound to the number keys, in order.
    #[serde(rename = "filter_preset")]
    pub saved_filters: Vec<SavedFilter>,
    #[serde(rename = "alert")]
    pub alerts: Vec<AlertRule>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            refresh_ms: 800,
            sort_by: SortBy::Cpu,
            sort_desc: true,
            filter: String::new(),
            tree: false,
            show_threads: false,
            group_by: GroupBy::None,
            layout: Layout::Dashboard,
            theme: "default".into(),
            group_sensors: true,
            show_virtual_ifaces: false,
            virtual_iface_prefixes: netclass::default_prefixes(),
            history_len: 240,
            thresholds: Thresholds::default(),
            panels: Panels::default(),
            panel_order: ["net", "psi", "sensors", "disks", "power", "gpu", "cgroup"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            gpu: GpuConfig::default(),
            procs: ProcsConfig::default(),
            export: ExportConfig::default(),
            audit: AuditConfig::default(),
            serve: ServeConfig::default(),
            record: RecordConfig::default(),
            remote: RemoteConfig::default(),
            colors: BTreeMap::new(),
            saved_filters: default_saved_filters(),
            alerts: crate::alerts::default_rules(),
        }
    }
}

impl Config {
    pub fn parse(s: &str) -> Config {
        toml::from_str(s).unwrap_or_default()
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Clamp anything a hand-edited file could set to a value that would break
    /// rendering or make the metrics meaningless.
    pub fn sanitize(mut self) -> Config {
        self.refresh_ms = self.refresh_ms.clamp(crate::MIN_REFRESH_MS, crate::MAX_REFRESH_MS);
        self.history_len = self.history_len.clamp(16, 4096);
        self.thresholds.warn = self.thresholds.warn.clamp(0.0, 1.0);
        self.thresholds.crit = self.thresholds.crit.clamp(self.thresholds.warn, 1.0);
        self.thresholds.temp_warn = self.thresholds.temp_warn.clamp(0.0, 1.0);
        self.thresholds.temp_crit = self.thresholds.temp_crit.clamp(self.thresholds.temp_warn, 1.0);
        if crate::theme::Theme::preset(&self.theme).is_none() {
            self.theme = "default".into();
        }
        // The number keys only reach nine presets; more would be unreachable.
        self.saved_filters.truncate(9);
        // An unknown column name would otherwise render as a blank strip with
        // no hint about the typo.
        self.procs.columns.retain(|c| crate::ui::procs::column_by_name(c).is_some());
        self
    }

    /// A saved filter's query with `$USER` expanded, so the shipped defaults
    /// mean something on every machine.
    pub fn saved_filter_query(&self, index: usize) -> Option<String> {
        let f = self.saved_filters.get(index)?;
        let user = std::env::var("USER").or_else(|_| std::env::var("LOGNAME")).unwrap_or_default();
        Some(f.query.replace("$USER", &user))
    }
}

pub fn config_path() -> PathBuf {
    let mut base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    base.push("crabmon");
    base.push("config.toml");
    base
}

pub fn load_from(path: &PathBuf) -> Config {
    match fs::read_to_string(path) {
        Ok(s) => Config::parse(&s).sanitize(),
        Err(_) => Config::default(),
    }
}

/// Best-effort save; a read-only config dir must never take the app down.
pub fn save_to(path: &PathBuf, cfg: &Config) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, cfg.to_toml());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_config_round_trips_through_toml() {
        let cfg = Config::default();
        let parsed = Config::parse(&cfg.to_toml());
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn a_partial_file_keeps_defaults_for_everything_else() {
        let cfg = Config::parse("refresh_ms = 1500\ntheme = \"nord\"\n");
        assert_eq!(cfg.refresh_ms, 1500);
        assert_eq!(cfg.theme, "nord");
        assert_eq!(cfg.sort_by, SortBy::Cpu);
        assert!(!cfg.panel_order.is_empty());
    }

    #[test]
    fn a_corrupt_file_falls_back_to_defaults_instead_of_failing_to_start() {
        assert_eq!(Config::parse("this is not toml {{{"), Config::default());
        assert_eq!(Config::parse(""), Config::default());
    }

    #[test]
    fn sanitize_enforces_the_refresh_floor_sysinfo_needs() {
        let fast = Config { refresh_ms: 10, ..Default::default() }.sanitize();
        assert_eq!(fast.refresh_ms, crate::MIN_REFRESH_MS);
        let slow = Config { refresh_ms: 999_999, ..Default::default() }.sanitize();
        assert_eq!(slow.refresh_ms, crate::MAX_REFRESH_MS);
    }

    #[test]
    fn sanitize_repairs_inverted_and_out_of_range_thresholds() {
        let cfg = Config {
            thresholds: Thresholds { warn: 1.5, crit: 0.1, temp_warn: -1.0, temp_crit: 0.5 },
            ..Default::default()
        }
        .sanitize();
        assert_eq!(cfg.thresholds.warn, 1.0);
        assert!(cfg.thresholds.crit >= cfg.thresholds.warn);
        assert_eq!(cfg.thresholds.temp_warn, 0.0);
    }

    #[test]
    fn unknown_column_names_are_dropped_rather_than_drawn_blank() {
        let cfg = Config {
            procs: ProcsConfig {
                columns: vec!["pid".into(), "nonsense".into(), "cpu".into()],
                ..Default::default()
            },
            ..Default::default()
        }
        .sanitize();
        assert_eq!(cfg.procs.columns, vec!["pid".to_string(), "cpu".to_string()]);
    }

    #[test]
    fn an_unknown_theme_name_falls_back_to_default() {
        let cfg = Config { theme: "chartreuse".into(), ..Default::default() }.sanitize();
        assert_eq!(cfg.theme, "default");
    }

    #[test]
    fn alerts_survive_a_write_read_cycle() {
        let cfg = Config::default();
        let parsed = Config::parse(&cfg.to_toml());
        assert_eq!(parsed.alerts.len(), cfg.alerts.len());
        assert_eq!(parsed.alerts[0].name, "cpu-saturated");
    }

    #[test]
    fn saving_and_loading_uses_the_given_path() {
        let path = std::env::temp_dir().join(format!("crabmon-cfg-{}.toml", std::process::id()));
        let cfg = Config { refresh_ms: 1234, tree: true, ..Default::default() };
        save_to(&path, &cfg);
        let back = load_from(&path);
        assert_eq!(back.refresh_ms, 1234);
        assert!(back.tree);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loading_a_missing_file_yields_defaults() {
        let missing = PathBuf::from("/nonexistent/crabmon/config.toml");
        assert_eq!(load_from(&missing), Config::default());
    }
}
