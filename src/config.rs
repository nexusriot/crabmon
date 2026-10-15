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
    /// OOM kills, crashes and IO errors from `/dev/kmsg`.
    pub kernel: bool,
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
            kernel: true,
        }
    }
}

/// Ceiling on the flight recorder's two frame budgets. At the default trim of
/// 100 processes a frame is tens of kilobytes live, so this is a few hundred
/// megabytes at the very top — deliberate, but bounded.
pub const MAX_FLIGHT_FRAMES: usize = 5_000;

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
    /// Frames of context the flight recorder keeps in memory, ready to be
    /// written out when an alert fires. 0 — the default — switches it off; a
    /// monitor that writes files unasked is not what anyone installed. At the
    /// default 800 ms refresh, 750 frames is ten minutes and roughly 40 MB.
    pub flight: usize,
    /// Frames written after the alert, so a dump shows the recovery and not
    /// only the run-up.
    pub flight_after: usize,
    /// Where dumps are written. Empty means the working directory.
    pub flight_dir: String,
    /// Frames per segment of a rotating recording. 0 switches rotation off,
    /// which is the default: a monitor that writes files unasked is not what
    /// anyone installed.
    pub rotate_frames: usize,
    /// Segments kept, including the one being written. At the default 800 ms
    /// refresh, 4 segments of 1800 frames is about ninety minutes.
    pub rotate_keep: usize,
    /// Where segments are written. Empty means `flight_dir`, and failing
    /// that the working directory.
    pub rotate_dir: String,
}

impl Default for RecordConfig {
    fn default() -> Self {
        Self {
            top_n: 100,
            omit_paths: false,
            max_cmd_len: 200,
            flight: 0,
            flight_after: 30,
            flight_dir: String::new(),
            rotate_frames: 0,
            rotate_keep: 4,
            rotate_dir: String::new(),
        }
    }
}

impl RecordConfig {
    pub fn limits(&self) -> crate::record::Limits {
        crate::record::Limits {
            top_n: self.top_n,
            omit_paths: self.omit_paths,
            max_cmd_len: self.max_cmd_len,
        }
    }
}

/// Settings for `--serve`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServeConfig {
    /// Per-process series to export. 0 disables them.
    pub top_procs: usize,
    /// A bearer token every scrape must present, or empty for none.
    ///
    /// The exporter publishes every command line on the machine, which is
    /// the argv of every process including the ones people put secrets in.
    /// Binding it anywhere but loopback with nothing in front of it was one
    /// flag away, and there was no way to ask for a password.
    ///
    /// `CRABMON_SERVE_TOKEN` overrides this, which is how to keep the secret
    /// out of a config file that is in version control.
    pub token: String,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self { top_procs: 20, token: String::new() }
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
    /// Count each process's open descriptors and read its limit, which adds
    /// the FD column and makes `fd>` and `fd%>` filters and `kind = "fd"`
    /// alerts work. Off by default: unlike PORTS it is sampled for every
    /// process, not only the visible rows, because a filter that only saw the
    /// rows already on screen would be a filter over the wrong set.
    pub fds: bool,
    /// Read each process's proportional and unique memory from
    /// `/proc/<pid>/smaps_rollup`, which adds the PSS column and makes
    /// `pss>` filters work. Off by default for the same reason `fds` is: one
    /// more read per process per sample.
    ///
    /// Worth the cost when the question is "which program is using the
    /// memory" on a machine that forks — a browser, a database with workers
    /// — where RSS counts every shared page once per process holding it.
    pub pss: bool,
    /// Rows to keep pinned across restarts is meaningless — PIDs are recycled —
    /// but the marker column can be switched off.
    pub pin_marker: bool,
}

impl Default for ProcsConfig {
    fn default() -> Self {
        Self { columns: Vec::new(), ports: false, fds: false, pss: false, pin_marker: true }
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
            panel_order: crate::ui::SIDE_PANELS.iter().map(|s| s.to_string()).collect(),
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
    pub fn serve_token(&self) -> Option<String> {
        // The environment wins: a token in a config file is a secret at rest
        // in a file people commit.
        std::env::var("CRABMON_SERVE_TOKEN")
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .or_else(|| Some(self.serve.token.trim().to_string()).filter(|t| !t.is_empty()))
    }

    pub fn parse(s: &str) -> Config {
        Config::parse_checked(s).unwrap_or_default()
    }

    /// As `parse`, but says why a file was rejected.
    ///
    /// `parse` swallowing the error is how a single typo — a stray bracket, an
    /// alert rule with a misspelled `kind` — silently discarded *every*
    /// setting in the file, alert rules included, with nothing on screen to
    /// say so. A monitor whose alerts have quietly stopped existing is the
    /// worst possible failure mode for one.
    pub fn parse_checked(s: &str) -> Result<Config, String> {
        toml::from_str(s).map_err(|e| e.message().replace('\n', " "))
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Clamp anything a hand-edited file could set to a value that would break
    /// rendering or make the metrics meaningless.
    pub fn sanitize(mut self) -> Config {
        self.refresh_ms = self.refresh_ms.clamp(crate::MIN_REFRESH_MS, crate::MAX_REFRESH_MS);
        self.history_len = self.history_len.clamp(16, 4096);
        // TOML has a real `nan` literal, and `NaN.clamp(0.0, 1.0)` returns NaN
        // rather than repairing it — which then hits `clamp(NaN, 1.0)`, whose
        // `min <= max` assertion aborts the process. A hand-edited file must
        // degrade to defaults, not stop crabmon from starting at all.
        let defaults = Thresholds::default();
        let sane = |v: f64, fallback: f64| if v.is_finite() { v } else { fallback };
        self.thresholds.warn = sane(self.thresholds.warn, defaults.warn).clamp(0.0, 1.0);
        self.thresholds.crit =
            sane(self.thresholds.crit, defaults.crit).clamp(self.thresholds.warn, 1.0);
        self.thresholds.temp_warn =
            sane(self.thresholds.temp_warn, defaults.temp_warn).clamp(0.0, 1.0);
        self.thresholds.temp_crit = sane(self.thresholds.temp_crit, defaults.temp_crit)
            .clamp(self.thresholds.temp_warn, 1.0);
        if crate::theme::Theme::preset(&self.theme).is_none() {
            self.theme = "default".into();
        }
        // The number keys only reach nine presets; more would be unreachable.
        self.saved_filters.truncate(9);
        // An unknown column name would otherwise render as a blank strip with
        // no hint about the typo.
        self.procs.columns.retain(|c| crate::ui::procs::column_by_name(c).is_some());
        // The same for the right-hand column. `panel_order` was read straight
        // into the draw loop, where an unrecognised name fell through a `match`
        // arm and simply did nothing — so a typo, or a panel renamed between
        // releases, silently dropped that panel with nothing anywhere to say
        // why. Dropping the name here at least makes the file and the screen
        // agree, and a name that survives is one that draws something.
        self.panel_order.retain(|p| crate::ui::is_side_panel(p));
        // ...and a file that lists none of them leaves the column blank
        // forever, which reads as a rendering bug rather than as a setting.
        if self.panel_order.is_empty() {
            self.panel_order = crate::ui::SIDE_PANELS.iter().map(|s| s.to_string()).collect();
        }
        // A panel named twice would be drawn twice, splitting the column
        // between two copies of the same thing.
        let mut seen = std::collections::HashSet::new();
        self.panel_order.retain(|p| seen.insert(p.clone()));
        // A rotating recording is bounded disk, and the bound has to be one
        // somebody can reason about: four segments of a million frames is
        // not a setting, it is a full filesystem.
        self.record.rotate_frames = self.record.rotate_frames.min(1_000_000);
        self.record.rotate_keep = self.record.rotate_keep.clamp(1, 64);
        // The flight recorder's ring is live memory, and it holds whole
        // frames. A hand-edited `flight = 100000000` would be an OOM in the
        // process that exists to make memory pressure visible.
        self.record.flight = self.record.flight.min(MAX_FLIGHT_FRAMES);
        self.record.flight_after = self.record.flight_after.min(MAX_FLIGHT_FRAMES);
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
    load_checked(path).0
}

/// Load a config, and say what was wrong with it.
///
/// A missing file is not an error — that is simply a first run. A file that
/// exists and cannot be parsed is: every setting in it has just been ignored,
/// and the caller has to tell someone rather than carry on with defaults as
/// though the file said nothing.
pub fn load_checked(path: &PathBuf) -> (Config, Option<String>) {
    match fs::read_to_string(path) {
        Ok(s) => match Config::parse_checked(&s) {
            Ok(cfg) => (cfg.sanitize(), None),
            Err(e) => (Config::default(), Some(e)),
        },
        Err(_) => (Config::default(), None),
    }
}

/// Best-effort save; a read-only config dir must never take the app down.
///
/// Written to a sibling temp file and renamed over the target, because
/// `fs::write` truncates first: a crash, a full disk or a kill between the
/// truncate and the write left the user with an empty config and every
/// setting gone. A rename is atomic on the same filesystem, so the old file
/// survives intact until the new one is complete.
pub fn save_to(path: &PathBuf, cfg: &Config) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let body = cfg.to_toml();
    // `to_toml` swallows its own errors into an empty string; writing that
    // over a good config would lose it just as thoroughly as a truncation.
    if body.trim().is_empty() {
        return;
    }
    let tmp = path.with_extension(format!("toml.{}.tmp", std::process::id()));
    if fs::write(&tmp, &body).is_ok() && fs::rename(&tmp, path).is_ok() {
        return;
    }
    let _ = fs::remove_file(&tmp);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The failure this is here to prevent: a file with one typo in it used
    /// to be discarded whole, silently — every threshold, every column and
    /// every alert rule gone, with the TUI then writing the defaults back
    /// over the file on exit.
    #[test]
    fn an_unparseable_file_is_reported_rather_than_silently_discarded() {
        let bad = "refresh_ms = 1500\n[[alert]\nname = \"hot\"\n";
        assert!(Config::parse_checked(bad).is_err());

        let path =
            std::env::temp_dir().join(format!("crabmon-cfg-bad-{}.toml", std::process::id()));
        std::fs::write(&path, bad).unwrap();
        let (cfg, err) = load_checked(&path);
        assert!(err.is_some(), "a broken file parsed clean");
        assert_eq!(cfg, Config::default());
        let _ = std::fs::remove_file(&path);
    }

    /// An alert rule naming a kind that does not exist is the same failure
    /// wearing a different hat: it takes the whole file with it.
    #[test]
    fn a_bad_alert_kind_is_an_error_not_a_config_with_no_alerts() {
        let out = Config::parse_checked(
            "[[alert]]\nname = \"hot\"\nkind = \"cpu-ish\"\nthreshold = 90.0\n",
        );
        assert!(out.is_err(), "{out:?}");
    }

    #[test]
    fn a_missing_file_is_a_first_run_not_an_error() {
        let path =
            std::env::temp_dir().join(format!("crabmon-cfg-absent-{}.toml", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let (cfg, err) = load_checked(&path);
        assert_eq!(err, None);
        assert_eq!(cfg, Config::default());
    }

    /// `fs::write` truncates before it writes; the save has to leave either
    /// the old file or the new one, never half of either.
    #[test]
    fn saving_replaces_the_file_atomically_and_leaves_no_temporary_behind() {
        let dir = std::env::temp_dir().join(format!("crabmon-cfg-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("config.toml");

        let mut cfg = Config { refresh_ms: 1500, ..Default::default() };
        save_to(&path, &cfg);
        assert_eq!(load_checked(&path).0.refresh_ms, 1500);

        cfg.refresh_ms = 2500;
        save_to(&path, &cfg);
        assert_eq!(load_checked(&path).0.refresh_ms, 2500);

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temporary files left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
    fn an_unknown_panel_name_is_dropped_rather_than_silently_drawing_nothing() {
        // `panel_order` went straight into the draw loop, where a name no arm
        // matched fell through and did nothing at all — so a typo cost you a
        // panel with nothing anywhere to say why.
        let cfg = Config::parse("panel_order = [\"net\", \"nonsense\", \"disks\"]").sanitize();
        assert_eq!(cfg.panel_order, vec!["net", "disks"]);
        for name in &cfg.panel_order {
            assert!(crate::ui::is_side_panel(name), "{name} is not a panel");
        }
    }

    #[test]
    fn a_panel_order_that_names_nothing_usable_falls_back_to_the_defaults() {
        // An empty right-hand column reads as a rendering bug, not a setting.
        let cfg = Config::parse("panel_order = [\"typo\", \"alsotypo\"]").sanitize();
        assert_eq!(cfg.panel_order, Config::default().panel_order);
        let cfg = Config::parse("panel_order = []").sanitize();
        assert_eq!(cfg.panel_order, Config::default().panel_order);
    }

    #[test]
    fn a_panel_named_twice_is_drawn_once() {
        // Two slots for one panel splits the column between two copies of it.
        let cfg = Config::parse("panel_order = [\"net\", \"disks\", \"net\"]").sanitize();
        assert_eq!(cfg.panel_order, vec!["net", "disks"]);
    }

    /// The default must survive its own validator, or every start would
    /// silently rewrite the shipped config.
    #[test]
    fn every_default_panel_is_one_the_side_column_can_draw() {
        let cfg = Config::default();
        assert_eq!(cfg.clone().sanitize().panel_order, cfg.panel_order);
        assert_eq!(cfg.panel_order.len(), crate::ui::SIDE_PANELS.len());
    }

    #[test]
    fn sanitize_enforces_the_refresh_floor_sysinfo_needs() {
        let fast = Config { refresh_ms: 10, ..Default::default() }.sanitize();
        assert_eq!(fast.refresh_ms, crate::MIN_REFRESH_MS);
        let slow = Config { refresh_ms: 999_999, ..Default::default() }.sanitize();
        assert_eq!(slow.refresh_ms, crate::MAX_REFRESH_MS);
    }

    #[test]
    fn the_flight_recorders_frame_budgets_are_clamped() {
        // The ring holds whole frames in memory. A hand-edited
        // `flight = 100000000` would be an out-of-memory kill in the process
        // that exists to make memory pressure visible.
        let cfg =
            Config::parse("[record]\nflight = 100000000\nflight_after = 99999999\n").sanitize();
        assert_eq!(cfg.record.flight, MAX_FLIGHT_FRAMES);
        assert_eq!(cfg.record.flight_after, MAX_FLIGHT_FRAMES);

        // ...and the default is off, so nothing writes files unasked.
        assert_eq!(Config::default().record.flight, 0);
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

    /// The shipped `mine` filter is `user:$USER`. Without the expansion it
    /// matches a user literally called `$USER`, i.e. nothing, and the number
    /// key bound to it silently empties the table.
    #[test]
    fn a_saved_filter_expands_the_user_placeholder() {
        let cfg = Config {
            saved_filters: vec![
                SavedFilter { name: "mine".into(), query: "user:$USER".into() },
                SavedFilter { name: "busy".into(), query: "cpu>5".into() },
            ],
            ..Default::default()
        };
        let user = std::env::var("USER").or_else(|_| std::env::var("LOGNAME")).unwrap_or_default();

        let mine = cfg.saved_filter_query(0).unwrap();
        assert_eq!(mine, format!("user:{user}"));
        assert!(!mine.contains("$USER"), "the placeholder reached the filter parser");

        // A query with no placeholder is passed through untouched.
        assert_eq!(cfg.saved_filter_query(1).as_deref(), Some("cpu>5"));
    }

    #[test]
    fn a_number_key_with_no_saved_filter_behind_it_does_nothing() {
        // The keys 1..9 are always bound; only the first few have a filter.
        let cfg = Config { saved_filters: vec![], ..Default::default() };
        assert_eq!(cfg.saved_filter_query(0), None);
        assert_eq!(Config::default().saved_filter_query(99), None);
    }

    /// TOML has a real `nan` literal, and `NaN.clamp(0.0, 1.0)` returns NaN
    /// rather than repairing it — which then tripped `clamp`'s `min <= max`
    /// assertion and aborted every mode before the UI existed.
    #[test]
    fn a_non_finite_threshold_falls_back_instead_of_aborting_the_process() {
        for field in ["warn", "crit", "temp_warn", "temp_crit"] {
            for value in ["nan", "inf", "-inf"] {
                let cfg = Config::parse(&format!("[thresholds]\n{field} = {value}\n")).sanitize();
                let t = &cfg.thresholds;
                for (name, v) in [
                    ("warn", t.warn),
                    ("crit", t.crit),
                    ("temp_warn", t.temp_warn),
                    ("temp_crit", t.temp_crit),
                ] {
                    assert!(v.is_finite(), "{field}={value} left {name} at {v}");
                    assert!((0.0..=1.0).contains(&v), "{field}={value} left {name} at {v}");
                }
                assert!(t.warn <= t.crit, "{field}={value} inverted the pair");
                assert!(t.temp_warn <= t.temp_crit, "{field}={value} inverted the temp pair");
            }
        }
    }
}
