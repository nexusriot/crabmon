//! Application state and input handling.
//!
//! `App` owns no terminal and performs no IO of its own beyond the metric
//! source it is handed, so the whole interaction model can be driven from tests.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;

use crate::alerts::{ActiveAlert, AlertEngine};
use crate::audit;
use crate::config::Config;
use crate::export::{self, ExportFormat};
use crate::filter::{self, Filter};
use crate::history::History;
use crate::metrics::procgroup::GroupBy;
use crate::metrics::{sockets, MetricSource, ProcRow, Snapshot};
use crate::sort::{self, SortBy};
use crate::theme::Theme;
use crate::tree::{self, TreeRow};
use crate::ui::Layout;

/// Samples kept per process for the sparkline column. Short on purpose: one
/// vector per process on a 2000-task machine adds up.
pub const PROC_SPARK_LEN: usize = 24;

/// Signals offered in the signal menu, in the order they are listed.
pub const SIGNAL_NAMES: [&str; 9] = [
    "SIGTERM", "SIGKILL", "SIGINT", "SIGHUP", "SIGQUIT", "SIGSTOP", "SIGCONT", "SIGUSR1", "SIGUSR2",
];

#[cfg(unix)]
pub fn signal_by_name(name: &str) -> Option<nix::sys::signal::Signal> {
    use nix::sys::signal::Signal::*;
    Some(match name {
        "SIGTERM" => SIGTERM,
        "SIGKILL" => SIGKILL,
        "SIGINT" => SIGINT,
        "SIGHUP" => SIGHUP,
        "SIGQUIT" => SIGQUIT,
        "SIGSTOP" => SIGSTOP,
        "SIGCONT" => SIGCONT,
        "SIGUSR1" => SIGUSR1,
        "SIGUSR2" => SIGUSR2,
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Filter,
    SignalMenu,
    ConfirmKill,
    Renice,
    Affinity,
    Detail,
    Help,
    Alerts,
    AuditLog,
}

/// What `run_app` must do after handling an event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    None,
    Quit,
    Export,
}

/// A process targeted by a pending signal, captured with enough identity to
/// notice PID reuse between opening the menu and confirming.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub pid: u32,
    pub name: String,
    pub start_time_unix: u64,
}

impl From<&ProcRow> for Target {
    fn from(r: &ProcRow) -> Target {
        Target { pid: r.pid, name: r.name.clone(), start_time_unix: r.start_time_unix }
    }
}

pub struct App {
    pub cfg: Config,
    pub theme: Theme,
    pub source: Box<dyn MetricSource>,
    pub snap: Snapshot,

    /// Tree metadata, positionally aligned with `rows()` when tree view is on.
    pub tree_rows: Vec<TreeRow>,
    pub selected: usize,
    pub selected_pid: Option<u32>,
    pub scroll: usize,
    /// Rows the process table could show at the last draw.
    pub page: usize,

    pub mode: Mode,
    pub filter_text: String,
    pub filter: Filter,
    pub filter_error: Option<String>,
    pub input: String,
    pub input_error: Option<String>,
    pub signal_index: usize,
    pub target: Option<Target>,
    /// The processes an open prompt will act on, captured when it opened. The
    /// identities must be frozen here: rebuilding them at confirm time would
    /// defeat the PID-reuse guard, because the recycled process would look like
    /// a perfectly valid target.
    pub pending_targets: Vec<Target>,

    pub refresh: Duration,
    pub last_refresh: Instant,
    pub status: Option<(String, Instant)>,

    pub cpu_hist: History,
    pub per_core_hist: Vec<History>,
    pub mem_hist: History,
    pub swap_hist: History,
    pub net_rx_hist: History,
    pub net_tx_hist: History,
    pub disk_r_hist: History,
    pub disk_w_hist: History,

    pub alerts: AlertEngine,
    pub active_alerts: Vec<ActiveAlert>,
    started: Instant,

    /// Set during draw so mouse clicks can be mapped back to rows and columns.
    pub proc_area: Rect,
    pub header_cols: Vec<(u16, u16, SortBy)>,

    /// Where to persist `cfg`. `None` disables saving entirely, which is what
    /// tests and embedders want — an `App` must never write to the user's real
    /// config file just because it was constructed.
    pub config_path: Option<PathBuf>,

    /// Sampling is suspended; the last snapshot stays on screen.
    pub paused: bool,
    /// PIDs tagged for a bulk action.
    pub tagged: HashSet<u32>,
    /// PIDs held at the top of the table, and shown even when the filter would
    /// hide them, so one process can be watched while the rest churns.
    pub pinned: HashSet<u32>,
    /// Listening ports for the rows last drawn, when the PORTS column is on.
    pub ports: HashMap<u32, Vec<u16>>,
    /// Which tick `ports` was filled for, and for which rows.
    ports_at: (u64, Vec<u32>),
    /// Bumped once per sample, so the port lookup runs per tick rather than
    /// per draw — the UI redraws several times a second even when idle.
    generation: u64,
    /// Selection and viewport of the grouped view, kept apart from the process
    /// table's so switching back and forth does not lose either.
    pub group_selected: usize,
    pub group_scroll: usize,
    group_page: usize,
    group_len: usize,
    /// First visible line of whichever popup is open, and how many lines it
    /// last had room for.
    pub popup_scroll: usize,
    pub popup_height: usize,
    /// The frame the last tick saw, for sources that sample asynchronously.
    last_frame_id: Option<u64>,
    stale: bool,
    /// Per-PID CPU history for the sparkline column.
    cpu_spark: HashMap<u32, Vec<f32>>,
    /// Sockets of the process whose detail pane is open, fetched on demand.
    pub detail_sockets: Vec<sockets::Socket>,
    /// Where actions are logged. `None` disables the audit log.
    pub audit_path: Option<PathBuf>,
    /// Newest-first audit entries, loaded when the viewer opens.
    pub audit_entries: Vec<String>,

    /// Backing store for `rows()`; private so the view can only be replaced
    /// through `rebuild_view`.
    rows: Vec<ProcRow>,
}

impl App {
    pub fn new(cfg: Config, mut source: Box<dyn MetricSource>) -> Self {
        let theme = Theme::preset(&cfg.theme).unwrap_or_default().with_overrides(&cfg.colors);
        let snap = source.snapshot(Duration::from_secs(0));
        let hist = || History::new(cfg.history_len);
        let mut app = App {
            refresh: Duration::from_millis(cfg.refresh_ms),
            filter_text: cfg.filter.clone(),
            filter: filter::parse(&cfg.filter).unwrap_or_default(),
            alerts: AlertEngine::new(cfg.alerts.clone()),
            theme,
            source,
            tree_rows: Vec::new(),
            selected: 0,
            selected_pid: None,
            scroll: 0,
            page: 1,
            mode: Mode::Normal,
            filter_error: None,
            input: String::new(),
            input_error: None,
            signal_index: 0,
            target: None,
            pending_targets: Vec::new(),
            last_refresh: Instant::now(),
            status: None,
            cpu_hist: hist(),
            per_core_hist: (0..snap.cpu.per_core.len()).map(|_| hist()).collect(),
            mem_hist: hist(),
            swap_hist: hist(),
            net_rx_hist: hist(),
            net_tx_hist: hist(),
            disk_r_hist: hist(),
            disk_w_hist: hist(),
            active_alerts: Vec::new(),
            started: Instant::now(),
            proc_area: Rect::default(),
            header_cols: Vec::new(),
            config_path: None,
            paused: false,
            tagged: HashSet::new(),
            pinned: HashSet::new(),
            ports: HashMap::new(),
            ports_at: (0, Vec::new()),
            generation: 0,
            group_selected: 0,
            group_scroll: 0,
            group_page: 1,
            group_len: 0,
            popup_scroll: 0,
            popup_height: 1,
            last_frame_id: None,
            stale: false,
            cpu_spark: HashMap::new(),
            detail_sockets: Vec::new(),
            audit_path: None,
            audit_entries: Vec::new(),
            rows: Vec::new(),
            snap,
            cfg,
        };
        app.last_frame_id = app.source.frame_id();
        app.rebuild_view();
        app
    }

    // -------------------------------------------------------------- refresh

    pub fn due(&self) -> bool {
        !self.paused && self.last_refresh.elapsed() >= self.refresh
    }

    /// `(position, length)` when the source has a timeline (a replay).
    pub fn timeline(&self) -> Option<(usize, usize)> {
        self.source.timeline()
    }

    pub fn source_label(&self) -> Option<String> {
        self.source.label()
    }

    pub fn tick(&mut self) {
        let dt = self.last_refresh.elapsed();
        self.snap = self.source.snapshot(dt);
        self.last_refresh = Instant::now();
        self.generation = self.generation.wrapping_add(1);

        // A source that samples on another thread hands back the same frame
        // until a new one is ready. Pushing that into the charts would draw a
        // flat line that looks like real measurement; skipping it leaves the
        // last real sample on screen, which is what actually happened.
        let id = self.source.frame_id();
        let fresh = id.is_none() || id != self.last_frame_id;
        self.last_frame_id = id;
        self.stale = !fresh;
        if !fresh {
            return;
        }

        self.record_history();
        self.active_alerts = self.alerts.evaluate(&self.snap, self.started.elapsed().as_secs());
        self.rebuild_view();
    }

    /// Whether the last tick produced nothing new, i.e. the source is still
    /// working on a sample. Shown in the status line so a frozen-looking
    /// dashboard says why.
    pub fn is_stale(&self) -> bool {
        self.stale
    }

    fn record_history(&mut self) {
        self.cpu_hist.push(self.snap.cpu.avg());
        // Cores can appear or disappear (CPU hotplug, containers).
        self.per_core_hist
            .resize_with(self.snap.cpu.per_core.len(), || History::new(self.cfg.history_len));
        for (h, v) in self.per_core_hist.iter_mut().zip(&self.snap.cpu.per_core) {
            h.push(*v as f64);
        }
        self.mem_hist.push(self.snap.mem.ratio() * 100.0);
        self.swap_hist.push(self.snap.mem.swap_ratio() * 100.0);
        let (rx, tx) = self.snap.net_totals(self.cfg.show_virtual_ifaces);
        self.net_rx_hist.push(rx);
        self.net_tx_hist.push(tx);
        let (r, w) = self.snap.disk_io_totals();
        self.disk_r_hist.push(r);
        self.disk_w_hist.push(w);
        self.record_process_history();
    }

    /// A short CPU history per process, for the sparkline column. Bounded by
    /// evicting processes that are no longer in the snapshot.
    fn record_process_history(&mut self) {
        let live: HashSet<u32> = self.snap.procs.iter().map(|p| p.pid).collect();
        self.cpu_spark.retain(|pid, _| live.contains(pid));
        // Tags and pins outlive their processes otherwise: the "[N tagged]"
        // counter kept counting the dead, and a pinned PID could be recycled
        // onto an unrelated process and quietly hoisted to the top.
        self.tagged.retain(|pid| live.contains(pid));
        self.pinned.retain(|pid| live.contains(pid));
        for p in &self.snap.procs {
            let series = self.cpu_spark.entry(p.pid).or_default();
            if series.len() >= PROC_SPARK_LEN {
                series.remove(0);
            }
            series.push(p.cpu);
        }
    }

    pub fn cpu_spark(&self, pid: u32) -> &[f32] {
        self.cpu_spark.get(&pid).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Recompute the filtered, sorted (or tree-ordered) process view and keep
    /// the highlight on the same process rather than the same row number.
    pub fn rebuild_view(&mut self) {
        let hide_threads = !self.cfg.show_threads && threads_are_distinguishable(&self.snap.procs);
        let mut rows: Vec<ProcRow> = self
            .snap
            .procs
            .iter()
            // A pinned process stays visible through the filter and the thread
            // hiding: pinning it is a more specific instruction than either.
            .filter(|p| {
                self.pinned.contains(&p.pid)
                    || ((!hide_threads || p.threads.is_some()) && self.filter.matches(p))
            })
            .cloned()
            .collect();

        if self.cfg.tree {
            self.tree_rows = tree::flatten(&rows, self.cfg.sort_by, self.cfg.sort_desc);
            // `flatten` yields each index exactly once, so the rows can be moved
            // into tree order rather than cloned a second time.
            let mut ordered = Vec::with_capacity(self.tree_rows.len());
            for t in &self.tree_rows {
                ordered.push(std::mem::take(&mut rows[t.index]));
            }
            rows = ordered;
        } else {
            self.tree_rows.clear();
            sort::sort_rows(&mut rows, self.cfg.sort_by, self.cfg.sort_desc);
            // Pins float to the top of a flat list. In tree view they do not:
            // the ordering there is structural, and hoisting a child out of its
            // parent would draw a tree that is not one.
            if !self.pinned.is_empty() {
                let pinned = &self.pinned;
                let (mut top, rest): (Vec<ProcRow>, Vec<ProcRow>) =
                    rows.into_iter().partition(|r| pinned.contains(&r.pid));
                top.extend(rest);
                rows = top;
            }
        }

        self.rows = rows;

        self.selected = match self.selected_pid.and_then(|pid| self.index_of_pid(pid)) {
            Some(i) => i,
            None => self.selected.min(self.rows.len().saturating_sub(1)),
        };
        self.selected_pid = self.rows.get(self.selected).map(|r| r.pid);
        self.clamp_scroll();
    }

    /// The aggregate table shown when grouping is active.
    pub fn groups(&self) -> Vec<GroupRow> {
        group_rows(self.rows(), self.cfg.group_by)
    }

    pub fn grouped(&self) -> bool {
        self.cfg.group_by != GroupBy::None
    }

    /// Listening ports for the rows on screen, refreshed at most once per
    /// sample. The draw loop runs several times a second even when nothing
    /// changes, and this is the one column that costs syscalls.
    pub fn refresh_ports(&mut self, visible: &[u32]) {
        if self.ports_at.0 == self.generation && self.ports_at.1 == visible {
            return;
        }
        self.ports = sockets::listening_ports(visible);
        self.ports_at = (self.generation, visible.to_vec());
    }

    /// Told by the grouped view how many rows it can show, and how many there
    /// are, so the selection and viewport stay inside the table.
    pub fn set_group_page(&mut self, page: usize, len: usize) {
        self.group_page = page.max(1);
        self.group_len = len;
        self.clamp_group_scroll();
    }

    fn clamp_group_scroll(&mut self) {
        if self.group_len == 0 {
            self.group_selected = 0;
            self.group_scroll = 0;
            return;
        }
        self.group_selected = self.group_selected.min(self.group_len - 1);
        if self.group_selected < self.group_scroll {
            self.group_scroll = self.group_selected;
        } else if self.group_selected >= self.group_scroll + self.group_page {
            self.group_scroll = self.group_selected + 1 - self.group_page;
        }
        self.group_scroll = self.group_scroll.min(self.group_len.saturating_sub(self.group_page));
    }

    fn move_group_selection(&mut self, delta: isize) {
        if self.group_len == 0 {
            return;
        }
        let next = (self.group_selected as isize + delta).clamp(0, self.group_len as isize - 1);
        self.group_selected = next as usize;
        self.clamp_group_scroll();
    }

    /// Open the selected group: filter the process table down to it and leave
    /// the aggregate view. Without this the grouped view is a dead end — it can
    /// tell you docker.service is eating the machine but not which process.
    fn open_group(&mut self) {
        let groups = self.groups();
        let Some(group) = groups.get(self.group_selected) else { return };
        let name = group.name.clone();
        // Filter terms are whitespace-separated, so a name with a space in it
        // cannot be expressed as one. Say so rather than filtering to nonsense.
        if name.split_whitespace().count() > 1 {
            self.set_status(format!("cannot filter on a name with spaces: {name}"));
            return;
        }
        let field = match self.cfg.group_by {
            GroupBy::Service => "service",
            GroupBy::Container => "container",
            GroupBy::User => "user",
            GroupBy::None => return,
        };
        self.filter_text = format!("{field}:{}", name.to_lowercase());
        self.apply_filter();
        self.cfg.group_by = GroupBy::None;
        self.set_status(format!("{field}: {name}"));
    }

    /// Pin or unpin the selected process.
    fn toggle_pin(&mut self) {
        let Some(row) = self.selected_row() else { return };
        let (pid, name) = (row.pid, row.name.clone());
        if self.pinned.remove(&pid) {
            self.set_status(format!("unpinned {name}"));
        } else {
            self.pinned.insert(pid);
            self.set_status(format!("pinned {name} ({pid})"));
        }
        self.rebuild_view();
    }

    pub fn index_of_pid(&self, pid: u32) -> Option<usize> {
        self.rows.iter().position(|r| r.pid == pid)
    }

    pub fn rows(&self) -> &[ProcRow] {
        &self.rows
    }

    pub fn selected_row(&self) -> Option<&ProcRow> {
        self.rows.get(self.selected)
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), Instant::now()));
    }

    /// Status messages fade after a few seconds so they do not look like state.
    pub fn status_text(&self) -> Option<&str> {
        match &self.status {
            Some((m, at)) if at.elapsed() < Duration::from_secs(5) => Some(m.as_str()),
            _ => None,
        }
    }

    // ------------------------------------------------------------ selection

    fn clamp_scroll(&mut self) {
        let page = self.page.max(1);
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + page {
            self.scroll = self.selected + 1 - page;
        }
        let max_scroll = self.rows.len().saturating_sub(page);
        self.scroll = self.scroll.min(max_scroll);
    }

    pub fn set_page(&mut self, page: usize) {
        self.page = page.max(1);
        self.clamp_scroll();
    }

    pub fn select(&mut self, index: usize) {
        if self.rows.is_empty() {
            self.selected = 0;
            self.selected_pid = None;
            return;
        }
        self.selected = index.min(self.rows.len() - 1);
        self.selected_pid = self.rows.get(self.selected).map(|r| r.pid);
        self.clamp_scroll();
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.grouped() {
            self.move_group_selection(delta);
            return;
        }
        let next = (self.selected as isize + delta).max(0) as usize;
        self.select(next);
    }

    // ---------------------------------------------------------------- input

    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        // Ctrl-C must always quit: in raw mode the terminal will not do it for us.
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Action::Quit;
        }
        match self.mode {
            Mode::Normal => self.on_normal_key(key),
            Mode::Filter => {
                self.on_filter_key(key);
                Action::None
            }
            Mode::SignalMenu => {
                self.on_signal_menu_key(key);
                Action::None
            }
            Mode::ConfirmKill => {
                self.on_confirm_key(key);
                Action::None
            }
            Mode::Renice => {
                self.on_renice_key(key);
                Action::None
            }
            Mode::Affinity => {
                self.on_affinity_key(key);
                Action::None
            }
            Mode::Detail | Mode::Help | Mode::Alerts | Mode::AuditLog => {
                self.on_popup_key(key);
                Action::None
            }
        }
    }

    /// Enter a popup, always from the top: leaving a scrolled help open and
    /// coming back to the middle of it reads as a rendering bug.
    fn open_popup(&mut self, mode: Mode) {
        self.popup_scroll = 0;
        self.mode = mode;
    }

    /// Popups scroll rather than truncating. The key list alone is longer than
    /// an 80x24 terminal can show, so the rows past the bottom used to be
    /// simply unreachable.
    fn on_popup_key(&mut self, key: KeyEvent) {
        let page = self.popup_page();
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') | KeyCode::Char('?') => {
                self.popup_scroll = 0;
                self.mode = Mode::Normal;
            }
            KeyCode::Down | KeyCode::Char('j') => self.scroll_popup(1),
            KeyCode::Up | KeyCode::Char('k') => self.scroll_popup(-1),
            KeyCode::PageDown | KeyCode::Char(' ') => self.scroll_popup(page),
            KeyCode::PageUp => self.scroll_popup(-page),
            KeyCode::Home => self.popup_scroll = 0,
            KeyCode::End => self.scroll_popup(isize::MAX / 2),
            _ => {}
        }
    }

    fn scroll_popup(&mut self, delta: isize) {
        let max = self.popup_lines().saturating_sub(self.popup_page().max(1) as usize);
        let next = (self.popup_scroll as isize).saturating_add(delta).clamp(0, max as isize);
        self.popup_scroll = next as usize;
    }

    /// Rows the open popup holds, so scrolling can stop at the end of them.
    pub fn popup_lines(&self) -> usize {
        match self.mode {
            Mode::Help => crate::ui::popups::KEYS.len(),
            Mode::Alerts => self.alerts.rules.len() + self.alerts.history_len(),
            Mode::AuditLog => self.audit_entries.len(),
            Mode::Detail => crate::ui::popups::DETAIL_FIELDS + self.detail_sockets.len(),
            _ => 0,
        }
    }

    /// Set by the popup as it draws, so paging matches what is on screen.
    pub fn popup_page(&self) -> isize {
        self.popup_height.max(1) as isize
    }

    fn on_normal_key(&mut self, key: KeyEvent) -> Action {
        match key.code {
            // Esc no longer quits: it clears the filter, which is what a filtered
            // list makes you reach for.
            KeyCode::Esc => {
                if self.filter_text.is_empty() {
                    return Action::None;
                }
                self.filter_text.clear();
                self.apply_filter();
            }
            KeyCode::Char('q') if key.modifiers == KeyModifiers::NONE => return Action::Quit,

            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::PageDown => self.move_selection(self.page as isize),
            KeyCode::PageUp => self.move_selection(-(self.page as isize)),
            KeyCode::Home => {
                if self.grouped() {
                    self.move_group_selection(isize::MIN / 2);
                } else {
                    self.select(0);
                }
            }
            KeyCode::End => {
                if self.grouped() {
                    self.move_group_selection(isize::MAX / 2);
                } else {
                    self.select(self.rows.len().saturating_sub(1));
                }
            }

            KeyCode::Char('c') => self.set_sort(SortBy::Cpu),
            KeyCode::Char('m') => self.set_sort(SortBy::Mem),
            KeyCode::Char('p') => self.set_sort(SortBy::Pid),
            KeyCode::Char('n') => self.set_sort(SortBy::Name),
            KeyCode::Char('d') => self.set_sort(SortBy::Disk),
            KeyCode::Char('>') | KeyCode::Char('.') => self.set_sort(self.cfg.sort_by.next()),
            KeyCode::Char('<') | KeyCode::Char(',') => self.set_sort(self.cfg.sort_by.prev()),
            KeyCode::Char('s') => {
                self.cfg.sort_desc = !self.cfg.sort_desc;
                self.rebuild_view();
            }

            KeyCode::Char('/') => {
                self.mode = Mode::Filter;
            }
            KeyCode::Char('T') => {
                self.cfg.tree = !self.cfg.tree;
                self.set_status(if self.cfg.tree { "tree view" } else { "flat view" });
                self.rebuild_view();
            }
            KeyCode::Tab => {
                self.cfg.layout = self.cfg.layout.next();
                self.set_status(format!("layout: {}", self.cfg.layout.name()));
            }
            KeyCode::BackTab => {
                self.cfg.layout = self.cfg.layout.prev();
                self.set_status(format!("layout: {}", self.cfg.layout.name()));
            }
            KeyCode::Char('v') => {
                self.cfg.show_virtual_ifaces = !self.cfg.show_virtual_ifaces;
                self.set_status(if self.cfg.show_virtual_ifaces {
                    "counting virtual interfaces"
                } else {
                    "physical interfaces only"
                });
            }
            KeyCode::Char('H') => {
                self.cfg.show_threads = !self.cfg.show_threads;
                self.set_status(if self.cfg.show_threads {
                    "showing threads"
                } else {
                    "hiding threads"
                });
                self.rebuild_view();
            }
            KeyCode::Char('g') => {
                self.cfg.group_sensors = !self.cfg.group_sensors;
                self.set_status(if self.cfg.group_sensors {
                    "sensors grouped"
                } else {
                    "sensors expanded"
                });
            }

            // Pause sampling so a spike can actually be read.
            KeyCode::Char('z') => {
                self.paused = !self.paused;
                // A pause stops `tick`, so a "sampling…" left over from the
                // last one would sit in the status line for good.
                self.stale = false;
                self.set_status(if self.paused { "paused" } else { "resumed" });
            }
            // Scrub a recording.
            KeyCode::Char('[') => self.scrub(-1),
            KeyCode::Char(']') => self.scrub(1),
            KeyCode::Char('{') => self.scrub(-10),
            KeyCode::Char('}') => self.scrub(10),

            // Tag the selected process for a bulk action.
            KeyCode::Char(' ') => {
                if self.grouped() {
                    self.set_status("grouped view: press Enter to open a group first");
                    return Action::None;
                }
                if let Some(pid) = self.selected_row().map(|r| r.pid) {
                    if !self.tagged.remove(&pid) {
                        self.tagged.insert(pid);
                    }
                    self.move_selection(1);
                }
            }
            KeyCode::Char('U') => {
                let n = self.tagged.len();
                self.tagged.clear();
                self.set_status(format!("untagged {n}"));
            }

            KeyCode::Char('G') => {
                self.cfg.group_by = self.cfg.group_by.next();
                self.group_selected = 0;
                self.group_scroll = 0;
                self.set_status(format!("group by {}", self.cfg.group_by.label()));
            }
            KeyCode::Char('y') => self.yank(),
            KeyCode::Char('L') => self.open_audit_log(),

            // Saved filters on the number keys.
            KeyCode::Char(c @ '1'..='9') => {
                let index = c as usize - '1' as usize;
                match self.cfg.saved_filter_query(index) {
                    Some(query) => {
                        let name = self.cfg.saved_filters[index].name.clone();
                        self.filter_text = query;
                        self.apply_filter();
                        self.set_status(format!("filter: {name}"));
                    }
                    None => self.set_status(format!("no saved filter {}", index + 1)),
                }
            }
            KeyCode::Char('0') => {
                self.filter_text.clear();
                self.apply_filter();
                self.set_status("filter cleared");
            }

            KeyCode::Char('+') | KeyCode::Char('=') => self.nudge_refresh(-200),
            KeyCode::Char('-') | KeyCode::Char('_') => self.nudge_refresh(200),

            KeyCode::Enter => {
                if self.grouped() {
                    self.open_group();
                } else if let Some(pid) = self.selected_row().map(|r| r.pid) {
                    // Fetched here rather than every tick: joining /proc/net
                    // against a process's fd table is only cheap for one PID.
                    self.detail_sockets = sockets::for_pid(pid);
                    self.open_popup(Mode::Detail);
                }
            }
            KeyCode::Char('f') => self.toggle_pin(),
            KeyCode::Char('F') => {
                let n = self.pinned.len();
                self.pinned.clear();
                self.set_status(format!("unpinned {n}"));
                self.rebuild_view();
            }
            KeyCode::Char('?') | KeyCode::F(1) => self.open_popup(Mode::Help),
            KeyCode::Char('A') => self.open_affinity(),
            KeyCode::Char('r') => self.open_renice(),
            KeyCode::Char('t') => self.open_signal_menu(),
            KeyCode::Char('!') => self.open_popup(Mode::Alerts),
            KeyCode::Char('P') => return Action::Export,
            _ => {}
        }
        Action::None
    }

    /// Step through a recording. Inert on a live source.
    fn scrub(&mut self, delta: isize) {
        let Some((pos, len)) = self.source.timeline() else {
            self.set_status("not a recording");
            return;
        };
        let next = (pos as isize + delta).clamp(0, len.saturating_sub(1) as isize) as usize;
        self.source.seek(next);
        // Scrubbing implies stepping frame by frame rather than playing on.
        self.paused = true;
        // `peek`, not `snapshot`: snapshot would advance past the frame just
        // sought, so every step forward would move two frames.
        if let Some(snap) = self.source.peek() {
            self.snap = snap;
            self.rebuild_view();
        }
        self.set_status(format!("frame {}/{}", next + 1, len));
    }

    /// Copy the selected process's command line to the system clipboard.
    fn yank(&mut self) {
        let Some(row) = self.selected_row() else { return };
        let text = if row.cmd.is_empty() { row.name.clone() } else { row.cmd.clone() };
        let summary = crate::format::truncate_fit(&text, 40);
        match crate::clipboard::copy(&text) {
            Ok(()) => self.set_status(format!("copied: {summary}")),
            Err(e) => self.set_status(format!("copy failed: {e}")),
        }
    }

    fn open_audit_log(&mut self) {
        self.audit_entries = match &self.audit_path {
            Some(path) => audit::tail(path, 200),
            None => Vec::new(),
        };
        self.open_popup(Mode::AuditLog);
    }

    /// Record an action in the audit log, if one is configured.
    fn log_action(
        &mut self,
        action: audit::Action,
        pid: u32,
        name: &str,
        result: Result<(), &str>,
    ) {
        let Some(path) = self.audit_path.clone() else { return };
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        audit::append(&path, &audit::format_entry(now, &action, pid, name, result));
    }

    /// Processes a prompt will act on: whatever it captured when it opened.
    pub fn action_targets(&self) -> Vec<Target> {
        self.pending_targets.clone()
    }

    /// Processes a *new* prompt would target: everything tagged, or the
    /// selection when nothing is tagged.
    fn current_targets(&self) -> Vec<Target> {
        if self.tagged.is_empty() {
            return self.selected_row().map(Target::from).into_iter().collect();
        }
        self.rows().iter().filter(|r| self.tagged.contains(&r.pid)).map(Target::from).collect()
    }

    /// Targets for a prompt about to open, or `None` when the grouped view is
    /// on. The aggregate table shows no process rows, so acting from it would
    /// signal whichever process the invisible cursor happened to be on.
    fn targets_for_prompt(&mut self) -> Option<Vec<Target>> {
        if self.grouped() {
            self.set_status("grouped view: press Enter to open a group first");
            return None;
        }
        let targets = self.current_targets();
        if targets.is_empty() {
            return None;
        }
        Some(targets)
    }

    /// What the renice and affinity prompts call the thing they will change.
    /// Unlike signals these apply on one keystroke with no confirmation step,
    /// so the count has to be in the title or a bulk renice looks single.
    pub fn prompt_subject(&self) -> String {
        let targets = &self.pending_targets;
        match targets.len() {
            0 => "<none>".to_string(),
            1 => format!(
                "{} ({})",
                crate::format::truncate_fit(&targets[0].name, 20),
                targets[0].pid
            ),
            n => format!("{n} tagged processes"),
        }
    }

    /// Leave any prompt, forgetting what it had captured.
    fn close_prompt(&mut self) {
        self.target = None;
        self.pending_targets.clear();
        self.input_error = None;
        self.mode = Mode::Normal;
    }

    fn set_sort(&mut self, key: SortBy) {
        if self.cfg.sort_by == key {
            self.cfg.sort_desc = !self.cfg.sort_desc;
        } else {
            self.cfg.sort_by = key;
        }
        self.rebuild_view();
    }

    fn nudge_refresh(&mut self, delta_ms: i64) {
        let ms = (self.refresh.as_millis() as i64 + delta_ms)
            .clamp(crate::MIN_REFRESH_MS as i64, crate::MAX_REFRESH_MS as i64)
            as u64;
        self.refresh = Duration::from_millis(ms);
        self.cfg.refresh_ms = ms;
        // Persist immediately: the README always claimed this happened, and a
        // session that ended in anything but `q` used to lose the setting.
        self.persist();
        self.set_status(format!("refresh {ms} ms"));
    }

    fn apply_filter(&mut self) {
        match filter::parse(&self.filter_text) {
            Ok(f) => {
                self.filter = f;
                self.filter_error = None;
                self.cfg.filter = self.filter_text.clone();
            }
            Err(e) => self.filter_error = Some(e),
        }
        self.rebuild_view();
    }

    fn on_filter_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.filter_text.clear();
                self.apply_filter();
                self.mode = Mode::Normal;
            }
            KeyCode::Enter => self.mode = Mode::Normal,
            KeyCode::Backspace => {
                self.filter_text.pop();
                self.apply_filter();
            }
            KeyCode::Char(c) => {
                self.filter_text.push(c);
                self.apply_filter();
            }
            _ => {}
        }
    }

    fn open_signal_menu(&mut self) {
        let Some(targets) = self.targets_for_prompt() else { return };
        let Some(first) = targets.first().cloned() else { return };
        self.pending_targets = targets;
        self.target = Some(first);
        self.signal_index = 0;
        self.mode = Mode::SignalMenu;
    }

    fn on_signal_menu_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.close_prompt(),
            KeyCode::Down | KeyCode::Char('j') => {
                self.signal_index = (self.signal_index + 1).min(SIGNAL_NAMES.len() - 1)
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.signal_index = self.signal_index.saturating_sub(1)
            }
            KeyCode::Enter => self.mode = Mode::ConfirmKill,
            _ => {}
        }
    }

    /// Guard against PID reuse: the process must still be the one that was
    /// selected when the prompt opened. A PID recycled between opening a signal
    /// menu, renice or affinity prompt and confirming it would otherwise hit an
    /// unrelated process.
    fn target_matches(&self, t: &Target) -> bool {
        self.snap.procs.iter().any(|p| p.pid == t.pid && p.start_time_unix == t.start_time_unix)
    }

    /// A bulk action reports how many of its targets it actually reached; a
    /// single-process action already has a more specific message.
    fn summarise_bulk(&mut self, what: &str, done: usize, skipped: usize, total: usize) {
        if done == 0 && skipped > 0 {
            // Every target vanished between opening the prompt and confirming.
            self.set_status(format!("{what}: process is gone — nothing done"));
            return;
        }
        if total <= 1 {
            return;
        }
        let mut msg = format!("{what}: {done}/{total}");
        if skipped > 0 {
            msg.push_str(&format!(" ({skipped} gone)"));
        }
        self.set_status(msg);
    }

    fn on_confirm_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                let name = SIGNAL_NAMES[self.signal_index];
                let targets = self.action_targets();
                let mut sent = 0usize;
                let mut skipped = 0usize;
                for t in &targets {
                    if !self.target_matches(t) {
                        skipped += 1;
                        continue;
                    }
                    if self.send_signal(t, name) {
                        sent += 1;
                    }
                }
                self.summarise_bulk(name, sent, skipped, targets.len());
                self.tagged.clear();
                self.close_prompt();
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => self.close_prompt(),
            _ => {}
        }
    }

    #[cfg(unix)]
    fn send_signal(&mut self, t: &Target, name: &str) -> bool {
        let Some(sig) = signal_by_name(name) else { return false };
        let action = audit::Action::Signal { name: name.to_string() };
        match crate::metrics::sysinfo_source::send_signal(t.pid, sig) {
            Ok(()) => {
                self.set_status(format!("sent {name} to {} ({})", t.name, t.pid));
                self.log_action(action, t.pid, &t.name, Ok(()));
                true
            }
            Err(e) => {
                self.set_status(format!("{name} to {}: {e}", t.pid));
                self.log_action(action, t.pid, &t.name, Err(&e.to_string()));
                false
            }
        }
    }

    #[cfg(not(unix))]
    fn send_signal(&mut self, _t: &Target, _name: &str) -> bool {
        self.set_status("signals are Unix-only");
        false
    }

    fn open_renice(&mut self) {
        let Some(targets) = self.targets_for_prompt() else { return };
        let Some(first) = targets.first().cloned() else { return };
        let pid = first.pid;
        self.pending_targets = targets;
        self.target = Some(first);
        self.input = crate::metrics::sysinfo_source::get_priority(pid)
            .map(|n| n.to_string())
            .unwrap_or_else(|| "0".into());
        self.input_error = None;
        self.mode = Mode::Renice;
    }

    fn on_renice_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.input.clear();
                self.close_prompt();
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) if c.is_ascii_digit() || c == '-' => self.input.push(c),
            KeyCode::Enter => {
                if self.pending_targets.is_empty() {
                    self.close_prompt();
                    return;
                }
                match parse_nice(&self.input) {
                    Ok(nice) => {
                        let targets = self.action_targets();
                        let (mut done, mut skipped) = (0usize, 0usize);
                        for target in &targets {
                            if !self.target_matches(target) {
                                skipped += 1;
                                continue;
                            }
                            let action = audit::Action::Renice { nice };
                            match crate::metrics::sysinfo_source::set_priority(target.pid, nice) {
                                Ok(()) => {
                                    self.set_status(format!("{} niced to {nice}", target.name));
                                    self.log_action(action, target.pid, &target.name, Ok(()));
                                    done += 1;
                                }
                                Err(e) => {
                                    self.set_status(format!("renice {}: {e}", target.pid));
                                    self.log_action(
                                        action,
                                        target.pid,
                                        &target.name,
                                        Err(&e.to_string()),
                                    );
                                }
                            }
                        }
                        self.summarise_bulk("renice", done, skipped, targets.len());
                        self.tagged.clear();
                        self.input.clear();
                        self.close_prompt();
                    }
                    Err(e) => self.input_error = Some(e),
                }
            }
            _ => {}
        }
    }

    fn open_affinity(&mut self) {
        let Some(targets) = self.targets_for_prompt() else { return };
        let Some(first) = targets.first().cloned() else { return };
        self.pending_targets = targets;
        self.target = Some(first);
        self.input = format!("0-{}", self.snap.cpu.per_core.len().saturating_sub(1));
        self.input_error = None;
        self.mode = Mode::Affinity;
    }

    fn on_affinity_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.input.clear();
                self.close_prompt();
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) if c.is_ascii_digit() || c == '-' || c == ',' => self.input.push(c),
            KeyCode::Enter => {
                let Some(t) = self.target.clone() else {
                    self.mode = Mode::Normal;
                    return;
                };
                let ncpu = self.snap.cpu.per_core.len();
                let _ = &t;
                match crate::metrics::sysinfo_source::parse_cpu_list(&self.input, ncpu) {
                    Ok(cpus) => {
                        let list = self.input.clone();
                        let targets = self.action_targets();
                        let (mut done, mut skipped) = (0usize, 0usize);
                        for target in &targets {
                            if !self.target_matches(target) {
                                skipped += 1;
                                continue;
                            }
                            let action = audit::Action::Affinity { cpus: list.clone() };
                            match crate::metrics::sysinfo_source::set_affinity(target.pid, &cpus) {
                                Ok(()) => {
                                    self.set_status(format!("{} pinned to {list}", target.name));
                                    self.log_action(action, target.pid, &target.name, Ok(()));
                                    done += 1;
                                }
                                Err(e) => {
                                    self.set_status(format!("affinity {}: {e}", target.pid));
                                    self.log_action(
                                        action,
                                        target.pid,
                                        &target.name,
                                        Err(&e.to_string()),
                                    );
                                }
                            }
                        }
                        self.summarise_bulk("affinity", done, skipped, targets.len());
                        self.tagged.clear();
                        self.input.clear();
                        self.close_prompt();
                    }
                    Err(e) => self.input_error = Some(e),
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------- mouse

    pub fn on_mouse(&mut self, ev: MouseEvent) -> Action {
        if self.mode != Mode::Normal {
            return Action::None;
        }
        match ev.kind {
            MouseEventKind::ScrollDown => self.move_selection(3),
            MouseEventKind::ScrollUp => self.move_selection(-3),
            MouseEventKind::Down(MouseButton::Left) => {
                if !contains(self.proc_area, ev.column, ev.row) {
                    return Action::None;
                }
                // Row 0 is the border, row 1 the header.
                let header_row = self.proc_area.y + 1;
                if ev.row == header_row {
                    if let Some((_, _, key)) = self
                        .header_cols
                        .iter()
                        .copied()
                        .find(|(x0, x1, _)| ev.column >= *x0 && ev.column < *x1)
                    {
                        self.set_sort(key);
                    }
                } else if ev.row > header_row {
                    let offset = (ev.row - header_row - 1) as usize;
                    if self.grouped() {
                        self.group_selected = self.group_scroll + offset;
                        self.clamp_group_scroll();
                    } else {
                        self.select(self.scroll + offset);
                    }
                }
            }
            _ => {}
        }
        Action::None
    }

    // --------------------------------------------------------------- export

    pub fn export_format(&self) -> ExportFormat {
        ExportFormat::parse(&self.cfg.export.format).unwrap_or(ExportFormat::Json)
    }

    /// Write a snapshot of the *current view* (filter applied) to disk.
    pub fn export_now(&mut self) {
        let dir = if self.cfg.export.dir.is_empty() {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        } else {
            std::path::PathBuf::from(&self.cfg.export.dir)
        };
        let format = self.export_format();
        let mut snap = self.snap.clone();
        snap.procs = self.rows.clone();
        let snap = export::trim_procs(&snap, self.cfg.export.top_n);
        let path = export::export_path(&dir, snap.taken_at_unix, format);
        match std::fs::write(&path, export::render(&snap, format)) {
            Ok(()) => self.set_status(format!("wrote {}", path.display())),
            Err(e) => self.set_status(format!("export failed: {e}")),
        }
    }

    pub fn persist(&self) {
        if let Some(path) = &self.config_path {
            crate::config::save_to(path, &self.cfg);
        }
    }
}

/// Whether this platform lets us tell a thread from a process at all.
///
/// On Linux the sampler reports a task count for real processes and `None` for
/// their threads. Platforms that never report a task count would otherwise have
/// every row hidden, so the distinction is only applied when it exists.
fn threads_are_distinguishable(procs: &[ProcRow]) -> bool {
    procs.iter().any(|p| p.threads.is_some())
}

/// One row of the grouped process view.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupRow {
    pub name: String,
    pub procs: usize,
    pub cpu: f32,
    pub mem: u64,
    pub io_bps: f64,
}

/// Aggregate processes by systemd unit, container or user.
///
/// Processes with no value for the chosen key are collected under `-`, so the
/// totals still add up to the machine.
pub fn group_rows(rows: &[ProcRow], by: GroupBy) -> Vec<GroupRow> {
    if by == GroupBy::None {
        return Vec::new();
    }
    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<String, GroupRow> = HashMap::new();
    for p in rows {
        let key = match by {
            GroupBy::Service => p.service.clone(),
            GroupBy::Container => p.container.clone(),
            GroupBy::User => p.user.clone(),
            GroupBy::None => None,
        }
        .unwrap_or_else(|| "-".to_string());

        let entry = acc.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            GroupRow { name: key, procs: 0, cpu: 0.0, mem: 0, io_bps: 0.0 }
        });
        entry.procs += 1;
        entry.cpu += p.cpu;
        entry.mem = entry.mem.saturating_add(p.mem);
        entry.io_bps += p.io_bps();
    }
    let mut out: Vec<GroupRow> = order.into_iter().filter_map(|k| acc.remove(&k)).collect();
    out.sort_by(|a, b| {
        b.cpu
            .partial_cmp(&a.cpu)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.mem.cmp(&a.mem))
    });
    out
}

fn contains(r: Rect, x: u16, y: u16) -> bool {
    x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height
}

pub fn parse_nice(s: &str) -> Result<i32, String> {
    let n: i32 = s.trim().parse().map_err(|_| format!("not a number: {s}"))?;
    if !(-20..=19).contains(&n) {
        return Err("nice must be between -20 and 19".into());
    }
    Ok(n)
}

impl Layout {
    pub fn next(self) -> Layout {
        match self {
            Layout::Dashboard => Layout::Processes,
            Layout::Processes => Layout::Cpu,
            Layout::Cpu => Layout::Io,
            Layout::Io => Layout::Dashboard,
        }
    }
    pub fn prev(self) -> Layout {
        match self {
            Layout::Dashboard => Layout::Io,
            Layout::Processes => Layout::Dashboard,
            Layout::Cpu => Layout::Processes,
            Layout::Io => Layout::Cpu,
        }
    }
}
