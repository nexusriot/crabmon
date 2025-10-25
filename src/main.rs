use std::cmp::min;
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use dirs::config_dir;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Row, Table, TableState,
    },
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use sysinfo::{
    CpuRefreshKind, Disks, MemoryRefreshKind, Networks, Pid, ProcessRefreshKind, RefreshKind,
    System,
};

// ------------------------------ Config ------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppConfig {
    sort_by: SortBy,
    sort_desc: bool,
    refresh_ms: u64,
}
impl Default for AppConfig {
    fn default() -> Self {
        Self {
            sort_by: SortBy::Cpu,
            sort_desc: true,
            refresh_ms: 800,
        }
    }
}

fn config_path() -> PathBuf {
    let mut base = config_dir().unwrap_or_else(|| PathBuf::from("."));
    base.push("crabmon");
    let _ = fs::create_dir_all(&base);
    base.push("config.toml");
    base
}

fn load_config() -> AppConfig {
    let path = config_path();
    let s = fs::read_to_string(&path).unwrap_or_default();
    toml::from_str(&s).unwrap_or_default()
}

fn save_config(cfg: &AppConfig) {
    let path = config_path();
    let _ = fs::write(path, toml::to_string_pretty(cfg).unwrap_or_default());
}

// ------------------------------ App State ------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum SortBy {
    Cpu,
    Mem,
    Pid,
    Name,
}

struct NetSnapshot {
    rx: u64,
    tx: u64,
}

struct App {
    sys: System,
    last_refresh: Instant,

    // separate resources in sysinfo 0.30
    networks: Networks,
    disks: Disks,

    // UI / selection
    selected: usize,

    // Sorting & config
    sort_by: SortBy,
    sort_desc: bool,
    refresh: Duration,

    // Charts
    cpu_history: VecDeque<f64>, // avg CPU usage (0..100) over time
    history_len: usize,

    // Network rate calculation
    prev_net: BTreeMap<String, NetSnapshot>, // by interface name
    agg_rx_bps: f64,
    agg_tx_bps: f64,
}

impl App {
    fn new() -> Self {
        let cfg = load_config();

        // System refresh kind (networks/disks are managed separately)
        let refresh = RefreshKind::new()
            .with_cpu(CpuRefreshKind::new().with_cpu_usage())
            .with_memory(MemoryRefreshKind::new().with_ram().with_swap())
            .with_processes(ProcessRefreshKind::everything());

        let mut sys = System::new_with_specifics(refresh);
        sys.refresh_all();

        // init networks & disks lists fully refreshed
        let mut networks = Networks::new_with_refreshed_list();
        networks.refresh();
        let mut disks = Disks::new_with_refreshed_list();
        disks.refresh();

        // init history with zeros
        let history_len = 120; // ~ last N ticks in chart
        let mut cpu_history = VecDeque::with_capacity(history_len);
        cpu_history.extend(std::iter::repeat(0.0).take(history_len));

        // initial net snapshot
        let mut prev_net = BTreeMap::new();
        for (name, data) in networks.iter() {
            prev_net.insert(
                name.to_string(),
                NetSnapshot {
                    rx: data.total_received(),
                    tx: data.total_transmitted(),
                },
            );
        }

        Self {
            sys,
            last_refresh: Instant::now(),
            networks,
            disks,
            selected: 0,
            sort_by: cfg.sort_by,
            sort_desc: cfg.sort_desc,
            refresh: Duration::from_millis(cfg.refresh_ms),
            cpu_history,
            history_len,
            prev_net,
            agg_rx_bps: 0.0,
            agg_tx_bps: 0.0,
        }
    }

    fn refresh(&mut self) {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last_refresh);

        // refresh system metrics
        self.sys.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::new().with_cpu_usage())
                .with_memory(MemoryRefreshKind::new().with_ram().with_swap())
                .with_processes(ProcessRefreshKind::new().with_cpu().with_memory()),
        );

        // refresh external resources
        self.networks.refresh();
        self.disks.refresh();

        // push average CPU usage into history
        let avg_cpu = {
            let cpus = self.sys.cpus();
            if cpus.is_empty() {
                0.0
            } else {
                let sum: f32 = cpus.iter().map(|c| c.cpu_usage()).sum();
                (sum / (cpus.len() as f32)) as f64
            }
        }
        .clamp(0.0, 100.0);

        if self.cpu_history.len() >= self.history_len {
            self.cpu_history.pop_front();
        }
        self.cpu_history.push_back(avg_cpu);

        // network bps
        self.update_network_rates(dt);

        self.last_refresh = now;
    }

    fn update_network_rates(&mut self, dt: Duration) {
        if dt.as_secs_f64() == 0.0 {
            return;
        }
        let mut agg_rx = 0.0f64;
        let mut agg_tx = 0.0f64;
        let mut new_prev = BTreeMap::new();

        for (name, data) in self.networks.iter() {
            let rx = data.total_received();
            let tx = data.total_transmitted();
            if let Some(prev) = self.prev_net.get(name) {
                let drx = rx.saturating_sub(prev.rx) as f64;
                let dtx = tx.saturating_sub(prev.tx) as f64;
                agg_rx += drx / dt.as_secs_f64();
                agg_tx += dtx / dt.as_secs_f64();
            }
            new_prev.insert(name.to_string(), NetSnapshot { rx, tx });
        }

        self.prev_net = new_prev;
        self.agg_rx_bps = agg_rx;
        self.agg_tx_bps = agg_tx;
    }

    fn processes(&self) -> Vec<(Pid, String, f32, u64)> {
        let mut items: Vec<_> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, p)| (*pid, p.name().to_string(), p.cpu_usage(), p.memory()))
            .collect();

        match self.sort_by {
            SortBy::Cpu => {
                items.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            }
            SortBy::Mem => items.sort_by_key(|x| x.3),
            SortBy::Pid => items.sort_by_key(|x| x.0.as_u32()),
            SortBy::Name => items.sort_by(|a, b| a.1.to_lowercase().cmp(&b.1.to_lowercase())),
        }
        if self.sort_desc {
            items.reverse();
        }
        items
    }

    fn persist_config(&self) {
        let cfg = AppConfig {
            sort_by: self.sort_by,
            sort_desc: self.sort_desc,
            refresh_ms: self.refresh.as_millis() as u64,
        };
        save_config(&cfg);
    }
}

// ------------------------------ Main Loop ------------------------------

fn main() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_app(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> anyhow::Result<()> {
    let mut app = App::new();
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        let timeout = app
            .refresh
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_millis(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if handle_key(key, &mut app)? {
                    app.persist_config();
                    break;
                }
            }
        }

        if last_tick.elapsed() >= app.refresh {
            app.refresh();
            last_tick = Instant::now();
        }
    }
    Ok(())
}

// ------------------------------ Input ------------------------------

fn handle_key(key: KeyEvent, app: &mut App) -> anyhow::Result<bool> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => return Ok(true),

        (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
            let total = app.sys.processes().len();
            if total > 0 {
                app.selected = (app.selected + 1).min(total - 1);
            }
        }
        (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
            if app.selected > 0 {
                app.selected -= 1;
            }
        }

        (KeyCode::Char('c'), KeyModifiers::NONE) => app.sort_by = SortBy::Cpu,
        (KeyCode::Char('m'), KeyModifiers::NONE) => app.sort_by = SortBy::Mem,
        (KeyCode::Char('p'), KeyModifiers::NONE) => app.sort_by = SortBy::Pid,
        (KeyCode::Char('n'), KeyModifiers::NONE) => app.sort_by = SortBy::Name,
        (KeyCode::Char('s'), KeyModifiers::NONE) => app.sort_desc = !app.sort_desc,

        (KeyCode::Char('+'), _) | (KeyCode::Char('='), _) => {
            let ms = (app.refresh.as_millis() as i64 - 200).max(100) as u64;
            app.refresh = Duration::from_millis(ms);
            app.persist_config();
        }
        (KeyCode::Char('-'), _) => {
            let ms = (app.refresh.as_millis() as i64 + 200).min(5000) as u64;
            app.refresh = Duration::from_millis(ms);
            app.persist_config();
        }

        // send SIGTERM to selected PID (Unix only)
        (KeyCode::Char('t'), KeyModifiers::NONE) => {
            #[cfg(unix)]
            {
                if let Some((pid, _, _, _)) = app.processes().get(app.selected).cloned() {
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(pid.as_u32() as i32),
                        nix::sys::signal::Signal::SIGTERM,
                    );
                }
            }
        }

        _ => {}
    }
    Ok(false)
}

// ------------------------------ UI ------------------------------

fn ui(f: &mut Frame<'_>, app: &mut App) {
    let size = f.size();

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(size);

    // Left side: CPU overview / Memory / Procs
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // CPU overview (single composite chart)
            Constraint::Length(3), // Memory
            Constraint::Min(7),    // Processes
        ])
        .split(cols[0]);

    draw_cpu_overview(f, left[0], app);
    draw_mem(f, left[1], app);
    draw_procs(f, left[2], app);

    // Right side: Network + Disks
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Net
            Constraint::Min(5),    // Disks
        ])
        .split(cols[1]);

    draw_network(f, right[0], app);
    draw_disks(f, right[1], app);
}

fn draw_cpu_overview(f: &mut Frame<'_>, area: Rect, app: &App) {
    // current usage is last point in history
    let current = app
        .cpu_history
        .back()
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 100.0);

    // convert history -> (x,y) pairs for Chart
    let data: Vec<(f64, f64)> = app
        .cpu_history
        .iter()
        .enumerate()
        .map(|(i, y)| (i as f64, *y))
        .collect();

    let datasets = vec![Dataset::default()
        .name("avg cpu")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .data(&data)];

    let x_max = (app.cpu_history.len().saturating_sub(1)) as f64;
    let title = format!("CPU avg {:>4.1}%", current);

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(title))
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max.max(1.0)])
                .labels(vec![Span::raw("past"), Span::raw("now")]),
        )
        .y_axis(
            Axis::default()
                .bounds([0.0, 100.0])
                .labels(vec![Span::raw("0"), Span::raw("100%")]),
        );

    f.render_widget(chart, area);
}

fn draw_mem(f: &mut Frame<'_>, area: Rect, app: &App) {
    let total = app.sys.total_memory() as f64;
    let used = app.sys.used_memory() as f64;
    let ratio = if total > 0.0 { used / total } else { 0.0 };
    let label = format!(
        "Mem {:>5.1}%  ({:.1} / {:.1} GiB)  •  Refresh: {}ms",
        ratio * 100.0,
        kib_to_gib(used),
        kib_to_gib(total),
        app.refresh.as_millis()
    );

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Memory"))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::raw(label));
    f.render_widget(gauge, area);
}

fn draw_procs(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let mut rows_data = app.processes();

    let header_rows = 3usize;
    let max_rows = area.height.saturating_sub(header_rows as u16) as usize;
    if rows_data.is_empty() {
        rows_data.push((Pid::from_u32(0), "<no processes>".into(), 0.0, 0));
    }
    app.selected = min(app.selected, rows_data.len().saturating_sub(1));

    let start = app.selected.saturating_sub(max_rows.saturating_sub(1));
    let end = min(start + max_rows, rows_data.len());
    let visible = &rows_data[start..end];

    let header = Row::new(vec!["PID", "Name", "CPU %", "Mem (MiB)"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows = visible.iter().map(|(pid, name, cpu, mem_kib)| {
        Row::new(vec![
            Cell::from(pid.as_u32().to_string()),
            Cell::from(truncate_fit(name, 40)),
            Cell::from(format!("{:>5.1}", cpu)),
            Cell::from(format!("{:>7.1}", kib_to_mib(*mem_kib as f64))),
        ])
    });

    let title = Line::from(vec![
        Span::raw("Processes  "),
        Span::raw("[↑/k] up  [↓/j] down  [c] CPU  [m] Mem  [p] PID  [n] Name  [s] asc/desc  "),
        Span::raw("[+] faster  [-] slower  "),
        #[cfg(unix)]
        Span::raw("[t] SIGTERM  "),
        Span::raw("[q] quit"),
    ]);

    let widths = [
        Constraint::Length(8),
        Constraint::Percentage(60),
        Constraint::Length(8),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    let mut state = table_state(app, start);
    f.render_stateful_widget(table, area, &mut state);
}

fn draw_network(f: &mut Frame<'_>, area: Rect, app: &App) {
    let rx = human_bps(app.agg_rx_bps);
    let tx = human_bps(app.agg_tx_bps);
    let title = format!("Network  RX: {rx}  TX: {tx}");

    // Top 4 interfaces by total transferred
    let mut per: Vec<(String, u64, u64)> = app
        .networks
        .iter()
        .map(|(name, data)| (name.to_string(), data.total_received(), data.total_transmitted()))
        .collect();

    per.sort_by_key(|(_, rx, tx)| rx + tx);
    per.reverse();
    per.truncate(4);

    let header = Row::new(vec!["Iface", "RX total", "TX total"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows = per.into_iter().map(|(name, rx, tx)| {
        Row::new(vec![
            Cell::from(name),
            Cell::from(human_bytes(rx)),
            Cell::from(human_bytes(tx)),
        ])
    });

    let widths = [
        Constraint::Percentage(35),
        Constraint::Percentage(32),
        Constraint::Percentage(33),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(title));

    f.render_widget(table, area);
}

fn draw_disks(f: &mut Frame<'_>, area: Rect, app: &App) {
    let header = Row::new(vec!["Mount", "FS", "Used / Total"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let mut rows_vec = Vec::new();
    for d in app.disks.iter() {
        let mount = d.mount_point().to_string_lossy().to_string();
        let fs = d.file_system().to_string_lossy().to_string();
        let total = d.total_space() as f64;
        let avail = d.available_space() as f64;
        let used = (total - avail).max(0.0);
        rows_vec.push((
            mount,
            fs,
            format!("{} / {}", human_bytes(used as u64), human_bytes(total as u64)),
        ));
    }

    if rows_vec.is_empty() {
        rows_vec.push(("-".into(), "-".into(), "-".into()));
    }

    let rows = rows_vec.into_iter().map(|(m, fs, u)| {
        Row::new(vec![Cell::from(m), Cell::from(fs), Cell::from(u)])
    });

    let widths = [
        Constraint::Percentage(45),
        Constraint::Percentage(20),
        Constraint::Percentage(35),
    ];

    let table = Table::new(rows, widths)
        .block(Block::default().borders(Borders::ALL).title("Disks"))
        .header(header);

    f.render_widget(table, area);
}

// ------------------------------ Helpers ------------------------------

fn table_state(app: &App, start_index: usize) -> TableState {
    let mut state = TableState::default();
    state.select(Some(app.selected - start_index));
    state
}

fn kib_to_gib(kib: f64) -> f64 {
    kib / 1024.0 / 1024.0
}
fn kib_to_mib(kib: f64) -> f64 {
    kib / 1024.0
}

fn truncate_fit(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i + 1 >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn human_bytes(x: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = x as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{:.0} {}", v, UNITS[u])
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

fn human_bps(bps: f64) -> String {
    const UNITS: [&str; 6] = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s", "PiB/s"];
    let mut v = bps;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{:.0} {}", v, UNITS[u])
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}
