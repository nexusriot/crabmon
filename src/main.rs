use std::cmp::min;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Row, Table, TableState},
    Terminal, Frame,
};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, RefreshKind, System, ProcessRefreshKind,
};

const TICK: Duration = Duration::from_millis(800);

#[derive(Clone, Copy)]
enum SortBy {
    Cpu,
    Mem,
    Pid,
    Name,
}

struct App {
    sys: System,
    last_refresh: Instant,
    selected: usize,
    sort_by: SortBy,
    sort_desc: bool,
}

impl App {
    fn new() -> Self {
        let refresh = RefreshKind::new()
            .with_cpu(CpuRefreshKind::new().with_cpu_usage())
            .with_memory(MemoryRefreshKind::new().with_ram().with_swap())
            .with_processes(ProcessRefreshKind::everything());

        let mut sys = System::new_with_specifics(refresh);
        sys.refresh_all();

        Self {
            sys,
            last_refresh: Instant::now(),
            selected: 0,
            sort_by: SortBy::Cpu,
            sort_desc: true,
        }
    }

    fn refresh(&mut self) {
        self.sys.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::new().with_cpu_usage())
                .with_memory(MemoryRefreshKind::new().with_ram().with_swap())
                .with_processes(ProcessRefreshKind::new().with_cpu().with_memory()),
        );
        self.last_refresh = Instant::now();
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
}

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

        let timeout = TICK
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_millis(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if handle_key(key, &mut app)? {
                    break; // quit
                }
            }
        }

        if last_tick.elapsed() >= TICK {
            app.refresh();
            last_tick = Instant::now();
        }
    }
    Ok(())
}

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
        _ => {}
    }
    Ok(false)
}

fn ui(f: &mut Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // CPU
            Constraint::Length(3), // Memory
            Constraint::Min(5),    // Processes
        ])
        .split(f.size());

    draw_cpu(f, chunks[0], app);
    draw_mem(f, chunks[1], app);
    draw_procs(f, chunks[2], app);
}

fn draw_cpu(f: &mut Frame<'_>, area: Rect, app: &App) {
    let cpu = app.sys.global_cpu_info().cpu_usage(); // 0..100
    let ratio = (cpu as f64 / 100.0).clamp(0.0, 1.0);
    let label = format!("CPU {:>4.1}%", cpu);

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("CPU"))
        .ratio(ratio)
        .label(Span::raw(label));
    f.render_widget(gauge, area);
}

fn draw_mem(f: &mut Frame<'_>, area: Rect, app: &App) {
    // sysinfo memory units are KiB
    let total = app.sys.total_memory() as f64;
    let used = app.sys.used_memory() as f64;
    let ratio = if total > 0.0 { used / total } else { 0.0 };
    let label = format!(
        "Mem {:>5.1}%  ({:.1} / {:.1} GiB)",
        ratio * 100.0,
        kib_to_gib(used),
        kib_to_gib(total)
    );

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Memory"))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::raw(label));
    f.render_widget(gauge, area);
}

fn draw_procs(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let mut rows_data = app.processes();

    let header_rows = 3usize; // table header + borders
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
        Span::raw("[↑/k] up  [↓/j] down  [c] CPU  [m] Mem  [p] PID  [n] Name  [s] asc/desc  [q] quit"),
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

/// Truncate with ellipsis to fit within `max` columns.
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
