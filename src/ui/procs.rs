//! The process table: adaptive columns, tree view, and click-to-sort headers.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Row, Table, TableState};
use ratatui::Frame;

use crate::app::{App, Mode};
use crate::format::{compact_bytes, human_duration, truncate_fit};
use crate::metrics::ProcRow;
use crate::sort::SortBy;
use crate::ui::block;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Col {
    pub key: SortBy,
    pub title: &'static str,
    /// Fixed width; the name column flexes and is stored as 0.
    pub width: u16,
}

const PID: Col = Col { key: SortBy::Pid, title: "PID", width: 7 };
const USER: Col = Col { key: SortBy::User, title: "USER", width: 9 };
const STATE: Col = Col { key: SortBy::State, title: "S", width: 1 };
const CPU: Col = Col { key: SortBy::Cpu, title: "CPU%", width: 6 };
const MEM: Col = Col { key: SortBy::Mem, title: "MEM", width: 7 };
const VIRT: Col = Col { key: SortBy::Virt, title: "VIRT", width: 7 };
const THR: Col = Col { key: SortBy::Threads, title: "THR", width: 4 };
const NI: Col = Col { key: SortBy::Nice, title: "NI", width: 3 };
/// Not sortable in its own right; it visualises the CPU column.
const SPARK: Col = Col { key: SortBy::Cpu, title: "TREND", width: 8 };
const DISK: Col = Col { key: SortBy::Disk, title: "DISK", width: 9 };
const TIME: Col = Col { key: SortBy::Time, title: "TIME", width: 9 };
const NAME: Col = Col { key: SortBy::Name, title: "COMMAND", width: 0 };

/// Columns in display order, widest-first priority for what gets dropped.
const OPTIONAL: [Col; 8] = [STATE, USER, DISK, TIME, NI, SPARK, VIRT, THR];
const MIN_NAME: u16 = 10;

/// Choose the columns that fit in `width`, always keeping PID/CPU/MEM/COMMAND.
pub fn plan_columns(width: u16) -> Vec<Col> {
    let mut chosen = vec![PID, CPU, MEM];
    let cost = |cols: &[Col]| -> u16 {
        // One space of ratatui column spacing between every pair, plus the
        // flexible name column and its own spacing.
        cols.iter().map(|c| c.width + 1).sum::<u16>() + MIN_NAME
    };
    for col in OPTIONAL {
        let mut trial = chosen.clone();
        trial.push(col);
        if cost(&trial) <= width {
            chosen = trial;
        }
    }
    // Re-order into the canonical display order.
    let order = [PID, USER, STATE, CPU, SPARK, MEM, VIRT, THR, NI, DISK, TIME];
    let mut out: Vec<Col> = order.into_iter().filter(|c| chosen.contains(c)).collect();
    out.push(NAME);
    out
}

/// Screen x-ranges of each header cell, so a click can be mapped to a sort key.
pub fn column_ranges(inner: Rect, cols: &[Col]) -> Vec<(u16, u16, SortBy)> {
    let mut out = Vec::new();
    let mut x = inner.x;
    let fixed: u16 =
        cols.iter().map(|c| c.width).sum::<u16>() + cols.len().saturating_sub(1) as u16;
    let name_w = inner.width.saturating_sub(fixed);
    for col in cols {
        let w = if col.width == 0 { name_w } else { col.width };
        out.push((x, x + w, col.key));
        x += w + 1;
    }
    out
}

/// Braille sparkline of a short CPU history, scaled to the busiest sample so a
/// quiet process still shows its shape.
pub fn sparkline(samples: &[f32], width: usize) -> String {
    const RAMP: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if width == 0 || samples.is_empty() {
        return " ".repeat(width);
    }
    let peak = samples.iter().copied().fold(1.0f32, f32::max);
    let start = samples.len().saturating_sub(width);
    let mut out: String = samples[start..]
        .iter()
        .map(|v| {
            let idx = ((v / peak) * (RAMP.len() - 1) as f32).round() as usize;
            RAMP[idx.min(RAMP.len() - 1)]
        })
        .collect();
    while out.chars().count() < width {
        out.insert(0, ' ');
    }
    out
}

fn cell_text(col: Col, p: &ProcRow, name: &str, name_width: usize) -> String {
    match col.key {
        SortBy::Pid => p.pid.to_string(),
        SortBy::User => truncate_fit(p.user.as_deref().unwrap_or("-"), 9),
        SortBy::State => p.state.to_string(),
        SortBy::Cpu => format!("{:>5.1}", p.cpu),
        SortBy::Mem => format!("{:>6}", compact_bytes(p.mem)),
        SortBy::Virt => format!("{:>6}", compact_bytes(p.virt)),
        SortBy::Threads => p.threads.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
        SortBy::Disk => {
            if p.io_bps() < 1.0 {
                "        -".into()
            } else {
                format!("{:>8}", compact_bytes(p.io_bps() as u64))
            }
        }
        SortBy::Time => human_duration(p.run_time),
        SortBy::Nice => p.nice.map(|n| n.to_string()).unwrap_or_else(|| "-".into()),
        SortBy::Name => truncate_fit(name, name_width),
    }
}

fn title(app: &App) -> Line<'static> {
    if app.mode == Mode::Filter {
        let style = match app.filter_error {
            Some(_) => Style::default().fg(app.theme.crit),
            None => Style::default().fg(app.theme.warn),
        };
        let mut spans = vec![
            Span::styled(" Filter: ", style.add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}\u{2588}", app.filter_text), style),
        ];
        if let Some(err) = &app.filter_error {
            spans.push(Span::styled(format!("  {err}"), Style::default().fg(app.theme.crit)));
        }
        return Line::from(spans);
    }

    let label = match (app.cfg.tree, app.cfg.group_by) {
        (_, g) if g != crate::metrics::GroupBy::None => format!(" By {} ", g.label()),
        (true, _) => " Processes (tree) ".to_string(),
        _ => " Processes ".to_string(),
    };
    let mut spans = vec![Span::styled(label, Style::default().fg(app.theme.title))];
    if !app.filter_text.is_empty() {
        spans.push(Span::styled(
            format!("/{} ", app.filter_text),
            Style::default().fg(app.theme.warn),
        ));
    }
    if !app.tagged.is_empty() {
        spans.push(Span::styled(
            format!("[{} tagged] ", app.tagged.len()),
            Style::default().fg(app.theme.warn).add_modifier(Modifier::BOLD),
        ));
    }
    if app.paused {
        spans.push(Span::styled(
            "[PAUSED] ",
            Style::default().fg(app.theme.crit).add_modifier(Modifier::BOLD),
        ));
    }
    Line::from(spans)
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let b = block(app, "").title(title(app));
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height < 2 || inner.width < 20 {
        return;
    }
    if app.cfg.group_by != crate::metrics::GroupBy::None {
        draw_groups(f, inner, app);
        return;
    }

    let cols = plan_columns(inner.width);
    app.proc_area = area;
    app.header_cols = column_ranges(inner, &cols);

    // One line of the inner area is the header row.
    let page = inner.height.saturating_sub(1) as usize;
    app.set_page(page);

    let fixed: u16 =
        cols.iter().map(|c| c.width).sum::<u16>() + cols.len().saturating_sub(1) as u16;
    let name_width = inner.width.saturating_sub(fixed) as usize;

    let header = Row::new(cols.iter().map(|c| {
        let mut style = Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title);
        if c.key == app.cfg.sort_by {
            style = style.fg(app.theme.accent);
        }
        let marker = if c.key == app.cfg.sort_by {
            if app.cfg.sort_desc {
                "↓"
            } else {
                "↑"
            }
        } else {
            ""
        };
        Cell::from(Span::styled(format!("{}{marker}", c.title), style))
    }));

    let start = app.scroll.min(app.rows().len());
    let end = (start + page).min(app.rows().len());
    let tree_rows = app.tree_rows.clone();
    // Collected up front: `app` is borrowed immutably for the row iterator.
    let spark: std::collections::HashMap<u32, Vec<f32>> = if cols.iter().any(|c| c.title == "TREND")
    {
        app.rows()[start..end].iter().map(|p| (p.pid, app.cpu_spark(p.pid).to_vec())).collect()
    } else {
        std::collections::HashMap::new()
    };

    let rows: Vec<Row> = app.rows()[start..end]
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let display_name = if app.cfg.tree {
                match tree_rows.get(start + i) {
                    Some(t) => {
                        format!("{}{}", crate::tree::indent(t.depth, t.last_sibling), p.name)
                    }
                    None => p.name.clone(),
                }
            } else {
                p.name.clone()
            };
            let cpu_color = app.theme.usage(
                (p.cpu / 100.0) as f64,
                app.cfg.thresholds.warn,
                app.cfg.thresholds.crit,
            );
            let tagged = app.tagged.contains(&p.pid);
            Row::new(cols.iter().map(|c| {
                let text = if std::ptr::eq(c, &SPARK) || (c.title == "TREND") {
                    sparkline(spark.get(&p.pid).map(|v| v.as_slice()).unwrap_or(&[]), 8)
                } else {
                    cell_text(*c, p, &display_name, name_width)
                };
                let style = match c.key {
                    _ if tagged => Style::default().fg(app.theme.warn).add_modifier(Modifier::BOLD),
                    SortBy::Cpu => Style::default().fg(cpu_color),
                    SortBy::State if p.state == 'Z' || p.state == 'D' => {
                        Style::default().fg(app.theme.warn)
                    }
                    SortBy::Pid => Style::default().fg(app.theme.dim),
                    _ => Style::default().fg(app.theme.text),
                };
                Cell::from(Span::styled(text, style))
            }))
        })
        .collect();

    let widths: Vec<Constraint> = cols
        .iter()
        .map(|c| if c.width == 0 { Constraint::Min(MIN_NAME) } else { Constraint::Length(c.width) })
        .collect();

    let table = Table::new(rows, widths).header(header).highlight_style(
        Style::default().fg(app.theme.sel_fg).bg(app.theme.sel_bg).add_modifier(Modifier::BOLD),
    );

    let mut state = TableState::default();
    if !app.rows().is_empty() {
        state.select(Some(app.selected.saturating_sub(start)));
    }
    f.render_stateful_widget(table, inner, &mut state);
}

/// The aggregate view: one row per systemd unit, container or user.
fn draw_groups(f: &mut Frame<'_>, inner: Rect, app: &App) {
    let groups = app.groups();
    let header = Row::new(["GROUP", "PROCS", "CPU%", "MEM", "DISK"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    let name_w = inner.width.saturating_sub(34) as usize;

    let rows = groups.iter().take(inner.height.saturating_sub(1) as usize).map(|g| {
        let color = app.theme.usage(
            (g.cpu / 100.0) as f64,
            app.cfg.thresholds.warn,
            app.cfg.thresholds.crit,
        );
        Row::new(vec![
            Cell::from(truncate_fit(&g.name, name_w.max(8))),
            Cell::from(format!("{:>5}", g.procs)),
            Cell::from(Span::styled(format!("{:>5.1}", g.cpu), Style::default().fg(color))),
            Cell::from(format!("{:>7}", compact_bytes(g.mem))),
            Cell::from(if g.io_bps < 1.0 {
                "      -".to_string()
            } else {
                format!("{:>7}", compact_bytes(g.io_bps as u64))
            }),
        ])
    });
    let widths = [
        Constraint::Min(8),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
    ];
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn essential_columns_survive_a_narrow_terminal() {
        let cols = plan_columns(30);
        let keys: Vec<SortBy> = cols.iter().map(|c| c.key).collect();
        assert!(keys.contains(&SortBy::Pid));
        assert!(keys.contains(&SortBy::Cpu));
        assert!(keys.contains(&SortBy::Mem));
        assert_eq!(*keys.last().unwrap(), SortBy::Name, "COMMAND is always last");
    }

    #[test]
    fn a_wide_terminal_shows_every_column_once() {
        let cols = plan_columns(200);
        assert_eq!(cols.len(), OPTIONAL.len() + 4, "PID, CPU, MEM, COMMAND plus the optionals");
        let mut titles: Vec<&str> = cols.iter().map(|c| c.title).collect();
        let before = titles.len();
        titles.sort_unstable();
        titles.dedup();
        assert_eq!(titles.len(), before, "no column may appear twice");
    }

    #[test]
    fn the_trend_column_sorts_by_cpu_since_that_is_what_it_draws() {
        let cols = plan_columns(200);
        let spark = cols.iter().find(|c| c.title == "TREND").expect("TREND column");
        assert_eq!(spark.key, SortBy::Cpu);
    }

    #[test]
    fn sparklines_are_padded_to_the_requested_width() {
        assert_eq!(sparkline(&[], 8).chars().count(), 8);
        assert_eq!(sparkline(&[1.0, 2.0], 8).chars().count(), 8);
        assert_eq!(sparkline(&[1.0; 40], 8).chars().count(), 8);
        assert_eq!(sparkline(&[5.0], 0), "");
    }

    #[test]
    fn a_sparkline_scales_to_its_own_peak_so_quiet_processes_still_show_shape() {
        // Two samples an order of magnitude apart must not both render flat.
        let s = sparkline(&[1.0, 10.0], 2);
        let chars: Vec<char> = s.chars().collect();
        assert_ne!(chars[0], chars[1], "{s}");
        // An idle process stays at the floor rather than being scaled up.
        assert_eq!(sparkline(&[0.0, 0.0], 2), "▁▁");
    }

    #[test]
    fn columns_are_added_by_priority_as_width_grows() {
        let narrow = plan_columns(40).len();
        let medium = plan_columns(80).len();
        let wide = plan_columns(160).len();
        assert!(narrow <= medium && medium <= wide, "{narrow} {medium} {wide}");
    }

    #[test]
    fn planned_columns_always_fit_the_available_width() {
        for w in 20u16..200 {
            let cols = plan_columns(w);
            let fixed: u16 = cols.iter().map(|c| c.width + 1).sum::<u16>();
            assert!(fixed <= w + MIN_NAME, "columns overflow at width {w}");
        }
    }

    #[test]
    fn header_ranges_are_contiguous_and_inside_the_panel() {
        let inner = Rect { x: 1, y: 1, width: 100, height: 20 };
        let cols = plan_columns(inner.width);
        let ranges = column_ranges(inner, &cols);
        assert_eq!(ranges.len(), cols.len());
        assert_eq!(ranges[0].0, inner.x);
        for w in ranges.windows(2) {
            assert!(w[0].1 <= w[1].0, "columns must not overlap");
        }
        let last = ranges.last().unwrap();
        assert!(last.1 <= inner.x + inner.width, "last column overflows the panel");
    }

    #[test]
    fn cells_render_the_right_field_for_each_column() {
        let p = ProcRow {
            pid: 4242,
            name: "firefox".into(),
            cpu: 12.34,
            mem: 475_815_936,
            virt: 2_000_000_000,
            state: 'R',
            user: Some("vlad".into()),
            run_time: 3_661,
            threads: Some(42),
            read_bps: 1024.0,
            write_bps: 1024.0,
            ..Default::default()
        };
        assert_eq!(cell_text(PID, &p, "firefox", 20), "4242");
        assert_eq!(cell_text(CPU, &p, "firefox", 20).trim(), "12.3");
        assert_eq!(cell_text(MEM, &p, "firefox", 20).trim(), "454M");
        assert_eq!(cell_text(STATE, &p, "firefox", 20), "R");
        assert_eq!(cell_text(USER, &p, "firefox", 20), "vlad");
        assert_eq!(cell_text(TIME, &p, "firefox", 20), "01:01:01");
        assert_eq!(cell_text(THR, &p, "firefox", 20), "42");
        assert_eq!(cell_text(DISK, &p, "firefox", 20).trim(), "2.0K");
    }

    #[test]
    fn idle_processes_show_a_dash_rather_than_zero_io() {
        let p = ProcRow { pid: 1, ..Default::default() };
        assert_eq!(cell_text(DISK, &p, "x", 10).trim(), "-");
        assert_eq!(cell_text(USER, &p, "x", 10), "-");
        assert_eq!(cell_text(THR, &p, "x", 10), "-");
    }
}
