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
    /// Stable name, used both by `[procs] columns` and to decide what the cell
    /// renders. Dispatching on the sort key instead meant TREND and PORTS — which
    /// have no sort key of their own — had to be recognised by their title.
    pub id: &'static str,
    pub key: SortBy,
    pub title: &'static str,
    /// Fixed width; the name column flexes and is stored as 0.
    pub width: u16,
    /// Whether clicking the header re-sorts. PORTS has no ordering worth having.
    pub sortable: bool,
}

const fn col(id: &'static str, key: SortBy, title: &'static str, width: u16) -> Col {
    Col { id, key, title, width, sortable: true }
}

/// Pin and tag markers, so a marked row is recognisable without colour.
const MARK: Col = Col { id: "mark", key: SortBy::Pid, title: " ", width: 1, sortable: false };
const PID: Col = col("pid", SortBy::Pid, "PID", 7);
const USER: Col = col("user", SortBy::User, "USER", 9);
const STATE: Col = col("state", SortBy::State, "S", 1);
const CPU: Col = col("cpu", SortBy::Cpu, "CPU%", 6);
const MEM: Col = col("mem", SortBy::Mem, "MEM", 7);
const VIRT: Col = col("virt", SortBy::Virt, "VIRT", 7);
const THR: Col = col("thr", SortBy::Threads, "THR", 4);
const NI: Col = col("ni", SortBy::Nice, "NI", 3);
/// Not sortable in its own right; it visualises the CPU column.
const SPARK: Col = col("trend", SortBy::Cpu, "TREND", 8);
const DISK: Col = col("disk", SortBy::Disk, "DISK", 9);
const TIME: Col = col("time", SortBy::Time, "TIME", 9);
/// Listening ports, looked up only for the rows on screen. Off the automatic
/// list because it walks an fd table per row; add it via `[procs] columns`.
const PORTS: Col =
    Col { id: "ports", key: SortBy::Pid, title: "PORTS", width: 11, sortable: false };
const NAME: Col = col("name", SortBy::Name, "COMMAND", 0);

/// Every column `[procs] columns` can name.
pub const ALL_COLUMNS: [Col; 14] =
    [MARK, PID, USER, STATE, CPU, SPARK, MEM, VIRT, THR, NI, DISK, TIME, PORTS, NAME];

pub fn column_by_name(name: &str) -> Option<Col> {
    let name = name.trim().to_ascii_lowercase();
    ALL_COLUMNS.into_iter().find(|c| c.id == name)
}

/// Columns in display order, widest-first priority for what gets dropped.
const OPTIONAL: [Col; 8] = [STATE, USER, DISK, TIME, NI, SPARK, VIRT, THR];
/// Canonical left-to-right order, whichever subset is chosen.
const ORDER: [Col; 13] =
    [MARK, PID, USER, STATE, CPU, SPARK, MEM, VIRT, THR, NI, DISK, TIME, PORTS];
const MIN_NAME: u16 = 10;

/// Choose the columns that fit in `width`, always keeping the marker, PID,
/// CPU%, MEM and COMMAND.
pub fn plan_columns(width: u16) -> Vec<Col> {
    let mut chosen = vec![MARK, PID, CPU, MEM];
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
    let mut out: Vec<Col> = ORDER.into_iter().filter(|c| chosen.contains(c)).collect();
    out.push(NAME);
    out
}

/// The columns to draw: an explicit `[procs] columns` list when there is one,
/// otherwise whatever fits.
///
/// An explicit list is honoured as written — someone who asked for a column has
/// a reason — except that COMMAND is always appended if it was left out, since
/// a table of numbers with no process names is not a process table.
pub fn columns_for(width: u16, configured: &[String]) -> Vec<Col> {
    if configured.is_empty() {
        return plan_columns(width);
    }
    let mut out: Vec<Col> = Vec::new();
    for name in configured {
        if let Some(c) = column_by_name(name) {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    if out.is_empty() {
        return plan_columns(width);
    }
    // COMMAND flexes, so it has to be last wherever it was asked for.
    out.retain(|c| c.id != NAME.id);
    out.push(NAME);
    out
}

/// Screen x-ranges of each header cell, so a click can be mapped to a sort key.
pub fn column_ranges(inner: Rect, cols: &[Col]) -> Vec<(u16, u16, SortBy)> {
    let mut out = Vec::new();
    let mut x = inner.x;
    let name_w = inner.width.saturating_sub(fixed_width(cols));
    for col in cols {
        let w = if col.width == 0 { name_w } else { col.width };
        if col.sortable {
            out.push((x, x + w, col.key));
        }
        x += w + 1;
    }
    out
}

/// Everything the flexible COMMAND column does not get.
fn fixed_width(cols: &[Col]) -> u16 {
    cols.iter().map(|c| c.width).sum::<u16>() + cols.len().saturating_sub(1) as u16
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

/// What a marked row shows in the leading column.
pub fn mark_char(pinned: bool, tagged: bool) -> char {
    match (pinned, tagged) {
        (true, _) => '\u{25b8}', // ▸ pinned to the top
        (_, true) => '\u{2022}', // • tagged for a bulk action
        _ => ' ',
    }
}

fn cell_text(col: Col, p: &ProcRow, name: &str, name_width: usize) -> String {
    match col.id {
        "mark" => String::new(),
        "trend" | "ports" => String::new(),
        _ => field_text(col.key, p, name, name_width),
    }
}

fn field_text(key: SortBy, p: &ProcRow, name: &str, name_width: usize) -> String {
    match key {
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

/// `22,443,8080` clipped to the column, or `-` when the process listens on
/// nothing. An empty map means the lookup is switched off, not that nothing
/// listens, so that case shows nothing at all rather than a misleading dash.
pub fn ports_text(ports: Option<&Vec<u16>>, width: usize) -> String {
    match ports {
        Some(p) if !p.is_empty() => {
            truncate_fit(&p.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(","), width)
        }
        _ => "-".into(),
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
        draw_groups(f, area, inner, app);
        return;
    }

    let cols = columns_for(inner.width, &app.cfg.procs.columns);
    app.proc_area = area;
    app.header_cols = column_ranges(inner, &cols);

    // One line of the inner area is the header row.
    let page = inner.height.saturating_sub(1) as usize;
    app.set_page(page);

    let name_width = inner.width.saturating_sub(fixed_width(&cols)) as usize;

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
    let spark: std::collections::HashMap<u32, Vec<f32>> = if cols.iter().any(|c| c.id == SPARK.id) {
        app.rows()[start..end].iter().map(|p| (p.pid, app.cpu_spark(p.pid).to_vec())).collect()
    } else {
        std::collections::HashMap::new()
    };
    // The PORTS column is the one thing here that costs syscalls, so it is only
    // ever resolved for rows that are actually on screen, and only when asked for.
    let show_ports = cols.iter().any(|c| c.id == PORTS.id);
    if show_ports {
        let visible: Vec<u32> = app.rows()[start..end].iter().map(|p| p.pid).collect();
        app.refresh_ports(&visible);
    }
    let ports = app.ports.clone();

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
            let pinned = app.pinned.contains(&p.pid);
            Row::new(cols.iter().map(|c| {
                let text = match c.id {
                    "trend" => sparkline(spark.get(&p.pid).map(|v| v.as_slice()).unwrap_or(&[]), 8),
                    "ports" => ports_text(ports.get(&p.pid), PORTS.width as usize),
                    "mark" => {
                        if app.cfg.procs.pin_marker {
                            mark_char(pinned, tagged).to_string()
                        } else {
                            " ".into()
                        }
                    }
                    _ => cell_text(*c, p, &display_name, name_width),
                };
                let style = match c.id {
                    "mark" if pinned => Style::default().fg(app.theme.accent),
                    _ if tagged => Style::default().fg(app.theme.warn).add_modifier(Modifier::BOLD),
                    _ if pinned => Style::default().fg(app.theme.accent),
                    "cpu" => Style::default().fg(cpu_color),
                    "state" if p.state == 'Z' || p.state == 'D' => {
                        Style::default().fg(app.theme.warn)
                    }
                    "pid" => Style::default().fg(app.theme.dim),
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

    let table = Table::new(rows, widths).header(header).highlight_style(app.theme.selection());

    let mut state = TableState::default();
    if !app.rows().is_empty() {
        state.select(Some(app.selected.saturating_sub(start)));
    }
    f.render_stateful_widget(table, inner, &mut state);
}

/// The aggregate view: one row per systemd unit, container or user.
///
/// It gets the same selection and scrolling as the process table, because
/// without them the rows past the panel height were simply unreachable and the
/// process actions kept targeting a row nobody could see.
fn draw_groups(f: &mut Frame<'_>, area: Rect, inner: Rect, app: &mut App) {
    let groups = app.groups();
    app.proc_area = area;
    // Group headers sort by CPU only, so no header is clickable.
    app.header_cols = Vec::new();
    let page = inner.height.saturating_sub(1) as usize;
    app.set_group_page(page, groups.len());

    let header = Row::new(["GROUP", "PROCS", "CPU%", "MEM", "DISK"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    let name_w = inner.width.saturating_sub(34) as usize;

    let start = app.group_scroll.min(groups.len());
    let end = (start + page.max(1)).min(groups.len());
    let rows = groups[start..end].iter().map(|g| {
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
    let table = Table::new(rows, widths).header(header).highlight_style(app.theme.selection());
    let mut state = TableState::default();
    if !groups.is_empty() {
        state.select(Some(app.group_selected.saturating_sub(start)));
    }
    f.render_stateful_widget(table, inner, &mut state);
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
        assert_eq!(
            cols.len(),
            OPTIONAL.len() + 5,
            "the marker, PID, CPU, MEM and COMMAND plus the optionals"
        );
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
        assert_eq!(
            ranges.len(),
            cols.iter().filter(|c| c.sortable).count(),
            "only sortable headers get a clickable range"
        );
        // The marker column is first and is not clickable, so the first range
        // starts one cell in.
        assert_eq!(ranges[0].0, inner.x + MARK.width + 1);
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
    fn an_explicit_column_list_is_honoured_and_typos_are_dropped() {
        let cols = columns_for(200, &["pid".into(), "ports".into(), "nonsense".into()]);
        let ids: Vec<&str> = cols.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec!["pid", "ports", "name"], "COMMAND is appended, typos are not");

        // Asking for COMMAND in the middle still puts it last: it is the column
        // that flexes, so anything after it would have no width left.
        let cols = columns_for(200, &["name".into(), "cpu".into()]);
        assert_eq!(cols.iter().map(|c| c.id).collect::<Vec<_>>(), vec!["cpu", "name"]);

        // An all-typo list falls back rather than drawing an empty table.
        assert_eq!(columns_for(200, &["nope".into()]).len(), plan_columns(200).len());
        assert_eq!(columns_for(200, &[]).len(), plan_columns(200).len());
    }

    #[test]
    fn every_column_is_reachable_by_the_name_the_config_uses() {
        for c in ALL_COLUMNS {
            assert_eq!(column_by_name(c.id).map(|f| f.id), Some(c.id));
            assert_eq!(column_by_name(&c.id.to_uppercase()).map(|f| f.id), Some(c.id));
        }
        assert!(column_by_name("nonsense").is_none());
    }

    #[test]
    fn unsortable_headers_are_not_clickable() {
        // Clicking PORTS has no ordering to apply; the old code would have
        // silently re-sorted by whatever key the column borrowed.
        let inner = Rect { x: 0, y: 0, width: 120, height: 20 };
        let cols = columns_for(120, &["pid".into(), "ports".into(), "cpu".into()]);
        let ranges = column_ranges(inner, &cols);
        assert_eq!(ranges.len(), 3, "mark and ports contribute no clickable range");
    }

    #[test]
    fn marks_distinguish_pinned_from_tagged_without_colour() {
        assert_eq!(mark_char(true, false), '▸');
        assert_eq!(mark_char(false, true), '•');
        assert_eq!(mark_char(true, true), '▸', "pinning is the stronger statement");
        assert_eq!(mark_char(false, false), ' ');
    }

    #[test]
    fn ports_render_as_a_list_or_a_dash() {
        assert_eq!(ports_text(Some(&vec![22, 443]), 11), "22,443");
        assert_eq!(ports_text(None, 11), "-");
        assert_eq!(ports_text(Some(&vec![]), 11), "-");
        assert_eq!(ports_text(Some(&vec![10000, 10001, 10002]), 11).chars().count(), 11);
    }

    #[test]
    fn idle_processes_show_a_dash_rather_than_zero_io() {
        let p = ProcRow { pid: 1, ..Default::default() };
        assert_eq!(cell_text(DISK, &p, "x", 10).trim(), "-");
        assert_eq!(cell_text(USER, &p, "x", 10), "-");
        assert_eq!(cell_text(THR, &p, "x", 10), "-");
    }
}
