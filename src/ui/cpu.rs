//! CPU panels: the average history chart plus the per-core view that the
//! project has been promising since the "one indicator per cpu" commit.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::format::human_hz_mhz;
use crate::ui::block;

/// Eighth-block ramp used for the compact per-core strip.
pub const RAMP: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub fn ramp_char(pct: f32) -> char {
    let idx = ((pct.clamp(0.0, 100.0) / 100.0) * (RAMP.len() - 1) as f32).round() as usize;
    RAMP[idx.min(RAMP.len() - 1)]
}

/// `[|||||     ]` style bar for the per-core list.
pub fn bar(pct: f32, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let filled = ((pct.clamp(0.0, 100.0) / 100.0) * width as f32).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "|".repeat(filled), " ".repeat(width - filled))
}

fn title(app: &App) -> String {
    let avg = app.snap.cpu.avg();
    match app.snap.cpu.max_freq() {
        Some(f) => format!(" CPU {avg:>5.1}%  {}  ", human_hz_mhz(f)),
        None => format!(" CPU {avg:>5.1}%  "),
    }
}

/// Dashboard variant: average history chart with a one-line per-core strip.
pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, title(app));
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    let strip_h = if inner.height >= 3 { 1 } else { 0 };
    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(strip_h)])
        .split(inner);

    draw_avg_chart(f, slots[0], app);
    if strip_h > 0 {
        f.render_widget(Paragraph::new(core_strip(app)), slots[1]);
    }
}

fn draw_avg_chart(f: &mut Frame<'_>, area: Rect, app: &App) {
    let data = app.cpu_hist.points();
    if data.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("collecting…", Style::default().fg(app.theme.dim))),
            area,
        );
        return;
    }
    // No `.name()`: a single series would otherwise draw a legend box over the
    // top-right of the chart.
    let datasets = vec![Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(app.theme.accent))
        .data(&data)];
    let x_max = (data.len().saturating_sub(1)) as f64;
    let chart = Chart::new(datasets).x_axis(Axis::default().bounds([0.0, x_max.max(1.0)])).y_axis(
        Axis::default().bounds([0.0, 100.0]).labels(vec![
            Span::styled("0", Style::default().fg(app.theme.dim)),
            Span::styled("100", Style::default().fg(app.theme.dim)),
        ]),
    );
    f.render_widget(chart, area);
}

/// One coloured block per core, in core order.
fn core_strip(app: &App) -> Line<'static> {
    let spans: Vec<Span> = app
        .snap
        .cpu
        .per_core
        .iter()
        .map(|pct| {
            Span::styled(
                ramp_char(*pct).to_string(),
                Style::default().fg(app.theme.usage(
                    (*pct / 100.0) as f64,
                    app.cfg.thresholds.warn,
                    app.cfg.thresholds.crit,
                )),
            )
        })
        .collect();
    Line::from(spans)
}

/// Full-screen `Layout::Cpu`: one row per core with a bar, percentage and clock.
pub fn draw_detail(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, title(app));
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 || inner.width < 12 {
        return;
    }

    let cores = app.snap.cpu.per_core.len();
    if cores == 0 {
        f.render_widget(Paragraph::new("no cpu data"), inner);
        return;
    }

    // Lay the cores out in as many columns of ~22 chars as fit, give the list
    // exactly the rows it needs, and let the average chart have the rest.
    // Wide enough for the clock column when the platform reports one.
    let has_freq = app.snap.cpu.freq_mhz.iter().any(|f| *f > 0);
    let per_col = if has_freq { 31u16 } else { 22u16 };
    let ncols = ((inner.width / per_col).max(1) as usize).min(cores);
    let rows_per_col = cores.div_ceil(ncols);
    let list_h = (rows_per_col as u16).min(inner.height);

    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(list_h), Constraint::Min(0)])
        .split(inner);

    let col_areas = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(vec![Constraint::Ratio(1, ncols as u32); ncols])
        .split(slots[0]);

    // A single bar width for every column: `Ratio` splits leave some columns a
    // character narrower, and per-column widths would make the bars ragged.
    let narrowest = col_areas.iter().map(|a| a.width).min().unwrap_or(0);
    // Everything on a core row except the bar itself: "NNN " + "[" + "]" +
    // "PPP.P%" is 12 columns, and the clock adds " " + 9 right-aligned.
    let reserved = if has_freq { 22 } else { 12 };
    let bar_w = narrowest.saturating_sub(reserved).clamp(4, 40) as usize;

    for (col, col_area) in col_areas.iter().enumerate() {
        let start = col * rows_per_col;
        let end = (start + rows_per_col).min(cores);
        if start >= end {
            continue;
        }
        let lines: Vec<Line> = (start..end)
            .map(|i| {
                let pct = app.snap.cpu.per_core[i];
                let color = app.theme.usage(
                    (pct / 100.0) as f64,
                    app.cfg.thresholds.warn,
                    app.cfg.thresholds.crit,
                );
                let mut spans = vec![
                    Span::styled(format!("{i:>3} "), Style::default().fg(app.theme.dim)),
                    Span::styled("[", Style::default().fg(app.theme.border)),
                    Span::styled(bar(pct, bar_w), Style::default().fg(color)),
                    Span::styled("]", Style::default().fg(app.theme.border)),
                    Span::styled(format!("{pct:>5.1}%"), Style::default().fg(color)),
                ];
                if has_freq {
                    let mhz = app.snap.cpu.freq_mhz.get(i).copied().unwrap_or(0);
                    spans.push(Span::styled(
                        format!(" {:>9}", human_hz_mhz(mhz)),
                        Style::default().fg(app.theme.dim),
                    ));
                }
                Line::from(spans)
            })
            .collect();
        f.render_widget(Paragraph::new(lines), *col_area);
    }

    if slots[1].height > 0 {
        draw_avg_chart(f, slots[1], app);
    }
}

/// Rendered by the process detail popup.
pub fn core_legend(app: &App) -> Line<'static> {
    Line::from(Span::styled(
        format!("{} cores", app.snap.cpu.per_core.len()),
        Style::default().add_modifier(Modifier::DIM),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ramp_spans_idle_to_saturated() {
        assert_eq!(ramp_char(0.0), ' ');
        assert_eq!(ramp_char(100.0), '█');
        assert_eq!(ramp_char(50.0), '▄');
        // Out-of-range readings must not index past the ramp.
        assert_eq!(ramp_char(-10.0), ' ');
        assert_eq!(ramp_char(1000.0), '█');
    }

    /// The clock column was clipped by exactly one character until the reserved
    /// width matched what the row actually prints.
    #[test]
    fn a_core_row_fits_the_width_reserved_for_it() {
        let bar_w = 9usize;
        let with_clock = format!("{:>3} [{}]{:>5.1}% {:>9}", 0, bar(50.0, bar_w), 50.0, "2.40 GHz");
        assert_eq!(with_clock.chars().count(), 22 + bar_w);

        let without = format!("{:>3} [{}]{:>5.1}%", 0, bar(50.0, bar_w), 50.0);
        assert_eq!(without.chars().count(), 12 + bar_w);
    }

    /// The detail popup has no room for the per-core grid, so it says how many
    /// cores the percentages it shows are spread over. Reading the count off
    /// the snapshot rather than the host is what makes a replay of an 8-core
    /// machine still say 8 on a 4-core laptop.
    #[test]
    fn the_detail_legend_counts_the_cores_in_the_snapshot() {
        let mut snap = crate::metrics::Snapshot::default();
        snap.cpu.per_core = vec![0.0; 8];
        let app = App::new(
            crate::Config::default(),
            Box::new(crate::record::ReplaySource::from_frames(vec![snap], "t")),
        );
        let text: String = core_legend(&app).spans.iter().map(|s| s.content.to_string()).collect();
        assert_eq!(text, "8 cores");

        // A source that reports no cores at all must not print a bare "cores".
        let empty = App::new(
            crate::Config::default(),
            Box::new(crate::record::ReplaySource::from_frames(
                vec![crate::metrics::Snapshot::default()],
                "t",
            )),
        );
        let text: String =
            core_legend(&empty).spans.iter().map(|s| s.content.to_string()).collect();
        assert_eq!(text, "0 cores");
    }

    #[test]
    fn bars_are_exactly_the_requested_width() {
        for pct in [0.0, 13.0, 50.0, 99.9, 100.0, 250.0] {
            assert_eq!(bar(pct, 10).chars().count(), 10, "pct {pct}");
        }
        assert_eq!(bar(100.0, 4), "||||");
        assert_eq!(bar(0.0, 4), "    ");
        assert_eq!(bar(50.0, 0), "");
    }
}
