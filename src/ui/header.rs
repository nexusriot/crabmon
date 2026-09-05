//! The host banner and the status line.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::{App, Mode};
use crate::format::{human_bytes, human_duration};
use crate::ui::block;

/// Load average is only alarming relative to the core count.
pub fn load_ratio(load1: f64, cores: usize) -> f64 {
    if cores == 0 {
        0.0
    } else {
        load1 / cores as f64
    }
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let h = &app.snap.host;
    let cores = match h.physical_cores {
        Some(p) if p != h.logical_cores => format!("{p}c/{}t", h.logical_cores),
        _ => format!("{} cpu", h.logical_cores),
    };
    let load_color = app.theme.usage(
        load_ratio(h.load[0], h.logical_cores),
        app.cfg.thresholds.warn,
        app.cfg.thresholds.crit,
    );

    let left = Line::from(vec![
        Span::styled(
            h.hostname.clone(),
            Style::default().fg(app.theme.accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(format!("{} {}", h.os, h.kernel), Style::default().fg(app.theme.dim)),
    ]);
    let right = Line::from(vec![
        Span::styled(format!("{cores}  "), Style::default().fg(app.theme.text)),
        Span::styled("load ", Style::default().fg(app.theme.dim)),
        Span::styled(
            format!("{:.2} {:.2} {:.2}", h.load[0], h.load[1], h.load[2]),
            Style::default().fg(load_color),
        ),
        Span::styled(
            format!("  up {}", human_duration(h.uptime_secs)),
            Style::default().fg(app.theme.dim),
        ),
    ]);

    let inner = block(app, " crabmon ").title_alignment(Alignment::Left);
    let area_inner = inner.inner(area);
    f.render_widget(inner, area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area_inner);
    f.render_widget(Paragraph::new(left), cols[0]);
    f.render_widget(Paragraph::new(right).alignment(Alignment::Right), cols[1]);
}

/// One-line status bar: alerts win over transient status, which wins over hints.
pub fn draw_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    let line = if !app.active_alerts.is_empty() {
        let msg =
            app.active_alerts.iter().map(|a| a.message.clone()).collect::<Vec<_>>().join("  •  ");
        Line::from(Span::styled(
            format!(" ALERT  {msg} "),
            Style::default().fg(app.theme.sel_fg).bg(app.theme.crit).add_modifier(Modifier::BOLD),
        ))
    } else if let Some(status) = app.status_text() {
        Line::from(Span::styled(format!(" {status}"), Style::default().fg(app.theme.accent)))
    } else if app.mode == Mode::Filter {
        Line::from(Span::styled(
            " type to filter · Enter apply · Esc clear",
            Style::default().fg(app.theme.dim),
        ))
    } else if let Some((pos, len)) = app.timeline() {
        Line::from(vec![
            Span::styled(
                format!(" ⏸ frame {}/{len} ", pos + 1,),
                Style::default().fg(app.theme.sel_fg).bg(app.theme.accent),
            ),
            Span::styled(
                format!(
                    " {} · {} shown · [ ] to scrub · z to play",
                    app.source_label().unwrap_or_else(|| "replay".into()),
                    app.rows().len(),
                ),
                Style::default().fg(app.theme.dim),
            ),
        ])
    } else if let Some(label) = app.source_label() {
        // A non-live source names itself, and says so loudly when it is broken:
        // otherwise an unreachable `--remote` host is just an empty dashboard.
        let broken = label.contains("UNREACHABLE");
        let style = if broken {
            Style::default().fg(app.theme.sel_fg).bg(app.theme.crit).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.accent)
        };
        Line::from(Span::styled(format!(" {label} "), style))
    } else {
        Line::from(Span::styled(
            format!(
                " {} shown of {} tasks · sort {}{} · {} ms · [?] help",
                app.rows().len(),
                app.snap.procs.len(),
                app.cfg.sort_by.label(),
                if app.cfg.sort_desc { "↓" } else { "↑" },
                app.refresh.as_millis(),
            ),
            Style::default().fg(app.theme.dim),
        ))
    };
    f.render_widget(Paragraph::new(line), area);
}

/// Memory summary reused by the header of the memory panel.
pub fn mem_summary(used: u64, total: u64) -> String {
    format!("{} / {}", human_bytes(used), human_bytes(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_is_judged_per_core() {
        assert_eq!(load_ratio(16.0, 16), 1.0);
        assert_eq!(load_ratio(8.0, 16), 0.5);
        assert_eq!(load_ratio(1.0, 0), 0.0, "no cores must not divide by zero");
    }

    #[test]
    fn memory_summary_uses_real_units() {
        assert_eq!(mem_summary(5_866_434_560, 16_035_467_264), "5.5 GiB / 14.9 GiB");
    }
}
