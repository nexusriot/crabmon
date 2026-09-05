//! Pressure stall gauges.
//!
//! `some` is the share of time at least one task was stalled; `full` is the
//! share where nothing ran at all. `full` is the number that explains a machine
//! that feels frozen while the CPU graph looks idle.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::app::App;
use crate::metrics::psi::Pressure;
use crate::ui::block;

/// A tiny inline bar, since a full gauge widget per row would not fit.
pub fn mini_bar(percent: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let filled = ((percent.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled.min(width)), "░".repeat(width - filled.min(width)))
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, " Pressure ");
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }
    if !app.snap.psi.is_available() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "no PSI (needs Linux 4.20+ with CONFIG_PSI)",
                Style::default().fg(app.theme.dim),
            )),
            inner,
        );
        return;
    }

    let header = Row::new(["", "some", "full", "10s"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    let bar_w = (inner.width as usize).saturating_sub(22).clamp(4, 20);

    let rows: Vec<Row> =
        [("cpu", app.snap.psi.cpu), ("mem", app.snap.psi.memory), ("io", app.snap.psi.io)]
            .into_iter()
            .filter_map(|(name, p)| p.map(|p| (name, p)))
            .map(|(name, p): (&str, Pressure)| {
                // Thresholds are absolute: 10% of wall time stalled is already bad.
                let color = app.theme.usage(p.some.avg10 / 100.0, 0.10, 0.40);
                let full =
                    p.full.map(|f| format!("{:>5.1}%", f.avg10)).unwrap_or_else(|| "    -".into());
                Row::new(vec![
                    Cell::from(Span::styled(name, Style::default().fg(app.theme.text))),
                    Cell::from(Span::styled(
                        format!("{:>5.1}%", p.some.avg10),
                        Style::default().fg(color),
                    )),
                    Cell::from(Span::styled(full, Style::default().fg(color))),
                    Cell::from(Span::styled(
                        mini_bar(p.some.avg10, bar_w),
                        Style::default().fg(color),
                    )),
                ])
            })
            .collect();

    let widths =
        [Constraint::Length(4), Constraint::Length(7), Constraint::Length(7), Constraint::Min(4)];
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bar_is_always_the_requested_width() {
        for pct in [0.0, 2.55, 50.0, 100.0, 250.0, -5.0] {
            assert_eq!(mini_bar(pct, 10).chars().count(), 10, "pct {pct}");
        }
        assert_eq!(mini_bar(0.0, 4), "░░░░");
        assert_eq!(mini_bar(100.0, 4), "████");
        assert_eq!(mini_bar(50.0, 4), "██░░");
        assert_eq!(mini_bar(50.0, 0), "");
    }
}
