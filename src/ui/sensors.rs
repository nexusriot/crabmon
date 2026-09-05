//! Temperature sensors, grouped by driver so 22 `coretemp Core N` rows do not
//! push the NVMe and chassis sensors off the panel.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::app::App;
use crate::format::truncate_fit;
use crate::metrics::group_sensors;
use crate::ui::block;

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = if app.cfg.group_sensors { " Sensors (°C) " } else { " Sensors (°C, all) " };
    let b = block(app, title);
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    if app.snap.sensors.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "no temperature sensors",
                Style::default().fg(app.theme.dim),
            )),
            inner,
        );
        return;
    }

    let label_w = (inner.width as usize).saturating_sub(18).max(6);
    let header = Row::new(["Sensor", "Temp", "Crit"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));

    let rows: Vec<Row> = if app.cfg.group_sensors {
        group_sensors(&app.snap.sensors)
            .into_iter()
            .map(|g| {
                let color = app.theme.temp(
                    g.max_temp,
                    g.critical,
                    app.cfg.thresholds.temp_warn,
                    app.cfg.thresholds.temp_crit,
                );
                let label = if g.count > 1 {
                    format!("{} ×{}", g.label, g.count)
                } else {
                    g.label.clone()
                };
                Row::new(vec![
                    Cell::from(truncate_fit(&label, label_w)),
                    Cell::from(Span::styled(
                        format!("{:.1}", g.max_temp),
                        Style::default().fg(color),
                    )),
                    Cell::from(g.critical.map(|c| format!("{c:.0}")).unwrap_or_else(|| "-".into())),
                ])
            })
            .collect()
    } else {
        app.snap
            .sensors
            .iter()
            .map(|s| {
                let color = app.theme.temp(
                    s.temp,
                    s.critical,
                    app.cfg.thresholds.temp_warn,
                    app.cfg.thresholds.temp_crit,
                );
                Row::new(vec![
                    Cell::from(truncate_fit(&s.label, label_w)),
                    Cell::from(Span::styled(format!("{:.1}", s.temp), Style::default().fg(color))),
                    Cell::from(s.critical.map(|c| format!("{c:.0}")).unwrap_or_else(|| "-".into())),
                ])
            })
            .collect()
    };

    let widths = [Constraint::Min(6), Constraint::Length(7), Constraint::Length(6)];
    f.render_widget(Table::new(rows, widths).header(header), inner);
}
