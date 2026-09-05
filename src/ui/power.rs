//! Battery, mains and package power.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::app::App;
use crate::format::human_duration;
use crate::metrics::power::Battery;
use crate::ui::block;

/// `-12.4 W` while discharging, `+20.0 W` while charging, `-` when unknown.
pub fn watts_text(power_w: Option<f64>) -> String {
    match power_w {
        Some(w) if w.abs() < 0.05 => "  0.0 W".to_string(),
        Some(w) => format!("{w:+5.1} W"),
        None => "      -".to_string(),
    }
}

pub fn remaining_text(b: &Battery) -> String {
    match b.time_remaining_secs {
        Some(secs) if b.is_discharging() => format!("{} left", human_duration(secs)),
        Some(secs) => format!("{} to full", human_duration(secs)),
        None => b.status.to_lowercase(),
    }
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = match app.snap.power.ac_online {
        Some(true) => " Power (AC) ",
        Some(false) => " Power (battery) ",
        None => " Power ",
    };
    let b = block(app, title);
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    if app.snap.power.batteries.is_empty() {
        let text = match app.snap.power.rapl_watts {
            Some(w) => format!("package {w:.1} W"),
            None => "no battery".to_string(),
        };
        f.render_widget(
            Paragraph::new(Span::styled(text, Style::default().fg(app.theme.dim))),
            inner,
        );
        return;
    }

    let header = Row::new(["Batt", "Charge", "Draw", "Remaining"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));

    let rows: Vec<Row> = app
        .snap
        .power
        .batteries
        .iter()
        .map(|b| {
            // A nearly-flat battery is the alarming case, so the ratio inverts.
            let color = app.theme.usage(1.0 - b.percent / 100.0, 0.80, 0.90);
            Row::new(vec![
                Cell::from(Span::styled(b.name.clone(), Style::default().fg(app.theme.text))),
                Cell::from(Span::styled(
                    format!("{:>3.0}%", b.percent),
                    Style::default().fg(color),
                )),
                Cell::from(Span::styled(
                    watts_text(b.power_w),
                    Style::default().fg(app.theme.text),
                )),
                Cell::from(Span::styled(remaining_text(b), Style::default().fg(app.theme.dim))),
            ])
        })
        .collect();

    let widths =
        [Constraint::Length(6), Constraint::Length(6), Constraint::Length(8), Constraint::Min(8)];
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sign_of_the_draw_is_visible_at_a_glance() {
        assert_eq!(watts_text(Some(-12.4)), "-12.4 W");
        assert_eq!(watts_text(Some(20.0)), "+20.0 W");
        assert_eq!(watts_text(Some(0.0)), "  0.0 W");
        assert_eq!(watts_text(None).trim(), "-");
    }

    #[test]
    fn remaining_time_says_which_direction_it_is_going() {
        let discharging = Battery {
            status: "Discharging".into(),
            time_remaining_secs: Some(3_661),
            ..Default::default()
        };
        assert_eq!(remaining_text(&discharging), "01:01:01 left");

        let charging = Battery {
            status: "Charging".into(),
            time_remaining_secs: Some(600),
            ..Default::default()
        };
        assert_eq!(remaining_text(&charging), "10:00 to full");

        let idle = Battery { status: "Not charging".into(), ..Default::default() };
        assert_eq!(remaining_text(&idle), "not charging");
    }
}
