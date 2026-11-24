//! Battery, mains and package power.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use crate::app::App;
use crate::format::human_duration;
use crate::metrics::backlight::Backlight;
use crate::metrics::power::Battery;
use crate::metrics::soc::SocSample;
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

/// The clock line: board, current against ceiling, governor.
pub fn clock_text(soc: &SocSample) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(model) = &soc.model {
        // "Raspberry Pi 4 Model B Rev 1.5" is most of a line by itself.
        parts.push(model.split(" Rev ").next().unwrap_or(model).to_string());
    }
    if let Some(cur) = soc.cur_khz {
        let ghz = |k: u64| format!("{:.2}", k as f64 / 1_000_000.0);
        parts.push(match soc.max_khz {
            Some(max) => format!("{}/{} GHz", ghz(cur), ghz(max)),
            None => format!("{} GHz", ghz(cur)),
        });
    }
    if let Some(g) = &soc.governor {
        parts.push(g.clone());
    }
    (!parts.is_empty()).then(|| parts.join("  "))
}

/// The throttle line and whether it is a problem right now (`Some(true)`),
/// an old one (`Some(false)`) or nothing to report (`None`).
pub fn throttle_text(soc: &SocSample) -> Option<(String, Option<bool>)> {
    let flags = soc.throttle?;
    let now = flags.now();
    if !now.is_empty() {
        return Some((now.join(", ") + " now", Some(true)));
    }
    let earlier = flags.earlier();
    if !earlier.is_empty() {
        return Some((earlier.join(", ") + " since boot", Some(false)));
    }
    Some(("power and clocks ok".to_string(), None))
}

/// `Screen 50%  intel_backlight`, one per backlight.
pub fn backlight_text(b: &Backlight) -> String {
    format!("Screen {:>3.0}%  {}", b.percent(), b.name)
}

/// Lines the board's own state and the backlights need under the batteries.
pub fn soc_height(app: &App) -> u16 {
    let p = &app.snap.power;
    clock_text(&p.soc).is_some() as u16
        + p.soc.throttle.is_some() as u16
        + p.backlights.len() as u16
}

/// The panel's height: borders, the battery table (header and one row each,
/// or one line when there is none), and the board's lines.
pub fn height(app: &App) -> u16 {
    let p = &app.snap.power;
    let table = if p.batteries.is_empty() { 1 } else { p.batteries.len() as u16 + 1 };
    // A board with no battery has nothing but its own lines to show.
    let table = if p.batteries.is_empty() && soc_height(app) > 0 { 0 } else { table };
    table + soc_height(app) + 2
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

    let soc = &app.snap.power.soc;
    let mut lines: Vec<Line> = Vec::new();
    if let Some(text) = clock_text(soc) {
        lines.push(Line::from(Span::styled(text, Style::default().fg(app.theme.text))));
    }
    if let Some((text, bad)) = throttle_text(soc) {
        let color = match bad {
            Some(true) => app.theme.crit,
            Some(false) => app.theme.warn,
            None => app.theme.dim,
        };
        lines.push(Line::from(Span::styled(text, Style::default().fg(color))));
    }
    for b in &app.snap.power.backlights {
        lines
            .push(Line::from(Span::styled(backlight_text(b), Style::default().fg(app.theme.text))));
    }
    let soc_h = (lines.len() as u16).min(inner.height);

    let table_area = Rect { height: inner.height - soc_h, ..inner };
    let soc_area = Rect { y: inner.y + table_area.height, height: soc_h, ..inner };

    if app.snap.power.batteries.is_empty() {
        // Without a battery the board's lines are the panel; the old "no
        // battery" is only worth saying where there is nothing else.
        if lines.is_empty() {
            let text = match app.snap.power.rapl_watts {
                Some(w) => format!("package {w:.1} W"),
                None => "no battery".to_string(),
            };
            f.render_widget(
                Paragraph::new(Span::styled(text, Style::default().fg(app.theme.dim))),
                inner,
            );
        } else {
            f.render_widget(Paragraph::new(lines), Rect { height: inner.height, ..inner });
        }
        return;
    }

    let header = Row::new(["Batt", "Charge", "Draw", "Remaining", "Health", "Cyc"])
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
                Cell::from(Span::styled(health_text(b), Style::default().fg(app.theme.dim))),
                Cell::from(Span::styled(cycles_text(b), Style::default().fg(app.theme.dim))),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Min(8),
        Constraint::Length(7),
        Constraint::Length(5),
    ];
    f.render_widget(Table::new(rows, widths).header(header), table_area);
    if soc_h > 0 {
        f.render_widget(Paragraph::new(lines), soc_area);
    }
}

/// `92%` of design capacity, `-` where the gauge does not say.
pub fn health_text(b: &Battery) -> String {
    b.health_percent.map(|h| format!("{h:.0}%")).unwrap_or_else(|| "-".into())
}

/// Charge cycles, `-` where the gauge does not count them.
pub fn cycles_text(b: &Battery) -> String {
    b.cycle_count.map(|c| c.to_string()).unwrap_or_else(|| "-".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::soc::ThrottleFlags;

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

    #[test]
    fn the_board_line_shows_the_clock_against_its_ceiling() {
        let soc = SocSample {
            model: Some("Raspberry Pi 4 Model B Rev 1.5".into()),
            cur_khz: Some(600_000),
            max_khz: Some(1_500_000),
            governor: Some("ondemand".into()),
            ..Default::default()
        };
        assert_eq!(
            clock_text(&soc).as_deref(),
            Some("Raspberry Pi 4 Model B  0.60/1.50 GHz  ondemand")
        );
        assert_eq!(clock_text(&SocSample::default()), None);
    }

    #[test]
    fn the_throttle_line_distinguishes_now_from_earlier() {
        let with = |bits| SocSample { throttle: Some(ThrottleFlags(bits)), ..Default::default() };
        let (text, bad) = throttle_text(&with(0x5)).unwrap();
        assert_eq!(text, "undervoltage, throttled now");
        assert_eq!(bad, Some(true));

        let (text, bad) = throttle_text(&with(0x10000)).unwrap();
        assert_eq!(text, "undervoltage since boot");
        assert_eq!(bad, Some(false));

        let (text, bad) = throttle_text(&with(0)).unwrap();
        assert_eq!(text, "power and clocks ok");
        assert_eq!(bad, None);

        // No firmware to ask is not the same as asking and hearing "fine".
        assert!(throttle_text(&SocSample::default()).is_none());
    }

    #[test]
    fn the_backlight_reads_as_a_percentage_with_its_name() {
        let b = Backlight {
            name: "intel_backlight".into(),
            brightness: 48_000,
            max: 96_000,
            kind: "raw".into(),
        };
        assert_eq!(backlight_text(&b), "Screen  50%  intel_backlight");
    }

    #[test]
    fn health_and_cycles_say_dash_where_the_gauge_is_silent() {
        let b =
            Battery { health_percent: Some(91.6), cycle_count: Some(212), ..Default::default() };
        assert_eq!(health_text(&b), "92%");
        assert_eq!(cycles_text(&b), "212");
        assert_eq!(health_text(&Battery::default()), "-");
        assert_eq!(cycles_text(&Battery::default()), "-");
    }
}
