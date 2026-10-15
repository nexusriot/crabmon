//! The kernel-events panel: what the kernel has complained about.
//!
//! A dashboard shows memory *now*. The process the OOM killer chose is by
//! definition no longer in the table, so the one question asked after an
//! incident — "what got killed, and when" — was the one nothing on screen
//! could answer.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use super::block;
use crate::app::App;
use crate::format::{human_duration, truncate_fit};
use crate::metrics::kmsg::{EventKind, KernelEvent};

/// Rows the panel asks for, plus its borders.
pub fn height(app: &App) -> u16 {
    let rows = app.snap.kernel.len().clamp(1, 4) as u16;
    rows + 2
}

/// Whether there is anything to say: events, or a reason there are none.
pub fn has_content(app: &App) -> bool {
    !app.snap.kernel.is_empty() || app.snap.kernel_unreadable.is_some()
}

/// The colour a kind is drawn in. An OOM kill is not a warning.
fn colour(app: &App, kind: EventKind) -> ratatui::style::Color {
    match kind {
        EventKind::OomKill | EventKind::Oops => app.theme.crit,
        EventKind::IoError => app.theme.warn,
        EventKind::Segfault => app.theme.warn,
    }
}

/// `firefox (4242)`, or the message's own words when it names nobody.
pub fn subject_text(event: &KernelEvent, width: usize) -> String {
    let text = match (&event.process, event.pid) {
        (Some(name), Some(pid)) => format!("{name} ({pid})"),
        (Some(name), None) => name.clone(),
        (None, _) => event.text.clone(),
    };
    truncate_fit(&text, width)
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, " Kernel ");
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    if let Some(reason) = &app.snap.kernel_unreadable {
        // Not an empty list: "nothing has gone wrong" and "I am not allowed
        // to look" are different answers, and only one of them is reassuring.
        f.render_widget(
            Paragraph::new(Span::styled(
                truncate_fit(&format!("unreadable: {reason}"), inner.width as usize),
                Style::default().fg(app.theme.dim),
            )),
            inner,
        );
        return;
    }

    if app.snap.kernel.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("nothing reported", Style::default().fg(app.theme.dim))),
            inner,
        );
        return;
    }

    let uptime = app.snap.host.uptime_secs;
    let visible = inner.height as usize;
    let subject_w = (inner.width as usize).saturating_sub(14);
    // Newest first: the last thing that went wrong is the one being asked
    // about.
    let rows: Vec<Row> = app
        .snap
        .kernel
        .iter()
        .rev()
        .take(visible)
        .map(|e| {
            Row::new(vec![
                Cell::from(Span::styled(
                    e.kind.label(),
                    Style::default().fg(colour(app, e.kind)).add_modifier(Modifier::BOLD),
                )),
                Cell::from(Span::styled(
                    subject_text(e, subject_w),
                    Style::default().fg(app.theme.text),
                )),
                Cell::from(Span::styled(
                    format!("{} ago", human_duration(e.age_secs(uptime))),
                    Style::default().fg(app.theme.dim),
                )),
            ])
        })
        .collect();

    let widths = [
        ratatui::layout::Constraint::Length(4),
        ratatui::layout::Constraint::Min(8),
        ratatui::layout::Constraint::Length(10),
    ];
    f.render_widget(Table::new(rows, widths), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::kmsg::parse_record;

    #[test]
    fn a_named_victim_is_shown_with_its_pid() {
        let e = parse_record("4,1,1000000,-;Out of memory: Killed process 4242 (firefox)").unwrap();
        assert_eq!(subject_text(&e, 40), "firefox (4242)");
    }

    /// A line that names nobody still has to say something.
    #[test]
    fn an_anonymous_event_falls_back_to_its_own_words() {
        let e = parse_record("3,1,1000000,-;blk_update_request: I/O error, dev sda").unwrap();
        assert!(subject_text(&e, 60).contains("I/O error"));
    }

    #[test]
    fn a_narrow_panel_truncates_rather_than_overflowing() {
        let e = parse_record("4,1,1000000,-;Out of memory: Killed process 4242 (firefox)").unwrap();
        assert!(subject_text(&e, 6).chars().count() <= 6);
    }
}
