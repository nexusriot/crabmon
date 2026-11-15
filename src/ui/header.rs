//! The host banner and the status line.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{App, Mode};
use crate::format::{human_bytes, human_duration};
use crate::ui::block;

/// Load average is only alarming relative to the core count.
pub fn load_ratio(load1: f64, cores: usize) -> f64 {
    if cores == 0 { 0.0 } else { load1 / cores as f64 }
}

/// `2 units failed, 1 crash-looping`, or nothing when every unit is well.
/// Restarting on its own is not trouble yet: one restart is what
/// `Restart=on-failure` is for.
pub fn unit_trouble(snap: &crate::metrics::Snapshot) -> Option<String> {
    use crate::metrics::units::UnitState;
    let failed = snap.units_in(UnitState::Failed);
    let looping = snap.units_in(UnitState::CrashLoop);
    let unit = |n: usize| if n == 1 { "unit" } else { "units" };
    match (failed, looping) {
        (0, 0) => None,
        (f, 0) => Some(format!("{f} {} failed", unit(f))),
        (0, l) => Some(format!("{l} {} crash-looping", unit(l))),
        (f, l) => Some(format!("{f} {} failed, {l} crash-looping", unit(f))),
    }
}

/// `unit_trouble` for a narrow header: `2 failed, 1 looping`.
pub fn unit_trouble_compact(snap: &crate::metrics::Snapshot) -> Option<String> {
    use crate::metrics::units::UnitState;
    match (snap.units_in(UnitState::Failed), snap.units_in(UnitState::CrashLoop)) {
        (0, 0) => None,
        (f, 0) => Some(format!("{f} failed")),
        (0, l) => Some(format!("{l} looping")),
        (f, l) => Some(format!("{f} failed, {l} looping")),
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

    let inner = block(app, " crabmon ").title_alignment(Alignment::Left);
    let area_inner = inner.inner(area);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area_inner);

    // A broken service is said here, on every layout, ahead of the OS name:
    // the grouped view says which and why, but nobody opens it to check.
    // When the half-line is short, the hostname gives way to it — on an
    // 80-column terminal a long hostname used to clip it to `1 unit fai`.
    let room = cols[0].width as usize;
    // Long, then compact, then cut to fit, always leaving a gap before the
    // right half: at 50 columns even the compact form ran straight into
    // the CPU count (`1 failed, 1 lo1 cpu`).
    let trouble = unit_trouble(&app.snap).map(|t| {
        let t = match t.chars().count() + 10 > room {
            true => unit_trouble_compact(&app.snap).unwrap_or(t),
            false => t,
        };
        crate::format::truncate_fit(&t, room.saturating_sub(1).max(1))
    });
    let host_room = match &trouble {
        // The warning, the gap after it, and a gap before the right half.
        Some(t) => room.saturating_sub(t.chars().count() + 4),
        None => room,
    };
    let mut left = Vec::new();
    // The hostname gives way entirely before the warning is cut further;
    // a stub of three letters names nothing.
    if trouble.is_none() || host_room >= 6 {
        left.push(Span::styled(
            crate::format::truncate_fit(&h.hostname, host_room.max(1)),
            Style::default().fg(app.theme.accent).add_modifier(Modifier::BOLD),
        ));
        left.push(Span::styled("  ", Style::default()));
    }
    let fits_os = trouble.as_ref().is_none_or(|t| t.chars().count() + 3 < room);
    if let Some(units) = trouble {
        left.push(Span::styled(
            format!("{units}  "),
            Style::default().fg(app.theme.crit).add_modifier(Modifier::BOLD),
        ));
    }
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
    // The OS takes what is left. A gap is kept before the right half only
    // when that half is full — it ran into the core count on a narrow
    // terminal — and the kernel version goes before the OS name is cut, so
    // a 100-column header says `Linux (Ubuntu 26.04)` rather than
    // `Linux (Ubuntu 26.0…`.
    let right_len: usize = right.spans.iter().map(|s| s.content.chars().count()).sum();
    let gap = usize::from(right_len >= cols[1].width as usize);
    let used: usize = left.iter().map(|s| s.content.chars().count()).sum();
    let os_room = room.saturating_sub(used + gap);
    let full = format!("{} {}", h.os, h.kernel);
    let os_text = match () {
        _ if full.chars().count() <= os_room => full,
        _ if h.os.chars().count() <= os_room => h.os.clone(),
        _ => crate::format::truncate_fit(&h.os, os_room),
    };
    if fits_os && os_room >= 4 {
        left.push(Span::styled(os_text, Style::default().fg(app.theme.dim)));
    }
    let left = Line::from(left);

    f.render_widget(inner, area);
    f.render_widget(Paragraph::new(left), cols[0]);
    f.render_widget(Paragraph::new(right).alignment(Alignment::Right), cols[1]);
}

/// The recording squeezed into `width` cells: where the replay is, and
/// where the markers are — `!` an alert firing, `k` a kernel event — so an
/// hour of recording shows at a glance which minutes are worth reading.
/// A cell covering both kinds shows the alert.
pub fn timeline_bar<'a>(app: &App, pos: usize, len: usize, width: usize) -> Vec<Span<'a>> {
    let width = width.min(len.max(1));
    let cell_of = |frame: usize| (frame * width / len.max(1)).min(width.saturating_sub(1));
    let mut cells = vec![' '; width];
    if let Some(t) = &app.replay_timeline {
        for m in &t.markers {
            let cell = &mut cells[cell_of(m.position)];
            *cell = match (m.kind, *cell) {
                (crate::record::MarkerKind::Alert, _) => '!',
                (crate::record::MarkerKind::Kernel, '!') => '!',
                (crate::record::MarkerKind::Kernel, _) => 'k',
            };
        }
    }
    let here = cell_of(pos);
    cells
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let (text, style) = match (i == here, c) {
                (true, _) => ('●', Style::default().fg(app.theme.accent)),
                (false, '!') => {
                    ('!', Style::default().fg(app.theme.crit).add_modifier(Modifier::BOLD))
                }
                (false, 'k') => ('k', Style::default().fg(app.theme.warn)),
                _ => ('─', Style::default().fg(app.theme.dim)),
            };
            Span::styled(text.to_string(), style)
        })
        .collect()
}

/// The firing rules: the ones paging somebody in the red banner, and the
/// silenced ones after it, quietly — still firing, still worth knowing
/// about, but not what the red is for.
fn alert_spans<'a>(app: &App) -> Vec<Span<'a>> {
    let (muted, paging): (Vec<_>, Vec<_>) =
        app.active_alerts.iter().partition(|a| a.silenced.is_some());
    let mut spans = Vec::new();
    if !paging.is_empty() {
        let msg = paging.iter().map(|a| a.message.clone()).collect::<Vec<_>>().join("  •  ");
        spans.push(Span::styled(
            format!(" ALERT  {msg} "),
            Style::default().fg(app.theme.sel_fg).bg(app.theme.crit).add_modifier(Modifier::BOLD),
        ));
    }
    match (paging.is_empty(), muted.as_slice()) {
        (_, []) => {}
        (false, m) => spans.push(Span::styled(
            format!("  +{} silenced", m.len()),
            Style::default().fg(app.theme.dim),
        )),
        (true, m) => {
            let names = m
                .iter()
                .map(|a| format!("{} ({})", a.name, a.silenced.clone().unwrap_or_default()))
                .collect::<Vec<_>>()
                .join("  •  ");
            spans.push(Span::styled(
                format!(" silenced  {names}"),
                Style::default().fg(app.theme.warn),
            ));
        }
    }
    spans
}

/// One-line status bar: alerts win over transient status, which wins over hints.
pub fn draw_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    // A replay keeps its bar whatever else is going on: it is how the replay
    // is navigated, and an alert or a "jumped to" message that replaced it
    // left nobody able to see where in the recording they were. What would
    // have taken the whole line goes beside the bar instead.
    let line = if let Some((pos, len)) = app.timeline().filter(|_| app.mode != Mode::Filter) {
        let width = (area.width as usize / 4).clamp(8, 48);
        let mut spans = vec![
            Span::styled(
                format!(" ⏸ frame {}/{len} ", pos + 1,),
                Style::default().fg(app.theme.sel_fg).bg(app.theme.accent),
            ),
            Span::raw(" "),
        ];
        spans.extend(timeline_bar(app, pos, len, width));
        spans.push(Span::raw(" "));
        // The answer to a key just pressed — "no later markers" — outranks
        // a standing alert for its few seconds, or the key looks ignored.
        if let Some(status) = app.status_text() {
            spans.push(Span::styled(status.to_string(), Style::default().fg(app.theme.accent)));
        } else if !app.active_alerts.is_empty() {
            spans.extend(alert_spans(app));
        } else {
            spans.push(Span::styled(
                format!(
                    "{} · {} shown · [ ] to scrub · ( ) markers · z to play",
                    app.source_label().unwrap_or_else(|| "replay".into()),
                    app.rows().len(),
                ),
                Style::default().fg(app.theme.dim),
            ));
        }
        Line::from(spans)
    } else if !app.active_alerts.is_empty() {
        // The answer to a key just pressed, or a hook or notification that
        // failed, goes first for its few seconds: an alert used to own the
        // whole line, so while one fired every other message was invisible.
        let mut spans = Vec::new();
        if let Some(status) = app.status_text() {
            spans.push(Span::styled(format!(" {status} "), Style::default().fg(app.theme.accent)));
        }
        spans.extend(alert_spans(app));
        Line::from(spans)
    } else if let Some(status) = app.status_text() {
        Line::from(Span::styled(format!(" {status}"), Style::default().fg(app.theme.accent)))
    } else if app.mode == Mode::Filter {
        Line::from(Span::styled(
            " type to filter · Enter apply · Esc clear",
            Style::default().fg(app.theme.dim),
        ))
    } else if let Some(label) = app.source_label() {
        // A non-live source names itself, and says so loudly when it is broken:
        // otherwise an unreachable `--remote` host is just an empty dashboard.
        // This outranks the sampling notice: hiding UNREACHABLE behind
        // "sampling…" is exactly the unexplained blank it exists to prevent.
        let broken = label.contains("UNREACHABLE");
        let style = if broken {
            Style::default().fg(app.theme.sel_fg).bg(app.theme.crit).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.accent)
        };
        Line::from(vec![
            Span::styled(format!(" {label} "), style),
            Span::styled(sampling_note(app), Style::default().fg(app.theme.warn)),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                format!(
                    " {} shown of {} tasks · sort {}{} · {} ms · [?] help",
                    app.rows().len(),
                    app.snap.procs.len(),
                    app.cfg.sort_by.label(),
                    if app.cfg.sort_desc { "↓" } else { "↑" },
                    app.refresh.as_millis(),
                ),
                Style::default().fg(app.theme.dim),
            ),
            Span::styled(sampling_note(app), Style::default().fg(app.theme.warn)),
        ])
    };
    f.render_widget(Paragraph::new(line), area);
}

/// Appended while the source is still working on a sample, so a dashboard that
/// has stopped changing explains itself rather than looking broken.
fn sampling_note(app: &App) -> &'static str {
    if app.is_stale() { " · sampling…" } else { "" }
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
