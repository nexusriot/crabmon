//! Overlays: signal menu, kill confirmation, renice, affinity, process detail,
//! the key-binding help and the alert list.

use ratatui::layout::{Alignment, Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, Wrap,
};
use ratatui::Frame;

use crate::app::{App, SIGNAL_NAMES};
use crate::format::{compact_bytes, human_bps, human_bytes, human_duration, truncate_fit};
use crate::ui::centered_rect;

/// Adds `n/total` to a popup title when it holds more than it can show. A
/// silently-truncated list looks like the whole list.
pub fn scroll_title(base: &str, app: &App, total: usize) -> String {
    let shown = app.popup_height.max(1);
    if total <= shown {
        return base.to_string();
    }
    let first = app.popup_scroll + 1;
    let last = (app.popup_scroll + shown).min(total);
    format!("{base}{first}-{last} of {total}  [j/k] scroll ")
}

fn popup_block<'a>(app: &App, title: String, danger: bool) -> Block<'a> {
    let color = if danger { app.theme.crit } else { app.theme.accent };
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(Span::styled(title, Style::default().fg(color).add_modifier(Modifier::BOLD)))
}

/// Frame for a popup that scrolls: records the height it got so `App` can page
/// by exactly what is on screen, and titles it with the visible range.
fn scroll_frame(
    f: &mut Frame<'_>,
    area: Rect,
    app: &mut App,
    base: &str,
    total: usize,
    pct: (u16, u16),
) -> Rect {
    let popup = centered_rect(pct.0, pct.1, area);
    let inner = Block::default().borders(Borders::ALL).inner(popup);
    app.popup_height = inner.height.saturating_sub(1).max(1) as usize;
    let title = scroll_title(base, app, total);
    frame(f, area, app, title, false, pct)
}

fn frame(
    f: &mut Frame<'_>,
    area: Rect,
    app: &App,
    title: String,
    danger: bool,
    pct: (u16, u16),
) -> Rect {
    let popup = centered_rect(pct.0, pct.1, area);
    f.render_widget(Clear, popup);
    let b = popup_block(app, title, danger);
    let inner = b.inner(popup);
    f.render_widget(b, popup);
    inner
}

pub fn signal_menu(f: &mut Frame<'_>, area: Rect, app: &App) {
    let who = app
        .target
        .as_ref()
        .map(|t| format!("{} ({})", truncate_fit(&t.name, 24), t.pid))
        .unwrap_or_else(|| "<none>".into());
    let inner = frame(f, area, app, format!(" Signal → {who} "), false, (36, 60));

    let items: Vec<ListItem> = SIGNAL_NAMES.iter().map(|n| ListItem::new(*n)).collect();
    let list = List::new(items).highlight_style(app.theme.selection()).highlight_symbol("▶ ");
    let mut state = ListState::default();
    state.select(Some(app.signal_index));
    f.render_stateful_widget(list, inner, &mut state);
}

pub fn confirm_kill(f: &mut Frame<'_>, area: Rect, app: &App) {
    let inner = frame(f, area, app, " Confirm ".into(), true, (46, 30));
    let targets = app.action_targets();
    let sig = SIGNAL_NAMES[app.signal_index];

    // A bulk signal must say how many processes it will hit, not just the first.
    let who = match targets.len() {
        0 => "<none>".to_string(),
        1 => format!("{} (pid {})?", truncate_fit(&targets[0].name, 30), targets[0].pid),
        n => format!("{n} tagged processes?"),
    };
    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("Send {sig} to"),
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(who, Style::default().fg(app.theme.warn))),
        Line::from(""),
        Line::from("[y] yes    [n] no"),
    ];
    f.render_widget(Paragraph::new(text).alignment(Alignment::Center), inner);
}

fn input_popup(f: &mut Frame<'_>, area: Rect, app: &App, title: String, prompt: &str, hint: &str) {
    let inner = frame(f, area, app, title, false, (46, 30));
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(prompt.to_string(), Style::default().fg(app.theme.text))),
        Line::from(Span::styled(
            format!("{}\u{2588}", app.input),
            Style::default().fg(app.theme.warn).add_modifier(Modifier::BOLD),
        )),
    ];
    if let Some(err) = &app.input_error {
        lines.push(Line::from(Span::styled(err.clone(), Style::default().fg(app.theme.crit))));
    }
    lines.push(Line::from(Span::styled(hint.to_string(), Style::default().fg(app.theme.dim))));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), inner);
}

pub fn renice(f: &mut Frame<'_>, area: Rect, app: &App) {
    input_popup(
        f,
        area,
        app,
        format!(" Renice {} ", app.prompt_subject()),
        "New nice value (-20 … 19):",
        "[Enter] apply   [Esc] cancel",
    );
}

pub fn affinity(f: &mut Frame<'_>, area: Rect, app: &App) {
    input_popup(
        f,
        area,
        app,
        format!(" CPU affinity {} ", app.prompt_subject()),
        "CPU list, e.g. 0-3,8:",
        "[Enter] apply   [Esc] cancel",
    );
}

/// Fixed rows the detail pane draws before the socket list. `App` needs the
/// count to know how far the pane can scroll.
pub const DETAIL_FIELDS: usize = 16;

pub fn detail(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let Some(p) = app.selected_row().cloned() else { return };
    let p = &p;
    let total = DETAIL_FIELDS + app.detail_sockets.len();
    let title = format!(" {} (pid {}) ", truncate_fit(&p.name, 32), p.pid);
    let inner = scroll_frame(f, area, app, &title, total, (72, 70));

    let field = |k: &str, v: String| -> Row<'static> {
        let k = k.to_string();
        Row::new(vec![
            Cell::from(Span::styled(k, Style::default().fg(app.theme.dim))),
            Cell::from(Span::styled(v, Style::default().fg(app.theme.text))),
        ])
    };

    let listening: Vec<String> = app
        .detail_sockets
        .iter()
        .filter(|s| s.state == "LISTEN")
        .filter_map(|s| crate::metrics::sockets::port_of(&s.local).map(|p| p.to_string()))
        .collect();

    let mut rows = vec![
        field("PID", p.pid.to_string()),
        field("Parent", p.ppid.map(|v| v.to_string()).unwrap_or_else(|| "-".into())),
        field(
            "User",
            format!(
                "{} ({})",
                p.user.clone().unwrap_or_else(|| "-".into()),
                p.uid.map(|u| u.to_string()).unwrap_or_else(|| "-".into())
            ),
        ),
        field("State", p.state.to_string()),
        field("CPU", format!("{:.1}%", p.cpu)),
        field("Memory", format!("{} rss / {} virt", human_bytes(p.mem), human_bytes(p.virt))),
        field("Threads", p.threads.map(|t| t.to_string()).unwrap_or_else(|| "-".into())),
        field("Disk", format!("R {}  W {}", human_bps(p.read_bps), human_bps(p.write_bps))),
        field("Running", human_duration(p.run_time)),
        field(
            "Nice",
            crate::metrics::sysinfo_source::get_priority(p.pid)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
        ),
        field("Service", p.service.clone().unwrap_or_else(|| "-".into())),
        field("Container", p.container.clone().unwrap_or_else(|| "-".into())),
        field("Exe", if p.exe.is_empty() { "-".into() } else { p.exe.clone() }),
        field("Cwd", if p.cwd.is_empty() { "-".into() } else { p.cwd.clone() }),
        field("Listening", if listening.is_empty() { "-".into() } else { listening.join(", ") }),
        field("Sockets", app.detail_sockets.len().to_string()),
    ];
    debug_assert_eq!(rows.len(), DETAIL_FIELDS, "DETAIL_FIELDS must match what is drawn");

    // Open sockets follow the fields, and scroll with them.
    for sock in &app.detail_sockets {
        rows.push(field(
            &sock.protocol,
            format!(
                "{} → {}{}",
                sock.local,
                sock.remote,
                if sock.state.is_empty() { String::new() } else { format!("  {}", sock.state) }
            ),
        ));
    }

    let visible = inner.height.saturating_sub(2) as usize;
    let start = app.popup_scroll.min(rows.len().saturating_sub(1));
    let shown: Vec<Row> = rows.into_iter().skip(start).take(visible).collect();
    let table_h = shown.len() as u16;
    let widths = [Constraint::Length(10), Constraint::Min(10)];
    let table_area = Rect { height: table_h.min(inner.height), ..inner };
    f.render_widget(Table::new(shown, widths), table_area);

    if inner.height > table_h + 1 {
        let cmd_area =
            Rect { y: inner.y + table_h + 1, height: inner.height - table_h - 1, ..inner };
        let cmd = if p.cmd.is_empty() { "-" } else { &p.cmd };
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("Command", Style::default().fg(app.theme.dim))),
                Line::from(cmd.to_string()),
            ])
            .wrap(Wrap { trim: true }),
            cmd_area,
        );
    }
}

/// Every binding, in one place, so the help and the README cannot drift.
pub const KEYS: [(&str, &str); 38] = [
    ("q / Ctrl-C", "quit"),
    ("↑ ↓ / k j", "move selection"),
    ("PgUp PgDn", "move a page"),
    ("Home / End", "first / last process"),
    ("Enter", "process detail, or open the selected group"),
    ("/", "filter (user: pid: cpu> mem> re: !neg)"),
    ("Esc", "clear the filter"),
    ("c m p n d", "sort by cpu / mem / pid / name / disk"),
    ("< >", "previous / next sort column"),
    ("s", "reverse sort order"),
    ("click header", "sort by that column"),
    ("T", "toggle process tree"),
    ("H", "show or hide individual threads"),
    ("G", "group by service, container or user"),
    ("Space", "tag a process for a bulk action"),
    ("U", "untag everything"),
    ("f", "pin a process to the top of the table"),
    ("F", "unpin everything"),
    ("z", "pause and resume sampling"),
    ("[ ]", "step a recording back or forward"),
    ("{ }", "step a recording ten frames"),
    ("1-9", "apply a saved filter"),
    ("0", "clear the filter"),
    ("y", "copy the command line to the clipboard"),
    ("L", "action log"),
    ("Tab / S-Tab", "next / previous layout"),
    ("v", "count virtual interfaces in the total"),
    ("g", "group or expand sensors"),
    ("+ / -", "refresh faster / slower"),
    ("t", "signal menu (Unix)"),
    ("r", "renice (Unix)"),
    ("A", "CPU affinity (Linux)"),
    ("P", "export a snapshot"),
    ("!", "active alerts"),
    ("?", "this help"),
    ("mouse wheel", "scroll the process list"),
    ("mouse click", "select a process"),
    ("j k in a popup", "scroll it"),
];

pub fn help(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let inner = scroll_frame(f, area, app, " Keys ", KEYS.len(), (64, 80));
    let rows = KEYS.iter().skip(app.popup_scroll).take(inner.height as usize).map(|(k, v)| {
        Row::new(vec![
            Cell::from(Span::styled(
                *k,
                Style::default().fg(app.theme.accent).add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(*v, Style::default().fg(app.theme.text))),
        ])
    });
    let widths = [Constraint::Length(14), Constraint::Min(20)];
    f.render_widget(Table::new(rows, widths), inner);
}

pub fn alerts(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let total = app.alerts.rules.len() + app.alerts.history_len();
    let inner = scroll_frame(f, area, app, " Alerts ", total, (72, 60));
    if app.alerts.rules.is_empty() {
        f.render_widget(
            Paragraph::new("no alert rules configured\n\nAdd [[alert]] blocks to the config file.")
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    let active: Vec<&str> = app.active_alerts.iter().map(|a| a.name.as_str()).collect();
    let rule_rows: Vec<Row> = app
        .alerts
        .rules
        .iter()
        .map(|r| {
            let firing = active.contains(&r.name.as_str());
            let style = if firing {
                Style::default().fg(app.theme.crit).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.dim)
            };
            Row::new(vec![
                Cell::from(Span::styled(r.name.clone(), style)),
                Cell::from(format!("{:?}", r.kind).to_lowercase()),
                Cell::from(format!("{:.0}", r.threshold)),
                Cell::from(format!("{}s", r.for_secs)),
                Cell::from(if r.target.is_empty() { "any".to_string() } else { r.target.clone() }),
                Cell::from(if firing { "FIRING" } else { "ok" }),
            ])
        })
        .collect();

    // What happened, newest first. A rule list alone cannot tell you an alert
    // fired at 03:12 and cleared at 03:14 — which is the question you have the
    // morning after.
    let history_rows: Vec<Row> = app
        .alerts
        .history()
        .map(|e| {
            let (label, style) = if e.fired {
                ("FIRED", Style::default().fg(app.theme.crit))
            } else {
                ("cleared", Style::default().fg(app.theme.ok))
            };
            Row::new(vec![
                Cell::from(Span::styled(e.name.clone(), Style::default().fg(app.theme.text))),
                Cell::from(Span::styled(label, style)),
                Cell::from(format!("{:.1}", e.value)),
                Cell::from(""),
                Cell::from(relative_time(&e.at_unix.to_string())),
                Cell::from(""),
            ])
        })
        .collect();

    let mut rows = rule_rows;
    rows.extend(history_rows);
    let rows: Vec<Row> = rows
        .into_iter()
        .skip(app.popup_scroll)
        .take(inner.height.saturating_sub(1) as usize)
        .collect();

    let widths = [
        Constraint::Min(10),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Length(12),
        Constraint::Length(7),
    ];
    let header = Row::new(["Rule", "Kind", "Over", "For", "Target/When", "State"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

/// The local record of every signal, renice and affinity change crabmon made.
pub fn audit_log(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let total = app.audit_entries.len();
    let inner = scroll_frame(f, area, app, " Action log ", total, (76, 70));
    if app.audit_entries.is_empty() {
        let msg = match &app.audit_path {
            Some(p) => format!("no actions recorded yet\n\n{}", p.display()),
            None => "the action log is disabled in the config".to_string(),
        };
        f.render_widget(Paragraph::new(msg).alignment(Alignment::Center), inner);
        return;
    }
    let rows = app
        .audit_entries
        .iter()
        .skip(app.popup_scroll)
        .take(inner.height.saturating_sub(1) as usize)
        .map(|line| {
            let mut fields = line.split('\t');
            let when = fields.next().unwrap_or("");
            let action = fields.next().unwrap_or("");
            let pid = fields.next().unwrap_or("").trim_start_matches("pid=");
            let name = fields.next().unwrap_or("").trim_start_matches("name=");
            let outcome = fields.next().unwrap_or("");
            let style = if outcome.starts_with("failed") {
                Style::default().fg(app.theme.crit)
            } else {
                Style::default().fg(app.theme.text)
            };
            Row::new(vec![
                Cell::from(Span::styled(relative_time(when), Style::default().fg(app.theme.dim))),
                Cell::from(Span::styled(action.to_string(), Style::default().fg(app.theme.accent))),
                Cell::from(pid.to_string()),
                Cell::from(truncate_fit(name, 20)),
                Cell::from(Span::styled(outcome.to_string(), style)),
            ])
        });
    let widths = [
        Constraint::Length(10),
        Constraint::Length(18),
        Constraint::Length(8),
        Constraint::Length(20),
        Constraint::Min(6),
    ];
    let header = Row::new(["When", "Action", "PID", "Name", "Result"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

/// "3m ago" from a unix timestamp string; the raw value if it will not parse.
pub fn relative_time(unix: &str) -> String {
    let Ok(then) = unix.parse::<u64>() else { return unix.to_string() };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ago = now.saturating_sub(then);
    match ago {
        0..=59 => format!("{ago}s ago"),
        60..=3599 => format!("{}m ago", ago / 60),
        3600..=86_399 => format!("{}h ago", ago / 3600),
        _ => format!("{}d ago", ago / 86_400),
    }
}

/// Used by the detail popup's memory line and by tests.
pub fn short_mem(bytes: u64) -> String {
    compact_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_lists_every_interactive_key() {
        // Guards against a binding being added to `app.rs` and never documented.
        for needle in ["quit", "filter", "signal", "renice", "affinity", "tree", "layout", "export"]
        {
            assert!(
                KEYS.iter().any(|(_, v)| v.contains(needle)),
                "no help entry mentions {needle}"
            );
        }
    }

    #[test]
    fn help_entries_are_short_enough_for_the_popup() {
        for (k, _) in KEYS {
            assert!(k.len() <= 14, "key column overflows: {k}");
        }
    }

    #[test]
    fn short_mem_is_compact() {
        assert_eq!(short_mem(475_815_936), "454M");
    }
}
