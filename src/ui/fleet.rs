//! The fleet table: one row per host, so the machine in trouble is visible
//! before anyone has picked it.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Row, Table, TableState};
use ratatui::Frame;

use crate::app::App;
use crate::fleet::FleetHost;
use crate::format::{compact_bytes, human_duration, truncate_fit};
use crate::ui::block;

/// Columns, left to right. Everything a dashboard would make you visit each
/// host to see, reduced to the figures that say "this one".
///
/// STATUS is last and takes whatever width is left, which is what gives a
/// host that is not answering somewhere to put the reason. Squeezing it into
/// a numeric column truncated "unreachable: No route to host" to "unreac".
const HEADERS: [&str; 9] =
    ["HOST", "CPU%", "MEM", "SWAP", "LOAD", "DISK", "NET", "PROCS", "STATUS"];

/// Widest a host name is allowed to get before it is truncated. Beyond this
/// the name is pushing the numbers off to the right of a wide terminal, and
/// the numbers are the reason the table exists.
const MAX_NAME: u16 = 28;

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let hosts = app.fleet();
    let title = match hosts.len() {
        0 => " Fleet ".to_string(),
        n => format!(" Fleet — {n} hosts "),
    };
    let outer = block(app, title);
    let inner = outer.inner(area);
    f.render_widget(outer, area);

    if hosts.is_empty() {
        // The layout is reachable on a single-host run only by asking for it
        // by name, so say what it is for rather than drawing an empty table.
        let hint = ratatui::widgets::Paragraph::new(
            "no fleet: start crabmon with --remote host-a,host-b,… to watch several machines",
        )
        .style(Style::default().fg(app.theme.dim));
        f.render_widget(hint, inner);
        return;
    }

    let page = inner.height.saturating_sub(1) as usize;
    app.set_fleet_page(page, hosts.len());

    let header = Row::new(HEADERS.to_vec())
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    // The numeric columns and their spacing come to 52; leave at least ten
    // for the status, and never let the name have more than it needs.
    let name_w = inner.width.saturating_sub(62).clamp(10, MAX_NAME);

    let start = app.fleet_scroll.min(hosts.len().saturating_sub(1));
    let end = (start + page.max(1)).min(hosts.len());
    let rows: Vec<Row> = hosts[start..end].iter().map(|h| row(app, h, name_w as usize)).collect();

    let widths = [
        Constraint::Length(name_w + 1),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Min(10),
    ];
    let table = Table::new(rows, widths).header(header).highlight_style(app.theme.selection());
    let mut state = TableState::default();
    state.select(Some(app.fleet_selected.saturating_sub(start)));
    f.render_stateful_widget(table, inner, &mut state);
}

fn row<'a>(app: &App, h: &FleetHost, name_w: usize) -> Row<'a> {
    // The marker is how the selected host is recognisable once the cursor has
    // moved on: in the fleet table the highlight follows the keyboard, while
    // every other panel is still drawing the host that was *chosen*.
    let chosen = app.fleet_target().is_some_and(|t| t == h.target);
    let name = format!("{}{}", if chosen { "▸" } else { " " }, truncate_fit(&h.target, name_w));

    // A host with nothing to say gets the reason in the status column rather
    // than a row of zeroes, which would read as a machine sitting idle.
    let status = match (&h.error, h.has_data()) {
        (Some(e), _) => {
            Span::styled(format!("{}: {e}", h.status()), Style::default().fg(app.theme.crit))
        }
        (None, false) => Span::styled("connecting…", Style::default().fg(app.theme.dim)),
        (None, true) => Span::styled(
            format!("up {}", human_duration(h.snapshot.host.uptime_secs)),
            Style::default().fg(app.theme.dim),
        ),
    };

    if !h.has_data() {
        // Nothing has ever arrived, so there is nothing to put in the
        // numeric columns. Blank, not zero.
        let mut cells = vec![Cell::from(Span::styled(name, Style::default().fg(app.theme.dim)))];
        cells.extend((0..HEADERS.len() - 2).map(|_| Cell::from("")));
        cells.push(Cell::from(status));
        return Row::new(cells);
    }

    let s = &h.snapshot;
    let cpu = s.cpu.avg();
    let (rx, tx) = s.net_totals(app.cfg.show_virtual_ifaces);
    let cpu_color = app.theme.usage(cpu / 100.0, app.cfg.thresholds.warn, app.cfg.thresholds.crit);
    let mem_color =
        app.theme.usage(s.mem.ratio(), app.cfg.thresholds.warn, app.cfg.thresholds.crit);

    Row::new(vec![
        Cell::from(name),
        Cell::from(Span::styled(format!("{cpu:>5.1}"), Style::default().fg(cpu_color))),
        Cell::from(Span::styled(
            format!("{:>6.0}%", s.mem.ratio() * 100.0),
            Style::default().fg(mem_color),
        )),
        Cell::from(format!("{:>5.0}%", s.mem.swap_ratio() * 100.0)),
        Cell::from(format!("{:>5.2}", s.host.load[0])),
        Cell::from(match s.disk_util_max() {
            Some(u) => format!("{:>5.0}%", u * 100.0),
            None => "    -".to_string(),
        }),
        Cell::from(format!("{:>8}", compact_bytes((rx + tx) as u64))),
        Cell::from(format!("{:>5}", s.procs.len())),
        Cell::from(status),
    ])
}

/// The one-line summary under the table: how much of the fleet is answering,
/// and how long the selected host has been up.
pub fn summary(hosts: &[FleetHost], selected: usize) -> String {
    if hosts.is_empty() {
        return "no hosts".into();
    }
    let live = hosts.iter().filter(|h| h.is_live()).count();
    let up = hosts
        .get(selected)
        .filter(|h| h.is_live())
        .map(|h| format!(", {} up {}", h.target, human_duration(h.snapshot.host.uptime_secs)))
        .unwrap_or_default();
    format!("{live}/{} reachable{up}", hosts.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{HostInfo, Snapshot};

    fn host(target: &str, up: u64, error: Option<&str>, waiting: bool) -> FleetHost {
        FleetHost {
            target: target.into(),
            snapshot: Snapshot {
                host: HostInfo { uptime_secs: up, ..Default::default() },
                ..Default::default()
            },
            error: error.map(String::from),
            waiting,
        }
    }

    #[test]
    fn the_summary_counts_what_is_answering() {
        let hosts = vec![
            host("a", 90, None, false),
            host("b", 0, Some("no route"), false),
            host("c", 0, None, true),
        ];
        let s = summary(&hosts, 0);
        assert!(s.starts_with("1/3 reachable"), "{s}");
        assert!(s.contains("a up"), "{s}");
    }

    #[test]
    fn the_summary_does_not_claim_an_uptime_for_a_host_that_is_not_answering() {
        let hosts = vec![host("a", 0, Some("no route"), false)];
        let s = summary(&hosts, 0);
        assert_eq!(s, "0/1 reachable");
    }

    #[test]
    fn an_empty_fleet_summarises_as_nothing_rather_than_zero_of_zero() {
        assert_eq!(summary(&[], 0), "no hosts");
        assert_eq!(summary(&[host("a", 1, None, false)], 99), "1/1 reachable");
    }

    #[test]
    fn every_column_has_a_heading() {
        assert_eq!(HEADERS[0], "HOST");
        assert_eq!(
            *HEADERS.last().unwrap(),
            "STATUS",
            "the status column is last because it takes the leftover width"
        );
    }
}
