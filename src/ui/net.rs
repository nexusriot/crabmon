//! Network panels. The compact form lists interfaces with live rates; the
//! detail form adds RX/TX history charts.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table};

use crate::app::App;
use crate::format::{human_bps, human_bytes};
use crate::ui::block;

fn title(app: &App) -> String {
    let (rx, tx) = app.snap.net_totals(app.cfg.show_virtual_ifaces);
    let scope = if app.cfg.show_virtual_ifaces { "all" } else { "phys" };
    format!(" Network [{scope}]  ↓ {}  ↑ {} ", human_bps(rx), human_bps(tx))
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, title(app));
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    let header = Row::new(["Iface", "↓ rate", "↑ rate", "total"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));

    // TCP health takes the last line once there is room for it and a row.
    let tcp_line = app.snap.tcp.as_ref().filter(|_| inner.height >= 3).map(|t| {
        let mut spans = Vec::new();
        for (i, (text, alarming)) in tcp_summary(t).into_iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" · ", Style::default().fg(app.theme.dim)));
            }
            let fg = if alarming { app.theme.warn } else { app.theme.dim };
            spans.push(Span::styled(text, Style::default().fg(fg)));
        }
        Line::from(spans)
    });
    let table_area = Rect { height: inner.height - u16::from(tcp_line.is_some()), ..inner };

    let visible: Vec<_> = app
        .snap
        .nets
        .iter()
        .filter(|n| app.cfg.show_virtual_ifaces || !n.virtual_iface)
        .take(table_area.height.saturating_sub(1) as usize)
        .collect();

    let rows = visible.iter().map(|n| {
        let style = if n.virtual_iface {
            Style::default().fg(app.theme.dim)
        } else {
            Style::default().fg(app.theme.text)
        };
        Row::new(vec![
            Cell::from(Span::styled(n.name.clone(), style)),
            Cell::from(Span::styled(human_bps(n.rx_bps), Style::default().fg(app.theme.ok))),
            Cell::from(Span::styled(human_bps(n.tx_bps), Style::default().fg(app.theme.accent))),
            Cell::from(Span::styled(human_bytes(n.rx_total + n.tx_total), style)),
        ])
    });

    let widths = [
        Constraint::Percentage(28),
        Constraint::Percentage(24),
        Constraint::Percentage(24),
        Constraint::Percentage(24),
    ];
    f.render_widget(Table::new(rows, widths).header(header), table_area);
    if let Some(line) = tcp_line {
        let footer = Rect { y: inner.y + inner.height - 1, height: 1, ..inner };
        f.render_widget(Paragraph::new(line), footer);
    }
}

/// `tcp 87 est 40 tw 3 cw · retr 2.5% · drop 4/s · rst 0.5/s`.
///
/// Connection counts always, so the line has a stable shape; the rates only
/// once they are non-zero, because a listener dropping connections is the
/// thing to see and a row of `0/s` is the thing that hides it.
pub fn tcp_summary(t: &crate::metrics::TcpHealth) -> Vec<(String, bool)> {
    let mut counts = format!("tcp {} est", t.established);
    if let Some(tw) = t.time_wait {
        counts.push_str(&format!(" {tw} tw"));
    }
    if let Some(cw) = t.close_wait.filter(|n| *n > 0) {
        counts.push_str(&format!(" {cw} cw"));
    }
    let mut parts = vec![(counts, false)];
    if let Some(r) = t.retrans_percent.filter(|r| *r >= 0.05) {
        parts.push((format!("retr {r:.1}%"), r >= 1.0));
    }
    if let Some(d) = t.listen_drops_ps.filter(|d| *d > 0.0) {
        parts.push((format!("drop {}/s", rate_text(d)), true));
    }
    if t.resets_ps >= 0.05 {
        parts.push((format!("rst {}/s", rate_text(t.resets_ps)), false));
    }
    parts
}

fn rate_text(v: f64) -> String {
    if v >= 10.0 { format!("{v:.0}") } else { format!("{v:.1}") }
}

/// `Layout::Io` variant: RX/TX history chart above the interface table.
pub fn draw_detail(f: &mut Frame<'_>, area: Rect, app: &App) {
    let b = block(app, title(app));
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height < 3 {
        return;
    }

    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Min(2)])
        .split(inner);

    let rx = app.net_rx_hist.points();
    let tx = app.net_tx_hist.points();
    if !rx.is_empty() {
        let peak = app.net_rx_hist.max().max(app.net_tx_hist.max()).max(1024.0);
        let datasets = vec![
            Dataset::default()
                .name("rx")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(app.theme.ok))
                .data(&rx),
            Dataset::default()
                .name("tx")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(app.theme.accent))
                .data(&tx),
        ];
        let chart = Chart::new(datasets)
            .x_axis(Axis::default().bounds([0.0, (rx.len().saturating_sub(1)).max(1) as f64]))
            .y_axis(Axis::default().bounds([0.0, peak]).labels(vec![
                Span::styled("0", Style::default().fg(app.theme.dim)),
                Span::styled(human_bps(peak), Style::default().fg(app.theme.dim)),
            ]));
        f.render_widget(chart, slots[0]);
    }

    let header =
        Row::new(["Iface", "↓ rate", "↑ rate", "↓ total", "↑ total", "err", "drop", "mac"])
            .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    let rows =
        app.snap.nets.iter().filter(|n| app.cfg.show_virtual_ifaces || !n.virtual_iface).map(|n| {
            Row::new(vec![
                Cell::from(n.name.clone()),
                Cell::from(human_bps(n.rx_bps)),
                Cell::from(human_bps(n.tx_bps)),
                Cell::from(human_bytes(n.rx_total)),
                Cell::from(human_bytes(n.tx_total)),
                Cell::from((n.errors_rx + n.errors_tx).to_string()),
                Cell::from((n.drops_rx + n.drops_tx).to_string()),
                Cell::from(n.mac.clone()),
            ])
        });
    let widths = [
        Constraint::Length(16),
        Constraint::Length(11),
        Constraint::Length(11),
        Constraint::Length(11),
        Constraint::Length(11),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Min(17),
    ];
    f.render_widget(Table::new(rows, widths).header(header), slots[1]);
}
