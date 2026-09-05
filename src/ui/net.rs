//! Network panels. The compact form lists interfaces with live rates; the
//! detail form adds RX/TX history charts.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Cell, Chart, Dataset, GraphType, Row, Table};
use ratatui::Frame;

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

    let visible: Vec<_> = app
        .snap
        .nets
        .iter()
        .filter(|n| app.cfg.show_virtual_ifaces || !n.virtual_iface)
        .take(inner.height.saturating_sub(1) as usize)
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
    f.render_widget(Table::new(rows, widths).header(header), inner);
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

    let header = Row::new(["Iface", "↓ rate", "↑ rate", "↓ total", "↑ total", "err", "mac"])
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
        Constraint::Min(17),
    ];
    f.render_widget(Table::new(rows, widths).header(header), slots[1]);
}
