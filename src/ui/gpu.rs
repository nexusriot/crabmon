//! GPU rows: busy percentage where the driver reports it, clock ratio where it
//! does not, plus VRAM and temperature.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Row, Table};
use ratatui::Frame;

use crate::app::App;
use crate::format::{human_bytes, human_hz_mhz, truncate_fit};
use crate::metrics::GpuInfo;
use crate::ui::block;

/// What goes in the "Load" column: a real busy figure, or a clock ratio marked
/// with `~` so it is not mistaken for utilisation.
pub fn load_text(g: &GpuInfo) -> String {
    if let Some(b) = g.busy_percent {
        return format!("{b:.0}%");
    }
    match (g.freq_ratio(), g.freq_mhz) {
        (Some(r), Some(mhz)) => format!("~{:.0}% {}", r * 100.0, human_hz_mhz(mhz)),
        (None, Some(mhz)) => human_hz_mhz(mhz),
        _ => "-".into(),
    }
}

pub fn vram_text(g: &GpuInfo) -> String {
    match (g.vram_used, g.vram_total) {
        (Some(u), Some(t)) => format!("{} / {}", human_bytes(u), human_bytes(t)),
        _ => "-".into(),
    }
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    if app.snap.gpus.is_empty() {
        return;
    }
    let b = block(app, " GPU ");
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    let header = Row::new(["Device", "Load", "VRAM", "Temp"])
        .style(Style::default().add_modifier(Modifier::BOLD).fg(app.theme.title));
    let name_w = (inner.width as usize).saturating_sub(34).max(8);

    let rows = app.snap.gpus.iter().map(|g| {
        let ratio =
            g.busy_percent.map(|b| b as f64 / 100.0).or_else(|| g.freq_ratio()).unwrap_or(0.0);
        let color = app.theme.usage(ratio, app.cfg.thresholds.warn, app.cfg.thresholds.crit);
        let temp = g.temp_c.map(|t| format!("{t:.0}°")).unwrap_or_else(|| "-".into());
        Row::new(vec![
            Cell::from(truncate_fit(&g.name, name_w)),
            Cell::from(Span::styled(load_text(g), Style::default().fg(color))),
            Cell::from(vram_text(g)),
            Cell::from(temp),
        ])
    });

    let widths =
        [Constraint::Min(8), Constraint::Length(14), Constraint::Length(18), Constraint::Length(6)];
    f.render_widget(Table::new(rows, widths).header(header), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_real_busy_counter_is_shown_as_a_plain_percentage() {
        let g = GpuInfo { busy_percent: Some(42.0), ..Default::default() };
        assert_eq!(load_text(&g), "42%");
    }

    #[test]
    fn a_clock_ratio_is_marked_so_it_is_not_read_as_utilisation() {
        let g = GpuInfo { freq_mhz: Some(600), max_freq_mhz: Some(2400), ..Default::default() };
        let t = load_text(&g);
        assert!(t.starts_with('~'), "{t}");
        assert!(t.contains("25%"), "{t}");
    }

    #[test]
    fn missing_data_renders_as_a_dash_not_zero() {
        let g = GpuInfo::default();
        assert_eq!(load_text(&g), "-");
        assert_eq!(vram_text(&g), "-");
    }

    #[test]
    fn vram_is_reported_in_real_units() {
        let g = GpuInfo {
            vram_used: Some(1_073_741_824),
            vram_total: Some(8_589_934_592),
            ..Default::default()
        };
        assert_eq!(vram_text(&g), "1.0 GiB / 8.0 GiB");
    }
}
