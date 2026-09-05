//! Per-mount usage gauges with live read/write throughput.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::Span;
use ratatui::widgets::{Gauge, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::format::{human_bps, human_bytes, truncate_fit};
use crate::metrics::DiskRow;
use crate::ui::block;

pub fn disk_label(d: &DiskRow, mount_width: usize) -> String {
    let io = if d.read_bps + d.write_bps < 1.0 {
        String::new()
    } else {
        format!("  R {}  W {}", human_bps(d.read_bps), human_bps(d.write_bps))
    };
    // A filesystem can run out of inodes with terabytes free, and then nothing
    // in a bytes-only gauge explains why writes are failing.
    let inodes = if d.inodes_are_the_problem() {
        format!("  {:.0}% inodes", d.inode_ratio() * 100.0)
    } else {
        String::new()
    };
    format!(
        "{}  {} / {}{inodes}{io}",
        truncate_fit(&d.mount, mount_width),
        human_bytes(d.used),
        human_bytes(d.total)
    )
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let (r, w) = app.snap.disk_io_totals();
    let title = if r + w < 1.0 {
        " Disks ".to_string()
    } else {
        format!(" Disks  R {}  W {} ", human_bps(r), human_bps(w))
    };
    let b = block(app, title);
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    if app.snap.disks.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("no disks", Style::default().fg(app.theme.dim))),
            inner,
        );
        return;
    }

    let shown = (inner.height as usize).min(app.snap.disks.len());
    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Length(1); shown])
        .split(inner);

    let mount_width = (inner.width as usize / 3).clamp(6, 24);
    for (d, slot) in app.snap.disks.iter().take(shown).zip(slots.iter()) {
        let ratio = d.ratio();
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(app.theme.usage(
                ratio,
                app.cfg.thresholds.warn,
                app.cfg.thresholds.crit,
            )))
            .ratio(ratio)
            .label(Span::styled(disk_label(d, mount_width), Style::default().fg(app.theme.title)));
        f.render_widget(gauge, *slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_disks_omit_the_io_suffix() {
        let d = DiskRow {
            mount: "/".into(),
            total: 500_000_000_000,
            used: 250_000_000_000,
            ..Default::default()
        };
        let label = disk_label(&d, 18);
        assert!(!label.contains(" R "), "{label}");
        assert!(label.contains("232.8 GiB / 465.7 GiB"), "{label}");
    }

    #[test]
    fn inode_pressure_is_surfaced_only_when_it_is_worse_than_the_bytes() {
        // Plenty of space, almost no inodes: the failure mode a bytes gauge hides.
        let starved = DiskRow {
            mount: "/var".into(),
            total: 1_000_000_000,
            used: 100_000_000,
            inodes_total: 65_536,
            inodes_used: 64_800,
            ..Default::default()
        };
        assert!(disk_label(&starved, 18).contains("99% inodes"), "{}", disk_label(&starved, 18));

        // A filesystem where bytes are the binding constraint says nothing.
        let normal = DiskRow {
            mount: "/".into(),
            total: 100,
            used: 90,
            inodes_total: 1000,
            inodes_used: 100,
            ..Default::default()
        };
        assert!(!disk_label(&normal, 18).contains("inodes"));

        // Neither does one with no inode data at all.
        let unknown = DiskRow { mount: "/".into(), total: 100, used: 90, ..Default::default() };
        assert!(!disk_label(&unknown, 18).contains("inodes"));
    }

    #[test]
    fn busy_disks_show_read_and_write_rates() {
        let d = DiskRow {
            mount: "/var/lib/docker/overlay2".into(),
            total: 100,
            used: 50,
            read_bps: 2048.0,
            write_bps: 1_048_576.0,
            ..Default::default()
        };
        let label = disk_label(&d, 10);
        assert!(label.contains("R 2.0 KiB/s"), "{label}");
        assert!(label.contains("W 1.0 MiB/s"), "{label}");
        assert!(label.starts_with("/var/lib/…"), "{label}");
    }
}
