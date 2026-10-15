//! Per-mount usage gauges with live read/write throughput.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::Span;
use ratatui::widgets::{Gauge, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::format::{human_bps, human_bytes, human_duration, truncate_fit};
use crate::metrics::DiskRow;
use crate::ui::block;

/// Utilisation below this is ordinary background noise on any machine that is
/// doing anything at all, and annotating every row with it would crowd out the
/// figures the panel is actually for.
const BUSY_FROM: f64 = 0.5;

/// Longest ETA worth drawing. Past a week "it is filling" is not news, and
/// a label that always carries a number is one nobody reads.
pub const FILL_HORIZON: u64 = 7 * 24 * 3600;

/// `full in 3h 20m`, or nothing.
///
/// A percentage cannot ask this question: 95% of a 4 TB array is days of
/// warning and 95% of a 20 GB root is twenty minutes, and the same alert
/// fires far too late on one and far too early on the other.
pub fn fill_text(eta_secs: Option<u64>) -> String {
    match eta_secs {
        Some(eta) if eta <= FILL_HORIZON => format!("  full in {}", human_duration(eta)),
        _ => String::new(),
    }
}

pub fn disk_label(d: &DiskRow, mount_width: usize) -> String {
    disk_label_with_eta(d, mount_width, None)
}

/// As `disk_label`, with the fill prediction the app has been measuring.
pub fn disk_label_with_eta(d: &DiskRow, mount_width: usize, eta_secs: Option<u64>) -> String {
    let io = if d.read_bps + d.write_bps < 1.0 {
        String::new()
    } else {
        format!("  R {}  W {}", human_bps(d.read_bps), human_bps(d.write_bps))
    };
    // Throughput alone cannot say a device is saturated: an NVMe serving small
    // random reads is pinned at a few MB/s, and the rates above draw that as
    // very nearly idle. Shown only once it is high enough to be the answer to
    // a question, so an idle panel stays readable.
    let busy = match d.util {
        Some(u) if u >= BUSY_FROM => match d.await_ms {
            Some(ms) => format!("  {:.0}% busy {:.1}ms", u * 100.0, ms),
            None => format!("  {:.0}% busy", u * 100.0),
        },
        _ => String::new(),
    };
    // A filesystem can run out of inodes with terabytes free, and then nothing
    // in a bytes-only gauge explains why writes are failing.
    let inodes = if d.inodes_are_the_problem() {
        format!("  {:.0}% inodes", d.inode_ratio() * 100.0)
    } else {
        String::new()
    };
    format!(
        "{}  {} / {}{inodes}{}{busy}{io}",
        truncate_fit(&d.mount, mount_width),
        human_bytes(d.used),
        human_bytes(d.total),
        fill_text(eta_secs)
    )
}

pub fn draw(f: &mut Frame<'_>, area: Rect, app: &App) {
    let (r, w) = app.snap.disk_io_totals();
    let title = if r + w < 1.0 {
        " Disks ".to_string()
    } else {
        // The busiest device, not an average: two disks at 100% and 0% are not
        // a machine at 50%, and the pinned one is still what everything is
        // waiting on.
        let busy = match app.snap.disk_util_max() {
            Some(u) if u >= BUSY_FROM => format!("  {:.0}% busy", u * 100.0),
            _ => String::new(),
        };
        format!(" Disks  R {}  W {}{busy} ", human_bps(r), human_bps(w))
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
            .label(Span::styled(
                disk_label_with_eta(d, mount_width, app.fill_eta(&d.mount, d.total, d.used)),
                Style::default().fg(app.theme.title),
            ));
        f.render_widget(gauge, *slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "95% full" fires far too late on a 4 TB array and far too early on a
    /// 20 GB root. Time to full is the same number on both.
    #[test]
    fn a_filling_disk_says_when_it_will_be_full() {
        let d = DiskRow {
            mount: "/var".into(),
            total: 10_000_000_000,
            used: 9_000_000_000,
            ..Default::default()
        };
        let label = disk_label_with_eta(&d, 18, Some(3600));
        assert!(label.contains("full in"), "{label}");
        assert!(label.contains("01:00:00") || label.contains("1h"), "{label}");
    }

    /// A disk that is not filling, or that will take a year, says nothing:
    /// a label that always carries a number is one nobody reads.
    #[test]
    fn a_disk_that_is_not_filling_soon_says_nothing_about_it() {
        let d = DiskRow { mount: "/".into(), total: 100, used: 1, ..Default::default() };
        assert!(!disk_label_with_eta(&d, 18, None).contains("full in"));
        assert!(!disk_label_with_eta(&d, 18, Some(FILL_HORIZON + 1)).contains("full in"));
        assert!(disk_label_with_eta(&d, 18, Some(FILL_HORIZON)).contains("full in"));
    }

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
    fn a_saturated_device_says_so_even_when_its_throughput_looks_idle() {
        // 4 MB/s of small random reads, and the queue never empties. The rates
        // alone draw this as a disk doing almost nothing.
        let pinned = DiskRow {
            mount: "/".into(),
            total: 1_000_000_000_000,
            used: 100_000_000_000,
            read_bps: 4_000_000.0,
            util: Some(0.98),
            await_ms: Some(9.4),
            ..Default::default()
        };
        let label = disk_label(&pinned, 18);
        assert!(label.contains("98% busy"), "{label}");
        assert!(label.contains("9.4ms"), "{label}");
    }

    #[test]
    fn an_ordinary_disk_is_not_annotated_with_its_utilisation() {
        // Every machine doing anything has some non-zero utilisation, and a
        // percentage on every row crowds out what the panel is for.
        let ordinary = DiskRow {
            mount: "/".into(),
            total: 100,
            used: 50,
            read_bps: 2048.0,
            util: Some(0.04),
            await_ms: Some(0.3),
            ..Default::default()
        };
        assert!(!disk_label(&ordinary, 18).contains("busy"));

        // ...and a platform that publishes no utilisation says nothing at all,
        // rather than "0% busy", which would claim a measurement.
        let unknown = DiskRow { util: None, ..ordinary };
        assert!(!disk_label(&unknown, 18).contains("busy"));
    }

    #[test]
    fn a_busy_device_with_no_completed_requests_omits_the_latency() {
        // One enormous write in flight: the queue is full and nothing has
        // finished, so there is no average service time to report.
        let d = DiskRow {
            mount: "/".into(),
            total: 100,
            used: 50,
            write_bps: 500_000_000.0,
            util: Some(1.0),
            await_ms: None,
            ..Default::default()
        };
        let label = disk_label(&d, 18);
        assert!(label.contains("100% busy"), "{label}");
        assert!(!label.contains("ms"), "{label}");
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
