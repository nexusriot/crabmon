//! Memory, swap and cgroup gauges.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Span;
use ratatui::widgets::{Gauge, Paragraph};

use crate::app::App;
use crate::format::{human_bps, human_bytes};
use crate::ui::block;

pub fn memory_label(used: u64, total: u64, available: u64, buff_cache: u64) -> String {
    if total == 0 {
        return "Mem  (unknown)".into();
    }
    let pct = used as f64 / total as f64 * 100.0;
    let mut label = format!(
        "{pct:>5.1}%  {} / {}  ({} avail",
        human_bytes(used),
        human_bytes(total),
        human_bytes(available)
    );
    // "Used" on Linux is a derived figure; the reclaimable part is the context
    // that stops a scary-looking number from being scary.
    if buff_cache > 0 {
        label.push_str(&format!(", {} cache", human_bytes(buff_cache)));
    }
    label.push(')');
    label
}

pub fn swap_label(used: u64, total: u64) -> String {
    if total == 0 {
        return "no swap".into();
    }
    let pct = used as f64 / total as f64 * 100.0;
    format!("{pct:>5.1}%  {} / {}", human_bytes(used), human_bytes(total))
}

/// The swap gauge's label with the paging rate after it. Usage alone cannot
/// tell 800 MB of pages untouched since boot from a machine thrashing, and
/// the rate can: `in 0 B/s out 0 B/s` is the reassuring half of that.
pub fn swap_label_with_activity(
    used: u64,
    total: u64,
    vm: Option<&crate::metrics::VmActivity>,
) -> String {
    let mut label = swap_label(used, total);
    if let Some(vm) = vm {
        if total > 0 {
            label.push_str(&format!(
                "  in {} out {}",
                human_bps(vm.swap_in_bps),
                human_bps(vm.swap_out_bps)
            ));
        }
        if vm.major_faults_ps >= 1.0 {
            label.push_str(&format!("  {:.0} majflt/s", vm.major_faults_ps));
        }
    }
    label
}

fn gauge<'a>(app: &App, ratio: f64, label: String) -> Gauge<'a> {
    Gauge::default()
        .gauge_style(Style::default().fg(app.theme.usage(
            ratio,
            app.cfg.thresholds.warn,
            app.cfg.thresholds.crit,
        )))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::styled(label, Style::default().fg(app.theme.title)))
}

pub fn draw_memory(f: &mut Frame<'_>, area: Rect, app: &App) {
    let m = &app.snap.mem;
    let b = block(app, " Memory ");
    let inner = b.inner(area);
    f.render_widget(b, area);
    let label = memory_label(m.used, m.total, m.available, m.detail.buff_cache());
    f.render_widget(gauge(app, m.ratio(), label), inner);
}

pub fn draw_swap(f: &mut Frame<'_>, area: Rect, app: &App) {
    let m = &app.snap.mem;
    let b = block(app, " Swap ");
    let inner = b.inner(area);
    f.render_widget(b, area);
    let label = swap_label_with_activity(m.swap_used, m.swap_total, app.snap.vm.as_ref());
    f.render_widget(gauge(app, m.swap_ratio(), label), inner);
}

/// Shown only when the process is actually confined by a cgroup, where the
/// host's total memory would be misleading.
pub fn draw_cgroup(f: &mut Frame<'_>, area: Rect, app: &App) {
    let Some(cg) = &app.snap.cgroup else { return };
    let title = if cg.containerized { " cgroup (container) " } else { " cgroup " };
    let b = block(app, title);
    let inner = b.inner(area);
    f.render_widget(b, area);
    if inner.height == 0 {
        return;
    }

    match (cg.mem_ratio(), cg.mem_current, cg.mem_max) {
        (Some(ratio), Some(cur), Some(max)) => {
            let cpu = cg.cpu_quota_cores.map(|c| format!("  cpu {c:.2}")).unwrap_or_default();
            f.render_widget(
                gauge(app, ratio, format!("{} / {}{cpu}", human_bytes(cur), human_bytes(max))),
                inner,
            );
        }
        _ => {
            let cpu = cg
                .cpu_quota_cores
                .map(|c| format!("cpu quota {c:.2} cores"))
                .unwrap_or_else(|| "unlimited".into());
            f.render_widget(
                Paragraph::new(Span::styled(cpu, Style::default().fg(app.theme.dim))),
                inner,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_labels_report_gibibytes_not_tebibytes() {
        // Regression: the old code divided bytes by 2^20 and called it GiB.
        let label = memory_label(5_866_434_560, 16_035_467_264, 9_000_000_000, 0);
        assert!(label.contains("5.5 GiB / 14.9 GiB"), "{label}");
        assert!(label.starts_with(" 36.6%"), "{label}");
    }

    #[test]
    fn the_reclaimable_part_of_usage_is_spelled_out_when_known() {
        let label = memory_label(5_866_434_560, 16_035_467_264, 9_000_000_000, 6_442_450_944);
        assert!(label.contains("6.0 GiB cache"), "{label}");
        // ...and omitted on platforms that do not report it.
        assert!(!memory_label(1, 2, 1, 0).contains("cache"));
    }

    #[test]
    fn a_machine_with_no_swap_says_so() {
        assert_eq!(swap_label(0, 0), "no swap");
        assert_eq!(memory_label(0, 0, 0, 0), "Mem  (unknown)");
    }

    #[test]
    fn swap_activity_follows_usage_and_major_faults_only_when_there_are_some() {
        let vm = crate::metrics::VmActivity {
            swap_in_bps: 2.0 * 1024.0 * 1024.0,
            major_faults_ps: 0.4,
            ..Default::default()
        };
        let l = swap_label_with_activity(1 << 30, 4 << 30, Some(&vm));
        assert!(l.contains("in 2.0 MiB/s"), "{l}");
        assert!(l.contains("out 0 B/s"), "{l}");
        assert!(!l.contains("majflt"), "{l}");
        // No swap: no swap rates, but faults from file-backed pages still count.
        let busy = crate::metrics::VmActivity { major_faults_ps: 120.0, ..Default::default() };
        assert_eq!(swap_label_with_activity(0, 0, Some(&busy)), "no swap  120 majflt/s");
        assert_eq!(swap_label_with_activity(0, 0, None), "no swap");
    }

    #[test]
    fn swap_labels_show_usage_when_present() {
        let l = swap_label(1_073_741_824, 4_294_967_296);
        assert!(l.contains("1.0 GiB / 4.0 GiB"), "{l}");
    }
}
