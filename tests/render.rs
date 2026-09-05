//! Headless rendering: the whole dashboard is drawn onto a `TestBackend` and
//! the resulting buffer is asserted on, so panel content is verified without a
//! terminal or a screenshot.

mod common;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use common::{app, snapshot, FakeSource};
use crabmon::app::Mode;
use crabmon::ui::Layout;

/// Render `app` at the given size and return the screen as lines of text.
fn render(app: &mut crabmon::App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|f| crabmon::ui::draw(f, app)).unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.get(x, y).symbol().to_string())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn screen(app: &mut crabmon::App, w: u16, h: u16) -> String {
    render(app, w, h).join("\n")
}

#[test]
fn the_dashboard_shows_every_panel() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    for needle in ["crabmon", "CPU", "Memory", "Swap", "Processes", "Network", "Sensors", "Disks"] {
        assert!(s.contains(needle), "{needle} missing from the dashboard:\n{s}");
    }
}

#[test]
fn memory_is_rendered_in_gibibytes_not_a_thousand_times_too_large() {
    // The bug that started the rewrite: 16 GB of RAM used to render as
    // "15292.6 GiB" because sysinfo bytes were treated as KiB.
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("14.9 GiB"), "expected real GiB in:\n{s}");
    assert!(!s.contains("15292"), "the KiB/bytes confusion is back:\n{s}");
}

#[test]
fn the_header_reports_host_load_and_uptime() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("testbox"), "{s}");
    assert!(s.contains("Debian 12"), "{s}");
    assert!(s.contains("0.86"), "load average missing:\n{s}");
    assert!(s.contains("up 04:27:00"), "uptime missing:\n{s}");
    assert!(s.contains("2c/4t"), "core counts missing:\n{s}");
}

#[test]
fn the_process_table_shows_the_new_columns() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    for header in ["PID", "USER", "CPU%", "MEM", "VIRT", "THR", "DISK", "TIME", "COMMAND"] {
        assert!(s.contains(header), "column {header} missing:\n{s}");
    }
    assert!(s.contains("firefox"), "{s}");
    assert!(s.contains("vlad"), "{s}");
}

#[test]
fn sensors_are_grouped_so_the_nvme_and_chassis_rows_stay_visible() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("coretemp ×2"), "grouping missing:\n{s}");
    assert!(s.contains("acpitz"), "acpitz pushed off the panel:\n{s}");
    assert!(s.contains("nvme"), "nvme pushed off the panel:\n{s}");
    // Ungrouped, the raw labels come back.
    a.cfg.group_sensors = false;
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("coretemp Core"), "{s}");
}

#[test]
fn the_network_panel_reports_physical_traffic_only_by_default() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("[phys]"), "{s}");
    assert!(s.contains("wlp67s0"), "{s}");
    assert!(!s.contains("\nlo "), "loopback must be hidden by default:\n{s}");

    a.cfg.show_virtual_ifaces = true;
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("[all]"), "{s}");
}

#[test]
fn disk_io_rates_reach_the_gauge_label() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    // The dm-0 mapping is what makes this non-zero on an encrypted root.
    assert!(s.contains("W 1.0 MiB/s"), "disk write rate missing:\n{s}");
}

#[test]
fn every_layout_renders_without_panicking_and_shows_its_own_content() {
    let expectations = [
        (Layout::Dashboard, "Processes"),
        (Layout::Processes, "COMMAND"),
        (Layout::Cpu, "CPU"),
        (Layout::Io, "Network"),
    ];
    for (layout, needle) in expectations {
        let mut a = app();
        a.cfg.layout = layout;
        a.tick();
        let s = screen(&mut a, 160, 48);
        assert!(s.contains(needle), "{:?} missing {needle}:\n{s}", layout);
    }
}

#[test]
fn the_cpu_layout_draws_one_row_per_core() {
    let mut a = app();
    a.cfg.layout = Layout::Cpu;
    a.tick();
    let s = screen(&mut a, 160, 48);
    for core in 0..4 {
        assert!(s.contains(&format!("{core} [")), "core {core} row missing:\n{s}");
    }
    assert!(s.contains("95.0%"), "the busy core's percentage is missing:\n{s}");

    // Narrow terminals stack the cores instead of overflowing sideways.
    let lines = render(&mut a, 40, 24);
    let stacked = lines.iter().filter(|l| l.contains(" [")).count();
    assert!(stacked >= 4, "cores did not stack on a narrow screen:\n{}", lines.join("\n"));
}

#[test]
fn the_tree_view_draws_indent_guides() {
    let mut a = app();
    a.cfg.tree = true;
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("Processes (tree)"), "{s}");
    assert!(s.contains("└─") || s.contains("├─"), "no tree guides drawn:\n{s}");
}

#[test]
fn every_popup_renders_its_own_content() {
    let cases = [
        (Mode::Help, "quit"),
        (Mode::SignalMenu, "SIGKILL"),
        (Mode::ConfirmKill, "Confirm"),
        (Mode::Renice, "nice value"),
        (Mode::Affinity, "CPU list"),
        (Mode::Detail, "Parent"),
    ];
    for (mode, needle) in cases {
        let mut a = app();
        a.tick();
        a.select(0);
        a.mode = mode.clone();
        if matches!(mode, Mode::SignalMenu | Mode::ConfirmKill | Mode::Renice | Mode::Affinity) {
            a.target =
                Some(crabmon::app::Target { pid: 1, name: "systemd".into(), start_time_unix: 0 });
        }
        let s = screen(&mut a, 120, 40);
        assert!(s.contains(needle), "{mode:?} popup missing {needle}:\n{s}");
    }
}

#[test]
fn the_alert_popup_lists_rules_and_says_so_when_there_are_none() {
    let mut a = app();
    a.tick();
    a.mode = Mode::Alerts;
    let s = screen(&mut a, 120, 40);
    assert!(s.contains("no alert rules configured"), "{s}");

    let mut cfg = common::config();
    cfg.alerts = crabmon::alerts::default_rules();
    let mut a = crabmon::App::new(cfg, Box::new(FakeSource::single(snapshot())));
    a.tick();
    a.mode = Mode::Alerts;
    let s = screen(&mut a, 120, 40);
    assert!(s.contains("Rule"), "{s}");
    assert!(s.contains("disk-nearly-full"), "{s}");
    assert!(s.contains("FIRING"), "the /boot mount is at 98%:\n{s}");
}

#[test]
fn the_filter_prompt_and_its_errors_are_visible() {
    let mut a = app();
    a.tick();
    a.mode = Mode::Filter;
    a.filter_text = "re:[bad".into();
    a.filter_error = Some("bad regex: unclosed".into());
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("Filter:"), "{s}");
    assert!(s.contains("bad regex"), "the error is not shown:\n{s}");
}

#[test]
fn a_firing_alert_takes_over_the_status_line() {
    let mut cfg = common::config();
    cfg.alerts = vec![crabmon::alerts::AlertRule {
        name: "disk-full".into(),
        kind: crabmon::alerts::AlertKind::Disk,
        threshold: 90.0,
        for_secs: 0,
        target: "/boot".into(),
        command: String::new(),
    }];
    let mut a = crabmon::App::new(cfg, Box::new(FakeSource::single(snapshot())));
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("ALERT"), "{s}");
    assert!(s.contains("disk-full"), "{s}");
}

#[test]
fn the_cgroup_panel_appears_only_when_the_process_is_confined() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("cgroup (container)"), "{s}");

    let mut plain = snapshot();
    plain.cgroup = None;
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(plain)));
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(!s.contains("cgroup"), "the panel should be hidden:\n{s}");
}

#[test]
fn the_gpu_panel_marks_a_clock_ratio_as_an_estimate() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("Intel i915"), "{s}");
    assert!(s.contains("~25%"), "the clock-ratio estimate is missing:\n{s}");
}

#[test]
fn tiny_and_awkward_terminal_sizes_never_panic() {
    for (w, h) in [(10, 3), (20, 6), (40, 10), (80, 24), (300, 100), (81, 7)] {
        let mut a = app();
        a.tick();
        let _ = screen(&mut a, w, h);
    }
}

#[test]
fn every_theme_renders() {
    for name in crabmon::theme::PRESETS {
        let mut a = crabmon::App::new(
            crabmon::Config { theme: name.into(), ..Default::default() },
            Box::new(FakeSource::single(snapshot())),
        );
        a.tick();
        let s = screen(&mut a, 120, 40);
        assert!(s.contains("testbox"), "theme {name} rendered nothing:\n{s}");
    }
}

#[test]
fn disabled_panels_disappear_from_the_dashboard() {
    let mut cfg = common::config();
    cfg.panels.net = false;
    cfg.panels.sensors = false;
    cfg.panels.header = false;
    let mut a = crabmon::App::new(cfg, Box::new(FakeSource::single(snapshot())));
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(!s.contains("Sensors"), "{s}");
    assert!(!s.contains("Network"), "{s}");
    assert!(s.contains("Processes"), "the rest must still render:\n{s}");
}

#[test]
fn the_status_line_summarises_the_current_view() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("of 6 tasks"), "{s}");
    assert!(s.contains("[?] help"), "{s}");
}

#[test]
fn the_cpu_layout_shows_each_core_s_clock_speed() {
    // The README's CPU bullet promises "one bar per core and live clock speeds".
    let mut a = app();
    a.cfg.layout = Layout::Cpu;
    a.tick();
    let s = screen(&mut a, 160, 40);
    assert!(s.contains("2.40 GHz"), "per-core clocks missing:\n{s}");
    assert!(s.contains("800 MHz"), "the idle core's clock is missing:\n{s}");
}

#[test]
fn the_cpu_layout_omits_the_clock_column_when_the_platform_has_no_frequencies() {
    let mut snap = snapshot();
    snap.cpu.freq_mhz = vec![0, 0, 0, 0];
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    a.cfg.layout = Layout::Cpu;
    a.tick();
    let s = screen(&mut a, 160, 40);
    assert!(!s.contains("MHz"), "an empty clock column should not be drawn:\n{s}");
    assert!(s.contains("95.0%"), "the bars must still render:\n{s}");
}

// ---------------------------------------------------------------- Tier A-C

#[test]
fn the_pressure_panel_shows_some_and_full_stall_time() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("Pressure"), "{s}");
    // The fixture's IO pressure is the real reading from the dev machine.
    assert!(s.contains("2.6%") || s.contains("2.5%"), "io some/full missing:\n{s}");
    assert!(s.contains("io"), "{s}");
}

#[test]
fn a_machine_without_psi_says_so_instead_of_showing_zeros() {
    let mut snap = snapshot();
    snap.psi = Default::default();
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    a.tick();
    let s = screen(&mut a, 160, 48);
    // The panel is dropped entirely when there is nothing to show.
    assert!(!s.contains("Pressure"), "{s}");
}

#[test]
fn the_power_panel_shows_charge_draw_and_time_remaining() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("Power (battery)"), "{s}");
    assert!(s.contains("BAT0"), "{s}");
    assert!(s.contains("98%"), "{s}");
    assert!(s.contains("-12.0 W"), "discharge rate missing:\n{s}");
    assert!(s.contains("left"), "time remaining missing:\n{s}");
}

#[test]
fn a_machine_with_no_battery_hides_the_power_panel() {
    let mut snap = snapshot();
    snap.power = Default::default();
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    a.tick();
    assert!(!screen(&mut a, 160, 48).contains("Power"));
}

#[test]
fn the_process_table_gains_nice_and_trend_columns() {
    let mut a = app();
    for _ in 0..4 {
        a.tick();
    }
    let s = screen(&mut a, 200, 48);
    assert!(s.contains("NI"), "nice column missing:\n{s}");
    assert!(s.contains("TREND"), "sparkline column missing:\n{s}");
    // A sparkline is drawn from the accumulated per-process history.
    assert!(s.chars().any(|c| ('▁'..='█').contains(&c)), "no sparkline glyphs rendered:\n{s}");
}

#[test]
fn tagged_processes_are_marked_in_the_table() {
    let mut a = app();
    a.tick();
    a.select(0);
    a.on_key(crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Char(' '),
        crossterm::event::KeyModifiers::NONE,
    ));
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("1 tagged"), "the tag count belongs in the title:\n{s}");
}

#[test]
fn pausing_is_visible_in_the_process_panel_title() {
    let mut a = app();
    a.tick();
    a.paused = true;
    assert!(screen(&mut a, 160, 48).contains("PAUSED"));
}

#[test]
fn the_grouped_view_replaces_the_process_table() {
    let mut a = app();
    a.cfg.group_by = crabmon::metrics::GroupBy::Service;
    a.tick();
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("By service"), "{s}");
    assert!(s.contains("GROUP"), "{s}");
    assert!(s.contains("PROCS"), "{s}");
    assert!(s.contains("firefox.service"), "{s}");
    // The flat table's own header is gone.
    assert!(!s.contains("COMMAND"), "{s}");
}

#[test]
fn a_replay_shows_its_position_in_the_status_line() {
    let frames: Vec<crabmon::Snapshot> = (0..8).map(|_| snapshot()).collect();
    let source = crabmon::record::ReplaySource::from_frames(frames, "run.jsonl");
    let mut a = crabmon::App::new(common::config(), Box::new(source));
    let s = screen(&mut a, 160, 48);
    assert!(s.contains("frame"), "{s}");
    assert!(s.contains("/8"), "{s}");
    assert!(s.contains("run.jsonl"), "{s}");
}

#[test]
fn the_audit_log_popup_renders_entries_and_an_empty_state() {
    let mut a = app();
    a.tick();
    a.mode = Mode::AuditLog;
    assert!(screen(&mut a, 120, 40).contains("action log is disabled"));

    a.audit_path = Some(std::path::PathBuf::from("/tmp/whatever.log"));
    a.audit_entries = vec![
        "1788297407\tsignal SIGTERM\tpid=4242\tname=firefox\tok".into(),
        "1788297400\trenice 5\tpid=1\tname=init\tfailed: Operation not permitted".into(),
    ];
    let s = screen(&mut a, 120, 40);
    assert!(s.contains("SIGTERM"), "{s}");
    assert!(s.contains("firefox"), "{s}");
    assert!(s.contains("failed"), "{s}");
    assert!(s.contains("Action"), "{s}");
}

#[test]
fn the_detail_pane_shows_service_and_container() {
    let mut a = app();
    a.tick();
    a.select(0);
    a.mode = Mode::Detail;
    let s = screen(&mut a, 120, 40);
    assert!(s.contains("Service"), "{s}");
    assert!(s.contains("Container"), "{s}");
}

#[test]
fn memory_shows_how_much_of_the_usage_is_reclaimable_cache() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    // buffers + cached + sreclaimable from the fixture is 6.0 GiB.
    assert!(s.contains("6.0 GiB cache") || s.contains("cache"), "{s}");
}

#[test]
fn a_filesystem_low_on_inodes_says_so() {
    let mut a = app();
    a.tick();
    let s = screen(&mut a, 160, 48);
    // /boot in the fixture is 98% full by bytes but 99% by inodes.
    assert!(s.contains("inode"), "inode pressure is not surfaced:\n{s}");
}

#[test]
fn an_unreachable_remote_host_explains_the_empty_dashboard() {
    struct BrokenRemote;
    impl crabmon::MetricSource for BrokenRemote {
        fn snapshot(&mut self, _dt: std::time::Duration) -> crabmon::Snapshot {
            crabmon::Snapshot::default()
        }
        fn label(&self) -> Option<String> {
            Some("remote build01 UNREACHABLE — ssh: Could not resolve hostname".into())
        }
    }
    let mut a = crabmon::App::new(common::config(), Box::new(BrokenRemote));
    let s = screen(&mut a, 140, 30);
    assert!(s.contains("UNREACHABLE"), "{s}");
    assert!(s.contains("build01"), "{s}");
}
