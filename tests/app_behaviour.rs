//! End-to-end behaviour of the interaction model, driven through `App` with a
//! fixture metric source — no terminal, no host access.

mod common;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use common::{app, proc_row, snapshot, FakeSource};
use crabmon::app::{Action, Mode};
use crabmon::sort::SortBy;

fn key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

fn code(c: KeyCode) -> KeyEvent {
    KeyEvent::new(c, KeyModifiers::NONE)
}

#[test]
fn q_quits_and_ctrl_c_quits_from_every_mode() {
    let mut a = app();
    assert_eq!(a.on_key(key('q')), Action::Quit);

    // The old build swallowed Ctrl-C entirely: raw mode meant the terminal
    // would not deliver SIGINT and nothing handled the key.
    let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
    for mode in [Mode::Normal, Mode::Filter, Mode::SignalMenu, Mode::Help] {
        let mut a = app();
        a.mode = mode.clone();
        assert_eq!(a.on_key(ctrl_c), Action::Quit, "ctrl-c ignored in {mode:?}");
    }
}

#[test]
fn ctrl_q_does_not_quit() {
    let mut a = app();
    let ctrl_q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL);
    assert_eq!(a.on_key(ctrl_q), Action::None);
}

#[test]
fn escape_clears_the_filter_instead_of_quitting() {
    let mut a = app();
    a.on_key(key('/'));
    for c in "firefox".chars() {
        a.on_key(key(c));
    }
    a.on_key(code(KeyCode::Enter));
    assert_eq!(a.rows().len(), 1);

    assert_eq!(a.on_key(code(KeyCode::Esc)), Action::None, "esc must not quit");
    assert_eq!(a.rows().len(), 5, "esc clears the filter");
}

#[test]
fn selection_follows_the_process_not_the_row_number() {
    let mut a = app();
    a.set_page(10);
    // Sort by name ascending, then select `firefox`.
    a.on_key(key('n'));
    let pos = a.rows().iter().position(|r| r.name == "firefox").unwrap();
    a.select(pos);
    assert_eq!(a.selected_row().unwrap().name, "firefox");

    // Re-sorting by CPU used to leave the highlight on whatever row index the
    // cursor happened to be on.
    a.on_key(key('c'));
    assert_eq!(a.selected_row().unwrap().name, "firefox");
}

#[test]
fn sorting_keys_set_the_column_and_toggle_direction_on_repeat() {
    let mut a = app();
    a.on_key(key('m'));
    assert_eq!(a.cfg.sort_by, SortBy::Mem);
    let first = a.cfg.sort_desc;
    a.on_key(key('m'));
    assert_eq!(a.cfg.sort_desc, !first, "pressing the same key reverses");

    a.on_key(key('>'));
    assert_ne!(a.cfg.sort_by, SortBy::Mem);
    a.on_key(key('<'));
    assert_eq!(a.cfg.sort_by, SortBy::Mem);
}

#[test]
fn descending_cpu_sort_puts_the_busiest_process_first() {
    let mut a = app();
    a.on_key(key('n')); // move off CPU first, since a repeat press reverses
    a.on_key(key('c'));
    assert!(a.cfg.sort_desc);
    assert_eq!(a.rows()[0].name, "firefox");
}

#[test]
fn navigation_keys_move_and_clamp_the_selection() {
    let mut a = app();
    a.set_page(3);
    let n = a.rows().len();

    a.on_key(code(KeyCode::End));
    assert_eq!(a.selected, n - 1);
    a.on_key(code(KeyCode::Down));
    assert_eq!(a.selected, n - 1, "must not walk off the end");

    a.on_key(code(KeyCode::Home));
    assert_eq!(a.selected, 0);
    a.on_key(code(KeyCode::Up));
    assert_eq!(a.selected, 0, "must not walk off the start");

    a.on_key(code(KeyCode::PageDown));
    assert_eq!(a.selected, 3);
    a.on_key(code(KeyCode::PageUp));
    assert_eq!(a.selected, 0);
}

#[test]
fn scrolling_up_moves_the_cursor_not_just_the_window() {
    // The old viewport pinned the highlight to the bottom row, so moving up
    // scrolled the list while the cursor stayed put.
    let procs = (1..=50u32).map(|i| proc_row(i, Some(1), "p", 0.0, 0)).collect();
    let mut a = crabmon::App::new(
        common::config(),
        Box::new(FakeSource::single(crabmon::Snapshot { procs, ..snapshot() })),
    );
    a.set_page(10);
    a.on_key(code(KeyCode::End));
    let bottom_scroll = a.scroll;

    a.on_key(code(KeyCode::Up));
    assert_eq!(a.scroll, bottom_scroll, "the window must not move yet");
    assert_eq!(a.selected, a.rows().len() - 2, "the cursor moved instead");
}

#[test]
fn the_filter_language_is_live_and_reports_errors() {
    let mut a = app();
    a.on_key(key('/'));
    for c in "user:vlad cpu>1".chars() {
        a.on_key(key(c));
    }
    assert_eq!(a.mode, Mode::Filter);
    assert!(a.rows().iter().all(|r| r.cpu > 1.0));
    assert!(a.filter_error.is_none());

    let mut a = app();
    a.on_key(key('/'));
    for c in "re:[bad".chars() {
        a.on_key(key(c));
    }
    assert!(a.filter_error.is_some(), "an unclosed class must be reported");
}

#[test]
fn tree_view_orders_children_under_their_parents() {
    let mut a = app();
    a.on_key(key('T'));
    assert!(a.cfg.tree);
    let names: Vec<&str> = a.rows().iter().map(|r| r.name.as_str()).collect();
    let sshd = names.iter().position(|n| *n == "sshd").unwrap();
    let bash = names.iter().position(|n| *n == "bash").unwrap();
    let firefox = names.iter().position(|n| *n == "firefox").unwrap();
    assert!(sshd < bash && bash < firefox, "{names:?}");
    assert_eq!(a.tree_rows.len(), a.rows().len());
}

#[test]
fn the_signal_flow_needs_two_confirmations_and_survives_cancelling() {
    let mut a = app();
    a.select(0);
    a.on_key(key('t'));
    assert_eq!(a.mode, Mode::SignalMenu);
    assert!(a.target.is_some());

    a.on_key(code(KeyCode::Esc));
    assert_eq!(a.mode, Mode::Normal);
    assert!(a.target.is_none(), "cancelling must forget the target");

    a.on_key(key('t'));
    a.on_key(code(KeyCode::Enter));
    assert_eq!(a.mode, Mode::ConfirmKill);
    a.on_key(key('n'));
    assert_eq!(a.mode, Mode::Normal);
    assert!(a.target.is_none());
}

#[test]
fn a_recycled_pid_is_not_signalled() {
    // Frame 2 has the same PID with a different start time: the kernel reused
    // it for a different process while the confirmation prompt was open.
    let first = snapshot();
    let mut second = snapshot();
    second.procs[0].start_time_unix += 999;

    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::new(vec![first, second])));
    a.cfg.sort_by = SortBy::Pid;
    a.cfg.sort_desc = false;
    a.rebuild_view();
    a.select(0);
    a.on_key(key('t'));
    a.on_key(code(KeyCode::Enter));

    a.tick(); // the process is replaced underneath us
    a.on_key(key('y'));
    assert_eq!(a.mode, Mode::Normal);
    assert!(
        a.status_text().unwrap_or_default().contains("gone"),
        "expected a refusal, got {:?}",
        a.status_text()
    );
}

#[test]
fn refresh_nudging_respects_the_floor_sysinfo_needs() {
    let mut a = app();
    for _ in 0..50 {
        a.on_key(key('+'));
    }
    assert_eq!(a.refresh.as_millis() as u64, crabmon::MIN_REFRESH_MS);
    for _ in 0..200 {
        a.on_key(key('-'));
    }
    assert_eq!(a.refresh.as_millis() as u64, crabmon::MAX_REFRESH_MS);
}

#[test]
fn layout_and_display_toggles_round_trip() {
    let mut a = app();
    let start = a.cfg.layout;
    a.on_key(code(KeyCode::Tab));
    assert_ne!(a.cfg.layout, start);
    a.on_key(code(KeyCode::BackTab));
    assert_eq!(a.cfg.layout, start);

    let virt = a.cfg.show_virtual_ifaces;
    a.on_key(key('v'));
    assert_eq!(a.cfg.show_virtual_ifaces, !virt);

    let grouped = a.cfg.group_sensors;
    a.on_key(key('g'));
    assert_eq!(a.cfg.group_sensors, !grouped);
}

#[test]
fn popups_open_and_close_without_leaving_a_stuck_mode() {
    for (open, mode) in
        [(key('?'), Mode::Help), (key('!'), Mode::Alerts), (code(KeyCode::Enter), Mode::Detail)]
    {
        let mut a = app();
        a.select(0);
        a.on_key(open);
        assert_eq!(a.mode, mode);
        a.on_key(code(KeyCode::Esc));
        assert_eq!(a.mode, Mode::Normal);
    }
}

#[test]
fn renice_input_rejects_out_of_range_values_before_calling_the_kernel() {
    assert!(crabmon::app::parse_nice("0").is_ok());
    assert_eq!(crabmon::app::parse_nice("-20").unwrap(), -20);
    assert_eq!(crabmon::app::parse_nice("19").unwrap(), 19);
    assert!(crabmon::app::parse_nice("20").is_err());
    assert!(crabmon::app::parse_nice("-21").is_err());
    assert!(crabmon::app::parse_nice("x").is_err());

    let mut a = app();
    a.select(0);
    a.on_key(key('r'));
    assert_eq!(a.mode, Mode::Renice);
    a.input.clear();
    for c in "99".chars() {
        a.on_key(key(c));
    }
    a.on_key(code(KeyCode::Enter));
    assert_eq!(a.mode, Mode::Renice, "an invalid value keeps the prompt open");
    assert!(a.input_error.is_some());
}

#[test]
fn affinity_input_rejects_cpus_that_do_not_exist() {
    let mut a = app();
    a.select(0);
    a.on_key(key('A'));
    assert_eq!(a.mode, Mode::Affinity);
    a.input.clear();
    for c in "0-99".chars() {
        a.on_key(key(c));
    }
    a.on_key(code(KeyCode::Enter));
    assert_eq!(a.mode, Mode::Affinity);
    assert!(a.input_error.unwrap().contains("does not exist"));
}

#[test]
fn export_is_requested_rather_than_performed_by_the_key_handler() {
    let mut a = app();
    assert_eq!(a.on_key(key('P')), Action::Export);
}

#[test]
fn the_mouse_wheel_scrolls_and_clicks_select() {
    let mut a = app();
    a.set_page(10);
    a.proc_area = ratatui::layout::Rect { x: 0, y: 0, width: 80, height: 10 };

    let wheel = |kind| MouseEvent { kind, column: 5, row: 5, modifiers: KeyModifiers::NONE };
    a.on_mouse(wheel(MouseEventKind::ScrollDown));
    assert_eq!(a.selected, 3);
    a.on_mouse(wheel(MouseEventKind::ScrollUp));
    assert_eq!(a.selected, 0);

    // Row 0 is the border, row 1 the header, so row 3 is the second entry.
    a.on_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 5,
        row: 3,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(a.selected, 1);
}

#[test]
fn clicking_a_header_cell_sorts_by_that_column() {
    let mut a = app();
    a.proc_area = ratatui::layout::Rect { x: 0, y: 0, width: 100, height: 20 };
    a.header_cols = vec![(0, 7, SortBy::Pid), (8, 14, SortBy::Cpu)];
    a.on_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 10,
        row: 1,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(a.cfg.sort_by, SortBy::Cpu);
}

#[test]
fn history_grows_on_each_tick_and_stays_bounded() {
    let mut a = crabmon::App::new(
        crabmon::Config { history_len: 16, theme: "mono".into(), ..Default::default() },
        Box::new(FakeSource::single(snapshot())),
    );
    assert_eq!(a.cpu_hist.len(), 0, "no fake flat line before the first sample");
    for _ in 0..40 {
        a.tick();
    }
    assert_eq!(a.cpu_hist.len(), 16);
    assert_eq!(a.per_core_hist.len(), 4, "one series per core");
    assert!(a.net_rx_hist.last() > 0.0);
}

#[test]
fn aggregate_network_history_excludes_loopback_by_default() {
    let mut a = app();
    a.tick();
    // 125_000 from wlp67s0; `lo` contributes 900_000 and must be excluded.
    assert_eq!(a.net_rx_hist.last(), 125_000.0);
    a.on_key(key('v'));
    a.tick();
    assert_eq!(a.net_rx_hist.last(), 1_025_000.0);
}

#[test]
fn alerts_fire_once_their_hold_time_elapses() {
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
    assert_eq!(a.active_alerts.len(), 1);
    assert_eq!(a.active_alerts[0].name, "disk-full");
}

#[test]
fn an_empty_process_list_does_not_panic_any_handler() {
    let empty = crabmon::Snapshot { procs: vec![], ..snapshot() };
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(empty)));
    assert!(a.selected_row().is_none());
    for k in ['j', 'k', 't', 'r', 'A', 'c', 'T', '/'] {
        a.on_key(key(k));
    }
    a.on_key(code(KeyCode::Enter));
    a.on_key(code(KeyCode::End));
    assert_eq!(a.selected, 0);
}

#[test]
fn an_app_with_no_config_path_never_touches_the_filesystem() {
    // Regression: `App` used to save straight to the user's real config file,
    // so merely pressing `-` in a test rewrote ~/.config/crabmon/config.toml.
    let mut a = app();
    assert!(a.config_path.is_none());
    for _ in 0..10 {
        a.on_key(key('-'));
    }
    a.persist();
    assert!(a.refresh.as_millis() > 800, "the in-memory value still changes");
}

#[test]
fn persisting_writes_the_current_settings_to_the_given_path() {
    let path = std::env::temp_dir().join(format!("crabmon-persist-{}.toml", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut a = app();
    a.config_path = Some(path.clone());
    a.on_key(key('T'));
    a.on_key(key('m'));
    a.persist();

    let back = crabmon::config::load_from(&path);
    assert!(back.tree);
    assert_eq!(back.sort_by, SortBy::Mem);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn threads_are_hidden_by_default_and_toggle_back_with_h() {
    // On Linux the sampler reports every task, so a stock process list is
    // several times longer than the number of actual processes.
    let mut a = app();
    assert!(!a.cfg.show_threads);
    assert!(
        a.rows().iter().all(|r| r.name != "StreamTrans"),
        "threads should be hidden by default"
    );
    let processes = a.rows().len();

    a.on_key(key('H'));
    assert!(a.cfg.show_threads);
    assert_eq!(a.rows().len(), processes + 1);
    assert!(a.rows().iter().any(|r| r.name == "StreamTrans"));

    a.on_key(key('H'));
    assert_eq!(a.rows().len(), processes);
}

#[test]
fn hiding_threads_does_nothing_on_platforms_that_cannot_tell_them_apart() {
    // macOS and Windows never report a task count. Filtering on it there would
    // hide every single row.
    let mut snap = snapshot();
    for p in &mut snap.procs {
        p.threads = None;
    }
    let total = snap.procs.len();
    let a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    assert!(!a.cfg.show_threads);
    assert_eq!(a.rows().len(), total, "no rows may be hidden");
}

#[test]
fn a_hidden_thread_cannot_be_selected_or_signalled() {
    let mut a = app();
    a.on_key(code(KeyCode::End));
    assert!(a.selected_row().is_some());
    assert_ne!(a.selected_row().unwrap().name, "StreamTrans");
}

/// The PID-reuse guard must cover every action that targets a process, not just
/// the signal path — the README promises it for all three.
#[test]
fn a_recycled_pid_is_not_reniced_or_repinned() {
    for open_key in ['r', 'A'] {
        let first = snapshot();
        let mut second = snapshot();
        second.procs[0].start_time_unix += 999;

        let mut a =
            crabmon::App::new(common::config(), Box::new(FakeSource::new(vec![first, second])));
        a.cfg.sort_by = SortBy::Pid;
        a.cfg.sort_desc = false;
        a.rebuild_view();
        a.select(0);
        a.on_key(key(open_key));

        // The process is replaced while the prompt is open.
        a.tick();
        a.input = if open_key == 'r' { "5".into() } else { "0".into() };
        a.on_key(code(KeyCode::Enter));

        assert_eq!(a.mode, Mode::Normal, "{open_key}: the prompt should close");
        assert!(
            a.status_text().unwrap_or_default().contains("gone"),
            "{open_key}: expected a refusal, got {:?}",
            a.status_text()
        );
    }
}

// ---------------------------------------------------------------- Tier A-C

#[test]
fn pausing_stops_sampling_and_resuming_starts_it_again() {
    let mut a = crabmon::App::new(
        common::config(),
        Box::new(FakeSource::new(vec![snapshot(), snapshot(), snapshot()])),
    );
    a.on_key(key('z'));
    assert!(a.paused);
    assert!(!a.due(), "a paused app never becomes due");
    assert_eq!(a.status_text(), Some("paused"));

    a.on_key(key('z'));
    assert!(!a.paused);
}

#[test]
fn tagging_builds_a_bulk_target_set_and_clears_after_acting() {
    let mut a = app();
    a.set_page(10);
    a.select(0);
    let first = a.selected_row().unwrap().pid;
    a.on_key(key(' '));
    assert!(a.tagged.contains(&first));
    assert_eq!(a.selected, 1, "tagging advances, so a run can be tagged quickly");

    let second = a.selected_row().unwrap().pid;
    a.on_key(key(' '));
    assert_eq!(a.tagged.len(), 2);

    // A signal targets everything tagged, not just the cursor.
    a.on_key(key('t'));
    assert_eq!(a.action_targets().len(), 2);
    let pids: Vec<u32> = a.action_targets().iter().map(|t| t.pid).collect();
    assert!(pids.contains(&first) && pids.contains(&second));

    a.on_key(code(KeyCode::Esc));
    a.on_key(key('U'));
    assert!(a.tagged.is_empty());
}

#[test]
fn tagging_the_same_process_twice_untags_it() {
    let mut a = app();
    a.select(0);
    let pid = a.selected_row().unwrap().pid;
    a.on_key(key(' '));
    a.select(0);
    a.on_key(key(' '));
    assert!(!a.tagged.contains(&pid));
}

#[test]
fn saved_filters_are_bound_to_the_number_keys() {
    let mut cfg = common::config();
    cfg.saved_filters = vec![
        crabmon::config::SavedFilter { name: "busy".into(), query: "cpu>50".into() },
        crabmon::config::SavedFilter { name: "none".into(), query: "cpu>9999".into() },
    ];
    let mut a = crabmon::App::new(cfg, Box::new(FakeSource::single(snapshot())));

    a.on_key(key('1'));
    assert_eq!(a.filter_text, "cpu>50");
    assert_eq!(a.rows().len(), 1, "only firefox is over 50%");
    assert!(a.status_text().unwrap().contains("busy"));

    a.on_key(key('2'));
    assert!(a.rows().is_empty());

    a.on_key(key('0'));
    assert!(a.filter_text.is_empty());
    assert!(a.rows().len() > 1);

    // A key with no preset behind it says so rather than doing nothing.
    a.on_key(key('9'));
    assert!(a.status_text().unwrap().contains("no saved filter"));
}

#[test]
fn grouping_aggregates_processes_by_service() {
    let mut a = app();
    assert!(a.groups().is_empty(), "no grouping by default");

    a.on_key(key('G'));
    assert_eq!(a.cfg.group_by, crabmon::metrics::GroupBy::Service);
    let groups = a.groups();
    assert!(!groups.is_empty());
    // The fixture gives every process its own service.
    let total_procs: usize = groups.iter().map(|g| g.procs).sum();
    assert_eq!(total_procs, a.rows().len(), "every process lands in exactly one group");
    let total_cpu: f32 = groups.iter().map(|g| g.cpu).sum();
    let row_cpu: f32 = a.rows().iter().map(|r| r.cpu).sum();
    assert!((total_cpu - row_cpu).abs() < 0.01, "CPU must add up");
}

#[test]
fn grouping_puts_processes_without_a_group_under_a_placeholder() {
    let mut snap = snapshot();
    for p in &mut snap.procs {
        p.service = None;
    }
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    a.cfg.group_by = crabmon::metrics::GroupBy::Service;
    let groups = a.groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "-");
}

#[test]
fn scrubbing_a_recording_steps_frames_and_pauses() {
    let frames: Vec<crabmon::Snapshot> = (0..5)
        .map(|i| {
            let mut s = snapshot();
            s.taken_at_unix = 1000 + i;
            s
        })
        .collect();
    let source = crabmon::record::ReplaySource::from_frames(frames, "run.jsonl");
    let mut a = crabmon::App::new(common::config(), Box::new(source));

    assert_eq!(a.timeline(), Some((1, 5)), "constructing consumed the first frame");
    a.on_key(key(']'));
    assert!(a.paused, "scrubbing implies stepping, not playing");
    assert!(a.status_text().unwrap().contains("frame"));

    let (before, _) = a.timeline().unwrap();
    a.on_key(key('['));
    assert!(a.timeline().unwrap().0 < before);

    // Stepping past either end clamps instead of wrapping.
    for _ in 0..20 {
        a.on_key(key('}'));
    }
    assert_eq!(a.timeline().unwrap().0, 4);
    for _ in 0..20 {
        a.on_key(key('{'));
    }
    assert_eq!(a.timeline().unwrap().0, 0);
}

#[test]
fn scrubbing_a_live_source_says_it_is_not_a_recording() {
    let mut a = app();
    assert_eq!(a.timeline(), None);
    a.on_key(key(']'));
    assert_eq!(a.status_text(), Some("not a recording"));
    assert!(!a.paused);
}

#[test]
fn actions_are_written_to_the_audit_log() {
    let path = std::env::temp_dir().join(format!("crabmon-audit-app-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut a = app();
    a.audit_path = Some(path.clone());
    a.select(0);
    a.on_key(key('r'));
    a.input = "5".into();
    a.on_key(code(KeyCode::Enter));

    let log = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(log.contains("renice 5"), "{log}");
    assert!(log.contains("pid="), "{log}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_audit_log_viewer_opens_and_closes() {
    let mut a = app();
    a.on_key(key('L'));
    assert_eq!(a.mode, Mode::AuditLog);
    a.on_key(code(KeyCode::Esc));
    assert_eq!(a.mode, Mode::Normal);
}

#[test]
fn opening_the_detail_pane_fetches_that_process_s_sockets() {
    let mut a = app();
    a.select(0);
    a.on_key(code(KeyCode::Enter));
    assert_eq!(a.mode, Mode::Detail);
    // The fixture PIDs do not exist, so the list is empty — but the lookup ran
    // and did not panic, which is what the wiring test is for.
    assert!(a.detail_sockets.is_empty());
}

#[test]
fn per_process_cpu_history_accumulates_and_evicts_dead_processes() {
    let mut a = app();
    for _ in 0..3 {
        a.tick();
    }
    let pid = a.rows()[0].pid;
    assert_eq!(a.cpu_spark(pid).len(), 3);

    // A process that disappears must not leak its history.
    let mut gone = snapshot();
    gone.procs.clear();
    let mut a2 =
        crabmon::App::new(common::config(), Box::new(FakeSource::new(vec![snapshot(), gone])));
    a2.tick();
    a2.tick();
    assert!(a2.cpu_spark(1).is_empty());
}

#[test]
fn per_process_history_is_bounded() {
    let mut a = app();
    for _ in 0..(crabmon::app::PROC_SPARK_LEN * 3) {
        a.tick();
    }
    let pid = a.rows()[0].pid;
    assert_eq!(a.cpu_spark(pid).len(), crabmon::app::PROC_SPARK_LEN);
}

#[test]
fn nice_is_sortable_like_any_other_column() {
    let mut snap = snapshot();
    snap.procs[0].nice = Some(19);
    snap.procs[1].nice = Some(-5);
    let mut a = crabmon::App::new(common::config(), Box::new(FakeSource::single(snap)));
    a.cfg.sort_by = SortBy::Nice;
    a.cfg.sort_desc = false;
    a.rebuild_view();
    assert_eq!(a.rows()[0].nice, Some(-5));
}
