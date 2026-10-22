//! The process-control features actually take effect. These run against a real
//! sacrificial child process, because a unit test can only prove the arguments
//! were well-formed, not that the kernel accepted them.

#![cfg(unix)]

use std::process::{Child, Command};
use std::time::Duration;

use crabmon::metrics::sysinfo_source::{get_priority, set_priority};
// Affinity is Linux-only, and so is the one test that sets it.
#[cfg(target_os = "linux")]
use crabmon::metrics::sysinfo_source::set_affinity;

struct Sacrifice(Child);

impl Sacrifice {
    fn spawn() -> Sacrifice {
        let child = Command::new("sleep")
            .arg("30")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("failed to spawn a test process");
        // Give the kernel a moment to actually schedule it.
        std::thread::sleep(Duration::from_millis(50));
        Sacrifice(child)
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }
}

impl Drop for Sacrifice {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn renice_changes_the_process_priority_for_real() {
    // A child inherits its *parent's* nice value, which is 0 only if whoever
    // started the test harness was at 0. GitHub's macOS runners are not: the
    // test process sits at -10, so asserting a bare 0 here failed on a
    // perfectly healthy machine.
    let base = get_priority(std::process::id()).expect("our own nice value");
    let child = Sacrifice::spawn();
    assert_eq!(
        get_priority(child.pid()),
        Some(base),
        "a fresh child inherits its parent's nice value"
    );

    // Raising the nice value is always permitted; lowering it needs privileges,
    // so the test only goes in the direction an unprivileged user can. The
    // kernel clamps at 19, so the expectation is clamped the same way rather
    // than assuming there is room above `base`.
    for step in [7, 12] {
        let want = (base + step).min(19);
        set_priority(child.pid(), want).expect("setpriority failed");
        assert_eq!(get_priority(child.pid()), Some(want), "nice {want} did not take");
    }
}

#[test]
fn getting_the_priority_of_a_dead_process_reports_none_rather_than_zero() {
    // The -1/errno dance in `get_priority` is easy to get wrong in exactly this
    // case, and returning Some(-1) would render as a real nice value.
    let pid = {
        let child = Sacrifice::spawn();
        child.pid()
    };
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(get_priority(pid), None);
}

#[cfg(target_os = "linux")]
#[test]
fn affinity_pins_the_process_to_the_requested_cpus() {
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if ncpu < 2 {
        return; // nothing meaningful to assert on a single-CPU machine
    }
    let child = Sacrifice::spawn();
    set_affinity(child.pid(), &[0]).expect("sched_setaffinity failed");

    let set = nix::sched::sched_getaffinity(nix::unistd::Pid::from_raw(child.pid() as i32))
        .expect("sched_getaffinity failed");
    assert!(set.is_set(0).unwrap());
    for cpu in 1..ncpu {
        assert!(!set.is_set(cpu).unwrap(), "cpu {cpu} should have been excluded");
    }

    // And it can be widened again.
    let all: Vec<usize> = (0..ncpu).collect();
    set_affinity(child.pid(), &all).expect("widening the affinity failed");
    let set = nix::sched::sched_getaffinity(nix::unistd::Pid::from_raw(child.pid() as i32))
        .expect("sched_getaffinity failed");
    assert!(set.is_set(ncpu - 1).unwrap());
}

#[test]
fn signalling_a_process_actually_delivers() {
    use crabmon::app::signal_by_name;
    use crabmon::metrics::sysinfo_source::send_signal;

    let mut child = Sacrifice::spawn();
    send_signal(child.pid(), signal_by_name("SIGTERM").unwrap()).expect("kill failed");

    // The child must be gone shortly after.
    for _ in 0..50 {
        if matches!(child.0.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("the process survived SIGTERM");
}

#[test]
fn signalling_a_nonexistent_pid_is_an_error_not_a_silent_success() {
    use crabmon::app::signal_by_name;
    use crabmon::metrics::sysinfo_source::send_signal;

    // The old code discarded the result entirely, so a failed kill looked the
    // same as a successful one.
    let err = send_signal(u32::MAX / 2, signal_by_name("SIGTERM").unwrap());
    assert!(err.is_err(), "expected ESRCH");
}

/// A process that `exec`s keeps its PID, its parent and its start time, and
/// the sampler underneath crabmon reads a name only when it first meets a PID
/// — so a wrapper that hands off to the real program was reported under the
/// wrapper's name for the rest of its life. Every container entrypoint, every
/// unit with a shell in front of it, `sudo`, `nohup`: all of them showed as
/// `sh`, and `--watch nginx` could never match an nginx that had been exec'd
/// into.
///
/// This is a two-sample test on a real child because that is the only way to
/// see it: one snapshot is always right, and a fixture cannot exec.
#[cfg(target_os = "linux")]
#[test]
fn a_process_that_execs_is_reported_as_what_it_became() {
    use std::time::Instant;

    use crabmon::metrics::{MetricSource, SysinfoSource};

    // A copy of `sleep` under a name nothing else on the machine has, so the
    // assertion cannot be satisfied by some unrelated process.
    let dir = std::env::temp_dir().join(format!("crabmon-exec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let victim = dir.join("crabmon-execd");
    let Some(sleep) =
        ["/bin/sleep", "/usr/bin/sleep"].into_iter().find(|p| std::path::Path::new(p).exists())
    else {
        return; // no sleep to copy; nothing to assert
    };
    std::fs::copy(sleep, &victim).expect("copy sleep");

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(format!("sleep 1; exec {} 30", victim.display()))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn the wrapper");
    let pid = child.id();

    let name_of = |snap: &crabmon::Snapshot| {
        snap.procs.iter().find(|p| p.pid == pid).map(|p| (p.name.clone(), p.cmd.clone()))
    };

    let mut source = SysinfoSource::new(Vec::new(), false);
    // The first sample must see the wrapper, or the test proves nothing: it
    // would pass just as well on a sampler that never caches anything.
    let first = source.snapshot(Duration::ZERO);
    let (before, _) = name_of(&first).expect("the wrapper should be in the first snapshot");
    assert_eq!(before, "sh", "the wrapper should still be a shell at this point");

    // Poll the *same* source — a fresh one would read the process for the
    // first time and get the right answer however stale the cache is.
    let deadline = Instant::now() + Duration::from_secs(15);
    let found = loop {
        std::thread::sleep(Duration::from_millis(200));
        let snap = source.snapshot(Duration::from_millis(200));
        match name_of(&snap) {
            Some((name, cmd)) if name != "sh" => break Some((name, cmd)),
            _ if Instant::now() > deadline => break None,
            _ => {}
        }
    };

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);

    let (name, cmd) = found.expect("the exec'd process was still being reported as its wrapper");
    assert_eq!(name, "crabmon-execd", "the new program's name");
    assert!(cmd.contains("crabmon-execd"), "the command line still describes the wrapper: {cmd}");
}

#[cfg(target_os = "linux")]
fn app_for_child(child: &Sacrifice) -> crabmon::App {
    use crabmon::metrics::{MetricSource, SysinfoSource};
    struct Frozen(crabmon::Snapshot);
    impl MetricSource for Frozen {
        fn snapshot(&mut self, _: Duration) -> crabmon::Snapshot {
            self.0.clone()
        }
    }
    let mut source = SysinfoSource::new(Vec::new(), false);
    let mut snap = source.snapshot(Duration::ZERO);
    snap.procs.retain(|p| p.pid == child.pid());
    assert_eq!(snap.procs.len(), 1);
    assert!(snap.procs[0].start_time_ticks.is_some());
    crabmon::App::new(Default::default(), Box::new(Frozen(snap)))
}

#[cfg(target_os = "linux")]
fn confirm_control(app: &mut crabmon::App, key: char) {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
    app.on_key(press(KeyCode::Char(key)));
    if key == 't' {
        app.on_key(press(KeyCode::Enter));
        app.on_key(press(KeyCode::Char('y')));
    } else {
        app.input = match key {
            'r' => "19",
            'A' => "0",
            'I' => "idle",
            _ => unreachable!(),
        }
        .into();
        app.on_key(press(KeyCode::Enter));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn paused_controls_refuse_a_live_pid_with_the_wrong_incarnation() {
    for key in ['t', 'r', 'A', 'I'] {
        let mut child = Sacrifice::spawn();
        let base = get_priority(child.pid());
        let mut app = app_for_child(&child);
        app.paused = true;
        // Same PID and Unix second, a different precise identity. No tick
        // updates the snapshot before the confirmation reaches the kernel.
        app.snap.procs[0].start_time_ticks = app.snap.procs[0].start_time_ticks.map(|t| t + 1);
        app.rebuild_view();
        confirm_control(&mut app, key);
        assert!(
            app.status_text().unwrap_or_default().contains("gone"),
            "{key}: {:?}",
            app.status_text()
        );
        assert!(child.0.try_wait().unwrap().is_none(), "wrong incarnation was signalled");
        assert_eq!(get_priority(child.pid()), base);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn controls_refuse_an_exited_process_even_with_a_frozen_snapshot() {
    for key in ['t', 'r', 'A', 'I'] {
        let mut child = Sacrifice::spawn();
        let mut app = app_for_child(&child);
        child.0.kill().unwrap();
        child.0.wait().unwrap();
        confirm_control(&mut app, key);
        assert!(
            app.status_text().unwrap_or_default().contains("gone"),
            "{key}: {:?}",
            app.status_text()
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn app_controls_deliver_to_a_verified_live_child() {
    let mut child = Sacrifice::spawn();
    let mut app = app_for_child(&child);
    confirm_control(&mut app, 'r');
    assert_eq!(get_priority(child.pid()), Some(19));
    confirm_control(&mut app, 't');
    for _ in 0..50 {
        if child.0.try_wait().unwrap().is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("pidfd signal did not reach the verified child");
}
