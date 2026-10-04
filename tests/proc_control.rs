//! The process-control features actually take effect. These run against a real
//! sacrificial child process, because a unit test can only prove the arguments
//! were well-formed, not that the kernel accepted them.

#![cfg(unix)]

use std::process::{Child, Command};
use std::time::Duration;

use crabmon::metrics::sysinfo_source::{get_priority, set_affinity, set_priority};

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
    let child = Sacrifice::spawn();
    assert_eq!(get_priority(child.pid()), Some(0), "a fresh child starts at nice 0");

    // Raising the nice value is always permitted; lowering it needs privileges,
    // so the test only goes in the direction an unprivileged user can.
    set_priority(child.pid(), 7).expect("setpriority failed");
    assert_eq!(get_priority(child.pid()), Some(7));

    set_priority(child.pid(), 12).expect("setpriority failed");
    assert_eq!(get_priority(child.pid()), Some(12));
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
