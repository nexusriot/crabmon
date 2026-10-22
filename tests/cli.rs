//! Exercises the real binary: help, version, and the headless `--once` mode
//! that makes crabmon scriptable.

use std::process::Command;
use std::time::{Duration, Instant};

fn crabmon(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_crabmon")).args(args).output().expect("failed to run crabmon")
}

#[test]
fn help_and_version_exit_cleanly() {
    let out = crabmon(&["--help"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("USAGE"), "{text}");
    assert!(text.contains("--once"), "{text}");

    let out = crabmon(&["--version"]);
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(env!("CARGO_PKG_VERSION")),
        "version output does not name the package version"
    );
}

#[test]
fn an_unknown_flag_fails_with_a_usage_hint() {
    let out = crabmon(&["--definitely-not-a-flag"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown option"), "{err}");
    assert!(err.contains("--help"), "{err}");
}

#[test]
fn once_json_is_machine_readable_and_reports_real_units() {
    let out = crabmon(&["--once", "--top", "5", "--refresh", "200"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("--once must emit valid JSON");

    let total = v["mem"]["total"].as_u64().unwrap();
    assert!(total > 1_000_000_000, "memory should be a byte count, got {total}");
    assert!(v["procs"].as_array().unwrap().len() <= 5, "--top was ignored");
    assert!(!v["cpu"]["per_core"].as_array().unwrap().is_empty());
    assert!(!v["host"]["hostname"].as_str().unwrap().is_empty());
}

#[test]
fn once_csv_has_a_header_and_one_line_per_process() {
    let out = crabmon(&["--once", "--format", "csv", "--top", "3", "--refresh", "200"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    assert!(lines.next().unwrap().starts_with("pid,ppid,name,cpu_percent"));
    // Quoted command fields may contain newlines. Count CSV records, not
    // physical lines (a shell running a multiline script is a real example).
    let (mut quoted, mut records) = (false, 0);
    for ch in text.chars() {
        if ch == '"' {
            quoted = !quoted;
        }
        if ch == '\n' && !quoted {
            records += 1;
        }
    }
    assert!(records <= 4, "more than a header and three process records");
}

#[test]
fn once_honours_the_filter_language() {
    // This test process: it is running by definition, so it is in the list,
    // and a PID matches exactly one process on any platform. `pid:1` used to
    // stand in for that — init always exists — but only on Linux: Windows
    // has no PID 1, so the filter matched nothing and the run came back
    // empty. The assertion was about the platform, not about the filter.
    let me = std::process::id();
    let out = crabmon(&["--once", "--filter", &format!("pid:{me}"), "--refresh", "200"]);
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let procs = v["procs"].as_array().unwrap();
    assert_eq!(procs.len(), 1, "expected exactly one process, the test itself");
    assert_eq!(procs[0]["pid"], me);
}

#[test]
fn a_bad_sort_key_is_rejected_before_any_sampling_happens() {
    let out = crabmon(&["--once", "--sort", "nonsense"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown sort key"));
}

#[test]
fn headless_modes_refuse_to_silently_sample_the_wrong_host() {
    // `crabmon --once --remote prod-db > snap.json` used to print *this*
    // machine's snapshot with exit 0: plausible data about the wrong host,
    // and no way for a script to tell.
    for mode in [vec!["--once"], vec!["--stream"]] {
        let mut args = mode.clone();
        args.extend(["--remote", "somewhere.invalid"]);
        let out = crabmon(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?} was accepted");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("samples this host"), "{err}");

        let mut args = mode;
        args.extend(["--replay", "run.jsonl"]);
        let out = crabmon(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?} was accepted");
    }

    // `--serve` is the exception: it exports a fleet, labelling every series
    // by host. A recording still has no "now" to scrape.
    let out = crabmon(&["--serve", ":19999", "--replay", "run.jsonl"]);
    assert_eq!(out.status.code(), Some(2), "a replay was accepted as a scrape source");
}

#[test]
fn the_headless_modes_are_mutually_exclusive() {
    let out = crabmon(&["--once", "--stream"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("mutually exclusive"));
}

#[test]
fn diff_reports_what_changed_between_two_snapshots() {
    let dir = std::env::temp_dir();
    let a = dir.join(format!("crabmon-cli-diff-a-{}.json", std::process::id()));
    let b = dir.join(format!("crabmon-cli-diff-b-{}.json", std::process::id()));

    // Two real exports of this machine, taken moments apart.
    for path in [&a, &b] {
        let out = crabmon(&["--once", "--top", "20"]);
        assert!(out.status.success());
        std::fs::write(path, out.stdout).unwrap();
    }

    let out = crabmon(&["--diff", a.to_str().unwrap(), b.to_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    for needle in ["cpu", "load", "mem", "swap", "procs"] {
        assert!(text.contains(needle), "{needle} missing from the report:\n{text}");
    }

    // ...and as JSON, for something other than a human to read.
    let out = crabmon(&["--diff", a.to_str().unwrap(), b.to_str().unwrap(), "--format", "json"]);
    assert!(out.status.success());
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("--diff --format json must emit json");
    assert!(v["seconds_apart"].is_number(), "{v}");

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn diff_of_a_missing_file_fails_instead_of_printing_zeros() {
    let out = crabmon(&["--diff", "/nonexistent/crabmon-snapshot.json"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("crabmon"));
}

/// Read `want` lines from `--stream`, or as many as arrive before `deadline`.
/// Returns the raw text and the instant each line was read.
///
/// Sleeping a fixed wall-clock budget and reading whatever landed is what this
/// replaced: the first frame costs a priming sample plus a refresh interval
/// plus a sample, and on a debug build under a loaded CI runner that is most
/// of a second on its own — so the test failed for reasons that had nothing to
/// do with streaming.
fn stream_lines(args: &[&str], want: usize, deadline: Duration) -> (String, Vec<Instant>) {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_crabmon"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");

    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send((line, Instant::now())).is_err() {
                return;
            }
        }
    });

    let give_up = Instant::now() + deadline;
    let mut text = String::new();
    let mut at = Vec::new();
    while at.len() < want {
        let left = give_up.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match rx.recv_timeout(left) {
            Ok((line, now)) => {
                text.push_str(&line);
                text.push('\n');
                at.push(now);
            }
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    (text, at)
}

#[test]
fn stream_emits_one_parseable_snapshot_per_line() {
    let (text, at) = stream_lines(&["--stream", "--refresh", "200"], 2, Duration::from_secs(20));
    assert_eq!(at.len(), 2, "the stream produced {} frames:\n{text}", at.len());

    let frames = crabmon::record::parse_jsonl(&text).expect("the stream must be readable");
    assert_eq!(frames.len(), 2, "no frames were streamed");
    assert!(!frames[0].host.hostname.is_empty());
    assert!(!frames[0].procs.is_empty());
    // One frame per line: a pretty-printed document would break the reader.
    assert_eq!(text.lines().filter(|l| !l.trim().is_empty()).count(), frames.len());
}

/// `--stream` used to sleep the whole refresh interval and *then* sample, so a
/// frame cost `interval + sample_cost` while every rate in it was still a
/// counter delta divided by the nominal `interval` — every bytes-per-second
/// series overstated by that ratio. `sampler::Pacer` now marks the clock
/// before each sample and sleeps only the remainder.
///
/// The exact pacing is pinned by the `Pacer` unit tests, which run on a fixed
/// timeline. This is the end-to-end half: that the binary really does stream
/// on the cadence it was asked for rather than drifting a sample behind on
/// every frame.
///
/// The bound cannot be a fixed multiple of the interval. `Pacer` runs as fast
/// as it can when a sample costs more than the interval it was given, so the
/// honest expectation is `max(interval, sample_cost)` — and on a loaded
/// machine with a few thousand tasks a sample costs more than the 300 ms this
/// used to ask for, which failed the run for being *correct*. (Caught exactly
/// that way: at load 8 the frames averaged 1.17 s apart.)
///
/// So the cost is measured rather than assumed. A stream at the refresh floor
/// is paced by whichever of the two is larger, so its mean gap is an upper
/// bound on what one sample costs on this machine right now. The bug — which
/// adds the cost to the interval on *every* frame rather than overlapping it
/// — is still caught wherever it was ever visible, because that is precisely
/// where the cost is a sizeable fraction of the interval.
#[test]
fn stream_keeps_to_the_interval_it_was_asked_for() {
    // First to last rather than a mean of the gaps: these are the times this
    // process *read* each line, and a test thread that loses the CPU for a
    // moment reads several at once, which makes individual gaps meaningless
    // while leaving the span across all of them intact. The first gap also
    // carries the cold-cache cost of the opening samples — one
    // `/proc/<pid>/cgroup` read per process — so it is skipped.
    let mean_gap = |at: &[Instant]| (at[at.len() - 1] - at[1]) / (at.len() - 2) as u32;

    let floor = Duration::from_millis(crabmon::MIN_REFRESH_MS);
    let (floor_text, floor_at) =
        stream_lines(&["--stream", "--refresh", "200"], 5, Duration::from_secs(30));
    assert_eq!(floor_at.len(), 5, "only {} frames arrived:\n{floor_text}", floor_at.len());
    let cost = mean_gap(&floor_at).max(floor);

    let refresh = Duration::from_millis(1000);
    let (text, at) = stream_lines(&["--stream", "--refresh", "1000"], 5, Duration::from_secs(30));
    assert_eq!(at.len(), 5, "only {} frames arrived:\n{text}", at.len());
    let mean = mean_gap(&at);

    // Half an interval of slack for scheduling and for reading a line late:
    // both push this the same way, and the exact pacing is the `Pacer` unit
    // tests' job on a timeline that cannot drift.
    let bound = refresh.max(cost) + refresh / 2;
    assert!(
        mean < bound,
        "frames averaged {mean:?} apart for a {refresh:?} refresh, \
         against a {bound:?} bound (a sample costs about {cost:?} here):\n{text}"
    );
}

/// `--doctor` answers the question every empty panel raises — "is that
/// nothing, or is it something I am not allowed to see?" — in one place, and
/// exits non-zero when the config asks for something this host cannot do.
#[test]
fn doctor_reports_what_this_host_lets_crabmon_measure() {
    let out = crabmon(&["--doctor"]);
    let text = String::from_utf8_lossy(&out.stdout);
    for id in ["platform", "privileges", "config", "alert-rules"] {
        assert!(text.contains(id), "the report omits {id}:\n{text}");
    }
    // Every line is marked, so the shape is greppable.
    for line in text.lines().filter(|l| !l.trim_start().starts_with('→')) {
        assert!(line.starts_with(['+', '~', '-']), "a line with no status marker: {line:?}");
    }

    let out = crabmon(&["--doctor", "--format", "json"]);
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("--doctor --format json must be JSON");
    let checks = v["checks"].as_array().expect("checks");
    assert!(checks.len() >= 10, "only {} checks", checks.len());
    for c in checks {
        assert!(c["id"].is_string());
        assert!(["ok", "limited", "unavailable"].contains(&c["status"].as_str().unwrap()), "{c:?}");
    }
}

/// The report must be about the config actually in force. It used to read
/// the default path whatever `--config` said, so a pre-deploy check reported
/// on the operator's own file rather than the one being deployed.
#[test]
fn doctor_reports_on_the_config_that_is_actually_in_force() {
    let dir = scratch("doctor-path");
    let cfg = dir.join("named.toml");
    std::fs::write(&cfg, "refresh_ms = 800\n").unwrap();

    let out = crabmon(&["--config", cfg.to_str().unwrap(), "--doctor"]);
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains("config ")).expect("a config line");
    assert!(line.contains("named.toml"), "it named another file: {line}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ...and it is a check, not just a report: a config naming something that
/// cannot work here exits non-zero so a deploy can refuse it.
#[test]
fn doctor_exits_non_zero_when_the_config_asks_for_the_impossible() {
    let dir = scratch("doctor");
    let cfg = dir.join("crabmon.toml");
    // A rule over descriptors with the sampling it needs switched off: it can
    // never fire, and a rule that never fires reads as a quiet machine.
    std::fs::write(
        &cfg,
        "[procs]\nfds = false\n\n[[alert]]\nname = \"descriptors-high\"\nkind = \"fd\"\nthreshold = 90.0\n",
    )
    .unwrap();

    let out = crabmon(&["--config", cfg.to_str().unwrap(), "--doctor"]);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stdout));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("descriptors-high"), "it did not name the rule:\n{text}");

    // ...and a config that asks for nothing impossible is fine.
    std::fs::write(&cfg, "refresh_ms = 800\n").unwrap();
    let out = crabmon(&["--config", cfg.to_str().unwrap(), "--doctor"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));

    let _ = std::fs::remove_dir_all(&dir);
}

/// A scratch directory of this test's own, cleaned up afterwards.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("crabmon-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Recording used to need a terminal in front of it: `--record` was driven by
/// the interface's event loop, so every headless mode wrote nothing. That made
/// the always-on recorder — the whole point of `--record-rotate` — impossible
/// to deploy under systemd, which is the one place it belongs.
#[test]
fn stream_records_the_frames_it_prints() {
    let dir = scratch("stream-record");
    let path = dir.join("run.jsonl");

    let (text, at) = stream_lines(
        &["--stream", "--refresh", "200", "--record", path.to_str().unwrap()],
        5,
        Duration::from_secs(30),
    );
    assert_eq!(at.len(), 5, "only {} frames arrived:\n{text}", at.len());

    // Flushed per frame, so the file is complete even though the process was
    // killed rather than asked to stop.
    let body = std::fs::read_to_string(&path).expect("the recording must exist");
    let frames = crabmon::record::parse_jsonl(&body).expect("every recorded line must parse");
    assert!(frames.len() >= 3, "only {} frames reached the recording", frames.len());
    assert!(frames.iter().all(|f| f.taken_at_unix > 0), "a recorded frame has no timestamp");
    assert!(
        frames.iter().any(|f| !f.procs.is_empty()),
        "a recording with no processes in it is not evidence of anything"
    );

    // The frame on stdout keeps the full inventory a collector needs; the one
    // on disk is trimmed by `[record]`, which is what stops an always-on
    // recorder filling the disk.
    let streamed: Vec<crabmon::Snapshot> = crabmon::record::parse_jsonl(&text).expect("stdout");
    let widest_streamed = streamed.iter().map(|f| f.procs.len()).max().unwrap_or(0);
    let widest_recorded = frames.iter().map(|f| f.procs.len()).max().unwrap_or(0);
    assert!(
        widest_recorded <= widest_streamed,
        "the recording ({widest_recorded}) carries more than the stream ({widest_streamed})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `--watch` is the mode that is *waiting* for something to go wrong, so the
/// run-up it records is the evidence the condition exists to collect.
#[test]
fn a_watch_leaves_a_rotating_recording_of_the_run_up() {
    let dir = scratch("watch-rotate");

    let out = crabmon(&[
        "--watch",
        "name:definitely-no-such-process-exists",
        "--watch-timeout",
        "2",
        "--refresh",
        "200",
        "--record-rotate",
        dir.to_str().unwrap(),
    ]);
    // Nothing matched, so the watch exits 0 — and still leaves the recording.
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("rotating recording in"), "it never said where: {err}");

    let segments: Vec<_> = std::fs::read_dir(&dir)
        .expect("the rotation directory")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| crabmon::record::is_segment_name(n))
        .collect();
    assert!(!segments.is_empty(), "a watch wrote no segments: {segments:?}");

    let frames: usize = segments
        .iter()
        .map(|n| std::fs::read_to_string(dir.join(n)).expect("segment"))
        .map(|body| crabmon::record::parse_jsonl(&body).expect("every line parses").len())
        .sum();
    // Two, not "about as many as the interval predicts": how fast a machine
    // samples is the pacing test's subject, and tying this to it would make
    // a slow CI runner fail a test about whether recording happens at all.
    assert!(frames >= 2, "a watch recorded only {frames} frames");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ...and the modes that take exactly one sample still refuse, because there
/// is no sequence for a recording to be of.
#[test]
fn the_single_sample_modes_refuse_a_recording_flag() {
    // A name of this test's own, so the "it wrote nothing" assertion cannot
    // be satisfied or defeated by anything else in the working directory.
    let name = format!("crabmon-refused-{}.jsonl", std::process::id());
    let out = crabmon(&["--once", "--record", &name]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("needs a sequence of samples"), "{err}");
    assert!(!std::path::Path::new(&name).exists(), "it wrote the file anyway");
}

#[test]
fn watch_gives_up_and_exits_zero_when_nothing_matches() {
    // `--watch ... && page-someone` has to not page someone on a quiet machine.
    let out = crabmon(&[
        "--watch",
        "name:definitely-no-such-process-exists",
        "--watch-timeout",
        "1",
        "--refresh",
        "200",
    ]);
    assert_eq!(out.status.code(), Some(0), "a timeout is not a match");
    assert!(out.stdout.is_empty(), "nothing matched, so nothing should be printed");
}

#[test]
fn watch_exits_nonzero_and_prints_the_match() {
    // crabmon is always running while this test runs, so it matches itself.
    let out = crabmon(&["--watch", "crabmon", "--watch-timeout", "10", "--refresh", "200"]);
    assert_eq!(out.status.code(), Some(1), "a match must be distinguishable in the shell");
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the match is printed as json");
    assert!(!v["procs"].as_array().unwrap().is_empty(), "{v}");
}

#[test]
fn a_malformed_watch_query_is_rejected_rather_than_matching_nothing_forever() {
    let out = crabmon(&["--watch", "re:[unclosed", "--watch-timeout", "1"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("regex"));
}
