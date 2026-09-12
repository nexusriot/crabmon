//! Exercises the real binary: help, version, and the headless `--once` mode
//! that makes crabmon scriptable.

use std::process::Command;

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
    assert!(lines.count() <= 3);
}

#[test]
fn once_honours_the_filter_language() {
    // PID 1 always exists on Linux and is the only match for `pid:1`.
    let out = crabmon(&["--once", "--filter", "pid:1", "--refresh", "200"]);
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let procs = v["procs"].as_array().unwrap();
    assert_eq!(procs.len(), 1, "expected exactly one process");
    assert_eq!(procs[0]["pid"], 1);
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
    for mode in [vec!["--once"], vec!["--serve", ":19999"], vec!["--stream"]] {
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

#[test]
fn stream_emits_one_parseable_snapshot_per_line() {
    use std::io::Read;
    use std::process::Stdio;

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_crabmon"))
        .args(["--stream", "--refresh", "200"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");

    // Give it long enough for a couple of frames, then stop it.
    std::thread::sleep(std::time::Duration::from_millis(900));
    let _ = child.kill();
    let mut text = String::new();
    child.stdout.take().unwrap().read_to_string(&mut text).unwrap();
    let _ = child.wait();

    // Killing the writer can cut the final line in half, which is exactly what
    // the recording reader already tolerates — so read it the same way.
    let frames = crabmon::record::parse_jsonl(&text).expect("the stream must be readable");
    assert!(!frames.is_empty(), "no frames were streamed");
    assert!(!frames[0].host.hostname.is_empty());
    assert!(!frames[0].procs.is_empty());
    // One frame per line: a pretty-printed document would break the reader.
    assert!(text.lines().filter(|l| !l.trim().is_empty()).count() >= frames.len());
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
