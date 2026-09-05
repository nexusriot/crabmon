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
