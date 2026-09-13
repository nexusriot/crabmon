//! The build tooling checked against itself. `make` is a front end to
//! `scripts/build.sh` rather than a second implementation of it, and these
//! guard that arrangement: a target with nothing behind it, a command `make`
//! cannot reach, or one missing from the help are all silent failures
//! otherwise — the kind you find when you need the command, not before.

use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The commands `scripts/build.sh` dispatches on, taken from the `cmd_*`
/// functions themselves.
fn build_script_commands() -> Vec<String> {
    read("scripts/build.sh")
        .lines()
        .filter_map(|l| l.trim().strip_prefix("cmd_")?.split_once("()").map(|(n, _)| n.to_string()))
        .map(|n| n.replace('_', "-"))
        .collect()
}

/// The phony targets the Makefile declares.
fn makefile_targets() -> Vec<String> {
    let text = read("Makefile");
    let decl = text.split(".PHONY:").nth(1).expect("the Makefile should declare .PHONY targets");
    let mut targets = Vec::new();
    // The declaration is wrapped across lines with trailing backslashes.
    for line in decl.lines() {
        let wraps = line.trim_end().ends_with('\\');
        targets.extend(line.trim_end().trim_end_matches('\\').split_whitespace().map(String::from));
        if !wraps {
            break;
        }
    }
    targets
}

/// Two places that list the same commands is how a `make` target ends up
/// silently doing nothing, or a documented one not existing at all.
#[test]
fn every_make_target_is_a_command_the_build_script_implements() {
    let commands = build_script_commands();
    assert!(commands.contains(&"build".to_string()), "parsed nothing useful: {commands:?}");

    for target in makefile_targets() {
        assert!(
            commands.contains(&target),
            "make {target} has no cmd_{} in scripts/build.sh",
            target.replace('-', "_")
        );
    }
}

#[test]
fn every_build_script_command_is_reachable_through_make() {
    let targets = makefile_targets();
    for command in build_script_commands() {
        assert!(targets.contains(&command), "scripts/build.sh {command} has no make target");
    }
}

#[test]
fn the_build_script_help_lists_every_command_it_accepts() {
    // The help text is the only place the commands are described; one missing
    // from it is one nobody finds.
    let help = read("scripts/build.sh");
    let help = help.split("cmd_help()").nth(1).expect("cmd_help should exist");
    for command in build_script_commands() {
        assert!(help.contains(&format!("  {command} ")), "{command} is missing from the help");
    }
}

/// Run the script, from the repository root, and return `(status, stdout)`.
#[cfg(unix)]
fn run(args: &[&str]) -> (Option<i32>, String) {
    let out = std::process::Command::new("bash")
        .arg("scripts/build.sh")
        .args(args)
        .output()
        .expect("bash should be able to run the build script");
    (out.status.code(), String::from_utf8_lossy(&out.stdout).to_string())
}

/// Everything above reads the script as text. This runs it, which is the only
/// way to catch a syntax error or a dispatcher that resolves nothing — neither
/// of which shows up until someone needs a build.
#[cfg(unix)]
#[test]
fn the_script_runs_and_reports_the_version_the_crate_was_built_from() {
    let (status, stdout) = run(&["version"]);
    assert_eq!(status, Some(0), "build.sh version exited {status:?}");
    assert_eq!(stdout.trim(), env!("CARGO_PKG_VERSION"));
}

#[cfg(unix)]
#[test]
fn the_default_command_is_the_help_rather_than_a_build() {
    // Running the script bare must not start compiling anything.
    let (status, stdout) = run(&[]);
    assert_eq!(status, Some(0));
    assert!(stdout.contains("Usage:"), "{stdout}");
    assert!(stdout.contains("build"), "{stdout}");
}

/// A mistyped command must fail loudly. Exiting 0 on one would let a CI step
/// that runs the wrong name pass while doing nothing at all.
#[cfg(unix)]
#[test]
fn an_unknown_command_fails_instead_of_quietly_doing_nothing() {
    let (status, stdout) = run(&["biuld"]);
    assert_eq!(status, Some(2), "an unknown command should exit 2");
    assert!(stdout.is_empty(), "the error belongs on stderr: {stdout}");
}
