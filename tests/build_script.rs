//! The build tooling checked against itself. `make` is a front end to
//! `scripts/build.sh` rather than a second implementation of it, and these
//! guard that arrangement: a target with nothing behind it, a command `make`
//! cannot reach, or one missing from the help are all silent failures
//! otherwise — the kind you find when you need the command, not before.

use std::fs;

fn read(path: &str) -> String {
    // LF-normalised for the same reason `tests/docs.rs` does it: a Windows
    // checkout carries CRLF, and a stray `\r` on the end of a parsed target
    // name is a mismatch nobody would guess from the failure.
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}")).replace("\r\n", "\n")
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

/// The help is an unquoted heredoc, so a backtick in it runs a command. It once
/// printed `bin-X: command not found` and dropped the words it was quoting.
#[cfg(unix)]
#[test]
fn the_help_prints_its_backticks_instead_of_running_them() {
    let out = std::process::Command::new("bash")
        .args(["scripts/build.sh", "help"])
        .output()
        .expect("bash should be able to run the build script");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("`bin-X`") && stdout.contains("`pkg add`"), "{stdout}");
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

/// A static binary links nothing, so cargo-deb's `$auto` finds nothing and
/// leaves `Depends:` empty, which lintian flags. cargo-deb cannot omit the
/// field, so the script rewrites the package; losing that call silently would
/// put the warning back.
#[test]
fn cross_debs_are_repacked_without_an_empty_depends_field() {
    let script = read("scripts/build.sh");
    assert!(script.contains("strip_empty_depends()"), "the repacking helper is gone");
    let deb_cross = script.split("deb_cross()").nth(1).expect("deb_cross should exist");
    let body = deb_cross.split("\n}\n").next().unwrap();
    assert!(body.contains("strip_empty_depends"), "deb_cross no longer calls it:\n{body}");
}

/// The FreeBSD package is built by hand — pkg(8) is not available on the Linux
/// machine doing the building — so what `pkg add` will verify is checked here:
/// the manifests lead the archive, and every file's checksum and the flat size
/// in the manifest are those of the file actually packed.
#[cfg(unix)]
#[test]
fn the_freebsd_package_carries_manifests_and_checksums_that_match_its_files() {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !(have("tar") && have("xz") && have("sha256sum") && have("gzip")) {
        eprintln!("skipping: tar, xz, gzip and sha256sum are needed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("crabmon-pkg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("crabmon");
    std::fs::write(&fake, b"not really an ELF").unwrap();
    let out = dir.join("c.pkg");
    let status = Command::new("sh")
        .args(["scripts/freebsd/mkpkg.sh", ".", fake.to_str().unwrap(), "9.8.7"])
        .args(["FreeBSD:15:amd64", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());

    let unpack = dir.join("x");
    std::fs::create_dir_all(&unpack).unwrap();
    // tar -t lists in archive order, which is the order pkg reads.
    let listing = Command::new("tar").arg("-tJf").arg(&out).output().unwrap();
    let names: Vec<String> =
        String::from_utf8_lossy(&listing.stdout).lines().map(str::to_string).collect();
    assert_eq!(&names[..2], ["+COMPACT_MANIFEST", "+MANIFEST"], "{names:?}");
    assert!(names.contains(&"/usr/local/bin/crabmon".to_string()), "{names:?}");
    assert!(names.iter().all(|n| !n.ends_with('/')), "directory entries are not wanted: {names:?}");
    assert!(
        Command::new("tar")
            .arg("-xJf")
            .arg(&out)
            .arg("-C")
            .arg(&unpack)
            .status()
            .unwrap()
            .success()
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(unpack.join("+MANIFEST")).unwrap())
            .expect("the manifest is JSON");
    assert_eq!(manifest["name"], "crabmon");
    assert_eq!(manifest["version"], "9.8.7");
    assert_eq!(manifest["abi"], "FreeBSD:15:amd64");
    let compact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(unpack.join("+COMPACT_MANIFEST")).unwrap())
            .expect("the compact manifest is JSON too");
    assert_eq!(compact["version"], "9.8.7");
    assert!(compact.get("files").is_none());

    let files = manifest["files"].as_object().unwrap();
    assert_eq!(files.len(), names.len() - 2, "every packed file is in the manifest");
    let mut total = 0;
    for (path, sum) in files {
        let on_disk = unpack.join(path.trim_start_matches('/'));
        let sha = Command::new("sha256sum").arg(&on_disk).output().unwrap();
        let got = String::from_utf8_lossy(&sha.stdout);
        assert_eq!(got.split_whitespace().next().unwrap(), sum.as_str().unwrap(), "{path}");
        total += std::fs::metadata(&on_disk).unwrap().len();
    }
    assert_eq!(manifest["flatsize"].as_u64(), Some(total));
    let _ = std::fs::remove_dir_all(&dir);
}
