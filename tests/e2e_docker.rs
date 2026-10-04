//! End-to-end, in a container.
//!
//! Every other suite tests crabmon against *this* machine: whatever processes
//! happen to be running, whatever is mounted, whatever the developer's `/proc`
//! looks like. That is the right trade for most assertions and the wrong one
//! for the questions this suite asks — "is pid 1 reported", "does a process
//! that started between two samples show up as started", "does the exporter
//! answer on a machine with nothing else on it". Those need a machine whose
//! contents are known, which is what `tests/e2e/Dockerfile` builds: a pinned
//! toolchain, a pinned runtime, the binary built from this checkout, and a
//! process list the test itself controls.
//!
//! Hermetic, specifically: every container gets an empty network namespace, a
//! read-only image, a tmpfs for scratch and a fixed hostname and environment,
//! so a test cannot reach the network, cannot see the machine running the
//! suite, and cannot find anything an earlier test left behind. The exporter
//! is scraped over the container's own loopback rather than a published port.
//! Those properties are asserted by
//! `the_container_has_no_network_no_writable_image_and_a_fixed_identity` and
//! `nothing_a_container_writes_survives_into_the_next_one`, because a sandbox
//! nothing checks is a sandbox that quietly stops being one.
//!
//! The image itself is pinned — both bases by digest, the dependencies by
//! `Cargo.lock` — so the same checkout builds the same container. Building it
//! does fetch those dependencies, which is the one thing here that needs the
//! network; `tests/e2e/Dockerfile` says why that is left as it is.
//!
//! The suite is **opt-in**: it needs Docker and a few minutes on a cold cache,
//! which is not what `cargo test` should mean. Set `CRABMON_E2E=1` — or run
//! `make e2e`, which does — and it runs. Without it every test here reports
//! that it was skipped and passes, because a suite that fails on a laptop with
//! no Docker is a suite people delete.

use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

const IMAGE: &str = "crabmon-e2e:test";

/// Whether this run was asked for. Docker being installed is deliberately not
/// enough: these tests cost minutes, and `cargo test` should not.
fn enabled() -> bool {
    matches!(std::env::var("CRABMON_E2E").as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

/// `true` when the test should go ahead, after printing why if it should not.
///
/// Returning rather than panicking: a skipped end-to-end test must not fail
/// the suite on a machine that cannot run it, and cargo has no way to report a
/// test as skipped at runtime.
fn ready(name: &str) -> bool {
    if !enabled() {
        eprintln!("{name}: skipped (set CRABMON_E2E=1, or run `make e2e`)");
        return false;
    }
    match image() {
        Ok(()) => true,
        Err(e) => panic!("{name}: the e2e image could not be built: {e}"),
    }
}

/// Build the image once per run of this binary, however many tests ask for it.
fn image() -> Result<(), String> {
    static BUILT: OnceLock<Result<(), String>> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let out = Command::new("docker")
                .args(["build", "-f", "tests/e2e/Dockerfile", "-t", IMAGE, "."])
                .stdin(Stdio::null())
                .output()
                .map_err(|e| format!("docker could not be run: {e}"))?;
            if out.status.success() {
                return Ok(());
            }
            // The build log is long and only interesting when it failed.
            Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
        })
        .clone()
}

/// Run crabmon in a fresh container. The entrypoint is the binary, so `args`
/// is crabmon's own argv.
fn crabmon(args: &[&str]) -> Output {
    let mut argv = sandbox();
    argv.push(IMAGE);
    argv.extend_from_slice(args);
    Command::new("docker")
        .args(&argv)
        .stdin(Stdio::null())
        .output()
        .expect("docker run should be possible once the image has built")
}

/// Run a shell script in a fresh container, for the scenarios that need a
/// workload alongside crabmon.
///
/// bash rather than sh: `httpc` scrapes the exporter through `/dev/tcp`, which
/// is a bash feature, and a second shell in the image would be one more thing
/// the tests would have to agree about.
fn sh(script: &str) -> Output {
    let mut argv = sandbox();
    argv.extend_from_slice(&["--entrypoint", SHELL, IMAGE, "-c", script]);
    Command::new("docker")
        .args(&argv)
        .stdin(Stdio::null())
        .output()
        .expect("docker run should be possible once the image has built")
}

/// The hostname every container in this suite runs under.
///
/// Docker's default is the container id, different on every run, so anything
/// wanting to assert on the hostname could only check it was non-empty. Fixing
/// it makes "the snapshot describes *this* container" an equality rather than
/// a gesture at one.
const HOSTNAME: &str = "crabmon-e2e";

/// The shell `sh()` runs scripts in, named once so an assertion about the
/// wrapper process cannot drift away from what is actually wrapping it.
const SHELL: &str = "bash";

/// What every container in this suite is given, and nothing else.
///
/// - `--network none`: an empty network namespace. Nothing here can reach the
///   network, so nothing here can come to depend on it without the test that
///   needs it failing — and the exporter is scraped over the container's own
///   loopback rather than a published port.
/// - `--read-only` with a tmpfs on `/work`: the image layers cannot be written
///   to at all, and the scratch space is memory that dies with the container.
///   No test can leave anything behind for the next one to find.
/// - `--hostname`, `--env`: a fixed identity and a fixed environment, so the
///   same inputs produce the same snapshot.
/// - `--rm`: nothing accumulates on the machine running the suite.
fn sandbox() -> Vec<&'static str> {
    vec![
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/work:rw,exec,size=64m",
        "--hostname",
        HOSTNAME,
        "--env",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env",
        "LC_ALL=C",
        "--env",
        "TZ=UTC",
    ]
}

/// A shell fragment that blocks until `/work/run.jsonl` holds `n` whole
/// frames, or gives up after twenty seconds.
///
/// `sleep 2` in its place was a race: the first frame costs a priming sample
/// plus one `/proc/<pid>/cgroup` read per process, and on a loaded machine
/// that can outlast the sleep — leaving a recording whose "before" already
/// contains the workload the test needs it not to.
fn wait_for_frames(n: usize) -> String {
    format!(
        "for i in $(seq 100); do \
           [ \"$(wc -l < /work/run.jsonl)\" -ge {n} ] && break; sleep 0.2; \
         done"
    )
}

/// A shell fragment that writes `lines` to `/work/c.toml`.
///
/// `printf` rather than a heredoc: these scripts arrive at `sh` as one line,
/// and a heredoc terminator has to begin a line of its own — an indented `EOF`
/// never closes it, so the whole rest of the script is swallowed as config and
/// the container exits having done nothing.
fn write_config(lines: &[&str]) -> String {
    let quoted: Vec<String> =
        lines.iter().map(|l| format!("'{}'", l.replace('\'', "'\\''"))).collect();
    format!("printf '%s\\n' {} > /work/c.toml", quoted.join(" "))
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Assert success and hand back stdout, with stderr in the failure message —
/// a container that died has its reason there and nowhere else.
fn ok(out: &Output, what: &str) -> String {
    assert!(out.status.success(), "{what} exited {:?}:\n{}", out.status.code(), stderr(out));
    stdout(out)
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("not valid JSON: {e}\n{text}"))
}

/// The sandbox itself, asserted rather than assumed.
///
/// Every other test in this file is only as hermetic as the flags in
/// `sandbox()`, and a flag that stops working — a Docker upgrade, someone
/// loosening it to debug something and not putting it back — would not fail
/// anything here. It would just quietly let the next test depend on the
/// network, or on a file a previous one left behind.
#[test]
fn the_container_has_no_network_no_writable_image_and_a_fixed_identity() {
    if !ready("the_container_has_no_network_no_writable_image_and_a_fixed_identity") {
        return;
    }
    let out = sh("echo \"host=$(hostname)\"; \
         echo \"tz=$TZ locale=$LC_ALL\"; \
         echo \"ifaces=$(ls /sys/class/net | tr '\\n' ' ')\"; \
         ( echo > /dev/tcp/1.1.1.1/53 ) 2>/dev/null \
            && echo 'net=reachable' || echo 'net=unreachable'; \
         touch /usr/local/bin/scribble 2>/dev/null \
            && echo 'image=writable' || echo 'image=readonly'; \
         touch /work/scratch && echo 'work=writable'; \
         grep -q ' /work tmpfs ' /proc/mounts && echo 'work=tmpfs'");
    let text = ok(&out, "the sandbox");

    assert!(text.contains(&format!("host={HOSTNAME}")), "the hostname is not fixed:\n{text}");
    assert!(text.contains("tz=UTC locale=C"), "the environment is not fixed:\n{text}");
    // An empty network namespace has loopback and nothing else.
    assert!(text.contains("ifaces=lo "), "the container has interfaces it should not:\n{text}");
    assert!(text.contains("net=unreachable"), "the container can reach the network:\n{text}");
    assert!(text.contains("image=readonly"), "the image layers are writable:\n{text}");
    // ...and the one writable place is memory that dies with the container,
    // so nothing a test writes can reach the next one.
    assert!(text.contains("work=writable"), "{text}");
    assert!(text.contains("work=tmpfs"), "/work is not a tmpfs:\n{text}");
}

/// Two containers must not be able to see each other's leftovers, which is
/// the property that lets these tests run in parallel and in any order.
#[test]
fn nothing_a_container_writes_survives_into_the_next_one() {
    if !ready("nothing_a_container_writes_survives_into_the_next_one") {
        return;
    }
    let first = sh("echo marker > /work/left-behind; ls /work");
    assert!(ok(&first, "first container").contains("left-behind"));

    let second = sh("ls /work | wc -l");
    assert_eq!(ok(&second, "second container").trim(), "0", "a file outlived its container");
}

#[test]
fn the_binary_in_the_image_is_the_one_this_checkout_builds() {
    if !ready("the_binary_in_the_image_is_the_one_this_checkout_builds") {
        return;
    }
    let text = ok(&crabmon(&["--version"]), "--version");
    assert_eq!(text.trim(), format!("crabmon {}", env!("CARGO_PKG_VERSION")));
}

/// The container has a process namespace of its own, so pid 1 is the command
/// the test started and nothing else is moving. On the host this assertion
/// would be about whichever init that machine happens to run.
#[test]
fn a_snapshot_of_a_container_describes_that_container() {
    if !ready("a_snapshot_of_a_container_describes_that_container") {
        return;
    }
    let v = json(&ok(&crabmon(&["--once", "--refresh", "200"]), "--once"));

    assert!(v["mem"]["total"].as_u64().unwrap() > 0, "memory should be a byte count");
    assert!(!v["cpu"]["per_core"].as_array().unwrap().is_empty());

    let procs = v["procs"].as_array().expect("procs");
    let init = procs.iter().find(|p| p["pid"] == 1).expect("pid 1 must be in the snapshot");
    assert_eq!(init["name"], "crabmon", "pid 1 in this container is crabmon itself");
    // A fixed hostname makes this an equality rather than a gesture at one.
    assert_eq!(v["host"]["hostname"], HOSTNAME);
    // crabmon samples its own process namespace: in a container of two or
    // three tasks there is no room for the host's thousands to leak in.
    assert!(procs.len() < 50, "a bare container should not report {} processes", procs.len());
}

#[test]
fn a_filter_on_the_command_line_narrows_the_snapshot_it_prints() {
    if !ready("a_filter_on_the_command_line_narrows_the_snapshot_it_prints") {
        return;
    }
    let v = json(&ok(&crabmon(&["--once", "--filter", "pid:1", "--refresh", "200"]), "--once"));
    let procs = v["procs"].as_array().unwrap();
    assert_eq!(procs.len(), 1, "pid:1 matches exactly one process: {procs:?}");
    assert_eq!(procs[0]["pid"], 1);
}

/// A filter that cannot be parsed has to stop the run. The failure this
/// guards is the quiet one: a script that pipes `--once` into a dashboard and
/// gets a snapshot of everything, or of nothing, with exit 0.
#[test]
fn an_unusable_filter_fails_the_run_rather_than_printing_the_wrong_rows() {
    if !ready("an_unusable_filter_fails_the_run_rather_than_printing_the_wrong_rows") {
        return;
    }
    for (query, needle) in
        [("mem>-1M", "bad size"), ("re:[unclosed", "bad regex"), ("cpu>abc", "bad number")]
    {
        let out = crabmon(&["--once", "--filter", query]);
        assert_eq!(out.status.code(), Some(2), "{query} was accepted");
        assert!(stderr(&out).contains(needle), "{query}: {}", stderr(&out));
        assert!(out.stdout.is_empty(), "{query} printed rows anyway");
    }
}

#[test]
fn csv_output_carries_every_column_the_json_does() {
    if !ready("csv_output_carries_every_column_the_json_does") {
        return;
    }
    let text = ok(
        &crabmon(&["--once", "--format", "csv", "--top", "3", "--refresh", "200"]),
        "--once --format csv",
    );
    let mut lines = text.lines();
    let header = lines.next().expect("a header");
    assert_eq!(header, crabmon::export::CSV_COLUMNS.join(","));
    let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    assert!(!rows.is_empty() && rows.len() <= 3, "{} rows for --top 3", rows.len());
    for row in rows {
        assert_eq!(
            row.split(',').count().max(crabmon::export::CSV_COLUMNS.len()),
            row.split(',').count(),
            "row has fewer fields than the header: {row}"
        );
    }
}

/// `--watch` is the one mode whose whole contract is its exit code, and the
/// container is what makes it testable: the process it waits for does not
/// exist until the script starts it, and nothing else in the namespace could
/// match by accident.
#[test]
fn watch_blocks_until_its_process_appears_and_then_exits_one() {
    if !ready("watch_blocks_until_its_process_appears_and_then_exits_one") {
        return;
    }
    // A copy of `sleep` under a name nothing else in the image has, so the
    // query cannot match the harness: the shell running this script has
    // "e2e-victim" in its own command line, but a bare word is a *name*
    // search and the shell is not called that.
    //
    // The exit code goes out first and the JSON after it, because the body is
    // one pretty-printed document and "everything after the first line" is an
    // exact split — a marker printed at the end would have to be a string
    // that never appears in anyone's command line, and the script's own
    // command line is in the snapshot.
    //
    // The workload reaches its name through `exec`, which is how a wrapper
    // script hands off to the real program, and is exactly the case this used
    // to be blind to: the process already existed, under another name, when
    // the watch took its first sample.
    let out = sh("cp /bin/sleep /work/e2e-victim; \
         (sleep 1; exec /work/e2e-victim 30) & \
         crabmon --watch e2e-victim --watch-timeout 20 --refresh 200 > /work/match.json; \
         echo $?; cat /work/match.json");
    let text = ok(&out, "--watch");
    let (code, body) = text.split_once('\n').expect("the exit code, then the match");
    assert_eq!(code.trim(), "1", "a match must exit 1:\n{text}");

    let v = json(body.trim());
    let procs = v["procs"].as_array().expect("the match is printed as json");
    assert_eq!(procs.len(), 1, "exactly one process should have matched: {body}");
    assert_eq!(procs[0]["name"], "e2e-victim");
}

/// A wrapper that `exec`s the real program keeps its PID and its start time,
/// and the sampler reads a name only when it first meets a PID — so every
/// container entrypoint, and every unit with a shell in front of it, showed as
/// `sh` for as long as it ran, with no filter, grouping or alert able to see
/// what it had actually become.
///
/// This is a test the container makes possible: the process list is small
/// enough to follow one PID across the hand-off, which on a live machine would
/// be a needle in several thousand rows.
#[test]
fn a_process_that_execs_is_reported_as_what_it_became() {
    if !ready("a_process_that_execs_is_reported_as_what_it_became") {
        return;
    }
    // `$!` is the subshell's PID, and `exec` keeps it, so one row is followed
    // from wrapper to program.
    // The hand-off is driven by the recording rather than by a clock: the
    // subshell waits until it has been seen as the wrapper in a few frames,
    // then execs, and the outer script waits for a few more. A fixed sleep
    // here raced the priming sample on a loaded machine, and the failure —
    // "only 3 frames saw pid N" — said nothing about crabmon.
    let out = sh(&format!(
        "cp /bin/sleep /work/e2e-victim; : > /work/run.jsonl; \
         crabmon --stream --refresh 400 > /work/run.jsonl & \
         s=$!; {before}; \
         ( {hold}; exec /work/e2e-victim 20 ) & \
         echo $!; {after}; kill $s; cat /work/run.jsonl",
        before = wait_for_frames(2),
        hold = wait_for_frames(5),
        after = wait_for_frames(9),
    ));
    let text = ok(&out, "--stream");
    let (pid, body) = text.split_once('\n').expect("the pid, then the frames");
    let pid: u32 = pid.trim().parse().expect("the subshell's pid");

    let frames = crabmon::record::parse_jsonl(body).expect("every line must parse");
    let names: Vec<String> = frames
        .iter()
        .filter_map(|f| f.procs.iter().find(|p| p.pid == pid))
        .map(|p| p.name.clone())
        .collect();

    assert!(names.len() >= 4, "only {} frames saw pid {pid}", names.len());
    assert_eq!(
        names.first().map(String::as_str),
        Some(SHELL),
        "it should start out as the wrapper: {names:?}"
    );
    assert_eq!(
        names.last().map(String::as_str),
        Some("e2e-victim"),
        "the same pid must be reported as what it exec'd into: {names:?}"
    );

    // ...and the command line and executable follow it, rather than going on
    // describing a program that is no longer running.
    let last = frames
        .iter()
        .rev()
        .find_map(|f| f.procs.iter().find(|p| p.pid == pid))
        .expect("the last frame that has it");
    assert_eq!(last.cmd, "/work/e2e-victim 20", "the argv is still the wrapper's");
    assert_eq!(last.exe, "/work/e2e-victim");
}

#[test]
fn watch_that_times_out_exits_zero_and_prints_nothing() {
    if !ready("watch_that_times_out_exits_zero_and_prints_nothing") {
        return;
    }
    // `&& page-someone` must not page anyone on a quiet machine.
    let out = crabmon(&[
        "--watch",
        "name:no-such-process-in-this-container",
        "--watch-timeout",
        "2",
        "--refresh",
        "200",
    ]);
    assert_eq!(out.status.code(), Some(0), "a timeout is not a match: {}", stderr(&out));
    assert!(out.stdout.is_empty(), "nothing matched, so nothing should be printed");
}

/// Record frames across a change and ask the diff what changed. On a live
/// machine the answer is buried in whatever else started in those seconds; in
/// the container the workload is the only thing that did.
#[test]
fn a_recording_diffed_end_to_end_names_the_process_that_started_in_between() {
    if !ready("a_recording_diffed_end_to_end_names_the_process_that_started_in_between") {
        return;
    }
    // The workload starts *after* the recording is under way, so it is absent
    // from the first frame and present in the last — which is the only thing
    // that makes "started" the right answer rather than "grew".
    let out = sh(&format!(
        "cp /bin/sleep /work/e2e-victim; : > /work/run.jsonl; \
         crabmon --stream --refresh 300 > /work/run.jsonl & \
         stream=$!; {before}; /work/e2e-victim 30 & {after}; kill $stream; \
         crabmon --diff /work/run.jsonl",
        before = wait_for_frames(2),
        after = wait_for_frames(5),
    ));
    let text = ok(&out, "--diff");
    let started = text.split("started (").nth(1).unwrap_or_else(|| {
        panic!("the diff has no `started` section:\n{text}");
    });
    // Only the section itself, so a mention anywhere else cannot satisfy this.
    let started = started.split("\n\n").next().unwrap_or(started);
    assert!(
        started.lines().any(|l| l.contains("e2e-victim")),
        "the process that started between the ends is not listed as started:\n{text}"
    );
}

/// One JSON document per line, forever, on the cadence it was asked for.
#[test]
fn stream_writes_one_frame_per_line_on_its_configured_interval() {
    if !ready("stream_writes_one_frame_per_line_on_its_configured_interval") {
        return;
    }
    // Wait for a number of frames rather than for a wall-clock window: this
    // container shares a host with whatever else is running, and a fixed
    // window asserted on a frame *count* fails when the machine is merely
    // busy — which says nothing about crabmon. Four frames with a deadline
    // measures the same thing and only fails when they genuinely never come.
    let out = sh("crabmon --stream --refresh 300 > /work/run.jsonl & \
         s=$!; for i in $(seq 100); do \
           [ \"$(wc -l < /work/run.jsonl)\" -ge 4 ] && break; sleep 0.2; \
         done; kill $s; wc -l < /work/run.jsonl; cat /work/run.jsonl");
    let text = ok(&out, "--stream");
    let (count, body) = text.split_once('\n').expect("a count then the frames");
    let count: usize = count.trim().parse().expect("wc -l");

    assert!(count >= 4, "only {count} frames arrived in 20s at a 300 ms refresh");

    // The property this is really about: one whole JSON document per line.
    // A pretty-printed snapshot, or a line cut in half, breaks every reader
    // on the far end of a `--remote` pipe.
    let frames = crabmon::record::parse_jsonl(body).expect("every line must parse");
    assert_eq!(frames.len(), count, "a line that is not one whole frame");
    for f in &frames {
        assert_eq!(f.host.hostname, HOSTNAME);
        assert!(f.procs.iter().any(|p| p.pid == 1));
    }
}

/// The exporter, scraped from inside the container's own empty network
/// namespace. There is no published port and nothing else listening.
#[test]
fn the_exporter_answers_a_scrape_and_the_exposition_parses() {
    if !ready("the_exporter_answers_a_scrape_and_the_exposition_parses") {
        return;
    }
    let out = sh("crabmon --serve 127.0.0.1:9100 2>/dev/null & \
         for i in $(seq 15); do \
           httpc GET /metrics > /work/metrics && break; sleep 1; \
         done; \
         echo \"health=$(httpc GET /healthz code)\"; \
         echo \"missing=$(httpc GET /nope code)\"; \
         echo \"post=$(httpc POST /metrics code)\"; \
         echo '--- metrics ---'; cat /work/metrics");
    let text = ok(&out, "--serve");
    assert!(text.contains("health=200"), "{text}");
    assert!(text.contains("missing=404"), "{text}");
    assert!(text.contains("post=405"), "{text}");

    let metrics = text.split("--- metrics ---\n").nth(1).expect("the scrape body");
    assert!(metrics.contains("crabmon_up 1"), "{metrics}");
    assert!(metrics.contains("crabmon_memory_total_bytes"), "{metrics}");
    assert!(metrics.contains("crabmon_process_cpu_percent{"), "{metrics}");

    // Prometheus rejects a whole scrape over a repeated HELP line or a
    // repeated label set within a family, so neither may ever be emitted.
    let mut helps: Vec<&str> = metrics.lines().filter(|l| l.starts_with("# HELP ")).collect();
    let declared = helps.len();
    helps.sort_unstable();
    helps.dedup();
    assert_eq!(helps.len(), declared, "a metric family declares HELP twice:\n{metrics}");

    let mut series: Vec<&str> = metrics
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| l.rsplit_once(' ').map(|(name, _)| name))
        .collect();
    let emitted = series.len();
    assert!(emitted > 20, "only {emitted} samples in the scrape:\n{metrics}");
    series.sort_unstable();
    series.dedup();
    assert_eq!(series.len(), emitted, "the scrape repeats a series:\n{metrics}");

    // ...and every sample ends in a number, which is the rest of the format.
    for line in metrics.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let value = line.rsplit(' ').next().unwrap();
        assert!(value.parse::<f64>().is_ok(), "unparseable sample: {line}");
    }
}

/// Per-*process* network bytes is not something the kernel accounts, so the
/// honest unit is the network namespace — which is a container. Proving that
/// needs two namespaces and a process whose namespace is not crabmon's own,
/// which is exactly what a second container is.
#[test]
fn a_containers_network_is_measured_through_its_own_namespace() {
    if !ready("a_containers_network_is_measured_through_its_own_namespace") {
        return;
    }
    // A target container with a network of its own, and crabmon sharing its
    // *process* namespace so it can see those processes — while staying in a
    // different network namespace, which is the case that matters.
    let name = format!("crabmon-e2e-netns-{}", std::process::id());
    let started = Command::new("docker")
        .args([
            "run",
            "--rm",
            "-d",
            "--name",
            &name,
            "debian:bookworm-slim",
            "sh",
            "-c",
            "while true; do cat /proc/net/dev > /dev/null; sleep 0.2; done",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("docker run");
    assert!(started.status.success(), "{}", stderr(&started));

    let out = Command::new("docker")
        .args([
            "run",
            "--rm",
            &format!("--pid=container:{name}"),
            "--network",
            "none",
            IMAGE,
            "--once",
            "--refresh",
            "500",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("docker run");
    let _ = Command::new("docker").args(["rm", "-f", &name]).stdin(Stdio::null()).output();

    let v = json(&ok(&out, "--once"));
    let rows = v["netns"].as_array().expect("netns");
    assert!(rows.len() >= 2, "two namespaces were expected, got {rows:?}");

    // crabmon's own namespace is flagged, and never attributed to a
    // container: a container sharing the host's network would otherwise put
    // its id on the host's row.
    let host: Vec<_> = rows.iter().filter(|n| n["host"] == true).collect();
    assert_eq!(host.len(), 1, "exactly one namespace is this process's own: {rows:?}");
    assert!(host[0]["container"].is_null(), "{:?}", host[0]);

    // ...and the other one is the target container, counted once however many
    // processes it runs.
    let other: Vec<_> = rows.iter().filter(|n| n["host"] != true).collect();
    assert!(!other.is_empty(), "the target container has no row: {rows:?}");
    assert!(other.iter().any(|n| n["container"].is_string()), "none of {other:?} is a container");
    assert!(other.iter().all(|n| n["procs"].as_u64().unwrap_or(0) >= 1));
}

/// The flight recorder's whole point is to have a recording of an incident
/// nobody predicted, which it could not do without someone sitting in the TUI
/// at the moment it happened.
#[test]
fn alerts_fire_their_hooks_and_write_recordings_with_no_terminal() {
    if !ready("alerts_fire_their_hooks_and_write_recordings_with_no_terminal") {
        return;
    }
    // A rule that is true of any running machine, so the test does not have
    // to manufacture an incident, with a hook that leaves a file behind.
    //
    // `printf` rather than a heredoc: the script is one line by the time it
    // reaches `sh`, and a heredoc terminator has to start a line of its own.
    let out = sh(&format!(
        "{}; mkdir -p /work/dumps; \
         crabmon --stream --alerts --config /work/c.toml > /dev/null 2>/work/err & \
         s=$!; sleep 3; kill $s; \
         echo \"hook=$([ -f /work/hook-ran ] && echo yes || echo no)\"; \
         echo \"dumps=$(ls /work/dumps | wc -l)\"; \
         cat /work/err",
        write_config(&[
            "refresh_ms = 300",
            "[record]",
            "flight = 3",
            "flight_after = 0",
            "flight_dir = \"/work/dumps\"",
            "[[alert]]",
            "name = \"always\"",
            "kind = \"cpu\"",
            "threshold = 0.0",
            "for_secs = 0",
            "command = \"touch /work/hook-ran\"",
        ])
    ));
    let text = ok(&out, "--stream --alerts");

    assert!(text.contains("hook=yes"), "the alert hook never ran:\n{text}");
    assert!(text.contains("dumps=1"), "no flight recording was written:\n{text}");
    // The notices go to stderr so `--stream`'s stdout stays one frame a line.
    assert!(text.contains("always"), "the firing rule was never announced:\n{text}");
}

#[test]
fn the_exporter_publishes_alert_rules_when_it_is_asked_to() {
    if !ready("the_exporter_publishes_alert_rules_when_it_is_asked_to") {
        return;
    }
    let out = sh(&format!(
        "{}; crabmon --serve 127.0.0.1:9100 --alerts --config /work/c.toml 2>/dev/null & \
         for i in $(seq 10); do \
           httpc GET /metrics > /work/metrics && break; sleep 1; \
         done; \
         cat /work/metrics",
        write_config(&[
            "[[alert]]",
            "name = \"always\"",
            "kind = \"cpu\"",
            "threshold = 0.0",
            "for_secs = 0",
            "[[alert]]",
            "name = \"never\"",
            "kind = \"load\"",
            "threshold = 100000.0",
            "for_secs = 0",
        ])
    ));
    let text = ok(&out, "--serve --alerts");

    assert!(text.contains("crabmon_alert_active{rule=\"always\",kind=\"cpu\"} 1"), "{text}");
    // A rule that is not firing still needs a series, or it cannot be alerted
    // on and a dashboard built from it has nothing to draw.
    assert!(text.contains("crabmon_alert_active{rule=\"never\",kind=\"load\"} 0"), "{text}");
    assert!(
        text.contains("crabmon_alert_threshold{rule=\"never\",kind=\"load\"} 100000"),
        "{text}"
    );
}

/// A measurement watch, end to end, in both directions: a condition that is
/// true of any machine and one that is true of none.
#[test]
fn a_rule_watch_exits_on_the_condition_it_describes() {
    if !ready("a_rule_watch_exits_on_the_condition_it_describes") {
        return;
    }
    let out = sh(
        "crabmon --watch-rule 'cpu>=0' --watch-timeout 10 --refresh 200 > /work/hit.json;          echo $?;          crabmon --watch-rule 'load>=100000' --watch-timeout 2 --refresh 200 > /work/miss.json;          echo $?;          echo '--- hit ---'; cat /work/hit.json;          echo '--- miss ---'; cat /work/miss.json",
    );
    let text = ok(&out, "--watch-rule");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("1"), "a met condition exits 1:\n{text}");
    assert_eq!(lines.next(), Some("0"), "a timeout exits 0:\n{text}");

    let hit = text.split("--- hit ---\n").nth(1).unwrap_or("");
    let hit = hit.split("--- miss ---").next().unwrap_or("");
    let v = json(hit.trim());
    assert_eq!(v["rule"], "cpu>=0");
    assert!(v["value"].is_number(), "{v}");
    // A rule about the machine prints the condition, not an empty process
    // table dressed up as a snapshot.
    assert!(v["procs"].is_null(), "{v}");

    let miss = text.split("--- miss ---\n").nth(1).unwrap_or("x");
    assert!(miss.trim().is_empty(), "a timeout prints nothing: {miss:?}");
}

/// Exporting from inside a container must write a file the container can
/// actually read back, with the process list the snapshot described.
#[test]
fn an_exported_snapshot_round_trips_through_the_filesystem() {
    if !ready("an_exported_snapshot_round_trips_through_the_filesystem") {
        return;
    }
    let out = sh("crabmon --once --refresh 200 > /work/a.json; sleep 1; \
         crabmon --once --refresh 200 > /work/b.json; \
         crabmon --diff /work/a.json /work/b.json --format json");
    let v = json(&ok(&out, "--diff --format json"));
    assert!(v["seconds_apart"].is_number(), "{v}");
    assert_eq!(v["host_before"], v["host_after"], "both ends are the same container");
    assert!(v["mem_used"].is_array(), "{v}");
}
