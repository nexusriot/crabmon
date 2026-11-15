//! Running alert rules against a stream of snapshots.
//!
//! The engine, the hold-time logic, the recovery hooks and the flight recorder
//! all existed before this module, and all of them only ran inside the TUI.
//! That made the flight recorder — whose whole purpose is to have a recording
//! of an incident nobody predicted — depend on someone sitting in front of a
//! full-screen interface at the moment the incident happened, which is the one
//! thing that is never true at three in the morning. `--serve --alerts` and
//! `--stream --alerts` evaluate the same rules with no terminal at all.
//!
//! Nothing here spawns a process or writes a file on its own account except
//! the flight recording itself: hooks come back as strings for the caller to
//! run, so a test can drive a whole incident and assert on what *would* have
//! been run.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::alerts::{ActiveAlert, AlertEngine, HookCall};
use crate::config::Config;
use crate::metrics::Snapshot;
use crate::record::FlightRecorder;

/// Tracks actual frame arrivals, independently of how often clients scrape.
pub struct RemoteFreshness {
    last_id: Option<u64>,
    received: Option<Instant>,
    timeout: Duration,
    interrupted: bool,
}

impl RemoteFreshness {
    pub fn new(refresh: Duration) -> Self {
        Self {
            last_id: None,
            received: None,
            timeout: (refresh * 3).max(Duration::from_secs(5)),
            interrupted: true,
        }
    }

    /// Seconds since this process received the last new frame; absent before first contact.
    pub fn sample_age(&self, now: Instant) -> Option<f64> {
        self.received.map(|at| now.saturating_duration_since(at).as_secs_f64())
    }

    /// (new measurement, currently available). Clock injection keeps outage
    /// and fast-scrape regressions deterministic.
    pub fn update(
        &mut self,
        snap: &Snapshot,
        id: Option<u64>,
        reachable: bool,
        now: Instant,
        engine: Option<&mut AlertEngine>,
    ) -> (bool, bool) {
        self.update_received(snap, id, reachable, now, now, engine)
    }

    /// Use the worker's arrival timestamp, so delayed scrapes cannot rejuvenate old data.
    pub fn update_received(
        &mut self,
        snap: &Snapshot,
        id: Option<u64>,
        reachable: bool,
        now: Instant,
        received: Instant,
        engine: Option<&mut AlertEngine>,
    ) -> (bool, bool) {
        // Notice the gap *before* accepting a newly delivered frame, so a
        // reconnect cannot silently complete a pre-outage hold timer.
        let expired =
            self.received.is_some_and(|at| now.saturating_duration_since(at) >= self.timeout);
        let valid = reachable && snap.taken_at_unix != 0;
        let fresh = valid && (self.received.is_none() || id.is_none() || id != self.last_id);
        let delayed = now.saturating_duration_since(received) >= self.timeout;
        if !valid || expired || delayed || self.interrupted {
            if let Some(engine) = engine {
                engine.interrupt();
            }
            self.interrupted = true;
        }
        if fresh {
            self.last_id = id;
            self.received = Some(received);
            self.interrupted = delayed;
        }
        (fresh && !self.interrupted, valid && !self.interrupted)
    }
}

/// Everything one sample produced that the caller has to act on.
#[derive(Debug, Default, PartialEq)]
pub struct Observation {
    /// Rules over threshold and past their hold time, for the status line or
    /// the exporter.
    pub active: Vec<ActiveAlert>,
    /// Hooks to run, in the order the rules fired, each carrying what it
    /// should be told. Returned rather than spawned so that what a rule
    /// *would* do is assertable.
    pub commands: Vec<HookCall>,
    /// Every transition this sample, with the same context as `commands`,
    /// for `[[notify]]` — which wants to hear about rules with no hook too.
    pub notices: Vec<HookCall>,
    /// Lines worth telling the user: a recording started, a recording
    /// finished, a recording that could not be written.
    pub notes: Vec<String>,
}

impl Observation {
    pub fn is_quiet(&self) -> bool {
        self.active.is_empty() && self.commands.is_empty() && self.notes.is_empty()
    }
}

/// Feed one frame to the rules and, if there is one, the flight recorder.
///
/// The order is the part worth having in one place: the ring has to contain
/// the frame that tripped a rule *before* the dump is opened, or every
/// recording stops one frame short of the evidence it was written for.
pub fn observe_with(
    engine: &mut AlertEngine,
    flight: Option<&mut FlightRecorder>,
    snap: &Snapshot,
    now_secs: u64,
) -> Observation {
    observe_with_label(engine, flight, snap, now_secs, "")
}

/// As `observe_with`, with a host name folded into the names of any flight
/// dumps it opens.
///
/// A fleet runs one of these per host, and `cpu-saturated` firing on three
/// machines in the same second would otherwise be three writers opening one
/// file named after the rule.
pub fn observe_with_label(
    engine: &mut AlertEngine,
    flight: Option<&mut FlightRecorder>,
    snap: &Snapshot,
    now_secs: u64,
    label: &str,
) -> Observation {
    let active = engine.evaluate(snap, now_secs);
    let drained = drain_named(engine, flight, snap, &active, label);
    let mut commands = engine.take_commands();
    attach_context(&mut commands, &drained.dumps, label, snap);
    let mut notices = engine.take_notices();
    attach_context(&mut notices, &drained.dumps, label, snap);
    Observation { active, commands, notices, notes: drained.notes }
}

/// Fill in the parts of a hook's context that the engine cannot know: which
/// machine the rule was about, and which recording this pass opened for it.
///
/// Done here rather than in the engine because the engine evaluates rules and
/// has never known where it is running or what the recorder did afterwards —
/// the flight dump does not exist until `drain` has written it.
pub fn attach_context(
    commands: &mut [HookCall],
    dumps: &[(String, PathBuf)],
    label: &str,
    snap: &Snapshot,
) {
    for call in commands.iter_mut() {
        call.host = match label.is_empty() {
            false => label.to_string(),
            true => snap.host.hostname.clone(),
        };
        if let Some((_, path)) = dumps.iter().find(|(rule, _)| *rule == call.rule) {
            call.dump = path.display().to_string();
            call.summary = crate::incident::path_for(path).display().to_string();
        }
    }
}

/// What a drain produced.
#[derive(Debug, Default, PartialEq)]
pub struct Drained {
    /// Lines worth telling the user.
    pub notes: Vec<String>,
    /// The recording each rule opened on this pass, so the hook about to run
    /// can be told where the evidence is.
    pub dumps: Vec<(String, PathBuf)>,
}

/// The side-effect half on its own: offer the frame to the recorder, then open
/// a dump for anything that has just fired.
///
/// Separate from `observe_with` because the TUI evaluates its rules inside
/// `App::tick`, where there is no recorder to hand, and drains them from the
/// event loop, where there is. Both end up here, which is the point: the
/// ordering below is the kind of thing that is right in one copy and wrong in
/// the other.
pub fn drain(
    engine: &mut AlertEngine,
    flight: Option<&mut FlightRecorder>,
    snap: &Snapshot,
) -> Drained {
    drain_with_alerts(engine, flight, snap, &[])
}

/// As `drain`, and told which rules are currently firing so the summary
/// written beside a dump can say what tripped it and by how much.
pub fn drain_with_alerts(
    engine: &mut AlertEngine,
    flight: Option<&mut FlightRecorder>,
    snap: &Snapshot,
    active: &[ActiveAlert],
) -> Drained {
    drain_named(engine, flight, snap, active, "")
}

/// As `drain_with_alerts`, with a host name in the dump filenames.
pub fn drain_named(
    engine: &mut AlertEngine,
    flight: Option<&mut FlightRecorder>,
    snap: &Snapshot,
    active: &[ActiveAlert],
    label: &str,
) -> Drained {
    let mut out = Drained::default();
    let notes = &mut out.notes;
    let Some(flight) = flight else {
        // Nothing is listening for them, and leaving them to accumulate would
        // dump an hour of backlog into the first recorder that ever asks.
        engine.take_fired();
        return out;
    };
    if let Some(path) = flight.push(snap) {
        notes.push(format!("flight recording written to {}", path.display()));
    }
    for name in engine.take_fired() {
        let alert = active.iter().find(|a| a.name == name);
        let dump_name = match label.is_empty() {
            true => name.clone(),
            false => format!("{label}-{name}"),
        };
        match flight.trigger_with_summary(&dump_name, snap.taken_at_unix, alert, snap) {
            // `None` means a dump was already running and has been extended,
            // which needs no second announcement.
            Ok(Some(path)) => {
                notes.push(format!(
                    "{name}: recording {} frames to {} (+ {})",
                    flight.buffered(),
                    path.display(),
                    crate::incident::path_for(&path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default()
                ));
                out.dumps.push((name.clone(), path));
            }
            Ok(None) => {}
            Err(e) => notes.push(format!("flight recording failed: {e}")),
        }
    }
    out
}

/// Alert rules plus a flight recorder, for the modes with no `App` to hang
/// them off.
pub struct Supervisor {
    pub alerts: AlertEngine,
    flight: Option<FlightRecorder>,
    started: Instant,
    /// Prefixed to the names of this supervisor's flight dumps. A fleet runs
    /// one supervisor per host, and `cpu-saturated` firing on three machines
    /// at once would otherwise be three writers opening the same file.
    label: String,
}

impl Supervisor {
    /// Build from the config, exactly as the TUI does, so a rule behaves the
    /// same whether or not anyone is watching it.
    pub fn new(cfg: &Config) -> Supervisor {
        let dir = match cfg.record.flight_dir.is_empty() {
            true => PathBuf::from("."),
            false => PathBuf::from(&cfg.record.flight_dir),
        };
        Supervisor::with_flight_dir(cfg, &dir)
    }

    pub fn with_flight_dir(cfg: &Config, dir: &Path) -> Supervisor {
        Supervisor {
            alerts: AlertEngine::configured(cfg),
            flight: FlightRecorder::new(
                dir,
                cfg.record.flight,
                cfg.record.flight_after,
                cfg.record.limits(),
            ),
            started: Instant::now(),
            label: String::new(),
        }
    }

    /// Seconds since this supervisor started, which is the clock the hold
    /// times are measured against — the same one the TUI uses, so
    /// `for_secs = 30` means thirty seconds of crabmon running either way.
    fn uptime(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    /// Name this supervisor's host, so its dumps and its notes say which
    /// machine they are about.
    pub fn for_host(mut self, host: &str) -> Supervisor {
        self.label = host.to_string();
        self.alerts.host = host.to_string();
        self
    }

    pub fn host(&self) -> &str {
        &self.label
    }

    pub fn observe(&mut self, snap: &Snapshot) -> Observation {
        let now = self.uptime();
        let mut obs =
            observe_with_label(&mut self.alerts, self.flight.as_mut(), snap, now, &self.label);
        if !self.label.is_empty() {
            // Everything a fleet reports has to say which machine it is
            // about, or three hosts' alerts read as one machine in a very
            // bad way.
            for line in obs.notes.iter_mut() {
                *line = format!("{}: {line}", self.label);
            }
            for alert in obs.active.iter_mut() {
                alert.message = format!("{}: {}", self.label, alert.message);
            }
            // The same line the fleet view shows. `CRABMON_HOST` carries the
            // machine on its own, but a hook that echoes `CRABMON_MESSAGE`
            // into a chat channel should say what the screen says.
            for call in obs.commands.iter_mut().chain(obs.notices.iter_mut()) {
                call.message = format!("{}: {}", self.label, call.message);
            }
        }
        obs
    }

    pub fn is_dumping(&self) -> bool {
        self.flight.as_ref().is_some_and(|f| f.is_dumping())
    }
}

/// How long a hook may run before it is killed. A hook pages somebody; one
/// still going after a minute is wedged — a `curl` with no timeout against
/// an endpoint that accepted the connection and never answered — and before
/// this it was held, with its whole process group, for as long as crabmon ran.
pub const HOOK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

struct PendingHook {
    child: std::process::Child,
    started: std::time::Instant,
    timeout: std::time::Duration,
    command: String,
    /// Report a non-zero exit when reaped. A hook's exit status is its own
    /// business; a notification that did not arrive is worth saying.
    report_exit: bool,
    /// Killed already, and waiting to be reaped: a second kill would be
    /// aimed at a process group id that may have been reused.
    killed: bool,
}

/// Hook children that have been started and not yet reaped.
///
/// A spawned child stays a zombie in the process table until somebody waits
/// on it. Hooks are deliberately not waited on — an alert hook that blocks
/// the sampler is worse than no hook at all — so without this every firing
/// left a defunct `sh` behind, for the whole life of a process that is
/// typically left running for weeks and whose own process table is on screen.
/// A flapping rule made that unbounded.
#[derive(Default)]
struct Hooks {
    pending: Vec<PendingHook>,
}

impl Hooks {
    /// Reap any hook that has finished, and kill any that has outrun its
    /// timeout. Never blocks: a hook still running is kept and looked at
    /// again next time. Returns a note for each hook it killed.
    fn reap(&mut self) -> Vec<String> {
        let mut notes = Vec::new();
        self.pending.retain_mut(|hook| {
            match hook.child.try_wait() {
                Ok(Some(status)) => {
                    if hook.report_exit && !hook.killed && !status.success() {
                        notes.push(format!(
                            "{} failed ({})",
                            hook.command,
                            match status.code() {
                                Some(code) => curl_failure(code),
                                None => "killed".to_string(),
                            }
                        ));
                    }
                    return false;
                }
                Err(_) => return false,
                Ok(None) => {}
            }
            if !hook.killed && hook.started.elapsed() >= hook.timeout {
                kill_hook(&mut hook.child);
                hook.killed = true;
                notes.push(format!(
                    "{} still running after {}s, killed: {}",
                    if hook.report_exit { "notification" } else { "alert command" },
                    hook.timeout.as_secs(),
                    crate::format::truncate_fit(&hook.command, 80)
                ));
            }
            true
        });
        notes
    }

    /// Start each command, after reaping. Returns notes on hooks killed and
    /// errors for commands that could not be started.
    fn spawn(&mut self, commands: &[HookCall], timeout: std::time::Duration) -> Vec<String> {
        let mut notes = self.reap();
        for call in commands {
            let mut cmd = std::process::Command::new("sh");
            cmd.arg("-c").arg(&call.command).envs(call.env());
            if let Err(e) = self.start(cmd, None, call.command.clone(), false, timeout) {
                notes.push(format!("alert command failed: {e}"));
            }
        }
        notes
    }

    /// Start one child, detached and in a process group of its own, so one
    /// that outlives its timeout can be killed with whatever it started.
    /// `stdin` is written whole and closed; it is a few kilobytes at most,
    /// well inside a pipe's buffer, so the write does not wait on the child.
    fn start(
        &mut self,
        mut cmd: std::process::Command,
        stdin: Option<String>,
        command: String,
        report_exit: bool,
        timeout: std::time::Duration,
    ) -> std::io::Result<()> {
        use std::process::Stdio;
        cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        let mut child = cmd.spawn()?;
        if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
            use std::io::Write;
            let _ = pipe.write_all(text.as_bytes());
        }
        self.pending.push(PendingHook {
            child,
            started: std::time::Instant::now(),
            timeout,
            command,
            report_exit,
            killed: false,
        });
        Ok(())
    }
}

/// Kill a hook and everything it started. The hook is `sh -c`, so killing
/// only the shell would leave the `curl` it was waiting on running, now
/// reparented to init where nothing will ever reap it.
#[cfg(unix)]
fn kill_hook(child: &mut std::process::Child) {
    // Each hook leads its own process group (see `Hooks::spawn`), whose id is
    // the shell's pid, and the shell has not been reaped, so it is still ours.
    let pgid = nix::unistd::Pid::from_raw(child.id() as i32);
    if nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL).is_err() {
        let _ = child.kill();
    }
}

#[cfg(not(unix))]
fn kill_hook(child: &mut std::process::Child) {
    let _ = child.kill();
}

static HOOKS: std::sync::Mutex<Hooks> = std::sync::Mutex::new(Hooks { pending: Vec::new() });

/// Reap finished hooks and kill any past `HOOK_TIMEOUT`; see `Hooks::reap`.
pub fn reap_hooks() -> Vec<String> {
    HOOKS.lock().map(|mut h| h.reap()).unwrap_or_default()
}

/// What a curl exit status means, for a failed notification: `curl exit 28`
/// sent people to a man page in the middle of an incident.
pub fn curl_failure(code: i32) -> String {
    let why = match code {
        6 => "could not resolve the host",
        7 => "could not connect",
        22 => "the server answered with an HTTP error",
        28 => "timed out",
        35 | 51 | 58 | 60 | 77 | 91 => "TLS failed",
        52 => "the server sent nothing",
        56 => "the connection broke",
        _ => return format!("curl exit {code}"),
    };
    format!("{why}, curl exit {code}")
}

/// Run `curl --config -` with `config` on its stdin, as a hook: reaped
/// lazily, killed after `HOOK_TIMEOUT`, and reported under `label` if it
/// exits non-zero. Returns a note if it could not be started at all.
pub fn spawn_curl(label: &str, config: String) -> Vec<String> {
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["--config", "-"]);
    let Ok(mut hooks) = HOOKS.lock() else { return Vec::new() };
    match hooks.start(cmd, Some(config), label.to_string(), true, HOOK_TIMEOUT) {
        Ok(()) => Vec::new(),
        Err(e) => vec![format!("{label}: could not run curl: {e}")],
    }
}

/// How many hook children are still unreaped, for tests.
pub fn pending_hook_count() -> usize {
    HOOKS.lock().map(|h| h.pending.len()).unwrap_or(0)
}

/// Spawn hook commands, detached. Returns the ones that could not be started,
/// and any earlier hook killed for outrunning `HOOK_TIMEOUT`.
///
/// A hook is a user-configured shell command whose job is to page someone; it
/// is not waited on, and its output goes nowhere, because an alert hook that
/// blocks the sampler or scribbles on a full-screen interface is worse than no
/// hook at all. It is still *reaped*, lazily, so a finished hook does not stay
/// in the process table as a zombie, and killed if it is still running after
/// `HOOK_TIMEOUT`.
pub fn spawn_hooks(commands: &[HookCall]) -> Vec<String> {
    HOOKS.lock().map(|mut h| h.spawn(commands, HOOK_TIMEOUT)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::{AlertKind, AlertRule};

    /// The command lines an observation wants run.
    fn ran(obs: &Observation) -> Vec<&str> {
        obs.commands.iter().map(|c| c.command.as_str()).collect()
    }
    use crate::metrics::CpuSample;

    fn rule(name: &str, threshold: f64, for_secs: u64, command: &str) -> AlertRule {
        AlertRule {
            hysteresis: 0.0,
            name: name.into(),
            kind: AlertKind::Cpu,
            threshold,
            for_secs,
            over_secs: 0,
            below: false,
            target: String::new(),
            query: String::new(),
            command: command.into(),
            command_clear: String::new(),
        }
    }

    fn cpu(pct: f32) -> Snapshot {
        Snapshot {
            cpu: CpuSample { per_core: vec![pct], freq_mhz: vec![] },
            taken_at_unix: 1_700_000_000,
            ..Default::default()
        }
    }

    fn cfg_with(rules: Vec<AlertRule>) -> Config {
        Config { alerts: rules, ..Default::default() }
    }

    #[test]
    fn a_rule_fires_with_no_terminal_anywhere_in_sight() {
        // The whole point: the flight recorder and the hooks used to need
        // someone sitting in the TUI at the moment the incident happened.
        let mut sup = Supervisor::new(&cfg_with(vec![rule("hot", 90.0, 0, "page-someone")]));
        let quiet = sup.observe(&cpu(10.0));
        assert!(quiet.is_quiet(), "{quiet:?}");

        let fired = sup.observe(&cpu(95.0));
        assert_eq!(fired.active.len(), 1);
        assert_eq!(ran(&fired), vec!["page-someone"]);
        assert!(fired.active[0].message.contains("hot"), "{:?}", fired.active[0]);
    }

    #[test]
    fn a_hook_runs_once_per_crossing_not_once_per_sample() {
        // A rule that held for an hour used to re-spawn its hook on every
        // refresh if its state was ever cleared from under it.
        let mut sup = Supervisor::new(&cfg_with(vec![rule("hot", 90.0, 0, "page")]));
        assert_eq!(ran(&sup.observe(&cpu(95.0))), vec!["page"]);
        for _ in 0..5 {
            assert!(sup.observe(&cpu(95.0)).commands.is_empty(), "the hook ran twice");
        }
    }

    #[test]
    fn recovery_runs_the_clear_hook_so_an_alert_says_when_it_is_over() {
        let mut r = rule("hot", 90.0, 0, "page");
        r.command_clear = "all-clear".into();
        let mut sup = Supervisor::new(&cfg_with(vec![r]));

        assert_eq!(ran(&sup.observe(&cpu(95.0))), vec!["page"]);
        let recovered = sup.observe(&cpu(1.0));
        assert_eq!(ran(&recovered), vec!["all-clear"]);
        assert!(recovered.active.is_empty());
    }

    /// The ordering the flight recorder depends on: the ring must already
    /// hold the frame that tripped the rule when the dump opens, or every
    /// recording stops one frame short of the evidence.
    #[test]
    fn the_frame_that_tripped_the_rule_is_in_the_recording_it_triggered() {
        let dir = std::env::temp_dir().join(format!("crabmon-sup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut cfg = cfg_with(vec![rule("hot", 90.0, 0, "")]);
        cfg.record.flight = 4;
        cfg.record.flight_after = 0;
        let mut sup = Supervisor::with_flight_dir(&cfg, &dir);

        sup.observe(&cpu(1.0));
        sup.observe(&cpu(2.0));
        let fired = sup.observe(&cpu(95.0));
        assert!(
            fired.notes.iter().any(|n| n.contains("hot: recording")),
            "no recording was announced: {:?}",
            fired.notes
        );

        // The summary sits beside it, so the recording has to be named
        // rather than taken as whatever the directory lists first.
        let dump = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .expect("a dump");
        let frames = crate::record::parse_jsonl(&std::fs::read_to_string(&dump).unwrap()).unwrap();
        let peak = frames.last().expect("frames").cpu.per_core[0];
        assert_eq!(peak, 95.0, "the recording stops before the frame that caused it");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A JSONL dump is evidence nobody can read without crabmon in front of
    /// it; "send me a recording" only lands as a bug report somebody opens
    /// if it arrives with a page saying what fired.
    #[test]
    fn a_dump_comes_with_a_summary_anyone_can_read() {
        let dir = std::env::temp_dir().join(format!("crabmon-sum-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut cfg = cfg_with(vec![rule("hot", 90.0, 0, "")]);
        cfg.record.flight = 4;
        cfg.record.flight_after = 0;
        let mut sup = Supervisor::with_flight_dir(&cfg, &dir);

        sup.observe(&cpu(1.0));
        let fired = sup.observe(&cpu(95.0));
        assert!(!fired.active.is_empty());

        let summary = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "md"))
            .expect("a summary beside the dump");
        let text = std::fs::read_to_string(&summary).unwrap();
        assert!(text.contains("# hot"), "{text}");
        assert!(text.contains("95"), "the measurement is the point: {text}");
        assert!(text.contains("crabmon --replay"), "{text}");
        assert!(
            fired.notes.iter().any(|n| n.contains(".md")),
            "the summary was never announced: {:?}",
            fired.notes
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With no recorder the fired-rule queue still has to be drained, or the
    /// first recorder ever constructed would dump an hour of backlog.
    #[test]
    fn fired_rules_do_not_pile_up_when_nothing_is_recording() {
        let mut sup = Supervisor::new(&cfg_with(vec![rule("hot", 90.0, 0, "")]));
        sup.observe(&cpu(95.0));
        sup.observe(&cpu(1.0));
        sup.observe(&cpu(95.0));
        assert!(sup.alerts.take_fired().is_empty(), "the queue was never drained");
    }

    /// A hook that has exited must not stay in the process table. `crabmon`
    /// is left running for weeks, and a flapping rule used to leave one
    /// defunct `sh` per crossing behind it.
    #[test]
    fn finished_hooks_are_reaped_rather_than_left_as_zombies() {
        assert!(spawn_hooks(&[HookCall::bare("true")]).is_empty());
        // The child needs a moment to exit; `try_wait` never blocks.
        for _ in 0..200 {
            reap_hooks();
            if pending_hook_count() == 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the hook was never reaped");
    }

    #[test]
    fn curl_failures_say_what_went_wrong() {
        assert_eq!(curl_failure(28), "timed out, curl exit 28");
        assert_eq!(curl_failure(22), "the server answered with an HTTP error, curl exit 22");
        assert_eq!(curl_failure(99), "curl exit 99");
    }

    /// A hook that never exits is killed — with what it started — rather
    /// than held for the life of the process.
    #[cfg(unix)]
    #[test]
    fn a_hook_that_outruns_its_timeout_is_killed_with_its_children() {
        let dir = std::env::temp_dir().join(format!("crabmon-hookkill-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pidfile = dir.join("child");
        // The shell backgrounds a sleeper and waits on it, which is the shape
        // of `sh -c 'curl ...'` that killing only the shell gets wrong.
        let cmd = format!("sleep 600 & echo $! > {}; wait", pidfile.display());
        // A set of its own: the global one is shared with every other test in
        // the process, whose hooks must not be killed.
        let mut hooks = Hooks::default();
        let timeout = std::time::Duration::from_millis(200);
        assert!(hooks.spawn(&[HookCall::bare(&cmd)], timeout).is_empty());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let child_pid = loop {
            if let Ok(text) = std::fs::read_to_string(&pidfile)
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                break pid;
            }
            assert!(std::time::Instant::now() < deadline, "the hook never started");
            std::thread::sleep(std::time::Duration::from_millis(20));
        };

        std::thread::sleep(timeout);
        let notes = hooks.reap();
        assert!(notes.iter().any(|n| n.contains("killed") && n.contains("sleep 600")), "{notes:?}");

        // Gone, not a zombie of ours: the sleeper's parent was the shell.
        let alive = || {
            std::fs::read_to_string(format!("/proc/{child_pid}/stat"))
                .map(|s| !s.contains(") Z "))
                .unwrap_or(false)
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while alive() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!alive(), "the hook's child survived the kill");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole point of the context: one generic script, told what fired.
    ///
    /// The variables themselves are asserted here, on every platform, because
    /// they are the contract. Actually spawning a shell to read them back is
    /// the test below, which is Unix-only — a hook *is* `sh -c`, and a
    /// Windows path handed to Git Bash loses its backslashes.
    #[test]
    fn a_hook_is_told_what_fired() {
        let mut sup =
            Supervisor::new(&cfg_with(vec![rule("hot", 90.0, 0, "page-oncall")])).for_host("web-1");
        let fired = sup.observe(&cpu(95.0));
        assert_eq!(fired.commands.len(), 1);

        // Every variable is set, even the ones that do not apply here, so a
        // hook written with `set -u` does not have to guess.
        let env = fired.commands[0].env();
        let names: Vec<&str> = env.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, crate::alerts::HOOK_VARS, "the documented set is the real one");
        let value =
            |name: &str| env.iter().find(|(n, _)| *n == name).map(|(_, v)| v.clone()).unwrap();
        assert_eq!(value("CRABMON_RULE"), "hot");
        assert_eq!(value("CRABMON_KIND"), "cpu");
        assert_eq!(value("CRABMON_STATE"), "firing");
        assert_eq!(value("CRABMON_HOST"), "web-1", "a page must say which machine");
        assert!(value("CRABMON_MESSAGE").contains("web-1"), "{}", value("CRABMON_MESSAGE"));
        assert_eq!(value("CRABMON_AT"), "1700000000");
        assert!(value("CRABMON_VALUE").starts_with("95"), "{}", value("CRABMON_VALUE"));
        // The command itself is untouched: the context is beside it, not in it.
        assert_eq!(fired.commands[0].command, "page-oncall");
    }

    /// ...and the shell really does receive them.
    ///
    /// Unix-only: `spawn_hooks` runs `sh -c`, and writing the path of a
    /// scratch file into that command is not portable to the `sh` that
    /// happens to be on a Windows runner's PATH.
    #[cfg(unix)]
    #[test]
    fn a_hook_is_handed_its_context_by_the_shell() {
        let dir = std::env::temp_dir().join(format!("crabmon-hookenv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("seen");

        let mut sup = Supervisor::new(&cfg_with(vec![rule(
            "hot",
            90.0,
            0,
            &format!(
                "printf '%s|%s|%s|%s|%s\\n' \"$CRABMON_RULE\" \"$CRABMON_KIND\" \
             \"$CRABMON_STATE\" \"$CRABMON_VALUE\" \"$CRABMON_HOST\" > {}",
                out.display()
            ),
        )]))
        .for_host("web-1");
        let fired = sup.observe(&cpu(95.0));
        assert!(spawn_hooks(&fired.commands).is_empty());

        // Generous: a loaded CI runner can take a while to start a shell, and
        // the cost of waiting is nothing next to a test that fails for it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !out.exists() && std::time::Instant::now() < deadline {
            reap_hooks();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let seen = std::fs::read_to_string(&out).expect("the hook never ran");
        let fields: Vec<&str> = seen.trim().split('|').collect();
        assert_eq!(fields[0], "hot", "{seen}");
        assert_eq!(fields[1], "cpu", "{seen}");
        assert_eq!(fields[2], "firing", "{seen}");
        assert!(fields[3].starts_with("95"), "{seen}");
        assert_eq!(fields[4], "web-1", "{seen}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ...and the recovery hook says it is the same rule, going the other way.
    #[test]
    fn a_recovery_hook_is_told_it_is_a_recovery() {
        let mut r = rule("hot", 90.0, 0, "page");
        r.command_clear = "all-clear".into();
        let mut sup = Supervisor::new(&cfg_with(vec![r]));

        let fired = sup.observe(&cpu(95.0));
        assert!(fired.commands[0].firing);
        let recovered = sup.observe(&cpu(1.0));
        assert_eq!(recovered.commands.len(), 1);
        let call = &recovered.commands[0];
        assert!(!call.firing);
        assert_eq!(call.rule, "hot", "the recovery must name the rule it is for");
        assert_eq!(call.threshold, 90.0);
        assert!(call.value < 90.0);
        let env = call.env();
        let state = env.iter().find(|(n, _)| *n == "CRABMON_STATE").unwrap();
        assert_eq!(state.1, "cleared");
        assert!(call.message.contains("recovered"), "{}", call.message);
    }

    #[test]
    fn a_hook_that_cannot_be_started_is_reported_rather_than_lost() {
        // `sh -c` makes almost anything startable, so this is about the
        // return shape: a command that runs reports nothing.
        assert!(spawn_hooks(&[HookCall::bare("true")]).is_empty());
        assert!(spawn_hooks(&[]).is_empty());
    }

    #[test]
    fn a_supervisor_with_no_rules_has_nothing_to_say() {
        let mut sup = Supervisor::new(&cfg_with(Vec::new()));
        assert!(sup.observe(&cpu(100.0)).is_quiet());
        assert!(!sup.is_dumping());
    }
}

#[cfg(test)]
mod freshness_tests {
    use super::*;
    use crate::watch::parse_rule;

    #[test]
    fn blank_failed_and_duplicate_frames_cannot_fire_remote_rules() {
        let mut rule = parse_rule("proc:nginx<1", 2).unwrap();
        rule.command = "page".into();
        let mut engine = AlertEngine::new(vec![rule]);
        let mut gate = RemoteFreshness::new(Duration::from_millis(200));
        let start = Instant::now();
        let blank = Snapshot::default();
        assert_eq!(gate.update(&blank, Some(0), true, start, Some(&mut engine)), (false, false));
        assert!(engine.take_commands().is_empty());
        let snap = Snapshot { taken_at_unix: 1000, ..Default::default() };
        let (fresh, up) = gate.update(&snap, Some(1), true, start, Some(&mut engine));
        assert!(fresh && up);
        assert!(engine.evaluate(&snap, 0).is_empty());
        for t in 1..5 {
            assert_eq!(
                gate.update(
                    &snap,
                    Some(1),
                    true,
                    start + Duration::from_secs(t),
                    Some(&mut engine)
                ),
                (false, true)
            );
        }
        assert!(engine.take_commands().is_empty());
        // A timeout followed directly by a new frame must also reset the hold.
        assert_eq!(
            gate.update(&snap, Some(2), true, start + Duration::from_secs(6), Some(&mut engine)),
            (true, true)
        );
        assert!(engine.evaluate(&snap, 6).is_empty());
        assert_eq!(
            gate.update(&snap, Some(2), false, start + Duration::from_secs(7), Some(&mut engine)),
            (false, false)
        );
        assert_eq!(
            gate.update(&snap, Some(2), true, start + Duration::from_secs(8), Some(&mut engine)),
            (false, false)
        );
        gate.update(&snap, Some(3), true, start + Duration::from_secs(9), Some(&mut engine));
        assert!(engine.evaluate(&snap, 9).is_empty());
        gate.update(&snap, Some(4), true, start + Duration::from_secs(11), Some(&mut engine));
        assert_eq!(engine.evaluate(&snap, 11).len(), 1);
        assert_eq!(
            engine.take_commands().iter().map(|c| c.command.as_str()).collect::<Vec<_>>(),
            vec!["page"]
        );
    }
}

#[cfg(test)]
mod age_tests {
    use super::*;
    #[test]
    fn a_delayed_scrape_cannot_make_an_old_worker_delivery_fresh() {
        let start = Instant::now();
        let mut gate = RemoteFreshness::new(Duration::from_millis(200));
        let snap = Snapshot { taken_at_unix: 100, ..Default::default() };
        let now = start + Duration::from_secs(8);
        assert_eq!(gate.update_received(&snap, Some(1), true, now, start, None), (false, false));
        assert_eq!(gate.sample_age(now), Some(8.0));
        assert_eq!(gate.update_received(&snap, Some(2), true, now, now, None), (true, true));
        assert_eq!(gate.sample_age(now), Some(0.0));
    }

    #[test]
    fn age_increases_on_repeated_frames_and_resets_only_on_arrival() {
        let start = Instant::now();
        let mut gate = RemoteFreshness::new(Duration::from_millis(200));
        assert_eq!(gate.sample_age(start), None);
        let snap = Snapshot { taken_at_unix: u64::MAX, ..Default::default() };
        assert_eq!(gate.update(&snap, Some(1), true, start, None), (true, true));
        let later = start + Duration::from_secs(6);
        assert_eq!(gate.update(&snap, Some(1), true, later, None), (false, false));
        assert_eq!(gate.sample_age(later), Some(6.0));
        assert_eq!(gate.update(&snap, Some(2), true, later, None), (true, true));
        assert_eq!(gate.sample_age(later), Some(0.0));
        gate.update(&snap, Some(2), false, later + Duration::from_secs(1), None);
        assert_eq!(gate.sample_age(later + Duration::from_secs(1)), Some(1.0));
    }
}
