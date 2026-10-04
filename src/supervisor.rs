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
use std::time::Instant;

use crate::alerts::{ActiveAlert, AlertEngine};
use crate::config::Config;
use crate::metrics::Snapshot;
use crate::record::FlightRecorder;

/// Everything one sample produced that the caller has to act on.
#[derive(Debug, Default, PartialEq)]
pub struct Observation {
    /// Rules over threshold and past their hold time, for the status line or
    /// the exporter.
    pub active: Vec<ActiveAlert>,
    /// Hook commands to run, in the order the rules fired. Returned rather
    /// than spawned so that what a rule *would* do is assertable.
    pub commands: Vec<String>,
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
    let active = engine.evaluate(snap, now_secs);
    let notes = drain(engine, flight, snap);
    let commands = engine.take_commands();
    Observation { active, commands, notes }
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
) -> Vec<String> {
    let mut notes = Vec::new();
    let Some(flight) = flight else {
        // Nothing is listening for them, and leaving them to accumulate would
        // dump an hour of backlog into the first recorder that ever asks.
        engine.take_fired();
        return notes;
    };
    if let Some(path) = flight.push(snap) {
        notes.push(format!("flight recording written to {}", path.display()));
    }
    for name in engine.take_fired() {
        match flight.trigger(&name, snap.taken_at_unix) {
            // `None` means a dump was already running and has been extended,
            // which needs no second announcement.
            Ok(Some(path)) => notes.push(format!(
                "{name}: recording {} frames to {}",
                flight.buffered(),
                path.display()
            )),
            Ok(None) => {}
            Err(e) => notes.push(format!("flight recording failed: {e}")),
        }
    }
    notes
}

/// Alert rules plus a flight recorder, for the modes with no `App` to hang
/// them off.
pub struct Supervisor {
    pub alerts: AlertEngine,
    flight: Option<FlightRecorder>,
    started: Instant,
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
            alerts: AlertEngine::new(cfg.alerts.clone()),
            flight: FlightRecorder::new(
                dir,
                cfg.record.flight,
                cfg.record.flight_after,
                cfg.record.limits(),
            ),
            started: Instant::now(),
        }
    }

    /// Seconds since this supervisor started, which is the clock the hold
    /// times are measured against — the same one the TUI uses, so
    /// `for_secs = 30` means thirty seconds of crabmon running either way.
    fn uptime(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    pub fn observe(&mut self, snap: &Snapshot) -> Observation {
        let now = self.uptime();
        observe_with(&mut self.alerts, self.flight.as_mut(), snap, now)
    }

    pub fn is_dumping(&self) -> bool {
        self.flight.as_ref().is_some_and(|f| f.is_dumping())
    }
}

/// Spawn hook commands, detached. Returns the ones that could not be started.
///
/// A hook is a user-configured shell command whose job is to page someone; it
/// is not waited on, and its output goes nowhere, because an alert hook that
/// blocks the sampler or scribbles on a full-screen interface is worse than no
/// hook at all.
pub fn spawn_hooks(commands: &[String]) -> Vec<String> {
    commands
        .iter()
        .filter_map(|cmd| {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .err()
                .map(|e| format!("alert command failed: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::{AlertKind, AlertRule};
    use crate::metrics::CpuSample;

    fn rule(name: &str, threshold: f64, for_secs: u64, command: &str) -> AlertRule {
        AlertRule {
            name: name.into(),
            kind: AlertKind::Cpu,
            threshold,
            for_secs,
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
        assert_eq!(fired.commands, vec!["page-someone"]);
        assert!(fired.active[0].message.contains("hot"), "{:?}", fired.active[0]);
    }

    #[test]
    fn a_hook_runs_once_per_crossing_not_once_per_sample() {
        // A rule that held for an hour used to re-spawn its hook on every
        // refresh if its state was ever cleared from under it.
        let mut sup = Supervisor::new(&cfg_with(vec![rule("hot", 90.0, 0, "page")]));
        assert_eq!(sup.observe(&cpu(95.0)).commands, vec!["page"]);
        for _ in 0..5 {
            assert!(sup.observe(&cpu(95.0)).commands.is_empty(), "the hook ran twice");
        }
    }

    #[test]
    fn recovery_runs_the_clear_hook_so_an_alert_says_when_it_is_over() {
        let mut r = rule("hot", 90.0, 0, "page");
        r.command_clear = "all-clear".into();
        let mut sup = Supervisor::new(&cfg_with(vec![r]));

        assert_eq!(sup.observe(&cpu(95.0)).commands, vec!["page"]);
        let recovered = sup.observe(&cpu(1.0));
        assert_eq!(recovered.commands, vec!["all-clear"]);
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

        let dump = std::fs::read_dir(&dir).unwrap().flatten().next().expect("a dump").path();
        let frames = crate::record::parse_jsonl(&std::fs::read_to_string(&dump).unwrap()).unwrap();
        let peak = frames.last().expect("frames").cpu.per_core[0];
        assert_eq!(peak, 95.0, "the recording stops before the frame that caused it");

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

    #[test]
    fn a_hook_that_cannot_be_started_is_reported_rather_than_lost() {
        // `sh -c` makes almost anything startable, so this is about the
        // return shape: a command that runs reports nothing.
        assert!(spawn_hooks(&["true".to_string()]).is_empty());
        assert!(spawn_hooks(&[]).is_empty());
    }

    #[test]
    fn a_supervisor_with_no_rules_has_nothing_to_say() {
        let mut sup = Supervisor::new(&cfg_with(Vec::new()));
        assert!(sup.observe(&cpu(100.0)).is_quiet());
        assert!(!sup.is_dumping());
    }
}
