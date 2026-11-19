//! systemd unit health: which services have failed, and which are restarting
//! in a loop.
//!
//! A service with `Restart=on-failure` that dies on startup does not look
//! broken from anywhere else on the dashboard. Its process lives a second,
//! exits, and systemd starts another five seconds later; at any given sample
//! there is usually nothing running under the unit at all, so the grouped view
//! has no row for it, and a `proc:nginx<1` rule flickers rather than holds.
//! The number that says "this has been broken for twenty minutes" is the one
//! systemd keeps itself: `NRestarts`, the automatic restarts since the unit was
//! last started by hand.
//!
//! It is read with `systemctl show`, which needs no privileges, on a thread of
//! its own: it takes ~100 ms on an ordinary machine, which is too much to put
//! in front of every sample, and the restart count is cumulative, so reading it
//! every few seconds loses nothing. `systemctl show` only reports units that
//! are loaded — every running and every failed unit, which is what matters.
//! User units (`systemctl --user`) are not read.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// What a snapshot from a crabmon that does not report units says about them.
pub fn not_reported() -> Option<String> {
    Some("not reported by this crabmon".into())
}

/// What crabmon makes of a unit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnitState {
    #[default]
    Ok,
    /// Restarted automatically within the window, or waiting to be.
    Restarting,
    /// Restarted automatically `crash_loop` times or more within the window.
    #[serde(rename = "crashloop")]
    CrashLoop,
    /// `ActiveState=failed`: given up on, by systemd or by its start limit.
    Failed,
}

impl UnitState {
    pub fn label(self) -> &'static str {
        match self {
            UnitState::Ok => "ok",
            UnitState::Restarting => "restarting",
            UnitState::CrashLoop => "crash-loop",
            UnitState::Failed => "failed",
        }
    }

    /// Worst first, for sorting and for picking a colour.
    pub fn severity(self) -> u8 {
        match self {
            UnitState::Ok => 0,
            UnitState::Restarting => 1,
            UnitState::CrashLoop => 2,
            UnitState::Failed => 3,
        }
    }
}

/// One unit, as a snapshot carries it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UnitHealth {
    /// The unit's id, `nginx.service` — the same string a process's
    /// `service` holds, which is what joins the two in the grouped view.
    pub name: String,
    /// `ActiveState`, `SubState` and `Result` exactly as systemd spells them.
    pub active: String,
    pub sub: String,
    pub result: String,
    /// `NRestarts`: automatic restarts since the unit was last started by hand.
    pub restarts: u64,
    /// Of those, how many happened within the window crabmon has watched.
    pub recent_restarts: u64,
    /// `ExecMainStatus`: the last exit status of the main process.
    pub exit_status: Option<i32>,
    pub state: UnitState,
    /// The window `recent_restarts` was counted over, by the crabmon that
    /// counted them — under `--remote`, the far end's `[units] window_secs`,
    /// which the near end's config knows nothing about. 0 in frames that
    /// predate it, read as the default ten minutes.
    pub window_secs: u64,
    /// The unit's last journal lines, oldest first, for units in trouble
    /// only: "crash-looping" says *that* it is broken, and these say why.
    /// Read by the crabmon on the unit's own machine, so they travel over
    /// `--remote`, into recordings and into notifications. Empty where the
    /// journal cannot be read, or the unit is healthy.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub log_tail: Vec<String>,
}

impl UnitHealth {
    /// `failed (exit-code 1)`, `crash-loop 7×/10m`: the state and the reason,
    /// as short as a table cell can carry it.
    pub fn summary(&self) -> String {
        let window = if self.window_secs == 0 { 600 } else { self.window_secs };
        let mins = (window / 60).max(1);
        match self.state {
            UnitState::Failed => match (self.result.as_str(), self.exit_status) {
                // A unit whose ExecStartPre or a oneshot step failed reports
                // the *main* process's status, which can be 0.
                ("exit-code", Some(code)) if code != 0 => format!("failed (exit {code})"),
                ("" | "success", _) => "failed".to_string(),
                (result, _) => format!("failed ({result})"),
            },
            UnitState::CrashLoop => format!("crash-loop {}×/{mins}m", self.recent_restarts),
            UnitState::Restarting if self.recent_restarts > 0 => {
                format!("restarted {}×/{mins}m", self.recent_restarts)
            }
            UnitState::Restarting => "restarting".to_string(),
            UnitState::Ok => "ok".to_string(),
        }
    }
}

impl UnitHealth {
    /// `summary` for a narrow column, at most about ten characters: the
    /// reason for a failure, the count for a loop.
    pub fn short(&self) -> String {
        match self.state {
            UnitState::Failed => match (self.result.as_str(), self.exit_status) {
                ("exit-code", Some(code)) if code != 0 => format!("exit {code}"),
                ("start-limit-hit", _) => "start-limit".into(),
                ("" | "success" | "exit-code", _) => "failed".into(),
                (result, _) => result.to_string(),
            },
            UnitState::CrashLoop => format!("loop {}×", self.recent_restarts),
            UnitState::Restarting => "restarting".into(),
            UnitState::Ok => format!("{} restarts", self.restarts),
        }
    }
}

/// What crabmon can do to a unit from the interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitVerb {
    Restart,
    ResetFailed,
}

impl UnitVerb {
    /// The `systemctl` verb, which is also how the action log spells it.
    pub fn name(self) -> &'static str {
        match self {
            UnitVerb::Restart => "restart",
            UnitVerb::ResetFailed => "reset-failed",
        }
    }
}

/// Run `systemctl <verb> <unit>`.
///
/// `--no-block` queues a restart rather than waiting for the unit to come
/// up, which can take a minute and would freeze the interface; whether it
/// worked shows in the next readings. `--no-ask-password`, because a polkit
/// password prompt would land in the middle of a full-screen interface that
/// is not reading the terminal — without the privilege the call fails, and
/// says so. `--` keeps a unit name from ever being read as an option.
pub fn control(verb: UnitVerb, unit: &str) -> Result<(), String> {
    let mut cmd = std::process::Command::new("systemctl");
    cmd.arg("--no-ask-password");
    if verb == UnitVerb::Restart {
        cmd.arg("--no-block");
    }
    let out = cmd
        .args([verb.name(), "--", unit])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("systemctl: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    Err(err.lines().find(|l| !l.trim().is_empty()).unwrap_or("systemctl failed").trim().to_string())
}

/// Journal lines carried per unit in trouble.
pub const LOG_LINES: usize = 12;
/// Units whose journal is read per pass, worst first: each read is a
/// `journalctl` run, and a machine with forty failed units needs the first
/// few explained, not all of them every five seconds.
pub const LOG_UNITS: usize = 5;
/// A journal line longer than this is cut: a stack trace on one line is
/// still evidence, but not a kilobyte of it in every frame.
const LOG_LINE_MAX: usize = 300;

/// The last `lines` journal lines of `unit`, oldest first. `--unit=` rather
/// than `-u NAME`, so a unit name can never be read as an option, and
/// without a pager or a prompt. A journal crabmon may not read — another
/// user's, without the `systemd-journal` group — is an empty answer, not an
/// error: the unit's state is still worth showing.
pub fn read_log(unit: &str, lines: usize) -> Vec<String> {
    let out = std::process::Command::new("journalctl")
        .arg(format!("--unit={unit}"))
        .args(["-n", &lines.to_string(), "-o", "short-iso", "--no-pager", "-q"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| crate::format::truncate_fit(&without_hostname(l), LOG_LINE_MAX))
            .collect(),
        _ => Vec::new(),
    }
}

/// A `short-iso` journal line without its second field, the hostname: every
/// line names the same machine, and on a real host it took a third of the
/// width a popup or a chat message has for the part that says why.
fn without_hostname(line: &str) -> String {
    let mut parts = line.splitn(3, ' ');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(when), Some(_host), Some(rest)) if when.contains('T') => format!("{when} {rest}"),
        _ => line.to_string(),
    }
}

/// `lines` with credentials hidden when `redact` is on.
pub fn redact_lines(lines: Vec<String>, redact: bool) -> Vec<String> {
    match redact {
        true => lines.iter().map(|l| crate::export::redact_log(l)).collect(),
        false => lines,
    }
}

/// Fill in the log tails of the worst units in trouble.
pub fn attach_logs(
    units: &mut [UnitHealth],
    cache: &mut LogCache,
    read: &impl Fn(&str) -> Vec<String>,
) {
    for unit in units.iter_mut().filter(|u| u.state != UnitState::Ok).take(LOG_UNITS) {
        let key = LogCache::key(unit);
        unit.log_tail = match cache.0.get(&unit.name) {
            Some((k, tail)) if *k == key => tail.clone(),
            _ => {
                let tail = read(&unit.name);
                cache.0.insert(unit.name.clone(), (key, tail.clone()));
                tail
            }
        };
    }
    let names: std::collections::HashSet<&str> = units.iter().map(|u| u.name.as_str()).collect();
    cache.0.retain(|name, _| names.contains(name.as_str()));
}

/// Fill the log tails that the cache already holds for the units' current
/// state, and say whether any is missing — so the new states can be
/// published before the slow part.
pub fn attach_cached_logs(units: &mut [UnitHealth], cache: &LogCache) -> bool {
    let mut missing = false;
    for unit in units.iter_mut().filter(|u| u.state != UnitState::Ok).take(LOG_UNITS) {
        match cache.0.get(&unit.name) {
            Some((k, tail)) if *k == LogCache::key(unit) => unit.log_tail = tail.clone(),
            _ => missing = true,
        }
    }
    missing
}

/// Journal tails already read, by unit, with the state they were read in.
///
/// A unit's journal only says something new when the unit did something
/// new: it restarted again, changed state, or exited differently. Reading
/// it every five seconds regardless cost a second a unit on a busy one-core
/// host, and the fresh unit states waited behind every read.
#[derive(Debug, Default)]
pub struct LogCache(std::collections::HashMap<String, (String, Vec<String>)>);

impl LogCache {
    fn key(u: &UnitHealth) -> String {
        format!("{}|{}|{}|{}|{:?}", u.restarts, u.active, u.sub, u.result, u.exit_status)
    }
}

/// What an alert about units adds to its notification: the log tail of the
/// first unit it names, under the unit's name. Empty when there is none.
pub fn detail(units: &[UnitHealth], figure: &str, query: &str) -> String {
    let named = subject(units, figure, query);
    let Some(unit) = units
        .iter()
        .filter(|u| names(u, query) && !u.log_tail.is_empty())
        .find(|u| named.contains(&u.name))
    else {
        return String::new();
    };
    let tail = &unit.log_tail[unit.log_tail.len().saturating_sub(5)..];
    format!("{}:\n{}", unit.name, tail.join("\n"))
}

/// The figures a `unit` alert can measure.
pub const UNIT_SERIES: [&str; 3] = ["crashloop", "failed", "restarts"];

/// Whether `query` names `unit`: its full id, or its id without
/// `.service`. Empty names every unit.
pub fn names(unit: &UnitHealth, query: &str) -> bool {
    let query = query.trim();
    query.is_empty() || unit.name == query || unit.name.strip_suffix(".service") == Some(query)
}

/// What a `unit` alert measures: `crashloop` and `failed` count units in
/// that state, `restarts` is the most recent restarts of any. `None` for a
/// figure that does not exist, which a rule's parser refuses first.
pub fn measure(units: &[UnitHealth], figure: &str, query: &str) -> Option<f64> {
    let matching = units.iter().filter(|u| names(u, query));
    let count = |state: UnitState| matching.clone().filter(|u| u.state == state).count() as f64;
    match figure.trim() {
        "" | "crashloop" => Some(count(UnitState::CrashLoop)),
        "failed" => Some(count(UnitState::Failed)),
        "restarts" => Some(matching.map(|u| u.recent_restarts).max().unwrap_or(0) as f64),
        _ => None,
    }
}

/// Which units a `unit` alert is about, for its message: those in the state
/// it counts, or the one with the most restarts. "unit-failed 2 ≥ 1" pages
/// somebody who then has to go and find out which two; at most three are
/// named, then how many more.
pub fn subject(units: &[UnitHealth], figure: &str, query: &str) -> String {
    let matching = units.iter().filter(|u| names(u, query));
    let named: Vec<&str> = match figure.trim() {
        "restarts" => matching
            .filter(|u| u.recent_restarts > 0)
            .max_by_key(|u| u.recent_restarts)
            .map(|u| vec![u.name.as_str()])
            .unwrap_or_default(),
        "failed" => {
            matching.filter(|u| u.state == UnitState::Failed).map(|u| u.name.as_str()).collect()
        }
        _ => {
            matching.filter(|u| u.state == UnitState::CrashLoop).map(|u| u.name.as_str()).collect()
        }
    };
    match named.len() {
        0 => String::new(),
        1..=3 => format!(" ({})", named.join(", ")),
        n => format!(" ({}, +{} more)", named[..3].join(", "), n - 3),
    }
}

/// `systemctl show`'s output: blank-line separated blocks of `Key=Value`.
pub fn parse_show(out: &str) -> Vec<UnitHealth> {
    let mut units = Vec::new();
    for block in out.split("\n\n") {
        let mut unit = UnitHealth::default();
        for line in block.lines() {
            let Some((key, value)) = line.split_once('=') else { continue };
            match key {
                "Id" => unit.name = value.to_string(),
                "ActiveState" => unit.active = value.to_string(),
                "SubState" => unit.sub = value.to_string(),
                "Result" => unit.result = value.to_string(),
                "NRestarts" => unit.restarts = value.parse().unwrap_or(0),
                "ExecMainStatus" => unit.exit_status = value.parse().ok(),
                _ => {}
            }
        }
        if !unit.name.is_empty() {
            units.push(unit);
        }
    }
    units
}

/// Restart counts over time, per unit, so "restarted 7 times in the last ten
/// minutes" can be told apart from "restarted 7 times since March".
#[derive(Debug, Default)]
pub struct RestartTracker {
    seen: HashMap<String, VecDeque<(Instant, u64)>>,
}

impl RestartTracker {
    /// Fill in `recent_restarts` and `state` for each unit, as of `now`.
    ///
    /// A unit seen for the first time has no recent restarts — its count is
    /// since it was last started by hand, which could be weeks — so a crash
    /// loop already under way when crabmon starts is called one once it has
    /// restarted `crash_loop` more times. `Restarting` is reported straight
    /// away from `SubState=auto-restart`. A count that goes *down* is a unit
    /// somebody restarted by hand, and starts its history again.
    pub fn update(
        &mut self,
        units: &mut [UnitHealth],
        now: Instant,
        window: Duration,
        crash_loop: u64,
    ) {
        let mut present = std::collections::HashSet::new();
        for unit in units.iter_mut() {
            present.insert(unit.name.clone());
            let history = self.seen.entry(unit.name.clone()).or_default();
            if history.back().is_some_and(|(_, n)| unit.restarts < *n) {
                history.clear();
            }
            history.push_back((now, unit.restarts));
            // Keep one point at or before the window's start, so the count
            // is measured across the whole window rather than from inside it.
            while history.len() > 1
                && history.get(1).is_some_and(|(t, _)| now.duration_since(*t) >= window)
            {
                history.pop_front();
            }
            let oldest = history.front().map_or(unit.restarts, |(_, n)| *n);
            unit.recent_restarts = unit.restarts.saturating_sub(oldest);
            unit.window_secs = window.as_secs();
            unit.state = classify(unit, crash_loop);
        }
        self.seen.retain(|name, _| present.contains(name));
    }
}

fn classify(unit: &UnitHealth, crash_loop: u64) -> UnitState {
    if unit.active == "failed" {
        UnitState::Failed
    } else if crash_loop > 0 && unit.recent_restarts >= crash_loop {
        UnitState::CrashLoop
    } else if unit.sub == "auto-restart" || unit.recent_restarts > 0 {
        UnitState::Restarting
    } else {
        UnitState::Ok
    }
}

/// Whether a unit belongs in a snapshot: anything not healthy, or that has
/// restarted at all. The other hundred-odd units on a machine say nothing,
/// and every one of them would be in every frame of every recording.
pub fn is_notable(unit: &UnitHealth) -> bool {
    unit.state != UnitState::Ok || unit.restarts > 0
}

/// Settings the watcher runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitSettings {
    pub interval: Duration,
    pub window: Duration,
    pub crash_loop: u64,
    /// Hide credentials in journal lines before they leave this thread —
    /// and so before they reach a snapshot, `--remote`, a recording or a
    /// notification. See `export::redact_log`.
    pub redact: bool,
}

#[derive(Default)]
struct Shared {
    /// `None` until the first reading has been taken. The flag says whether
    /// its journals have been read too.
    latest: std::sync::Mutex<Option<Reading>>,
    ready: std::sync::Condvar,
}

/// One published reading: the notable units, why there are none when they
/// could not be read, and whether their journals are in yet.
type Reading = (Vec<UnitHealth>, Option<String>, bool);

/// Why there are no units yet on a host where they will be: temporary, so a
/// rule that waits on them is not refused for it.
pub const STILL_READING: &str = "systemd units are still being read";

/// How long the first reading is waited for, so `--once` has one.
const FIRST_WAIT: Duration = Duration::from_secs(2);

/// How to wake every unit watcher in the process for an immediate reading.
static WAKE: std::sync::Mutex<Vec<std::sync::mpsc::SyncSender<()>>> =
    std::sync::Mutex::new(Vec::new());

/// Read the units again now rather than at the next interval. Called after
/// crabmon restarts or resets a unit: the status line said `ok` while the
/// unit was still drawn failed for up to five seconds, found on a real host.
pub fn refresh_now() {
    if let Ok(mut all) = WAKE.lock() {
        // A full channel already has a reading queued; a closed one belongs
        // to a watcher that has stopped.
        all.retain(|w| {
            !matches!(w.try_send(()), Err(std::sync::mpsc::TrySendError::Disconnected(_)))
        });
    }
}

/// `systemctl show` on a thread of its own, every `interval`.
pub struct UnitWatcher {
    shared: std::sync::Arc<Shared>,
    started: Instant,
    /// Dropped with the watcher, which is how the thread learns to stop.
    _stop: std::sync::mpsc::Sender<()>,
}

impl UnitWatcher {
    pub fn spawn(settings: UnitSettings) -> UnitWatcher {
        let redact = settings.redact;
        Self::spawn_with(settings, query_systemctl, move |unit: &str| {
            redact_lines(read_log(unit, LOG_LINES), redact)
        })
    }

    fn spawn_with(
        settings: UnitSettings,
        query: impl Fn() -> Result<String, String> + Send + 'static,
        logs: impl Fn(&str) -> Vec<String> + Send + 'static,
    ) -> UnitWatcher {
        let shared = std::sync::Arc::new(Shared::default());
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let (wake, woken) = std::sync::mpsc::sync_channel::<()>(1);
        if let Ok(mut all) = WAKE.lock() {
            all.push(wake);
        }
        let worker = shared.clone();
        let _ = std::thread::Builder::new().name("crabmon-units".into()).spawn(move || {
            let mut tracker = RestartTracker::default();
            let mut cache = LogCache::default();
            let publish = |result: Result<Vec<UnitHealth>, String>, complete: bool| {
                if let Ok(mut latest) = worker.latest.lock() {
                    *latest = Some(match result {
                        Ok(units) => (units, None, complete),
                        Err(e) => (Vec::new(), Some(e), true),
                    });
                }
                worker.ready.notify_all();
            };
            loop {
                let result = query().map(|out| {
                    let mut units = parse_show(&out);
                    tracker.update(
                        &mut units,
                        Instant::now(),
                        settings.window,
                        settings.crash_loop,
                    );
                    units.retain(is_notable);
                    units.sort_by(|a, b| {
                        b.state
                            .severity()
                            .cmp(&a.state.severity())
                            .then(b.recent_restarts.cmp(&a.recent_restarts))
                            .then_with(|| a.name.cmp(&b.name))
                    });
                    units
                });
                match result {
                    Ok(mut units) => {
                        // The states first, with whatever journal is already
                        // known; then the journals that changed, if any.
                        if attach_cached_logs(&mut units, &cache) {
                            publish(Ok(units.clone()), false);
                            attach_logs(&mut units, &mut cache, &logs);
                        }
                        publish(Ok(units), true);
                    }
                    Err(e) => publish(Err(e), true),
                }
                // Asleep until the next reading is due, or until something
                // changed a unit and wants it read now.
                if let Err(std::sync::mpsc::RecvTimeoutError::Disconnected) =
                    woken.recv_timeout(settings.interval)
                {
                    std::thread::sleep(settings.interval);
                }
                if let Err(std::sync::mpsc::TryRecvError::Disconnected) = stopped.try_recv() {
                    break;
                }
            }
        });
        UnitWatcher { shared, started: Instant::now(), _stop: stop }
    }

    /// The last reading: the notable units, worst first, and why there is
    /// none when it could not be taken. Waits up to two seconds for the very
    /// first reading, so `--once` has one; never after that.
    ///
    /// For its first two seconds a watcher waits for a complete first
    /// reading, journals included, and settles for the states without them
    /// when they are slower than that. With nothing at all by then it says
    /// so: an empty list here read as "no unit has failed", which is what a
    /// busy host printed for a unit that had.
    pub fn latest(&self) -> (Vec<UnitHealth>, Option<String>) {
        let not_yet = || (Vec::new(), Some(STILL_READING.to_string()));
        let Ok(mut latest) = self.shared.latest.lock() else { return not_yet() };
        let deadline = self.started + FIRST_WAIT;
        while !matches!(&*latest, Some((_, _, true))) {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else { break };
            latest = match self.shared.ready.wait_timeout(latest, left) {
                Ok((guard, _)) => guard,
                Err(_) => return not_yet(),
            };
        }
        match &*latest {
            Some((units, err, _)) => (units.clone(), err.clone()),
            None => not_yet(),
        }
    }
}

fn query_systemctl() -> Result<String, String> {
    let out = std::process::Command::new("systemctl")
        .args([
            "show",
            "--no-pager",
            "-p",
            "Id,ActiveState,SubState,Result,NRestarts,ExecMainStatus",
            "*.service",
        ])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("systemctl: {e}"))?;
    if !out.status.success() {
        return Err("systemctl show failed: not a systemd machine, or no access to it".into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Whether `systemctl show` answers here, for `--doctor`.
pub fn probe() -> Result<usize, String> {
    query_systemctl().map(|out| parse_show(&out).len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHOW: &str = "Id=nginx.service\nActiveState=activating\nSubState=auto-restart\n\
Result=exit-code\nNRestarts=4\nExecMainStatus=1\n\n\
Id=sshd.service\nActiveState=active\nSubState=running\nResult=success\nNRestarts=0\n\
ExecMainStatus=0\n\n\
Id=backup.service\nActiveState=failed\nSubState=failed\nResult=exit-code\nNRestarts=0\n\
ExecMainStatus=2\n";

    /// A frame from an older crabmon has no `units_unreadable` field. Taking
    /// that as `None` would read as "systemd says nothing has failed".
    #[test]
    fn a_frame_without_unit_fields_is_not_reported_rather_than_healthy() {
        let old: crate::metrics::Snapshot =
            serde_json::from_str(r#"{"taken_at_unix": 5}"#).unwrap();
        assert!(old.units_unreadable.is_some());
        let mut new = crate::metrics::Snapshot { taken_at_unix: 5, ..Default::default() };
        new.units_unreadable = None;
        let back: crate::metrics::Snapshot =
            serde_json::from_str(&serde_json::to_string(&new).unwrap()).unwrap();
        assert_eq!(back.units_unreadable, None, "a read that found nothing stays read");
    }

    #[test]
    fn systemctl_show_parses_into_units() {
        let units = parse_show(SHOW);
        assert_eq!(units.len(), 3);
        assert_eq!(units[0].name, "nginx.service");
        assert_eq!(units[0].sub, "auto-restart");
        assert_eq!(units[0].restarts, 4);
        assert_eq!(units[2].exit_status, Some(2));
        assert!(parse_show("").is_empty());
        assert!(
            parse_show("garbage\n\nNRestarts=3\n").is_empty(),
            "a block with no Id is not a unit"
        );
    }

    fn unit(name: &str, active: &str, sub: &str, restarts: u64) -> UnitHealth {
        UnitHealth {
            name: name.into(),
            active: active.into(),
            sub: sub.into(),
            restarts,
            ..Default::default()
        }
    }

    #[test]
    fn restarts_inside_the_window_make_a_crash_loop_and_old_ones_age_out() {
        let window = Duration::from_secs(600);
        let t0 = Instant::now();
        let mut tracker = RestartTracker::default();
        let at = |secs| t0 + Duration::from_secs(secs);
        let mut step = |secs, restarts| {
            let mut u = vec![unit("api.service", "active", "running", restarts)];
            tracker.update(&mut u, at(secs), window, 3);
            u.remove(0)
        };
        // Seen first with 40 restarts since March: none of those are recent.
        assert_eq!(step(0, 40).state, UnitState::Ok);
        assert_eq!(step(60, 41).state, UnitState::Restarting);
        assert_eq!(step(120, 42).recent_restarts, 2);
        let looping = step(180, 43);
        assert_eq!((looping.state, looping.recent_restarts), (UnitState::CrashLoop, 3));
        // Quiet for a whole window: the loop is over.
        let later = step(900, 43);
        assert_eq!((later.state, later.recent_restarts), (UnitState::Ok, 0));
    }

    #[test]
    fn a_count_that_goes_down_is_a_manual_restart_not_negative_restarts() {
        let window = Duration::from_secs(600);
        let t0 = Instant::now();
        let mut tracker = RestartTracker::default();
        let mut u = vec![unit("api.service", "active", "running", 5)];
        tracker.update(&mut u, t0, window, 3);
        u[0].restarts = 0;
        tracker.update(&mut u, t0 + Duration::from_secs(5), window, 3);
        assert_eq!((u[0].state, u[0].recent_restarts), (UnitState::Ok, 0));
    }

    #[test]
    fn failed_and_waiting_to_restart_are_reported_without_any_history() {
        let mut units = parse_show(SHOW);
        RestartTracker::default().update(&mut units, Instant::now(), Duration::from_secs(600), 3);
        let state = |name: &str| units.iter().find(|u| u.name == name).unwrap().state;
        assert_eq!(state("nginx.service"), UnitState::Restarting);
        assert_eq!(state("backup.service"), UnitState::Failed);
        assert_eq!(state("sshd.service"), UnitState::Ok);
        let notable: Vec<_> =
            units.iter().filter(|u| is_notable(u)).map(|u| u.name.as_str()).collect();
        assert_eq!(notable, ["nginx.service", "backup.service"]);
    }

    #[test]
    fn a_unit_rule_counts_states_or_takes_the_worst_restart_count() {
        let mk = |name: &str, state, recent| UnitHealth {
            name: name.into(),
            state,
            recent_restarts: recent,
            ..Default::default()
        };
        let units = vec![
            mk("api.service", UnitState::CrashLoop, 6),
            mk("worker.service", UnitState::CrashLoop, 4),
            mk("backup.service", UnitState::Failed, 0),
        ];
        assert_eq!(measure(&units, "", ""), Some(2.0));
        assert_eq!(measure(&units, "crashloop", "api"), Some(1.0));
        assert_eq!(measure(&units, "failed", ""), Some(1.0));
        assert_eq!(measure(&units, "restarts", ""), Some(6.0));
        assert_eq!(measure(&units, "restarts", "worker.service"), Some(4.0));
        assert_eq!(measure(&units, "restarts", "nginx"), Some(0.0));
        assert_eq!(measure(&units, "bogus", ""), None);
    }

    #[test]
    fn an_alert_names_the_units_it_is_about() {
        let mk = |name: &str, state, recent| UnitHealth {
            name: name.into(),
            state,
            recent_restarts: recent,
            ..Default::default()
        };
        let units = vec![
            mk("a.service", UnitState::Failed, 0),
            mk("b.service", UnitState::Failed, 0),
            mk("loop.service", UnitState::CrashLoop, 9),
            mk("c.service", UnitState::Failed, 0),
            mk("d.service", UnitState::Failed, 0),
        ];
        assert_eq!(subject(&units, "failed", ""), " (a.service, b.service, c.service, +1 more)");
        assert_eq!(subject(&units, "", ""), " (loop.service)");
        assert_eq!(subject(&units, "restarts", ""), " (loop.service)");
        assert_eq!(subject(&units, "failed", "nope"), "");
    }

    #[test]
    fn logs_are_read_for_the_worst_units_only_and_quoted_in_alerts() {
        let mk = |name: &str, state| UnitHealth { name: name.into(), state, ..Default::default() };
        let mut units: Vec<UnitHealth> =
            (0..8).map(|i| mk(&format!("u{i}.service"), UnitState::Failed)).collect();
        units.push(mk("fine.service", UnitState::Ok));
        let read = |unit: &str| (1..=8).map(|n| format!("{unit} line {n}")).collect::<Vec<_>>();
        attach_logs(&mut units, &mut LogCache::default(), &read);
        assert_eq!(units.iter().filter(|u| !u.log_tail.is_empty()).count(), LOG_UNITS);
        assert!(units.last().unwrap().log_tail.is_empty(), "a healthy unit is not read");

        let d = detail(&units, "failed", "u0");
        assert_eq!(
            d,
            "u0.service:\nu0.service line 4\nu0.service line 5\nu0.service line 6\nu0.service line 7\nu0.service line 8"
        );
        assert_eq!(detail(&units, "failed", "u7"), "", "no tail was read for it");
        assert_eq!(detail(&units, "crashloop", ""), "", "nothing crash-looping");
    }

    #[test]
    fn journal_lines_lose_the_hostname_and_nothing_else() {
        assert_eq!(
            without_hostname(
                "2026-11-15T08:05:04+00:00 vps0000000.example.net api[7]: bind: in use"
            ),
            "2026-11-15T08:05:04+00:00 api[7]: bind: in use"
        );
        // Anything not shaped like a short-iso line is kept as it is.
        assert_eq!(without_hostname("-- Boot 1a2b --"), "-- Boot 1a2b --");
    }

    /// Found on a real host, where each `journalctl` took a second: a unit's
    /// journal is read again only when the unit did something new.
    #[test]
    fn a_journal_is_reread_only_when_its_unit_changes() {
        let reads = std::cell::Cell::new(0);
        let read = |unit: &str| {
            reads.set(reads.get() + 1);
            vec![format!("{unit} @ read {}", reads.get())]
        };
        let mut cache = LogCache::default();
        let mut units = vec![UnitHealth {
            name: "api.service".into(),
            state: UnitState::CrashLoop,
            restarts: 7,
            ..Default::default()
        }];
        attach_logs(&mut units, &mut cache, &read);
        attach_logs(&mut units, &mut cache, &read);
        assert_eq!(reads.get(), 1, "nothing changed, nothing read");
        assert!(!attach_cached_logs(&mut units, &cache), "and the cache can answer");
        units[0].restarts = 8;
        assert!(attach_cached_logs(&mut units, &cache), "a restart is news");
        attach_logs(&mut units, &mut cache, &read);
        assert_eq!(reads.get(), 2);
        assert_eq!(units[0].log_tail, ["api.service @ read 2"]);
        // A unit that has left is forgotten.
        attach_logs(&mut [], &mut cache, &read);
        assert!(cache.0.is_empty());
    }

    /// Found on a busy real host: the first reading took longer than the
    /// two seconds `--once` waits, and the empty answer read as "nothing has
    /// failed". The states come first; and nothing at all says so.
    #[test]
    fn a_slow_first_reading_is_never_reported_as_healthy() {
        let settings = UnitSettings {
            interval: Duration::from_secs(600),
            window: Duration::from_secs(600),
            crash_loop: 3,
            redact: true,
        };
        // Journals slower than the wait: the states arrive without them.
        let slow_logs = UnitWatcher::spawn_with(
            settings,
            || Ok(SHOW.to_string()),
            |_: &str| {
                std::thread::sleep(Duration::from_secs(4));
                vec!["late".into()]
            },
        );
        let (units, err) = slow_logs.latest();
        assert_eq!(err, None);
        assert!(units.iter().any(|u| u.name == "backup.service" && u.state == UnitState::Failed));
        assert!(units.iter().all(|u| u.log_tail.is_empty()), "the journals were not waited for");

        // systemd itself slower than the wait: not read, rather than healthy.
        let slow_show = UnitWatcher::spawn_with(
            settings,
            || {
                std::thread::sleep(Duration::from_secs(4));
                Ok(SHOW.to_string())
            },
            |_: &str| Vec::new(),
        );
        let (units, err) = slow_show.latest();
        assert!(units.is_empty());
        assert_eq!(err.as_deref(), Some("systemd units are still being read"));
    }

    #[test]
    fn a_watcher_reads_again_at_once_when_asked() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = calls.clone();
        let settings = UnitSettings {
            interval: Duration::from_secs(600),
            window: Duration::from_secs(600),
            crash_loop: 3,
            redact: true,
        };
        let w = UnitWatcher::spawn_with(
            settings,
            move || {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(SHOW.to_string())
            },
            |_: &str| Vec::new(),
        );
        w.latest();
        let before = calls.load(std::sync::atomic::Ordering::SeqCst);
        refresh_now();
        let deadline = Instant::now() + Duration::from_secs(10);
        while calls.load(std::sync::atomic::Ordering::SeqCst) == before && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(calls.load(std::sync::atomic::Ordering::SeqCst) > before, "not re-read");
    }

    #[test]
    fn short_forms_keep_the_reason() {
        let mut u = unit("x", "failed", "failed", 0);
        u.state = UnitState::Failed;
        u.result = "oom-kill".into();
        assert_eq!(u.short(), "oom-kill");
        u.result = "exit-code".into();
        u.exit_status = Some(3);
        assert_eq!(u.short(), "exit 3");
        u.state = UnitState::CrashLoop;
        u.recent_restarts = 15;
        assert_eq!(u.short(), "loop 15×");
    }

    #[test]
    fn summaries_say_the_state_and_the_reason() {
        let mut u = unit("x", "failed", "failed", 0);
        u.state = UnitState::Failed;
        u.result = "exit-code".into();
        u.exit_status = Some(1);
        assert_eq!(u.summary(), "failed (exit 1)");
        u.exit_status = Some(0);
        assert_eq!(u.summary(), "failed (exit-code)", "status 0 is not why it failed");
        u.result = "start-limit-hit".into();
        assert_eq!(u.summary(), "failed (start-limit-hit)");
        u.state = UnitState::CrashLoop;
        u.recent_restarts = 7;
        assert_eq!(u.summary(), "crash-loop 7×/10m", "no window recorded: the default");
        // The window the counting crabmon used, not the reader's.
        u.window_secs = 1800;
        assert_eq!(u.summary(), "crash-loop 7×/30m");
    }

    #[test]
    fn the_watcher_publishes_notable_units_worst_first_and_errors_as_such() {
        let settings = UnitSettings {
            interval: Duration::from_millis(10),
            window: Duration::from_secs(600),
            crash_loop: 3,
            redact: true,
        };
        let w = UnitWatcher::spawn_with(
            settings,
            || Ok(SHOW.to_string()),
            |unit: &str| vec![format!("{unit} said goodbye")],
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while w.latest().0.is_empty() && Instant::now() < deadline {
            // `latest` waits for the first reading itself; this is belt and
            // braces for a loaded runner.
            std::thread::sleep(Duration::from_millis(5));
        }
        let names: Vec<String> = w.latest().0.into_iter().map(|u| u.name).collect();
        assert_eq!(names, ["backup.service", "nginx.service"]);

        // Both troubled units carry their journal, worst first.
        let tails: Vec<Vec<String>> = w.latest().0.into_iter().map(|u| u.log_tail).collect();
        assert_eq!(
            tails,
            [vec!["backup.service said goodbye"], vec!["nginx.service said goodbye"]]
        );

        let broken =
            UnitWatcher::spawn_with(settings, || Err("no systemd".into()), |_: &str| Vec::new());
        let deadline = Instant::now() + Duration::from_secs(10);
        while broken.latest().1.is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(broken.latest().1.as_deref(), Some("no systemd"));
    }
}
