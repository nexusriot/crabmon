//! The live events timeline: what happened, in order, on every host watched.
//!
//! Everything on the dashboard is *now*. The kernel panel keeps its last few
//! complaints, the `!` popup its rule transitions, the action log what was
//! done — but "the disk started failing, then the database unit restarted
//! twice, then the OOM killer took the worker, then someone restarted it" is
//! one story told across four places, and in a fleet across several hosts.
//! This is that story in one list, newest first, behind `E`.
//!
//! It is derived entirely from what crabmon already samples: each snapshot is
//! compared with the last one from the same host. Nothing new is read or
//! recorded, so a replay rebuilds the same timeline from the recording, and
//! `--remote` and fleet hosts get one with no change at the far end.
//!
//! Bounded at `CAPACITY` events; the oldest go first.

use std::collections::{HashMap, VecDeque};

use crate::alerts::AlertEngine;
use crate::metrics::Snapshot;
use crate::metrics::kmsg::EventKind;
use crate::metrics::storage::ArrayState;
use crate::metrics::units::UnitState;

/// Where an event came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Alert,
    Kernel,
    Unit,
    Storage,
    Host,
    Action,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Alert => "alert",
            Source::Kernel => "kernel",
            Source::Unit => "unit",
            Source::Storage => "storage",
            Source::Host => "host",
            Source::Action => "action",
        }
    }
}

/// How it should be coloured: a recovery, a note, a warning, trouble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Ok,
    Info,
    Warn,
    Crit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub at_unix: u64,
    pub source: Source,
    pub level: Level,
    /// The fleet target it happened on; empty for a single host.
    pub host: String,
    pub text: String,
}

/// Events kept. A busy night of flapping rules fits; a runaway cannot grow
/// the process.
pub const CAPACITY: usize = 500;

/// What was last seen from one host, to compare the next snapshot with.
#[derive(Debug, Default)]
struct HostState {
    /// A first snapshot has been taken in. Until then, the state a host is
    /// already in is reported as such — "at start" — rather than as a change.
    seen: bool,
    kernel_seq: Option<u64>,
    units: HashMap<String, (UnitState, u64)>,
    storage: HashMap<(String, String), ArrayState>,
    alert_seq: u64,
    live: Option<bool>,
}

#[derive(Debug, Default)]
pub struct Timeline {
    /// Oldest first, ordered by time.
    events: VecDeque<Event>,
    hosts: HashMap<String, HostState>,
}

impl Timeline {
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Newest first, as the popup lists them.
    pub fn newest_first(&self) -> impl Iterator<Item = &Event> {
        self.events.iter().rev()
    }

    /// Forget everything: a replay that jumps has not watched what lies
    /// between, and must not report the jump as a burst of changes.
    pub fn reset(&mut self) {
        self.events.clear();
        self.hosts.clear();
    }

    /// Insert in time order. A kernel backlog read at start carries times
    /// from before it, and belongs before what has been seen since.
    pub fn push(&mut self, event: Event) {
        let at = self.events.partition_point(|e| e.at_unix <= event.at_unix);
        self.events.insert(at, event);
        while self.events.len() > CAPACITY {
            self.events.pop_front();
        }
    }

    fn add(&mut self, at_unix: u64, source: Source, level: Level, host: &str, text: String) {
        self.push(Event { at_unix, source, level, host: host.to_string(), text });
    }

    /// Something done from the interface: a signal, a restart, a silence.
    pub fn action(&mut self, at_unix: u64, ok: bool, text: String) {
        let level = if ok { Level::Info } else { Level::Warn };
        self.add(at_unix, Source::Action, level, "", text);
    }

    /// Compare a host's new snapshot with its last one.
    pub fn observe(&mut self, host: &str, snap: &Snapshot) {
        let at = snap.taken_at_unix;
        let mut state = self.hosts.remove(host).unwrap_or_default();
        let first = !state.seen;
        let note = |text: String| if first { format!("{text} (at start)") } else { text };

        // Kernel events carry their own sequence number. The ring read at
        // start is history, timed by boot time plus the kernel's own stamp;
        // anything after that is timed by the sample that first saw it. The
        // kernel's printk clock drifts from wall time — 17 s behind on a
        // two-week-old VM — which put a segfault before the unit failure
        // that came first. One refresh late is the better error.
        let boot = snap.host.boot_time_unix;
        for e in &snap.kernel {
            if state.kernel_seq.is_some_and(|seen| e.seq <= seen) {
                continue;
            }
            let when = match (first, boot) {
                (true, b) if b > 0 => (b + e.monotonic_us / 1_000_000).min(at),
                _ => at,
            };
            let level = match e.kind {
                EventKind::Segfault => Level::Warn,
                _ => Level::Crit,
            };
            self.add(when, Source::Kernel, level, host, kernel_text(e));
        }
        if let Some(max) = snap.kernel.iter().map(|e| e.seq).max() {
            state.kernel_seq = Some(state.kernel_seq.map_or(max, |s| s.max(max)));
        }

        // Units. A snapshot without a reading — still starting, or systemd
        // not readable — says nothing about them, so it changes nothing.
        if snap.units_unreadable.is_none() {
            let mut now: HashMap<String, (UnitState, u64)> = HashMap::new();
            for u in &snap.units {
                now.insert(u.name.clone(), (u.state, u.restarts));
                let level = match u.state {
                    UnitState::Ok => Level::Ok,
                    UnitState::Restarting => Level::Warn,
                    _ => Level::Crit,
                };
                match state.units.get(&u.name) {
                    None if u.state != UnitState::Ok => {
                        let text = format!("{} {}", u.name, u.summary());
                        self.add(at, Source::Unit, level, host, note(text));
                    }
                    Some((was, _)) if *was != u.state => {
                        self.add(
                            at,
                            Source::Unit,
                            level,
                            host,
                            format!("{} {}", u.name, u.summary()),
                        );
                    }
                    Some((_, restarts)) if u.restarts > *restarts => {
                        let text = format!("{} restarted ({}×)", u.name, u.restarts);
                        self.add(at, Source::Unit, Level::Info, host, text);
                    }
                    _ => {}
                }
            }
            // A unit drops out of the snapshot once it is healthy and has
            // not restarted, so one that was in trouble and is gone is back.
            for (name, (was, _)) in &state.units {
                if *was != UnitState::Ok && !now.contains_key(name) {
                    self.add(at, Source::Unit, Level::Ok, host, format!("{name} ok"));
                }
            }
            state.units = now;
        }

        // Arrays and SMART disks. One that stops being reported — a disk
        // gone from the scan — is left alone rather than declared healthy.
        for a in &snap.storage {
            let key = (a.kind.clone(), a.name.clone());
            let level = match a.state {
                ArrayState::Healthy => Level::Ok,
                ArrayState::Rebuilding => Level::Warn,
                _ => Level::Crit,
            };
            let text = match a.detail.is_empty() {
                true => format!("{} {} {}", a.kind, a.name, a.state.label()),
                false => format!("{} {} {} {}", a.kind, a.name, a.state.label(), a.detail),
            };
            match state.storage.get(&key) {
                None if a.state.is_unhealthy() => {
                    self.add(at, Source::Storage, level, host, note(text));
                }
                Some(was) if *was != a.state => self.add(at, Source::Storage, level, host, text),
                _ => {}
            }
            state.storage.insert(key, a.state);
        }

        state.seen = true;
        self.hosts.insert(host.to_string(), state);
    }

    /// Rule transitions the host's engine has recorded since the last call.
    pub fn alerts(&mut self, host: &str, engine: &AlertEngine) {
        let seen = self.hosts.get(host).map_or(0, |s| s.alert_seq);
        let mut new: Vec<_> = engine.history().filter(|e| e.seq > seen).cloned().collect();
        new.reverse();
        let Some(last) = new.last().map(|e| e.seq) else { return };
        for e in new {
            let (level, verb) = match (e.fired, e.silenced) {
                (true, false) => (Level::Crit, "fired"),
                (true, true) => (Level::Warn, "fired (silenced)"),
                (false, _) => (Level::Ok, "cleared"),
            };
            let text = format!("{} {verb} at {}{}", e.name, trim_value(e.value), e.subject);
            self.add(e.at_unix, Source::Alert, level, host, text);
        }
        self.hosts.entry(host.to_string()).or_default().alert_seq = last;
    }

    /// A fleet host started or stopped answering.
    pub fn host_live(&mut self, host: &str, live: bool, why: Option<&str>, at_unix: u64) {
        let state = self.hosts.entry(host.to_string()).or_default();
        let was = state.live.replace(live);
        // The first answer is not news, a first failure is.
        let event = match (was, live) {
            (Some(true), false) | (None, false) => Some((
                Level::Crit,
                match why {
                    Some(why) => format!("not answering: {why}"),
                    None => "not answering".to_string(),
                },
            )),
            (Some(false), true) => Some((Level::Ok, "answering again".to_string())),
            _ => None,
        };
        if let Some((level, text)) = event {
            self.add(at_unix, Source::Host, level, host, text);
        }
    }
}

/// `OOM: python3 (482060) in cm-e2e.service`, the way the Kernel panel
/// names it; the raw line where it names no process. The raw OOM line is
/// `oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),...`, which is
/// unreadable in a list.
fn kernel_text(e: &crate::metrics::kmsg::KernelEvent) -> String {
    let what = match (&e.process, e.pid) {
        (Some(name), Some(pid)) => format!("{name} ({pid})"),
        (Some(name), None) => name.clone(),
        (None, _) => return format!("{}: {}", e.kind.label(), e.text),
    };
    match e.memcg() {
        Some(group) if e.kind == EventKind::OomKill => {
            format!("{}: {what} in {group}", e.kind.label())
        }
        _ => format!("{}: {what}", e.kind.label()),
    }
}

/// `91.3`, `2`, `12500000`: enough to recognise, without noise.
fn trim_value(v: f64) -> String {
    if v.fract().abs() < 1e-9 || v.abs() >= 1000.0 { format!("{v:.0}") } else { format!("{v:.1}") }
}

/// `14:02:11` UTC, for the popup's first column. UTC like every other time
/// crabmon prints, so a fleet's hosts and a silence window agree.
pub fn clock(unix: u64) -> String {
    let s = unix % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
}

/// `3m`, `2h`: how long before `now` it happened. `now` is the snapshot's
/// own time, so a replay says how long before the frame on screen.
pub fn ago(at: u64, now: u64) -> String {
    let d = now.saturating_sub(at);
    match d {
        0..=59 => format!("{d}s"),
        60..=3599 => format!("{}m", d / 60),
        3600..=86_399 => format!("{}h", d / 3600),
        _ => format!("{}d", d / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::kmsg::KernelEvent;
    use crate::metrics::storage::StorageArray;
    use crate::metrics::units::UnitHealth;

    fn snap(at: u64) -> Snapshot {
        Snapshot { taken_at_unix: at, units_unreadable: None, ..Default::default() }
    }

    fn unit(name: &str, state: UnitState, restarts: u64) -> UnitHealth {
        UnitHealth {
            name: name.into(),
            active: if state == UnitState::Failed { "failed".into() } else { "active".into() },
            state,
            restarts,
            ..Default::default()
        }
    }

    fn texts(t: &Timeline) -> Vec<String> {
        t.newest_first().map(|e| e.text.clone()).collect()
    }

    #[test]
    fn unit_trouble_present_at_start_is_marked_as_such_and_changes_are_not() {
        let mut t = Timeline::default();
        let mut s = snap(100);
        s.units = vec![unit("db.service", UnitState::Failed, 0)];
        t.observe("", &s);
        assert_eq!(texts(&t), ["db.service failed (at start)"], "the unit is named");

        // Same state again: nothing new.
        t.observe("", &s);
        assert_eq!(t.len(), 1);

        // It recovers and drops out of the snapshot.
        t.observe("", &snap(110));
        assert_eq!(texts(&t)[0], "db.service ok");
        assert_eq!(t.newest_first().next().unwrap().level, Level::Ok);
    }

    #[test]
    fn a_restart_and_a_state_change_are_each_one_event() {
        let mut t = Timeline::default();
        let mut s = snap(100);
        s.units = vec![unit("api.service", UnitState::Ok, 1)];
        t.observe("", &s);
        assert!(t.is_empty(), "a healthy unit at start is not news");
        s.units = vec![unit("api.service", UnitState::Ok, 2)];
        t.observe("", &s);
        assert_eq!(texts(&t), ["api.service restarted (2×)"]);
        s.units = vec![unit("api.service", UnitState::CrashLoop, 5)];
        t.observe("", &s);
        assert_eq!(t.len(), 2);
        assert_eq!(t.newest_first().next().unwrap().level, Level::Crit);
    }

    #[test]
    fn an_unread_unit_list_is_not_every_unit_recovering() {
        let mut t = Timeline::default();
        let mut s = snap(100);
        s.units = vec![unit("db.service", UnitState::Failed, 0)];
        t.observe("", &s);
        let mut unread = snap(105);
        unread.units_unreadable = Some("still reading".into());
        t.observe("", &unread);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn kernel_events_are_taken_once_each_at_their_own_time() {
        let mut t = Timeline::default();
        let mut s = snap(10_000);
        s.host.boot_time_unix = 1_000;
        let ev = |seq: u64, secs: u64, kind| KernelEvent {
            kind,
            seq,
            monotonic_us: secs * 1_000_000,
            text: format!("event {seq}"),
            pid: None,
            process: None,
        };
        s.kernel = vec![ev(1, 100, EventKind::OomKill), ev(2, 200, EventKind::Segfault)];
        t.observe("", &s);
        assert_eq!(t.len(), 2);
        assert_eq!(t.newest_first().last().unwrap().at_unix, 1_100, "boot time plus uptime");
        t.observe("", &s);
        assert_eq!(t.len(), 2, "seen once");
        // Seen live: timed by the sample, not by the drifting printk clock,
        // whose stamp here would put it 8000 s before the frame.
        s.taken_at_unix = 10_005;
        s.kernel.push(ev(3, 1_000, EventKind::IoError));
        t.observe("", &s);
        assert_eq!(texts(&t)[0], "I/O: event 3");
        assert_eq!(t.newest_first().next().unwrap().at_unix, 10_005);
    }

    #[test]
    fn an_oom_kill_is_named_by_victim_and_cgroup_not_by_its_raw_line() {
        let mut t = Timeline::default();
        let mut s = snap(10_000);
        s.kernel = vec![KernelEvent {
            kind: EventKind::OomKill,
            seq: 1,
            monotonic_us: 0,
            text: "oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=cm.service,\
                   mems_allowed=0,oom_memcg=/system.slice/cm.service,\
                   task_memcg=/system.slice/cm.service,task=python3,pid=482060,uid=0"
                .into(),
            pid: Some(482_060),
            process: Some("python3".into()),
        }];
        t.observe("", &s);
        assert_eq!(texts(&t), ["OOM: python3 (482060) in cm.service"]);
    }

    #[test]
    fn storage_changes_are_reported_and_a_degraded_array_at_start_too() {
        let mut t = Timeline::default();
        let row = |name: &str, state| StorageArray {
            kind: "smart".into(),
            name: name.into(),
            state,
            ..Default::default()
        };
        let mut s = snap(100);
        s.storage = vec![row("sda", ArrayState::Healthy), row("sdb", ArrayState::Degraded)];
        t.observe("", &s);
        assert_eq!(texts(&t), ["smart sdb degraded (at start)"]);
        s.storage = vec![row("sda", ArrayState::Failed), row("sdb", ArrayState::Degraded)];
        t.observe("", &s);
        assert_eq!(texts(&t)[0], "smart sda failed");
    }

    #[test]
    fn hosts_are_tracked_apart_and_going_away_is_news() {
        let mut t = Timeline::default();
        let mut s = snap(100);
        s.units = vec![unit("db.service", UnitState::Failed, 0)];
        t.observe("db-1", &s);
        t.observe("web-1", &snap(100));
        assert_eq!(t.newest_first().next().unwrap().host, "db-1");

        t.host_live("web-1", true, None, 100);
        assert_eq!(t.len(), 1, "the first answer is not news");
        t.host_live("web-1", false, Some("timed out"), 120);
        t.host_live("web-1", false, Some("timed out"), 125);
        t.host_live("web-1", true, None, 130);
        assert_eq!(texts(&t)[..2], ["answering again", "not answering: timed out"]);
    }

    #[test]
    fn events_are_kept_in_time_order_and_bounded() {
        let mut t = Timeline::default();
        t.action(50, true, "late".into());
        t.action(10, true, "early".into());
        assert_eq!(texts(&t), ["late", "early"]);
        for i in 0..CAPACITY + 10 {
            t.action(100 + i as u64, true, format!("{i}"));
        }
        assert_eq!(t.len(), CAPACITY);
        assert!(!texts(&t).contains(&"early".to_string()), "the oldest go first");
    }

    #[test]
    fn rule_transitions_arrive_once_each() {
        let rule = crate::watch::parse_rule("cpu>=90", 0).unwrap();
        let mut engine = AlertEngine::new(vec![rule]);
        let mut t = Timeline::default();
        let mut hot = snap(100);
        hot.cpu.per_core = vec![99.0];
        engine.evaluate(&hot, 0);
        t.alerts("", &engine);
        t.alerts("", &engine);
        assert_eq!(t.len(), 1);
        assert!(texts(&t)[0].contains("fired at 99"), "{:?}", texts(&t));
        let mut cool = snap(110);
        cool.cpu.per_core = vec![1.0];
        engine.evaluate(&cool, 10);
        t.alerts("", &engine);
        assert_eq!(texts(&t)[0], format!("{} cleared at 1", engine.rules[0].name));
    }

    #[test]
    fn a_storage_alert_says_which_disk_when_it_fires_and_when_it_clears() {
        let rule = crate::watch::parse_rule("storage>=1", 0).unwrap();
        let mut engine = AlertEngine::new(vec![rule]);
        let mut t = Timeline::default();
        let mut bad = snap(100);
        bad.storage = vec![StorageArray {
            kind: "smart".into(),
            name: "sdz".into(),
            state: ArrayState::Degraded,
            detail: "8 reallocated 38°C".into(),
            ..Default::default()
        }];
        engine.evaluate(&bad, 0);
        t.alerts("", &engine);
        engine.evaluate(&snap(110), 10);
        t.alerts("", &engine);
        let texts = texts(&t);
        assert!(texts[1].ends_with("fired at 1 (sdz degraded 8 reallocated)"), "{texts:?}");
        assert!(texts[0].ends_with("cleared at 0 (sdz degraded 8 reallocated)"), "{texts:?}");
    }

    #[test]
    fn times_are_utc_clock_and_short_ages() {
        assert_eq!(clock(86_400 + 3_661), "01:01:01");
        assert_eq!(ago(100, 100), "0s");
        assert_eq!(ago(100, 400), "5m");
        assert_eq!(ago(0, 7_200), "2h");
    }
}
