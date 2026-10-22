//! Threshold alerts. A rule fires only after its condition has held for
//! `for_secs`, which keeps a single busy frame from lighting up the status bar.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::filter;
use crate::metrics::Snapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertKind {
    Cpu,
    Mem,
    Swap,
    Load,
    Disk,
    Temp,
    Gpu,
    /// Pressure-stall percentage. `target` names the resource and the line:
    /// `cpu`, `mem`, `io`, or one of those suffixed `.full`.
    Psi,
    /// Busiest block device's utilisation. Unlike `disk`, which is capacity,
    /// this is the figure that says the device is the bottleneck.
    Io,
    /// Aggregate network throughput in bytes per second.
    Net,
    /// The closest any process is to its own descriptor limit. Needs
    /// `[procs] fds`.
    Fd,
    /// How many processes match `query`. With `below`, this is how you alert
    /// on something having *stopped*.
    Proc,
    /// Growth of a series over time rather than its level: "this process's
    /// RSS has risen by 200 MB in the last ten minutes".
    ///
    /// `target` names the series — `mem` or `swap` for the machine — or a
    /// `query` selects processes and the worst grower among them is the one
    /// measured. `threshold` is in bytes and `over_secs` is the window.
    Trend,
    /// How close a container or unit is to *its own* cgroup memory limit,
    /// as a percentage.
    ///
    /// `target` names a container id or a unit; empty means whichever is
    /// closest to its limit. The host's memory gauge cannot ask this: a
    /// container at 98% of a 512 MB limit is about to be OOM-killed on a
    /// machine showing 30% memory used.
    Cgroup,
    /// Seconds until a filesystem is full at the rate it is filling.
    ///
    /// `target` names a mount; empty means whichever will fill first. Use
    /// `below`, since the alarming direction is *less* time: "under three
    /// hours" is `threshold = 10800` with `below = true`.
    FillIn,
    /// Kernel events in the current sample. `target` names a kind — `oom`,
    /// `segfault`, `io-error`, `oops` — and an empty target counts all of
    /// them.
    ///
    /// This is the rule nothing else could express: every other kind
    /// measures a level, and "a process was killed twenty minutes ago"
    /// leaves no level behind to measure. Memory is back to normal by the
    /// time anyone looks, which is exactly why the kill went unexplained.
    Kernel,
}

/// Every kind a rule can be, so the docs can be checked against the code
/// rather than against a hand-maintained list — the same reason `ALL_SORTS`
/// and `ALL_GROUPINGS` exist.
pub const ALL_ALERT_KINDS: [AlertKind; 16] = [
    AlertKind::Cpu,
    AlertKind::Mem,
    AlertKind::Swap,
    AlertKind::Load,
    AlertKind::Disk,
    AlertKind::Temp,
    AlertKind::Gpu,
    AlertKind::Psi,
    AlertKind::Io,
    AlertKind::Net,
    AlertKind::Fd,
    AlertKind::Proc,
    AlertKind::Trend,
    AlertKind::FillIn,
    AlertKind::Cgroup,
    AlertKind::Kernel,
];

impl AlertKind {
    /// The spelling `kind = "..."` takes, which is also what serde writes.
    pub fn key_name(self) -> &'static str {
        match self {
            AlertKind::Cpu => "cpu",
            AlertKind::Mem => "mem",
            AlertKind::Swap => "swap",
            AlertKind::Load => "load",
            AlertKind::Disk => "disk",
            AlertKind::Temp => "temp",
            AlertKind::Gpu => "gpu",
            AlertKind::Psi => "psi",
            AlertKind::Io => "io",
            AlertKind::Net => "net",
            AlertKind::Fd => "fd",
            AlertKind::Proc => "proc",
            AlertKind::Trend => "trend",
            AlertKind::FillIn => "fillin",
            AlertKind::Cgroup => "cgroup",
            AlertKind::Kernel => "kernel",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub kind: AlertKind,
    /// Percent for cpu/mem/swap/disk/gpu/psi/io/fd, degrees for temp, bytes
    /// per second for net, a count for proc, absolute for load.
    pub threshold: f64,
    #[serde(default)]
    pub for_secs: u64,
    /// The window a `trend` rule measures growth over, in seconds. Ignored
    /// by every other kind. 0 means `DEFAULT_TREND_WINDOW`.
    ///
    /// Distinct from `for_secs`, which is how long the *answer* has to hold
    /// before the rule fires: a leak alert wants both — measured over ten
    /// minutes, and still true two minutes later.
    #[serde(default)]
    pub over_secs: u64,
    /// Fire when the measurement is *strictly under* the threshold instead of
    /// at or over it.
    ///
    /// Without this there is no way to say "fewer than one nginx", which is
    /// the condition anyone actually wants to be paged about. The comparison
    /// is strict where the default is inclusive, so `threshold = 1` on a
    /// `proc` rule means "none running" rather than "one or fewer".
    #[serde(default)]
    pub below: bool,
    /// Restricts disk rules to a mount point, temp rules to a sensor prefix,
    /// psi rules to a resource, net rules to an interface and fd rules to a
    /// process name.
    #[serde(default)]
    pub target: String,
    /// The filter query a `proc` rule counts matches of. Ignored by every
    /// other kind.
    #[serde(default)]
    pub query: String,
    /// Shell command run once when the alert becomes active.
    #[serde(default)]
    pub command: String,
    /// Shell command run once when it recovers. Without this an alert fires,
    /// pages someone, and never says it is over.
    #[serde(default)]
    pub command_clear: String,
}

/// A rule that is currently over threshold and past its hold time.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveAlert {
    pub name: String,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Default)]
struct Tracked {
    /// Monotonic seconds at which the condition first held.
    over_since: Option<u64>,
    fired: bool,
}

/// A rule crossing into or out of its alarm state.
#[derive(Debug, Clone, PartialEq)]
pub struct AlertEvent {
    pub name: String,
    pub fired: bool,
    /// Wall-clock time of the snapshot that caused the transition, for display.
    pub at_unix: u64,
    pub value: f64,
}

/// Transitions kept for the `!` popup. Enough to cover a night, small enough
/// that a flapping rule cannot grow the process without bound.
pub const MAX_HISTORY: usize = 200;

#[derive(Debug, Default)]
pub struct AlertEngine {
    pub rules: Vec<AlertRule>,
    /// Per-rule state, positionally aligned with `rules`.
    ///
    /// Keyed by position rather than by name because nothing makes names
    /// unique. Two rules called the same thing — the obvious pair being one
    /// over a threshold and one under it, now that `below` exists — shared a
    /// `fired` flag, so each cleared the other's every refresh and the hook
    /// command re-spawned on every single sample.
    state: Vec<Tracked>,
    /// `proc` rules' queries, parsed once. Re-parsing a regex per rule per
    /// refresh is not free, and a query that fails to parse must be inert
    /// rather than an error on every frame.
    queries: Vec<Option<filter::Filter>>,
    /// Commands the engine wants run; drained by the caller.
    pending_commands: Vec<String>,
    /// Rules that have just gone active, for the flight recorder. Kept apart
    /// from `pending_commands` because a rule with no hook still wants its
    /// dump — the hook is for paging someone, the dump is the evidence.
    pending_fired: Vec<String>,
    /// Newest-last transitions, bounded by `MAX_HISTORY`.
    history: Vec<AlertEvent>,
    /// Series behind the `trend` rules. Kept by the engine rather than by
    /// the TUI so that a leak is detected identically under `--serve
    /// --alerts` and `--stream --alerts`, where there is no `App` at all.
    trends: TrendTracker,
}

impl AlertEngine {
    /// Missing observations cannot establish a continuous hold. Already
    /// firing rules retain their state until a real recovery is observed.
    pub fn interrupt(&mut self) {
        for state in &mut self.state {
            if !state.fired {
                state.over_since = None;
            }
        }
    }

    pub fn new(rules: Vec<AlertRule>) -> Self {
        let state = vec![Tracked::default(); rules.len()];
        let queries = rules
            .iter()
            // `trend` rules carry a query for the same reason `proc` rules
            // do — it selects the processes being measured — so it has to be
            // parsed here too, or the rule silently measures the machine
            // instead of the processes it named.
            .map(|r| {
                matches!(r.kind, AlertKind::Proc | AlertKind::Trend)
                    .then(|| filter::parse(&r.query).ok())
                    .flatten()
            })
            .collect();
        Self {
            rules,
            state,
            queries,
            pending_commands: Vec::new(),
            pending_fired: Vec::new(),
            history: Vec::new(),
            trends: TrendTracker::default(),
        }
    }

    /// Evaluate every rule against `snap`. `now_secs` is a monotonic clock,
    /// injected so the hold-time logic is testable.
    pub fn evaluate(&mut self, snap: &Snapshot, now_secs: u64) -> Vec<ActiveAlert> {
        let mut active = Vec::new();
        let at_unix = snap.taken_at_unix;
        let mut events: Vec<AlertEvent> = Vec::new();
        self.observe_trends(snap);
        for (i, rule) in self.rules.iter().enumerate() {
            let Some((value, unit)) = measure(
                rule,
                snap,
                self.queries.get(i).and_then(|q| q.as_ref()),
                Some(&self.trends),
            ) else {
                continue;
            };
            let st = &mut self.state[i];
            // `below` inverts the comparison, so one code path covers "too
            // much CPU" and "fewer than one nginx". Strict, so that the
            // obvious `threshold = 1` on a `proc` rule means "none left"
            // rather than firing while the one process is still running.
            let over = if rule.below { value < rule.threshold } else { value >= rule.threshold };
            if !over {
                st.over_since = None;
                // Recovering is a transition worth reporting: an alert that
                // only ever says "started" leaves you guessing when it ended.
                if std::mem::take(&mut st.fired) {
                    if !rule.command_clear.is_empty() {
                        self.pending_commands.push(rule.command_clear.clone());
                    }
                    events.push(AlertEvent {
                        name: rule.name.clone(),
                        fired: false,
                        at_unix,
                        value,
                    });
                }
                continue;
            }
            let since = *st.over_since.get_or_insert(now_secs);
            if now_secs.saturating_sub(since) < rule.for_secs {
                continue;
            }
            if !st.fired {
                st.fired = true;
                if !rule.command.is_empty() {
                    self.pending_commands.push(rule.command.clone());
                }
                self.pending_fired.push(rule.name.clone());
                events.push(AlertEvent { name: rule.name.clone(), fired: true, at_unix, value });
            }
            // The process that is actually growing, when the rule is about
            // processes: "leak ≥ 200 MiB" is a fact about the machine, and
            // "leak: postgres +412 MiB in 10m" is something to act on.
            let subject = match rule.kind {
                AlertKind::Trend => self
                    .queries
                    .get(i)
                    .and_then(|q| q.as_ref())
                    .and_then(|f| self.trends.worst_proc_growth(f, snap, rule.window(), rule.below))
                    .map(|(_, span, name)| {
                        format!(" ({name} over {})", crate::format::human_duration(span))
                    })
                    .unwrap_or_default(),
                _ => String::new(),
            };
            active.push(ActiveAlert {
                name: rule.name.clone(),
                message: format!(
                    "{} {} {} {}{subject}",
                    rule.name,
                    quantity(value, unit),
                    if rule.below { "<" } else { "≥" },
                    quantity(rule.threshold, unit),
                ),
                value,
                threshold: rule.threshold,
            });
        }
        self.record(events);
        active
    }

    fn record(&mut self, events: Vec<AlertEvent>) {
        if events.is_empty() {
            return;
        }
        self.history.extend(events);
        let overflow = self.history.len().saturating_sub(MAX_HISTORY);
        if overflow > 0 {
            self.history.drain(..overflow);
        }
    }

    /// Feed the sample to the trend series, if any rule needs them.
    ///
    /// Per-process series are only kept when a `trend` rule actually has a
    /// query: tracking every process costs a map entry per process per
    /// sample, and most configurations never ask for it.
    fn observe_trends(&mut self, snap: &Snapshot) {
        let windowed =
            |kind: AlertKind| self.rules.iter().filter(move |r| r.kind == kind).map(|r| r.window());
        // Machine and per-mount series are kept always — a handful of
        // entries — so the disks panel can show a fill ETA without anybody
        // having written a rule. Per-process series are the expensive half
        // and are only kept when a rule asks for them.
        let window = windowed(AlertKind::Trend)
            .chain(windowed(AlertKind::FillIn))
            .max()
            .unwrap_or(DEFAULT_TREND_WINDOW);
        let track_procs =
            self.rules.iter().any(|r| r.kind == AlertKind::Trend && !r.query.is_empty());
        self.trends.observe(snap, window, track_procs);
    }

    /// The trend series, so the interface can draw what the rules measure.
    pub fn trends(&self) -> &TrendTracker {
        &self.trends
    }

    /// Series currently tracked, so the cost of a trend rule is assertable.
    pub fn tracked_trends(&self) -> usize {
        self.trends.tracked_procs()
    }

    pub fn take_commands(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_commands)
    }

    /// Names of the rules that became active since this was last called.
    pub fn take_fired(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_fired)
    }

    /// Transitions newest first, which is the order the popup reads them in.
    pub fn history(&self) -> impl Iterator<Item = &AlertEvent> {
        self.history.iter().rev()
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

/// One rule as the exporter sees it: what it is, what it last measured, and
/// whether it is firing.
///
/// A rule that is *not* firing still needs a series. A gauge that only appears
/// once something is wrong cannot be alerted on — `absent()` is not the same
/// question — and a dashboard built on it has nothing to draw until the
/// incident it was built for.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleState {
    pub name: String,
    pub kind: AlertKind,
    pub threshold: f64,
    /// The last measurement, or `None` where the rule measures something this
    /// machine does not report — a `gpu` rule with no GPU, a `psi` rule on a
    /// kernel without pressure-stall accounting.
    pub value: Option<f64>,
    pub active: bool,
}

impl AlertEngine {
    /// Every rule and its current state, for the Prometheus exporter.
    ///
    /// Measured here rather than remembered from `evaluate`, so a rule that is
    /// inert on this machine reports no value instead of a stale one.
    pub fn states(&self, snap: &Snapshot) -> Vec<RuleState> {
        self.rules
            .iter()
            .enumerate()
            .map(|(i, rule)| RuleState {
                name: rule.name.clone(),
                kind: rule.kind,
                threshold: rule.threshold,
                value: measure(
                    rule,
                    snap,
                    self.queries.get(i).and_then(|q| q.as_ref()),
                    Some(&self.trends),
                )
                .map(|(v, _)| v),
                active: self.state.get(i).is_some_and(|s| s.fired),
            })
            .collect()
    }
}

/// A measurement with its unit, in whatever spelling that unit deserves.
///
/// Bytes get `human_bytes`, because `+432013312.0B` is a number nobody reads
/// as 412 MiB, and a trend alert is read in a hurry.
fn quantity(value: f64, unit: &str) -> String {
    if unit != "B" {
        return format!("{value:.1}{unit}");
    }
    let sign = if value < 0.0 { "-" } else { "+" };
    format!("{sign}{}", crate::format::human_bytes(value.abs() as u64))
}

/// How far back a `trend` rule looks when it does not say.
///
/// Five minutes: long enough that a build or a garbage collection is not a
/// leak, short enough to catch one before the machine does.
pub const DEFAULT_TREND_WINDOW: u64 = 300;

impl AlertRule {
    /// The window this rule measures over.
    pub fn window(&self) -> u64 {
        match self.over_secs {
            0 => DEFAULT_TREND_WINDOW,
            n => n,
        }
    }
}

/// Growth over time, which is the one shape a threshold cannot express.
///
/// Every other alert kind compares a level against a line: "memory is over
/// 90%". That catches the machine that is already in trouble. It does not
/// catch the process whose RSS has gone up by 200 MB every ten minutes since
/// Tuesday and will take the box out on Friday — by the time that crosses a
/// line, the useful window for doing something about it has closed.
///
/// What is kept is deliberately small: a handful of points per series,
/// thinned so that a series spans its window in `POINTS` samples however
/// often crabmon is sampling. A full history of every process's RSS would be
/// megabytes of live memory in a program whose job is to make memory
/// pressure visible.
#[derive(Debug, Default)]
pub struct TrendTracker {
    /// Machine-wide series, by name: `mem`, `swap`.
    machine: HashMap<&'static str, Series>,
    /// Per-process RSS, keyed by pid *and* start time: a recycled PID is a
    /// different program, and charting the two as one series is how you get
    /// a leak alert for a process that has only just started.
    procs: HashMap<(u32, u64), ProcSeries>,
    /// Bytes used per mount point, which is what turns "95% full" into "full
    /// in three hours".
    disks: HashMap<String, Series>,
}

/// Points kept per series. Eight is enough to measure a slope across a
/// window and to survive a sample or two being missed.
const POINTS: usize = 8;

/// Processes tracked at once. A machine that forks thousands of short-lived
/// processes must not turn a leak detector into a memory leak.
const MAX_TRACKED: usize = 4096;

#[derive(Debug, Default, Clone)]
struct Series {
    /// Thinned history, oldest first: `(taken_at_unix, value)`, roughly one
    /// point per `window / POINTS`.
    points: VecDeque<(u64, u64)>,
    /// The current sample, always. Kept apart from `points` so that the
    /// thinning cannot slide the newest point forward and leave the series
    /// measuring the gap between two samples instead of the window.
    latest: Option<(u64, u64)>,
}

#[derive(Debug, Clone)]
struct ProcSeries {
    series: Series,
    name: String,
    /// The sample this was last seen in, so processes that have exited can
    /// be dropped rather than tracked for ever.
    last_seen: u64,
}

impl Series {
    /// Record a point, keeping about `POINTS` of them across `window`.
    fn push(&mut self, at: u64, value: u64, window: u64) {
        self.latest = Some((at, value));
        let spacing = (window / POINTS as u64).max(1);
        let due = match self.points.back() {
            None => true,
            Some((last_at, _)) => at.saturating_sub(*last_at) >= spacing,
        };
        if due {
            self.points.push_back((at, value));
        }
        // Points that have fallen out of the window, but never the last one:
        // a series with no baseline has no slope.
        while self.points.len() > 1
            && self.points.front().is_some_and(|(t, _)| at.saturating_sub(*t) > window)
        {
            self.points.pop_front();
        }
        while self.points.len() > POINTS {
            self.points.pop_front();
        }
    }

    /// Change from the oldest point still in the window to the current
    /// sample, and how long that took.
    ///
    /// `None` until the series actually spans half the window: a rule that
    /// fired on two samples taken a second apart would be reporting noise as
    /// a trend.
    fn growth(&self, window: u64) -> Option<(i64, u64)> {
        let (last_at, last) = self.latest?;
        let (first_at, first) = *self.points.front()?;
        let span = last_at.saturating_sub(first_at);
        if span == 0 || span * 2 < window {
            return None;
        }
        Some((last as i64 - first as i64, span))
    }
}

impl TrendTracker {
    /// Record one sample. `window` is the longest any rule is looking back,
    /// and `track_procs` says whether any rule needs per-process series at
    /// all — tracking every process costs a map entry per process per
    /// sample, and most configurations never ask for it.
    pub fn observe(&mut self, snap: &Snapshot, window: u64, track_procs: bool) {
        let at = snap.taken_at_unix;
        if at == 0 {
            return;
        }
        self.machine.entry("mem").or_default().push(at, snap.mem.used, window);
        self.machine.entry("swap").or_default().push(at, snap.mem.swap_used, window);

        // A handful of mounts, eight points each: cheap enough to keep
        // always, which is what lets the disks panel show an ETA without
        // anybody having written a rule.
        for d in &snap.disks {
            self.disks.entry(d.mount.clone()).or_default().push(at, d.used, window);
        }
        self.disks.retain(|mount, _| snap.disks.iter().any(|d| &d.mount == mount));

        if !track_procs {
            self.procs.clear();
            return;
        }
        for p in &snap.procs {
            let key = (p.pid, p.start_time_unix);
            if let Some(entry) = self.procs.get_mut(&key) {
                entry.series.push(at, p.mem, window);
                entry.last_seen = at;
                continue;
            }
            if self.procs.len() >= MAX_TRACKED {
                continue;
            }
            let mut series = Series::default();
            series.push(at, p.mem, window);
            self.procs.insert(key, ProcSeries { series, name: p.name.clone(), last_seen: at });
        }
        // A process that has exited keeps nothing: its leak is over.
        self.procs.retain(|_, entry| entry.last_seen == at);
    }

    /// Growth of a machine-wide series over its window.
    pub fn machine_growth(&self, name: &str, window: u64) -> Option<(i64, u64)> {
        self.machine.get(name)?.growth(window)
    }

    /// The matching process that has grown the most, with its name.
    ///
    /// The worst one, because a rule that averaged across matches would be
    /// quietest exactly when one process is running away from the rest.
    pub fn worst_proc_growth(
        &self,
        filter: &crate::filter::Filter,
        snap: &Snapshot,
        window: u64,
        shrinking: bool,
    ) -> Option<(i64, u64, String)> {
        let matched: std::collections::HashSet<(u32, u64)> = snap
            .procs
            .iter()
            .filter(|p| filter.matches(p))
            .map(|p| (p.pid, p.start_time_unix))
            .collect();
        self.procs
            .iter()
            .filter(|(key, _)| matched.contains(key))
            .filter_map(|(_, entry)| {
                entry.series.growth(window).map(|(d, span)| (d, span, entry.name.clone()))
            })
            // `below` rules are looking for the biggest *fall*, so the
            // extreme to report is the other end.
            .reduce(|a, b| match shrinking {
                true if b.0 < a.0 => b,
                false if b.0 > a.0 => b,
                _ => a,
            })
    }

    /// Series currently tracked, for tests and for the cost of it to be
    /// assertable.
    pub fn tracked_procs(&self) -> usize {
        self.procs.len()
    }

    /// Seconds until a filesystem is full at the rate it has been filling,
    /// or `None` when it is not filling or has not been watched long enough.
    ///
    /// This is the disk question worth paging on. A percentage cannot ask
    /// it: 95% of a 4 TB array is 200 GB of headroom and days of warning,
    /// while 95% of a 20 GB root is twenty minutes — the same alert fires
    /// far too late on one and far too early on the other. Time to full is
    /// the same number on both.
    pub fn fill_eta(&self, mount: &str, total: u64, used: u64, window: u64) -> Option<u64> {
        let (delta, span) = self.disks.get(mount)?.growth(window)?;
        // Not filling, or shrinking: there is no "when".
        if delta <= 0 || span == 0 {
            return None;
        }
        let free = total.checked_sub(used)?;
        let bytes_per_sec = delta as f64 / span as f64;
        (bytes_per_sec > 0.0).then(|| (free as f64 / bytes_per_sec) as u64)
    }

    /// The mount that will fill first, and when.
    pub fn soonest_full(&self, snap: &Snapshot, window: u64) -> Option<(String, u64)> {
        snap.disks
            .iter()
            .filter_map(|d| {
                self.fill_eta(&d.mount, d.total, d.used, window).map(|eta| (d.mount.clone(), eta))
            })
            .min_by_key(|(_, eta)| *eta)
    }
}

/// The measured value for a rule plus the unit used in its message.
///
/// `query` is the pre-parsed filter for `proc` rules, and `None` for every
/// other kind.
fn measure(
    rule: &AlertRule,
    snap: &Snapshot,
    query: Option<&filter::Filter>,
    trends: Option<&TrendTracker>,
) -> Option<(f64, &'static str)> {
    let v = match rule.kind {
        AlertKind::Cpu => snap.cpu.avg(),
        AlertKind::Mem => snap.mem.ratio() * 100.0,
        AlertKind::Swap => snap.mem.swap_ratio() * 100.0,
        AlertKind::Load => return Some((snap.host.load[0], "")),
        AlertKind::Disk => snap
            .disks
            .iter()
            .filter(|d| rule.target.is_empty() || d.mount == rule.target)
            .map(|d| d.ratio() * 100.0)
            .fold(f64::NAN, f64::max),
        AlertKind::Temp => {
            return snap
                .sensors
                .iter()
                .filter(|s| rule.target.is_empty() || s.label.starts_with(&rule.target))
                .map(|s| s.temp as f64)
                .fold(None, |acc: Option<f64>, t| Some(acc.map_or(t, |a| a.max(t))))
                .map(|t| (t, "°C"));
        }
        AlertKind::Gpu => {
            return snap
                .gpus
                .iter()
                .filter(|g| rule.target.is_empty() || g.name.contains(&rule.target))
                .filter_map(|g| g.busy_percent)
                .fold(None, |acc: Option<f64>, b| Some(acc.map_or(b as f64, |a| a.max(b as f64))))
                .map(|b| (b, "%"));
        }
        AlertKind::Psi => return psi_value(&rule.target, snap).map(|v| (v, "%")),
        // A slope, in bytes, over the rule's own window. `None` until the
        // series actually spans that window: a rule that fired on two
        // samples a second apart would be reporting noise as a trend.
        AlertKind::Trend => {
            let trends = trends?;
            let window = rule.window();
            // An empty query parses into the filter that matches
            // *everything*, so the machine series has to be chosen by the
            // query being absent rather than by the filter being empty.
            let growth = match query.filter(|_| !rule.query.trim().is_empty()) {
                Some(filter) => trends
                    .worst_proc_growth(filter, snap, window, rule.below)
                    .map(|(delta, _, _)| delta),
                None => {
                    let series = match rule.target.trim() {
                        "" | "mem" | "memory" | "rss" => "mem",
                        "swap" => "swap",
                        _ => return None,
                    };
                    trends.machine_growth(series, window).map(|(delta, _)| delta)
                }
            }?;
            return Some((growth as f64, "B"));
        }
        // The group's own limit, which the host's gauges cannot see: a
        // container at 98% of a 512 MB limit is about to be killed on a
        // machine reporting 30% memory used.
        AlertKind::Cgroup => {
            return match rule.target.trim() {
                "" => snap
                    .cgroups
                    .iter()
                    .filter_map(|c| c.mem_ratio())
                    .fold(None, |acc, r| Some(acc.map_or(r, |a: f64| a.max(r)))),
                name => snap.cgroup_named(name).and_then(|c| c.mem_ratio()),
            }
            .map(|r| (r * 100.0, "%"));
        }
        // Seconds, counted down rather than up: the alarming direction is
        // less time, so these rules are written with `below`.
        AlertKind::FillIn => {
            let trends = trends?;
            let window = rule.window();
            let eta = match rule.target.trim() {
                "" => trends.soonest_full(snap, window).map(|(_, eta)| eta),
                mount => snap
                    .disks
                    .iter()
                    .find(|d| d.mount == mount)
                    .and_then(|d| trends.fill_eta(&d.mount, d.total, d.used, window)),
            }?;
            return Some((eta as f64, "s"));
        }
        // A count, not a level. The snapshot carries the events the kernel
        // has reported, so a rule with `threshold = 1` fires on the first
        // OOM kill and clears once it has aged out of the window.
        AlertKind::Kernel => return kernel_count(&rule.target, snap).map(|n| (n as f64, "")),
        // Utilisation, not capacity: a device can be pinned at 4 MB/s of small
        // random reads while both its throughput and its free space look fine.
        AlertKind::Io => {
            return match rule.target.is_empty() {
                true => snap.disk_util_max(),
                false => snap
                    .disks
                    .iter()
                    .filter(|d| d.mount == rule.target || d.kernel_name == rule.target)
                    .filter_map(|d| d.util)
                    .fold(None, |acc: Option<f64>, u| Some(acc.map_or(u, |a| a.max(u)))),
            }
            .map(|u| (u * 100.0, "%"));
        }
        AlertKind::Net => {
            // An untargeted rule uses the same aggregate the chart draws, so a
            // VPN tunnel is not counted alongside the NIC carrying it.
            let (rx, tx) = match rule.target.is_empty() {
                true => snap.net_totals(false),
                false => snap
                    .nets
                    .iter()
                    .filter(|n| n.name == rule.target)
                    .fold((0.0, 0.0), |(r, t), n| (r + n.rx_bps, t + n.tx_bps)),
            };
            return Some((rx + tx, " B/s"));
        }
        AlertKind::Fd => {
            return snap
                .procs
                .iter()
                .filter(|p| rule.target.is_empty() || p.name.contains(&rule.target))
                .filter_map(|p| p.fd_ratio())
                .fold(None, |acc: Option<f64>, r| Some(acc.map_or(r, |a| a.max(r))))
                .map(|r| (r * 100.0, "%"));
        }
        AlertKind::Proc => {
            // A query that did not parse leaves the rule inert. Counting zero
            // instead would make every `below` rule fire for ever on a typo.
            let q = query?;
            return Some((snap.procs.iter().filter(|p| q.matches(p)).count() as f64, ""));
        }
    };
    if v.is_nan() { None } else { Some((v, "%")) }
}

/// The PSI figure a rule's `target` names: `cpu`, `mem`/`memory` or `io`,
/// optionally suffixed `.full`. An empty target is the worst `some` reading
/// across all three, which is the same number the status line shows.
/// The resource names a `psi` rule's target may use, with or without a
/// `.full` suffix. Exposed so a rule can be *rejected* for naming something
/// that does not exist rather than silently measuring nothing for ever.
pub const PSI_RESOURCES: [&str; 4] = ["cpu", "mem", "memory", "io"];

/// Whether `target` names a pressure line this machine's kernel could report.
pub fn psi_target_is_known(target: &str) -> bool {
    if target.is_empty() {
        return true;
    }
    let resource = target.split_once('.').map(|(r, _)| r).unwrap_or(target);
    let suffix = target.split_once('.').map(|(_, s)| s).unwrap_or("");
    let line_ok = suffix.is_empty()
        || suffix.eq_ignore_ascii_case("full")
        || suffix.eq_ignore_ascii_case("some");
    line_ok && PSI_RESOURCES.iter().any(|r| r.eq_ignore_ascii_case(resource))
}

/// How many kernel events of the named kind a snapshot carries.
///
/// `None` — not zero — when the kernel log could not be read at all, so a
/// rule on a host where `/dev/kmsg` needs root reports nothing rather than
/// "no process has ever been OOM-killed here".
pub fn kernel_count(target: &str, snap: &Snapshot) -> Option<usize> {
    if snap.kernel_unreadable.is_some() {
        return None;
    }
    if target.is_empty() {
        return Some(snap.kernel.len());
    }
    let kind = crate::metrics::kmsg::ALL_EVENT_KINDS
        .into_iter()
        .find(|k| k.key_name().eq_ignore_ascii_case(target))?;
    Some(snap.kernel_count(kind))
}

/// Whether `target` names a kernel event kind, so a rule can be refused for
/// naming one that does not exist rather than measuring nothing for ever.
pub fn kernel_target_is_known(target: &str) -> bool {
    target.is_empty()
        || crate::metrics::kmsg::ALL_EVENT_KINDS
            .iter()
            .any(|k| k.key_name().eq_ignore_ascii_case(target))
}

fn psi_value(target: &str, snap: &Snapshot) -> Option<f64> {
    if target.is_empty() {
        return snap.psi.is_available().then(|| snap.psi.worst_avg10());
    }
    let (resource, full) = match target.split_once('.') {
        Some((r, line)) => (r, line.eq_ignore_ascii_case("full")),
        None => (target, false),
    };
    let pressure = match resource.to_ascii_lowercase().as_str() {
        "cpu" => snap.psi.cpu,
        "mem" | "memory" => snap.psi.memory,
        "io" => snap.psi.io,
        _ => None,
    }?;
    // `full` is absent for CPU on most kernels — a single runnable task is
    // never "full" — so the rule reports nothing rather than falling back to
    // `some`, which measures something else entirely.
    if full { pressure.full.map(|l| l.avg10) } else { Some(pressure.some.avg10) }
}

/// Default rules written into a fresh config file.
pub fn default_rules() -> Vec<AlertRule> {
    vec![
        AlertRule {
            name: "cpu-saturated".into(),
            kind: AlertKind::Cpu,
            threshold: 90.0,
            for_secs: 30,
            over_secs: 0,
            below: false,
            target: String::new(),
            query: String::new(),
            command: String::new(),
            command_clear: String::new(),
        },
        AlertRule {
            name: "disk-nearly-full".into(),
            kind: AlertKind::Disk,
            threshold: 95.0,
            for_secs: 0,
            over_secs: 0,
            below: false,
            target: String::new(),
            query: String::new(),
            command: String::new(),
            command_clear: String::new(),
        },
    ]
}

#[cfg(test)]
mod trend_tests {
    use super::*;
    use crate::metrics::{MemSample, ProcRow};

    fn trend_rule(query: &str, threshold: f64, over_secs: u64) -> AlertRule {
        AlertRule {
            name: "leak".into(),
            kind: AlertKind::Trend,
            threshold,
            for_secs: 0,
            over_secs,
            below: false,
            target: if query.is_empty() { "mem".into() } else { String::new() },
            query: query.into(),
            command: String::new(),
            command_clear: String::new(),
        }
    }

    /// A machine whose memory climbs steadily over `secs`.
    fn climbing(at: u64, used: u64, rss: u64) -> Snapshot {
        Snapshot {
            mem: MemSample { total: 16 << 30, used, ..Default::default() },
            procs: vec![ProcRow {
                pid: 42,
                name: "postgres".into(),
                mem: rss,
                start_time_unix: 1,
                ..Default::default()
            }],
            taken_at_unix: at,
            ..Default::default()
        }
    }

    /// The shape no threshold can express: nothing is over any line, and the
    /// machine is going to fall over on Friday.
    #[test]
    fn a_steady_climb_fires_even_though_no_level_is_near_a_threshold() {
        let mut e = AlertEngine::new(vec![trend_rule("", (1 << 30) as f64, 600)]);
        let base = 1_700_000_000;
        let mut fired = None;
        for step in 0..=10u64 {
            // 4 GiB over ten minutes, in a 16 GiB machine: memory never goes
            // above a quarter, so every `mem` rule stays quiet.
            let used = (2 << 30) + step * (4 << 30) / 10;
            let active = e.evaluate(&climbing(base + step * 60, used, 1 << 20), step * 60);
            if !active.is_empty() && fired.is_none() {
                fired = Some(active[0].clone());
            }
        }
        let fired = fired.expect("a 4 GiB climb never fired a 1 GiB trend rule");
        assert!(fired.value >= (1u64 << 30) as f64, "{fired:?}");
        assert!(fired.message.contains("GiB"), "bytes must read as bytes: {}", fired.message);
    }

    /// ...and a flat machine does not.
    #[test]
    fn a_flat_series_never_fires() {
        let mut e = AlertEngine::new(vec![trend_rule("", (1 << 30) as f64, 600)]);
        let base = 1_700_000_000;
        for step in 0..=10u64 {
            let active = e.evaluate(&climbing(base + step * 60, 2 << 30, 1 << 20), step * 60);
            assert!(active.is_empty(), "a flat series fired: {active:?}");
        }
    }

    /// Two samples a second apart are noise, not a trend.
    #[test]
    fn a_series_that_has_not_spanned_its_window_reports_nothing_yet() {
        let mut e = AlertEngine::new(vec![trend_rule("", 1024.0, 600)]);
        let base = 1_700_000_000;
        assert!(e.evaluate(&climbing(base, 1 << 30, 1 << 20), 0).is_empty());
        assert!(e.evaluate(&climbing(base + 1, 8 << 30, 1 << 20), 1).is_empty());
    }

    /// The point of a per-process rule is to name the process.
    #[test]
    fn a_process_rule_names_the_one_that_is_growing() {
        let mut e = AlertEngine::new(vec![trend_rule("postgres", (100 << 20) as f64, 600)]);
        let base = 1_700_000_000;
        let mut fired = None;
        for step in 0..=10u64 {
            let rss = (100 << 20) + step * (400 << 20) / 10;
            let active = e.evaluate(&climbing(base + step * 60, 2 << 30, rss), step * 60);
            if !active.is_empty() && fired.is_none() {
                fired = Some(active[0].clone());
            }
        }
        let fired = fired.expect("a 400 MiB RSS climb never fired");
        assert!(fired.message.contains("postgres"), "{}", fired.message);
    }

    /// A recycled PID is a different program; charting the two as one series
    /// is how a freshly started process looks like a leak.
    #[test]
    fn a_recycled_pid_starts_a_new_series() {
        let mut e = AlertEngine::new(vec![trend_rule("postgres", (1 << 20) as f64, 600)]);
        let base = 1_700_000_000;
        for step in 0..6u64 {
            e.evaluate(&climbing(base + step * 60, 2 << 30, 1 << 20), step * 60);
        }
        // Same PID, different start time, and holding a great deal more.
        let mut reborn = climbing(base + 600, 2 << 30, 4 << 30);
        reborn.procs[0].start_time_unix = 2;
        let active = e.evaluate(&reborn, 600);
        assert!(active.is_empty(), "a new process was reported as a leak: {active:?}");
    }

    /// Per-process series cost a map entry per process per sample, so they
    /// are only kept when a rule actually asks for them.
    #[test]
    fn processes_are_only_tracked_when_a_rule_needs_them() {
        let mut machine_only = AlertEngine::new(vec![trend_rule("", (1 << 30) as f64, 600)]);
        machine_only.evaluate(&climbing(1_700_000_000, 1 << 30, 1 << 20), 0);
        assert_eq!(machine_only.tracked_trends(), 0, "processes were tracked unasked");

        let mut per_proc = AlertEngine::new(vec![trend_rule("postgres", (1 << 30) as f64, 600)]);
        per_proc.evaluate(&climbing(1_700_000_000, 1 << 30, 1 << 20), 0);
        assert_eq!(per_proc.tracked_trends(), 1);
    }

    /// A machine that forks constantly must not turn a leak detector into a
    /// memory leak.
    #[test]
    fn the_number_of_tracked_processes_is_bounded() {
        let mut e = AlertEngine::new(vec![trend_rule("re:.", (1 << 30) as f64, 600)]);
        let mut snap = climbing(1_700_000_000, 1 << 30, 1 << 20);
        snap.procs = (0..MAX_TRACKED as u32 * 2)
            .map(|pid| ProcRow {
                pid,
                name: format!("p{pid}"),
                mem: 1 << 20,
                start_time_unix: 1,
                ..Default::default()
            })
            .collect();
        e.evaluate(&snap, 0);
        assert!(e.tracked_trends() <= MAX_TRACKED, "tracked {}", e.tracked_trends());
    }

    /// A process that exits takes its series with it: its leak is over.
    #[test]
    fn an_exited_process_is_forgotten() {
        let mut e = AlertEngine::new(vec![trend_rule("postgres", (1 << 30) as f64, 600)]);
        e.evaluate(&climbing(1_700_000_000, 1 << 30, 1 << 20), 0);
        assert_eq!(e.tracked_trends(), 1);

        let mut gone = climbing(1_700_000_060, 1 << 30, 0);
        gone.procs.clear();
        e.evaluate(&gone, 60);
        assert_eq!(e.tracked_trends(), 0);
    }

    /// `below` turns the same rule into "this is being freed", which is the
    /// shape of a cache being dropped or a service restarting.
    #[test]
    fn below_measures_the_biggest_fall() {
        let mut rule = trend_rule("", -(1i64 << 30) as f64, 600);
        rule.below = true;
        let mut e = AlertEngine::new(vec![rule]);
        let base = 1_700_000_000;
        let mut fired = false;
        for step in 0..=10u64 {
            let used = (8 << 30) - step * (4u64 << 30) / 10;
            fired |= !e.evaluate(&climbing(base + step * 60, used, 1 << 20), step * 60).is_empty();
        }
        assert!(fired, "a 4 GiB fall never fired a -1 GiB rule");
    }

    /// The disk question worth paging on, which a percentage cannot ask.
    #[test]
    fn a_filling_disk_is_measured_in_time_not_percent() {
        use crate::metrics::DiskRow;

        let mut rule = trend_rule("", 3600.0, 600);
        rule.kind = AlertKind::FillIn;
        rule.target = "/var".into();
        rule.below = true;
        let mut e = AlertEngine::new(vec![rule]);

        let base = 1_700_000_000;
        let mut fired = None;
        for step in 0..=10u64 {
            // 10 GB filesystem gaining 100 MB a minute: ~40 minutes left at
            // the end of the window, which is under the hour the rule wants.
            let used = (5u64 << 30) + step * (100 << 20);
            let snap = Snapshot {
                disks: vec![DiskRow {
                    mount: "/var".into(),
                    total: 10 << 30,
                    used,
                    ..Default::default()
                }],
                taken_at_unix: base + step * 60,
                ..Default::default()
            };
            let active = e.evaluate(&snap, step * 60);
            if !active.is_empty() && fired.is_none() {
                fired = Some(active[0].clone());
            }
        }
        let fired = fired.expect("a filesystem filling in under an hour never fired");
        assert!(fired.value > 0.0 && fired.value < 3600.0, "{fired:?}");
    }

    /// A disk that is not filling has no "when", and a rule that invented
    /// one would fire on every idle machine for ever.
    #[test]
    fn a_disk_that_is_not_filling_measures_nothing() {
        use crate::metrics::DiskRow;

        let mut rule = trend_rule("", 3600.0, 600);
        rule.kind = AlertKind::FillIn;
        rule.below = true;
        let mut e = AlertEngine::new(vec![rule]);
        let base = 1_700_000_000;
        for step in 0..=10u64 {
            let snap = Snapshot {
                disks: vec![DiskRow {
                    mount: "/".into(),
                    total: 10 << 30,
                    // Shrinking, then flat: neither has a time to full.
                    used: (5u64 << 30) - step * (1 << 20),
                    ..Default::default()
                }],
                taken_at_unix: base + step * 60,
                ..Default::default()
            };
            assert!(e.evaluate(&snap, step * 60).is_empty(), "a draining disk fired");
        }
    }

    /// A container at 98% of a 512 MB limit is invisible in a host memory
    /// gauge showing 30% used.
    #[test]
    fn a_cgroup_rule_measures_a_group_against_its_own_limit() {
        use crate::metrics::cgroup::CgroupStat;

        let mut rule = trend_rule("", 90.0, 600);
        rule.kind = AlertKind::Cgroup;
        rule.target = "api".into();
        let mut e = AlertEngine::new(vec![rule]);

        let snap = Snapshot {
            cgroups: vec![
                CgroupStat {
                    name: "api".into(),
                    mem_current: Some(500 << 20),
                    mem_max: Some(512 << 20),
                    ..Default::default()
                },
                CgroupStat {
                    name: "idle".into(),
                    mem_current: Some(1 << 20),
                    mem_max: Some(512 << 20),
                    ..Default::default()
                },
            ],
            // The host has plenty: that is the whole point.
            mem: crate::metrics::MemSample { total: 64 << 30, used: 8 << 30, ..Default::default() },
            taken_at_unix: 1_700_000_000,
            ..Default::default()
        };
        let active = e.evaluate(&snap, 0);
        assert_eq!(active.len(), 1, "{active:?}");
        assert!(active[0].value > 97.0, "{:?}", active[0]);
    }

    /// An untargeted rule watches whichever group is closest to its limit.
    #[test]
    fn an_untargeted_cgroup_rule_takes_the_worst_group() {
        use crate::metrics::cgroup::CgroupStat;

        let mut rule = trend_rule("", 90.0, 600);
        rule.kind = AlertKind::Cgroup;
        // The helper fills in a machine series; an untargeted cgroup rule
        // means "whichever group is worst".
        rule.target = String::new();
        let mut e = AlertEngine::new(vec![rule]);
        let snap = Snapshot {
            cgroups: vec![
                CgroupStat {
                    name: "quiet".into(),
                    mem_current: Some(1),
                    mem_max: Some(100),
                    ..Default::default()
                },
                CgroupStat {
                    name: "loud".into(),
                    mem_current: Some(99),
                    mem_max: Some(100),
                    ..Default::default()
                },
            ],
            taken_at_unix: 1_700_000_000,
            ..Default::default()
        };
        assert_eq!(e.evaluate(&snap, 0).len(), 1);
    }

    /// A group with no limit has no ratio, and a rule that invented one
    /// would fire on every unlimited unit on the machine.
    #[test]
    fn a_group_with_no_limit_measures_nothing() {
        use crate::metrics::cgroup::CgroupStat;

        let mut rule = trend_rule("", 1.0, 600);
        rule.kind = AlertKind::Cgroup;
        rule.target = String::new();
        let mut e = AlertEngine::new(vec![rule]);
        let snap = Snapshot {
            cgroups: vec![CgroupStat {
                name: "unlimited".into(),
                mem_current: Some(8 << 30),
                mem_max: None,
                ..Default::default()
            }],
            taken_at_unix: 1_700_000_000,
            ..Default::default()
        };
        assert!(e.evaluate(&snap, 0).is_empty());
    }

    /// Per-mount series are kept whether or not a rule asks, because the
    /// panel draws the ETA and nobody should have to write a rule for that.
    #[test]
    fn mounts_are_tracked_even_with_no_rules_at_all() {
        use crate::metrics::DiskRow;

        let mut e = AlertEngine::new(Vec::new());
        for step in 0..=8u64 {
            let snap = Snapshot {
                disks: vec![DiskRow {
                    mount: "/".into(),
                    total: 1000,
                    used: 100 + step * 50,
                    ..Default::default()
                }],
                taken_at_unix: 1_700_000_000 + step * 60,
                ..Default::default()
            };
            e.evaluate(&snap, step * 60);
        }
        let eta = e.trends().fill_eta("/", 1000, 500, DEFAULT_TREND_WINDOW);
        assert!(eta.is_some(), "no ETA without a rule");
    }

    #[test]
    fn a_byte_quantity_reads_as_bytes_and_keeps_its_sign() {
        assert_eq!(quantity(1024.0, "B"), "+1.0 KiB");
        assert_eq!(quantity(-1024.0, "B"), "-1.0 KiB");
        assert_eq!(quantity(90.0, "%"), "90.0%");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, DiskRow, MemSample, ProcRow, Sensor};

    fn rule(name: &str, kind: AlertKind, threshold: f64, for_secs: u64) -> AlertRule {
        AlertRule {
            name: name.into(),
            kind,
            threshold,
            for_secs,
            over_secs: 0,
            below: false,
            target: String::new(),
            query: String::new(),
            command: String::new(),
            command_clear: String::new(),
        }
    }

    fn snap_cpu(pct: f32) -> Snapshot {
        Snapshot { cpu: CpuSample { per_core: vec![pct], freq_mhz: vec![] }, ..Default::default() }
    }

    #[test]
    fn every_kind_is_in_the_list_and_spells_itself_the_way_serde_does() {
        // `ALL_ALERT_KINDS` is what the doc tests check the README and man page
        // against, so a kind missing from it would take the docs with it.
        for kind in ALL_ALERT_KINDS {
            let rule = AlertRule { kind, ..rule("x", kind, 0.0, 0) };
            let toml = toml::to_string(&rule).unwrap();
            assert!(
                toml.contains(&format!("kind = \"{}\"", kind.key_name())),
                "{kind:?} serialises as something other than {}",
                kind.key_name()
            );
        }
        let mut names: Vec<&str> = ALL_ALERT_KINDS.iter().map(|k| k.key_name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two kinds share a spelling");
    }

    /// Every rule needs a series whether or not it is firing, and the value
    /// has to be measured now rather than remembered from the last crossing —
    /// otherwise a rule that recovered would export the number that tripped it
    /// for ever.
    #[test]
    fn rule_states_report_what_is_true_now_not_what_was_true_when_it_fired() {
        let mut e = AlertEngine::new(vec![rule("cpu", AlertKind::Cpu, 90.0, 0)]);
        let hot = snap_cpu(95.0);
        e.evaluate(&hot, 100);

        let states = e.states(&hot);
        assert_eq!(states.len(), 1);
        assert!(states[0].active);
        assert_eq!(states[0].value, Some(95.0));
        assert_eq!(states[0].threshold, 90.0);
        assert_eq!(states[0].kind, AlertKind::Cpu);

        // It recovers; the exported value follows the machine, not the rule.
        let cool = snap_cpu(3.0);
        e.evaluate(&cool, 200);
        let states = e.states(&cool);
        assert!(!states[0].active);
        assert_eq!(states[0].value, Some(3.0));
    }

    #[test]
    fn a_rule_measuring_something_this_machine_lacks_reports_no_value() {
        // A `gpu` rule on a machine with no GPU. Reporting 0 would read as an
        // idle GPU, which is a different claim from "there isn't one".
        let e = AlertEngine::new(vec![rule("gpu", AlertKind::Gpu, 90.0, 0)]);
        let states = e.states(&Snapshot::default());
        assert_eq!(states[0].value, None);
        assert!(!states[0].active);
    }

    #[test]
    fn every_rule_gets_a_state_even_the_ones_that_never_fire() {
        let e = AlertEngine::new(vec![
            rule("a", AlertKind::Cpu, 90.0, 0),
            rule("b", AlertKind::Mem, 90.0, 0),
            rule("c", AlertKind::Swap, 90.0, 0),
        ]);
        assert_eq!(e.states(&Snapshot::default()).len(), 3);
        assert!(AlertEngine::new(Vec::new()).states(&Snapshot::default()).is_empty());
    }

    #[test]
    fn a_rule_waits_out_its_hold_time_before_firing() {
        let mut e = AlertEngine::new(vec![rule("cpu", AlertKind::Cpu, 90.0, 30)]);
        let hot = snap_cpu(95.0);
        assert!(e.evaluate(&hot, 100).is_empty(), "must not fire on the first frame");
        assert!(e.evaluate(&hot, 120).is_empty(), "still inside the hold window");
        assert_eq!(e.evaluate(&hot, 130).len(), 1, "fires once the hold elapses");
    }

    #[test]
    fn dropping_below_threshold_resets_the_hold_timer() {
        let mut e = AlertEngine::new(vec![rule("cpu", AlertKind::Cpu, 90.0, 30)]);
        e.evaluate(&snap_cpu(95.0), 100);
        e.evaluate(&snap_cpu(10.0), 110);
        assert!(e.evaluate(&snap_cpu(95.0), 130).is_empty(), "the clock restarted at 110");
        assert_eq!(e.evaluate(&snap_cpu(95.0), 165).len(), 1);
    }

    #[test]
    fn a_zero_hold_rule_fires_immediately() {
        let mut e = AlertEngine::new(vec![rule("cpu", AlertKind::Cpu, 90.0, 0)]);
        assert_eq!(e.evaluate(&snap_cpu(91.0), 0).len(), 1);
    }

    #[test]
    fn the_command_hook_runs_once_per_activation() {
        let mut r = rule("cpu", AlertKind::Cpu, 90.0, 0);
        r.command = "notify-send hot".into();
        let mut e = AlertEngine::new(vec![r]);

        e.evaluate(&snap_cpu(95.0), 0);
        assert_eq!(e.take_commands(), vec!["notify-send hot".to_string()]);
        e.evaluate(&snap_cpu(95.0), 1);
        assert!(e.take_commands().is_empty(), "must not re-fire while still active");

        e.evaluate(&snap_cpu(5.0), 2);
        e.evaluate(&snap_cpu(95.0), 3);
        assert_eq!(e.take_commands().len(), 1, "re-arms after recovering");
    }

    #[test]
    fn disk_rules_can_target_a_single_mount() {
        let snap = Snapshot {
            disks: vec![
                DiskRow { mount: "/".into(), total: 100, used: 50, ..Default::default() },
                DiskRow { mount: "/boot".into(), total: 100, used: 99, ..Default::default() },
            ],
            ..Default::default()
        };
        let mut all = AlertEngine::new(vec![rule("disk", AlertKind::Disk, 95.0, 0)]);
        assert_eq!(all.evaluate(&snap, 0).len(), 1, "worst mount trips an untargeted rule");

        let mut targeted = AlertEngine::new(vec![AlertRule {
            target: "/".into(),
            ..rule("root", AlertKind::Disk, 95.0, 0)
        }]);
        assert!(targeted.evaluate(&snap, 0).is_empty());
    }

    #[test]
    fn temperature_rules_read_degrees_not_percent() {
        let snap = Snapshot {
            sensors: vec![
                Sensor { label: "coretemp Core 0".into(), temp: 95.0, critical: Some(110.0) },
                Sensor { label: "nvme x".into(), temp: 40.0, critical: None },
            ],
            ..Default::default()
        };
        let mut e = AlertEngine::new(vec![AlertRule {
            target: "coretemp".into(),
            ..rule("hot-cpu", AlertKind::Temp, 90.0, 0)
        }]);
        let active = e.evaluate(&snap, 0);
        assert_eq!(active.len(), 1);
        assert!(active[0].message.contains("°C"), "{}", active[0].message);
    }

    #[test]
    fn rules_with_nothing_to_measure_are_skipped_silently() {
        let mut e = AlertEngine::new(vec![rule("disk", AlertKind::Disk, 1.0, 0)]);
        assert!(e.evaluate(&Snapshot::default(), 0).is_empty());
    }

    #[test]
    fn a_below_rule_fires_when_the_thing_it_watches_stops() {
        // The condition anyone actually wants paging on: nginx is *gone*.
        let mut r = rule("nginx-down", AlertKind::Proc, 1.0, 0);
        r.below = true;
        r.query = "nginx".into();
        let mut e = AlertEngine::new(vec![r]);

        let running = Snapshot {
            procs: vec![ProcRow { pid: 1, name: "nginx".into(), ..Default::default() }],
            ..Default::default()
        };
        assert!(e.evaluate(&running, 0).is_empty(), "one nginx is not fewer than one");

        let gone = Snapshot {
            procs: vec![ProcRow { pid: 2, name: "sshd".into(), ..Default::default() }],
            ..Default::default()
        };
        let active = e.evaluate(&gone, 1);
        assert_eq!(active.len(), 1);
        assert!(active[0].message.contains('<'), "reads as <: {}", active[0].message);

        // ...and it recovers when the process comes back.
        assert!(e.evaluate(&running, 2).is_empty());
    }

    #[test]
    fn a_proc_rule_counts_what_its_query_matches() {
        let mut r = rule("too-many-workers", AlertKind::Proc, 3.0, 0);
        r.query = "user:app cpu>1".into();
        let mut e = AlertEngine::new(vec![r]);
        let worker = |pid: u32, cpu: f32, user: &str| ProcRow {
            pid,
            name: "worker".into(),
            cpu,
            user: Some(user.into()),
            ..Default::default()
        };
        let snap = Snapshot {
            procs: vec![
                worker(1, 5.0, "app"),
                worker(2, 5.0, "app"),
                worker(3, 0.1, "app"),
                worker(4, 9.0, "root"),
            ],
            ..Default::default()
        };
        assert!(e.evaluate(&snap, 0).is_empty(), "two match, the threshold is three");

        let mut busier = snap.clone();
        busier.procs.push(worker(5, 5.0, "app"));
        busier.procs.push(worker(6, 5.0, "app"));
        assert_eq!(e.evaluate(&busier, 1).len(), 1);
    }

    #[test]
    fn a_proc_rule_whose_query_is_broken_stays_inert() {
        // Counting zero instead would make every `below` rule fire for ever
        // on a typo, with the hook command going off each time.
        let mut r = rule("typo", AlertKind::Proc, 1.0, 0);
        r.below = true;
        r.query = "re:[unclosed".into();
        let mut e = AlertEngine::new(vec![r]);
        assert!(e.evaluate(&Snapshot::default(), 0).is_empty());
        assert!(e.take_commands().is_empty());
    }

    #[test]
    fn two_rules_sharing_a_name_no_longer_clear_each_others_state() {
        // Keyed by name, the pair below shared one `fired` flag: each refresh
        // the second rule cleared what the first had just set, so the hook
        // re-spawned on every single sample for as long as the condition held.
        let mut over = rule("cpu", AlertKind::Cpu, 90.0, 0);
        over.command = "page".into();
        let mut under = rule("cpu", AlertKind::Cpu, 10.0, 0);
        under.below = true;
        let mut e = AlertEngine::new(vec![over, under]);

        assert_eq!(e.evaluate(&snap_cpu(95.0), 0).len(), 1);
        assert_eq!(e.take_commands(), vec!["page".to_string()]);
        for t in 1..5 {
            e.evaluate(&snap_cpu(95.0), t);
            assert!(e.take_commands().is_empty(), "re-fired at t={t}");
        }
    }

    #[test]
    fn pressure_rules_name_a_resource_and_a_line() {
        use crate::metrics::psi::{Pressure, PressureLine, PsiSample};
        let line = |avg10: f64| PressureLine { avg10, ..Default::default() };
        let snap = Snapshot {
            psi: PsiSample {
                cpu: Some(Pressure { some: line(4.0), full: None }),
                memory: Some(Pressure { some: line(30.0), full: Some(line(12.0)) }),
                io: Some(Pressure { some: line(8.0), full: Some(line(1.0)) }),
            },
            ..Default::default()
        };

        let fires = |target: &str, threshold: f64| {
            let mut r = rule("psi", AlertKind::Psi, threshold, 0);
            r.target = target.into();
            !AlertEngine::new(vec![r]).evaluate(&snap, 0).is_empty()
        };
        assert!(fires("", 25.0), "an untargeted rule takes the worst of the three");
        assert!(fires("mem", 25.0));
        assert!(!fires("io", 25.0));
        assert!(fires("memory.full", 10.0), "the full line is a different number");
        assert!(!fires("mem.full", 25.0), "...and 12 is not 30");
        // CPU publishes no `full` line on most kernels: report nothing rather
        // than falling back to `some`, which measures something else.
        assert!(!fires("cpu.full", 0.0));
        assert!(!fires("nonsense", 0.0));
    }

    #[test]
    fn an_io_rule_fires_on_a_saturated_device_not_a_full_one() {
        // The disk is 10% full and pinned. A capacity rule sees nothing.
        let snap = Snapshot {
            disks: vec![
                DiskRow {
                    mount: "/".into(),
                    kernel_name: "dm-0".into(),
                    total: 100,
                    used: 10,
                    util: Some(0.97),
                    ..Default::default()
                },
                DiskRow {
                    mount: "/boot".into(),
                    kernel_name: "sda1".into(),
                    total: 100,
                    used: 10,
                    util: Some(0.01),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut capacity = AlertEngine::new(vec![rule("full", AlertKind::Disk, 90.0, 0)]);
        assert!(capacity.evaluate(&snap, 0).is_empty());

        let mut busy = AlertEngine::new(vec![rule("busy", AlertKind::Io, 90.0, 0)]);
        assert_eq!(busy.evaluate(&snap, 0).len(), 1, "the busiest device trips it");

        let mut targeted = AlertEngine::new(vec![AlertRule {
            target: "/boot".into(),
            ..rule("boot", AlertKind::Io, 90.0, 0)
        }]);
        assert!(targeted.evaluate(&snap, 0).is_empty());
    }

    #[test]
    fn a_net_rule_measures_bytes_per_second_and_skips_virtual_interfaces() {
        use crate::metrics::NetIface;
        let snap = Snapshot {
            nets: vec![
                NetIface { name: "eth0".into(), rx_bps: 60e6, tx_bps: 10e6, ..Default::default() },
                NetIface {
                    name: "wg0".into(),
                    rx_bps: 55e6,
                    tx_bps: 9e6,
                    virtual_iface: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // 70 MB/s on the wire. Counting the tunnel too would report 134 and
        // trip a rule set just above the link's real capacity.
        let mut e = AlertEngine::new(vec![rule("saturated", AlertKind::Net, 100e6, 0)]);
        assert!(e.evaluate(&snap, 0).is_empty());

        let mut lower = AlertEngine::new(vec![rule("busy", AlertKind::Net, 50e6, 0)]);
        let active = lower.evaluate(&snap, 0);
        assert_eq!(active.len(), 1);
        assert!(active[0].message.contains("B/s"), "{}", active[0].message);
    }

    #[test]
    fn an_fd_rule_measures_each_process_against_its_own_limit() {
        // 4000 descriptors against a million is nothing; 1000 against 1024 is
        // a daemon about to start failing every accept().
        let snap = Snapshot {
            procs: vec![
                ProcRow {
                    pid: 1,
                    name: "envoy".into(),
                    fds: Some(4_000),
                    fd_limit: Some(1_048_576),
                    ..Default::default()
                },
                ProcRow {
                    pid: 2,
                    name: "api".into(),
                    fds: Some(1_000),
                    fd_limit: Some(1_024),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut e = AlertEngine::new(vec![rule("fds", AlertKind::Fd, 90.0, 0)]);
        assert_eq!(e.evaluate(&snap, 0).len(), 1);

        let mut targeted = AlertEngine::new(vec![AlertRule {
            target: "envoy".into(),
            ..rule("envoy-fds", AlertKind::Fd, 90.0, 0)
        }]);
        assert!(targeted.evaluate(&snap, 0).is_empty());

        // With `[procs] fds` off nothing reports a count, and a rule with
        // nothing to measure is skipped rather than reading as 0%.
        let blind = Snapshot {
            procs: vec![ProcRow { pid: 1, name: "api".into(), ..Default::default() }],
            ..Default::default()
        };
        let mut none = AlertEngine::new(vec![rule("fds", AlertKind::Fd, 0.0, 0)]);
        assert!(none.evaluate(&blind, 0).is_empty());
    }

    #[test]
    fn memory_and_swap_rules_read_percentages() {
        let snap = Snapshot {
            mem: MemSample {
                total: 100,
                used: 96,
                swap_total: 100,
                swap_used: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut e = AlertEngine::new(vec![
            rule("mem", AlertKind::Mem, 95.0, 0),
            rule("swap", AlertKind::Swap, 50.0, 0),
        ]);
        let active = e.evaluate(&snap, 0);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "mem");
    }
}
