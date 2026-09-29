//! Threshold alerts. A rule fires only after its condition has held for
//! `for_secs`, which keeps a single busy frame from lighting up the status bar.

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
}

/// Every kind a rule can be, so the docs can be checked against the code
/// rather than against a hand-maintained list — the same reason `ALL_SORTS`
/// and `ALL_GROUPINGS` exist.
pub const ALL_ALERT_KINDS: [AlertKind; 12] = [
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
struct RuleState {
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
    state: Vec<RuleState>,
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
}

impl AlertEngine {
    pub fn new(rules: Vec<AlertRule>) -> Self {
        let state = vec![RuleState::default(); rules.len()];
        let queries = rules
            .iter()
            .map(|r| (r.kind == AlertKind::Proc).then(|| filter::parse(&r.query).ok()).flatten())
            .collect();
        Self {
            rules,
            state,
            queries,
            pending_commands: Vec::new(),
            pending_fired: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Evaluate every rule against `snap`. `now_secs` is a monotonic clock,
    /// injected so the hold-time logic is testable.
    pub fn evaluate(&mut self, snap: &Snapshot, now_secs: u64) -> Vec<ActiveAlert> {
        let mut active = Vec::new();
        let at_unix = snap.taken_at_unix;
        let mut events: Vec<AlertEvent> = Vec::new();
        for (i, rule) in self.rules.iter().enumerate() {
            let Some((value, unit)) =
                measure(rule, snap, self.queries.get(i).and_then(|q| q.as_ref()))
            else {
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
            active.push(ActiveAlert {
                name: rule.name.clone(),
                message: format!(
                    "{} {:.1}{} {} {:.1}{}",
                    rule.name,
                    value,
                    unit,
                    if rule.below { "<" } else { "≥" },
                    rule.threshold,
                    unit
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

/// The measured value for a rule plus the unit used in its message.
///
/// `query` is the pre-parsed filter for `proc` rules, and `None` for every
/// other kind.
fn measure(
    rule: &AlertRule,
    snap: &Snapshot,
    query: Option<&filter::Filter>,
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
                .map(|t| (t, "°C"))
        }
        AlertKind::Gpu => {
            return snap
                .gpus
                .iter()
                .filter(|g| rule.target.is_empty() || g.name.contains(&rule.target))
                .filter_map(|g| g.busy_percent)
                .fold(None, |acc: Option<f64>, b| Some(acc.map_or(b as f64, |a| a.max(b as f64))))
                .map(|b| (b, "%"))
        }
        AlertKind::Psi => return psi_value(&rule.target, snap).map(|v| (v, "%")),
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
            .map(|u| (u * 100.0, "%"))
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
                .map(|r| (r * 100.0, "%"))
        }
        AlertKind::Proc => {
            // A query that did not parse leaves the rule inert. Counting zero
            // instead would make every `below` rule fire for ever on a typo.
            let q = query?;
            return Some((snap.procs.iter().filter(|p| q.matches(p)).count() as f64, ""));
        }
    };
    if v.is_nan() {
        None
    } else {
        Some((v, "%"))
    }
}

/// The PSI figure a rule's `target` names: `cpu`, `mem`/`memory` or `io`,
/// optionally suffixed `.full`. An empty target is the worst `some` reading
/// across all three, which is the same number the status line shows.
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
    if full {
        pressure.full.map(|l| l.avg10)
    } else {
        Some(pressure.some.avg10)
    }
}

/// Default rules written into a fresh config file.
pub fn default_rules() -> Vec<AlertRule> {
    vec![
        AlertRule {
            name: "cpu-saturated".into(),
            kind: AlertKind::Cpu,
            threshold: 90.0,
            for_secs: 30,
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
            below: false,
            target: String::new(),
            query: String::new(),
            command: String::new(),
            command_clear: String::new(),
        },
    ]
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
