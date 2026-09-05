//! Threshold alerts. A rule fires only after its condition has held for
//! `for_secs`, which keeps a single busy frame from lighting up the status bar.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub kind: AlertKind,
    /// Percent for cpu/mem/swap/disk/gpu, degrees for temp, absolute for load.
    pub threshold: f64,
    #[serde(default)]
    pub for_secs: u64,
    /// Restricts disk rules to a mount point and temp rules to a sensor prefix.
    #[serde(default)]
    pub target: String,
    /// Shell command run once when the alert becomes active.
    #[serde(default)]
    pub command: String,
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

#[derive(Debug, Default)]
pub struct AlertEngine {
    pub rules: Vec<AlertRule>,
    state: HashMap<String, RuleState>,
    /// Commands the engine wants run; drained by the caller.
    pending_commands: Vec<String>,
}

impl AlertEngine {
    pub fn new(rules: Vec<AlertRule>) -> Self {
        Self { rules, state: HashMap::new(), pending_commands: Vec::new() }
    }

    /// Evaluate every rule against `snap`. `now_secs` is a monotonic clock,
    /// injected so the hold-time logic is testable.
    pub fn evaluate(&mut self, snap: &Snapshot, now_secs: u64) -> Vec<ActiveAlert> {
        let mut active = Vec::new();
        for rule in &self.rules {
            let Some((value, unit)) = measure(rule, snap) else {
                continue;
            };
            let st = self.state.entry(rule.name.clone()).or_default();
            if value < rule.threshold {
                st.over_since = None;
                st.fired = false;
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
            }
            active.push(ActiveAlert {
                name: rule.name.clone(),
                message: format!(
                    "{} {:.1}{} ≥ {:.1}{}",
                    rule.name, value, unit, rule.threshold, unit
                ),
                value,
                threshold: rule.threshold,
            });
        }
        active
    }

    pub fn take_commands(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_commands)
    }
}

/// The measured value for a rule plus the unit used in its message.
fn measure(rule: &AlertRule, snap: &Snapshot) -> Option<(f64, &'static str)> {
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
    };
    if v.is_nan() {
        None
    } else {
        Some((v, "%"))
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
            target: String::new(),
            command: String::new(),
        },
        AlertRule {
            name: "disk-nearly-full".into(),
            kind: AlertKind::Disk,
            threshold: 95.0,
            for_secs: 0,
            target: String::new(),
            command: String::new(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, DiskRow, MemSample, Sensor};

    fn rule(name: &str, kind: AlertKind, threshold: f64, for_secs: u64) -> AlertRule {
        AlertRule {
            name: name.into(),
            kind,
            threshold,
            for_secs,
            target: String::new(),
            command: String::new(),
        }
    }

    fn snap_cpu(pct: f32) -> Snapshot {
        Snapshot { cpu: CpuSample { per_core: vec![pct], freq_mhz: vec![] }, ..Default::default() }
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
