//! `--watch`: wait for a condition, headless.
//!
//! The alert engine already knows how to hold a condition for a while before
//! believing it, and the filter language already knows how to describe a set
//! of processes. `--watch` is the two of them pointed at a shell script: block
//! until something is true, say what was true, and exit non-zero so `&&` and
//! `||` work.
//!
//! Both halves are expressed as alert rules. A process watch is
//! `kind = "proc"` with a threshold of one — "at least one process matches" —
//! and `--watch-rule` is every other kind, spelled compactly. That is not a
//! coincidence to be tidied away later: this mode used to carry a second
//! hold-time state machine of its own, which is exactly the sort of thing that
//! ends up right in one copy and wrong in the other.

use std::time::{Duration, Instant};

use crate::alerts::{ALL_ALERT_KINDS, AlertKind, AlertRule};
use crate::metrics::Snapshot;

/// Exit code when the watch condition was met. 0 means it never was.
pub const MATCHED: i32 = 1;

/// Exit code when the watch could not be run at all — a query that does not
/// parse, a source that cannot sample, a write that fails.
///
/// Distinct from `MATCHED` because a caller spelling `--watch-rule` wrong
/// otherwise got the same `1` as a successful match, so
/// `crabmon --watch-rule 'psi:oi>20' || page-oncall` paged on a typo. It is
/// the same `2` the argument parser already uses for a usage error, which is
/// also what grep and friends reserve for "something went wrong".
pub const FAILED: i32 = 2;

/// How a watch ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The condition held for long enough. Carries the processes that matched,
    /// which is empty for the kinds that are not about processes.
    Matched(Vec<crate::metrics::ProcRow>),
    /// `--watch-timeout` elapsed first.
    TimedOut,
}

impl Outcome {
    pub fn exit_code(&self) -> i32 {
        match self {
            Outcome::Matched(_) => MATCHED,
            Outcome::TimedOut => 0,
        }
    }
}

/// The rule `--watch <query>` means: at least one process matches.
/// The snapshot a watch judges: this machine without the watcher itself and
/// whatever it runs (`systemctl`, `smartctl`). `crabmon --watch 'cmd:backup'`
/// carries `backup` in its own command line, so it matched itself on the
/// first sample and exited 1 — "it is running" — with no backup in sight.
/// Found on a real host.
///
/// Its ancestors are left out too, for the same reason one level up: whatever
/// started the watch usually carries the query in its own argv. `ssh host 'cd
/// /srv; crabmon --watch cmd:backup'` runs it under `bash -c '...cmd:backup'`,
/// and `sudo`, `timeout`, `nohup` and a cron line do the same, so the watch
/// matched its own parent and reported a backup that was not running — also
/// found on a real host. PID 1 is kept: it starts everything, and is never
/// the thing that started a watch on purpose.
pub fn without_self(snap: &Snapshot, me: u32) -> Snapshot {
    let parent_of: std::collections::HashMap<u32, u32> =
        snap.procs.iter().filter_map(|p| Some((p.pid, p.ppid?))).collect();
    let mut ancestors = std::collections::HashSet::new();
    let mut at = me;
    // Bounded, so a recycled PID that closes a loop cannot hang the watch.
    while let Some(&parent) = parent_of.get(&at) {
        if parent <= 1 || !ancestors.insert(parent) || ancestors.len() > 512 {
            break;
        }
        at = parent;
    }
    let mut judged = snap.clone();
    judged.procs.retain(|p| p.pid != me && p.ppid != Some(me) && !ancestors.contains(&p.pid));
    judged
}

pub fn process_rule(query: &str, for_secs: u64) -> AlertRule {
    AlertRule {
        name: format!("watch {query}"),
        kind: AlertKind::Proc,
        threshold: 1.0,
        hysteresis: 0.0,
        for_secs,
        over_secs: 0,
        below: false,
        target: String::new(),
        query: query.to_string(),
        command: String::new(),
        command_clear: String::new(),
    }
}

/// Parse `--watch-rule`: `kind[:target]<op><value>`.
///
/// ```text
/// cpu>=90          average CPU at or over 90%
/// io>0.9           the busiest device at or over 90% utilised  (see below)
/// psi:io>20        io pressure, 10-second average, at or over 20%
/// disk:/var>=95    that filesystem at or over 95% full
/// temp>=80         the hottest sensor at or over 80 °C
/// fd>=90           some process at or over 90% of its own fd limit
/// net>=1M          aggregate throughput at or over 1 MB/s
/// proc:nginx<1     fewer than one nginx, i.e. it has stopped
/// trend:postgres>200M/10m   postgres's RSS up 200 MB over ten minutes
/// ```
///
/// The kinds that measure over a window — `trend`, `fillin` and `oomkill` —
/// take it after the value as `/DURATION`, which is the `over_secs` an
/// `[[alert]]` sets; without one they use `DEFAULT_TREND_WINDOW`.
///
/// The operator is found by scanning from the *right*, because a `proc`
/// target is a filter query and a filter query can contain one of its own:
/// `proc:cpu>5<1` is "fewer than one process over 5% CPU", and splitting on
/// the first `>` would read it as a comparison against `5<1`.
///
/// `>` and `>=` both mean at-or-over and `<` means strictly under, which is
/// how an alert rule compares and therefore how this has to compare too —
/// a `--watch-rule` that fired on a threshold the identical `[[alert]]` would
/// not would be worse than one spelling fewer operators. `<=` is refused
/// rather than quietly read as `<`.
pub fn parse_rule(expr: &str, for_secs: u64) -> Result<AlertRule, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("empty rule".into());
    }
    let (idx, op) = last_operator(expr).ok_or_else(|| {
        format!("{expr}: no comparison; write something like cpu>=90 or proc:nginx<1")
    })?;
    if op == "<=" {
        return Err("<= is not a comparison an alert rule makes; use <".into());
    }
    let below = op.starts_with('<');
    let (head, value) = (expr[..idx].trim(), expr[idx + op.len()..].trim());
    let (value, window) = match value.rsplit_once('/') {
        Some((v, w)) => (v.trim(), Some(w.trim())),
        None => (value, None),
    };

    let (kind_name, target) = match head.split_once(':') {
        Some((k, t)) => (k.trim(), t.trim()),
        None => (head, ""),
    };
    let kind = ALL_ALERT_KINDS
        .into_iter()
        .find(|k| k.key_name().eq_ignore_ascii_case(kind_name))
        .ok_or_else(|| {
        format!(
            "unknown kind: {kind_name}\nknown kinds: {}",
            ALL_ALERT_KINDS.iter().map(|k| k.key_name()).collect::<Vec<_>>().join(", ")
        )
    })?;

    // `net` is a byte rate, so it takes the same suffixes the filter language
    // takes; everything else is a plain number — a percentage, a temperature,
    // a load average or a count.
    let threshold = match kind {
        // Both are quantities of bytes, so both take the filter language's
        // suffixes: `trend:rss>200M` reads the way it is meant.
        // Seconds, with the filter language's suffixes: `fillin:/var<3h`
        // reads the way it is meant.
        AlertKind::FillIn => crate::format::parse_duration(value)
            .map(|v| v as f64)
            .ok_or_else(|| format!("bad duration: {value}"))?,
        // `vm` is a byte rate too, except `vm:majfault`, where a bare number
        // still parses as itself.
        AlertKind::Net | AlertKind::Trend | AlertKind::Vm => crate::format::parse_size(value)
            .map(|v| v as f64)
            .ok_or_else(|| format!("bad size: {value}"))?,
        _ => value.parse::<f64>().map_err(|_| format!("bad number: {value}"))?,
    };
    if !threshold.is_finite() {
        return Err(format!("bad number: {value}"));
    }
    let over_secs = match window {
        None => 0,
        Some(_) if !matches!(kind, AlertKind::Trend | AlertKind::FillIn | AlertKind::OomKill) => {
            return Err(format!(
                "{} has no window; only trend, fillin and oomkill take /DURATION",
                kind.key_name()
            ));
        }
        Some(w) => match crate::format::parse_duration(w) {
            Some(secs) if secs > 0 => secs,
            _ => return Err(format!("bad window: {w} (expected a duration like 10m)")),
        },
    };

    // A `psi` rule naming a resource the kernel has never heard of measures
    // nothing, and a rule that measures nothing never fires: `psi:oi>20`
    // waited out the whole timeout and then exited 0, which a shell reads as
    // "the condition never happened" rather than "you spelled it wrong".
    if kind == AlertKind::Psi && !crate::alerts::psi_target_is_known(target) {
        return Err(format!(
            "unknown pressure resource: {target}\nknown resources: {} (each also .some/.full)",
            crate::alerts::PSI_RESOURCES.join(", ")
        ));
    }

    if kind == AlertKind::Cputime && !crate::metrics::kstat::cpu_mode_is_known(target) {
        return Err(format!(
            "unknown cpu mode: {target}\nknown modes: {}",
            crate::metrics::kstat::CPU_MODES.join(", ")
        ));
    }
    if kind == AlertKind::Tcp
        && !crate::metrics::netstat::TCP_SERIES.contains(&target)
        && !target.is_empty()
    {
        return Err(format!(
            "unknown tcp figure: {target}\nknown figures: {}",
            crate::metrics::netstat::TCP_SERIES.join(", ")
        ));
    }
    if kind == AlertKind::Vm && !crate::metrics::kstat::vm_target_is_known(target) {
        return Err(format!(
            "unknown vm series: {target}\nknown series: {}",
            crate::metrics::kstat::VM_SERIES.join(", ")
        ));
    }

    // A `trend` rule's target is either a machine series or a process
    // query, and one that is neither measures nothing for ever.
    //
    // A process query may be prefixed with the series to follow:
    // `trend:io:postgres` is "postgres's IO rate", where `trend:postgres`
    // remains "postgres's RSS".
    if kind == AlertKind::Trend && !is_machine_series(target) {
        let (series, query) = crate::alerts::split_trend_target(target);
        if query.trim().is_empty() {
            return Err(format!(
                "{} needs a query, e.g. trend:{}:postgres",
                series.key_name(),
                series.key_name()
            ));
        }
        crate::filter::parse(query).map_err(|e| format!("bad query: {e}"))?;
    }

    // `unit:restarts:nginx` — the figure, then which unit; the same
    // `target`/`query` split an `[[alert]]` is written in.
    if kind == AlertKind::Unit {
        let figure = target.split_once(':').map_or(target, |(f, _)| f);
        if !figure.is_empty() && !crate::metrics::units::UNIT_SERIES.contains(&figure) {
            return Err(format!(
                "unknown unit figure: {figure}\nknown figures: {} (then optionally :unit)",
                crate::metrics::units::UNIT_SERIES.join(", ")
            ));
        }
    }

    if kind == AlertKind::Kernel && !crate::alerts::kernel_target_is_known(target) {
        return Err(format!(
            "unknown kernel event: {target}\nknown events: {}",
            crate::metrics::kmsg::ALL_EVENT_KINDS
                .iter()
                .map(|k| k.key_name())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // A `proc` rule's target is the query it counts matches of, and it has to
    // parse here rather than leaving an inert rule that waits for ever.
    if kind == AlertKind::Proc {
        if target.is_empty() {
            return Err("proc needs a query, e.g. proc:nginx<1".into());
        }
        crate::filter::parse(target).map_err(|e| format!("bad query: {e}"))?;
    }

    Ok(AlertRule {
        name: expr.to_string(),
        kind,
        threshold,
        hysteresis: 0.0,
        for_secs,
        over_secs,
        below,
        // A `trend` target that is not one of the machine series is a
        // process query, which belongs in `query` like a `proc` rule's.
        // A `trend` over processes keeps the series it follows in `target`
        // and the query in `query`, which is the shape an `[[alert]]` is
        // written in — so the two spellings produce the same rule.
        target: match (kind, target) {
            (AlertKind::Proc, _) => String::new(),
            (AlertKind::Trend, q) if !is_machine_series(q) => {
                crate::alerts::split_trend_target(q).0.key_name().to_string()
            }
            (AlertKind::Unit, t) => t.split_once(':').map_or(t, |(f, _)| f).to_string(),
            (_, t) => t.to_string(),
        },
        query: match (kind, target) {
            (AlertKind::Proc, q) => q.to_string(),
            (AlertKind::Trend, q) if !is_machine_series(q) => {
                crate::alerts::split_trend_target(q).1.to_string()
            }
            (AlertKind::Unit, t) => t.split_once(':').map_or("", |(_, q)| q).to_string(),
            _ => String::new(),
        },
        command: String::new(),
        command_clear: String::new(),
    })
}

/// Why `rule` can never fire on this machine because the mount it names is
/// not one crabmon measures, or `None` when it can. `disk` and `fillin`
/// only; the disk list is sysinfo's, which leaves out tmpfs and the other
/// memory filesystems.
pub fn unmeasured_mount(rule: &AlertRule, snap: &crate::metrics::Snapshot) -> Option<String> {
    // A source that can never be read here makes the rule wait for nothing,
    // and a watch that times out exits 0 — "it never happened". Found with
    // `unit:failed>=1` on a host with `[units]` off.
    let unreadable = match rule.kind {
        // "Still being read" passes: the first reading on a busy host can
        // outlast the first sample, and the watch will see it on the next.
        AlertKind::Unit => snap
            .units_unreadable
            .as_deref()
            .filter(|why| *why != crate::metrics::units::STILL_READING)
            .map(|why| ("systemd units are", why)),
        AlertKind::Kernel => {
            snap.kernel_unreadable.as_deref().map(|why| ("the kernel log is", why))
        }
        _ => None,
    };
    if let Some((what, why)) = unreadable {
        return Some(format!(
            "{} cannot measure anything here: {what} not read ({why})",
            rule.name
        ));
    }
    if !matches!(rule.kind, AlertKind::Disk | AlertKind::FillIn) || rule.target.is_empty() {
        return None;
    }
    if snap.disks.iter().any(|d| d.mount == rule.target) {
        return None;
    }
    let known: Vec<&str> = snap.disks.iter().map(|d| d.mount.as_str()).collect();
    Some(format!(
        "no filesystem crabmon measures is mounted at {} (tmpfs and other memory filesystems \
         are not measured)\nmeasured: {}",
        rule.target,
        if known.is_empty() { "none".to_string() } else { known.join(", ") }
    ))
}

/// Whether a `trend` target names a machine-wide series rather than a
/// process query.
fn is_machine_series(target: &str) -> bool {
    matches!(target, "" | "mem" | "memory" | "rss" | "swap")
}

/// The rightmost comparison operator and where it starts.
fn last_operator(expr: &str) -> Option<(usize, &'static str)> {
    let bytes = expr.as_bytes();
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b'>' | b'<' => {
                let two = i + 1 < bytes.len() && bytes[i + 1] == b'=';
                return Some(match (bytes[i], two) {
                    (b'>', true) => (i, ">="),
                    (b'>', false) => (i, ">"),
                    (b'<', true) => (i, "<="),
                    (_, _) => (i, "<"),
                });
            }
            _ => {}
        }
    }
    None
}

/// Whether a watch should give up. `timeout_secs == 0` waits forever.
pub fn timed_out(started: Instant, timeout_secs: u64, now: Instant) -> bool {
    timeout_secs > 0 && now.duration_since(started) >= Duration::from_secs(timeout_secs)
}

/// How long a watch sleeps between samples, never below what the sampler needs.
pub fn interval(refresh_ms: u64) -> Duration {
    Duration::from_millis(refresh_ms.max(crate::MIN_REFRESH_MS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alerts::AlertEngine;
    use crate::metrics::{CpuSample, ProcRow, Snapshot};

    fn snap(states: &[char]) -> Snapshot {
        Snapshot {
            procs: states
                .iter()
                .enumerate()
                .map(|(i, s)| ProcRow {
                    pid: i as u32 + 1,
                    name: "worker".into(),
                    state: *s,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_process_watch_is_a_rule_that_counts_at_least_one_match() {
        let rule = process_rule("state:D", 0);
        assert_eq!(rule.kind, AlertKind::Proc);
        assert_eq!(rule.threshold, 1.0);
        assert!(!rule.below);
        assert_eq!(rule.query, "state:D");
    }

    #[test]
    fn a_watch_does_not_count_itself_or_what_it_runs() {
        use crate::metrics::ProcRow;
        let row = |pid, ppid, cmd: &str| ProcRow {
            pid,
            ppid,
            name: cmd.split(' ').next().unwrap().into(),
            cmd: cmd.into(),
            ..Default::default()
        };
        let snap = Snapshot {
            procs: vec![
                row(10, Some(1), "crabmon --watch cmd:backup.sh"),
                row(11, Some(10), "systemctl show"),
                row(20, Some(1), "/bin/sh backup.sh"),
            ],
            ..Default::default()
        };
        let judged = without_self(&snap, 10);
        assert_eq!(judged.procs.iter().map(|p| p.pid).collect::<Vec<_>>(), [20]);
        let mut e = AlertEngine::new(vec![process_rule("cmd:backup.sh", 0)]);
        let mut only_me = snap.clone();
        only_me.procs.truncate(2);
        assert!(e.evaluate(&without_self(&only_me, 10), 0).is_empty(), "matched itself");
    }

    #[test]
    fn a_watch_does_not_count_the_shell_or_sudo_that_started_it() {
        use crate::metrics::ProcRow;
        let row = |pid, ppid, cmd: &str| ProcRow {
            pid,
            ppid: Some(ppid),
            name: cmd.split(' ').next().unwrap().into(),
            cmd: cmd.into(),
            ..Default::default()
        };
        let snap = Snapshot {
            procs: vec![
                row(1, 0, "/sbin/init cmd:backup.sh"),
                row(5, 1, "bash -c cd /srv; sudo crabmon --watch cmd:backup.sh"),
                row(6, 5, "sudo crabmon --watch cmd:backup.sh"),
                row(7, 6, "crabmon --watch cmd:backup.sh"),
                row(8, 5, "sh backup.sh"),
            ],
            ..Default::default()
        };
        let judged = without_self(&snap, 7);
        // The sibling started by the same shell is a real match; init is kept.
        assert_eq!(judged.procs.iter().map(|p| p.pid).collect::<Vec<_>>(), [1, 8]);
    }

    #[test]
    fn a_query_that_never_matches_never_fires() {
        let mut e = AlertEngine::new(vec![process_rule("state:D", 0)]);
        assert!(e.evaluate(&snap(&['S', 'R']), 0).is_empty());
        assert!(e.evaluate(&snap(&['S', 'R']), 60).is_empty());
    }

    #[test]
    fn a_zero_hold_fires_on_the_first_matching_sample() {
        let mut e = AlertEngine::new(vec![process_rule("state:D", 0)]);
        assert_eq!(e.evaluate(&snap(&['S', 'D']), 0).len(), 1);
    }

    /// A machine that stalls for a second every minute must not satisfy
    /// `--watch-for 30` after half an hour of flickering: thirty *consecutive*
    /// seconds is the promise.
    #[test]
    fn the_hold_must_be_unbroken() {
        let mut e = AlertEngine::new(vec![process_rule("state:D", 30)]);
        assert!(e.evaluate(&snap(&['D']), 0).is_empty());
        assert!(e.evaluate(&snap(&['D']), 20).is_empty());
        assert!(e.evaluate(&snap(&['S']), 21).is_empty(), "the streak breaks");
        assert!(e.evaluate(&snap(&['D']), 40).is_empty(), "the clock restarted at 40");
        assert!(e.evaluate(&snap(&['D']), 69).is_empty(), "one second short");
        assert!(!e.evaluate(&snap(&['D']), 70).is_empty(), "thirty unbroken seconds");
    }

    #[test]
    fn a_timeout_of_zero_waits_forever() {
        let base = Instant::now();
        let at = |s: u64| base + Duration::from_secs(s);
        assert!(!timed_out(base, 0, at(86_400)));
        assert!(!timed_out(base, 60, at(59)));
        assert!(timed_out(base, 60, at(60)));
    }

    /// The disk question, from a shell script.
    #[test]
    fn a_fillin_rule_takes_a_duration_and_a_mount() {
        let r = parse_rule("fillin:/var<3h", 0).expect("a fill rule");
        assert_eq!(r.kind, AlertKind::FillIn);
        assert_eq!(r.threshold, 3.0 * 3600.0);
        assert_eq!(r.target, "/var");
        assert!(r.below, "the alarming direction is *less* time");

        assert!(parse_rule("fillin:/var<nonsense", 0).is_err());
        // An untargeted rule watches whichever mount fills first.
        assert_eq!(parse_rule("fillin<30m", 0).unwrap().target, "");
    }

    /// The leak question, from a shell script.
    #[test]
    fn a_trend_rule_takes_sizes_and_sorts_its_target_from_its_query() {
        let machine = parse_rule("trend:mem>1G", 0).expect("a machine trend rule");
        assert_eq!(machine.kind, AlertKind::Trend);
        assert_eq!(machine.threshold, (1u64 << 30) as f64);
        assert_eq!(machine.target, "mem");
        assert!(machine.query.is_empty());

        let per_proc = parse_rule("trend:postgres>200M", 0).expect("a per-process trend rule");
        assert_eq!(per_proc.query, "postgres");
        assert_eq!(per_proc.target, "rss", "a bare query still means resident memory");

        // A size, not a bare number: 200M is the point.
        assert!(parse_rule("trend:mem>nonsense", 0).is_err());
        assert!(parse_rule("trend:re:[>1G", 0).is_err(), "a broken query must be refused");
    }

    #[test]
    fn a_unit_rule_takes_a_figure_and_then_a_unit() {
        let r = parse_rule("unit:restarts:nginx>=3", 0).unwrap();
        assert_eq!(
            (r.kind, r.target.as_str(), r.query.as_str()),
            (AlertKind::Unit, "restarts", "nginx")
        );
        let r = parse_rule("unit:failed>=1", 0).unwrap();
        assert_eq!((r.target.as_str(), r.query.as_str()), ("failed", ""));
        assert_eq!(parse_rule("unit>=1", 0).unwrap().target, "");
        assert!(parse_rule("unit:nginx>=1", 0).unwrap_err().contains("unknown unit figure"));
    }

    /// Found on a real host: `fillin:/tmp/x<90` on a tmpfs measured nothing
    /// and timed out with exit 0, which a script reads as "it never filled".
    #[test]
    fn a_rule_about_an_unmeasured_mount_is_refused_by_name() {
        let snap = crate::metrics::Snapshot {
            disks: vec![crate::metrics::DiskRow { mount: "/".into(), ..Default::default() }],
            ..Default::default()
        };
        let err = unmeasured_mount(&parse_rule("fillin:/tmp/x<90", 0).unwrap(), &snap).unwrap();
        assert!(err.contains("/tmp/x") && err.contains("measured: /"), "{err}");
        assert!(unmeasured_mount(&parse_rule("disk:/>=90", 0).unwrap(), &snap).is_none());
        assert!(unmeasured_mount(&parse_rule("disk>=90", 0).unwrap(), &snap).is_none(), "any");
        assert!(unmeasured_mount(&parse_rule("cpu>=90", 0).unwrap(), &snap).is_none());

        // Found on a real host with `[units]` off: the watch timed out with
        // exit 0. A source that is not read refuses the rule the same way.
        let mut off = snap.clone();
        off.units_unreadable = Some("[units] enabled is off".into());
        off.kernel_unreadable = Some("needs root or dmesg_restrict=0".into());
        let e = unmeasured_mount(&parse_rule("unit:failed>=1", 0).unwrap(), &off).unwrap();
        assert!(e.contains("systemd units are not read ([units] enabled is off)"), "{e}");
        let e = unmeasured_mount(&parse_rule("kernel:oom>=1", 0).unwrap(), &off).unwrap();
        assert!(e.contains("the kernel log is not read (needs root"), "{e}");
        let mut reading = snap.clone();
        reading.units_unreadable = Some(crate::metrics::units::STILL_READING.into());
        assert!(
            unmeasured_mount(&parse_rule("unit:failed>=1", 0).unwrap(), &reading).is_none(),
            "a first reading still in progress is not a reason to refuse"
        );
        let mut on = snap.clone();
        on.units_unreadable = None;
        assert!(unmeasured_mount(&parse_rule("unit:failed>=1", 0).unwrap(), &on).is_none());
    }

    /// The window an `[[alert]]` sets as `over_secs`, from a shell.
    #[test]
    fn windowed_kinds_take_a_window_after_the_value() {
        let r = parse_rule("trend:postgres>200M/10m", 0).expect("a windowed trend rule");
        assert_eq!(r.over_secs, 600);
        assert_eq!(r.threshold, (200u64 << 20) as f64);
        assert_eq!(r.query, "postgres");
        assert_eq!(parse_rule("trend:postgres>200M", 0).unwrap().over_secs, 0);
        assert_eq!(parse_rule("fillin:/var<3h/30m", 0).unwrap().over_secs, 1800);
        assert_eq!(parse_rule("fillin:/var<3h/30m", 0).unwrap().threshold, 10800.0);
        assert_eq!(parse_rule("oomkill>=1/1h", 0).unwrap().over_secs, 3600);

        for bad in ["trend:mem>1G/", "trend:mem>1G/0", "trend:mem>1G/soon"] {
            assert!(parse_rule(bad, 0).unwrap_err().contains("bad window"), "{bad}");
        }
        assert!(parse_rule("cpu>=90/5m", 0).unwrap_err().contains("has no window"));
    }

    /// RSS is not the only thing that climbs. `trend:io:postgres` follows the
    /// same processes' disk rate and `trend:cpu:postgres` their CPU, in the
    /// same `target`/`query` shape an `[[alert]]` is written in — so the two
    /// spellings cannot come to mean different things.
    #[test]
    fn a_trend_rule_can_name_the_series_it_follows() {
        let io = parse_rule("trend:io:postgres>1M", 0).expect("an io trend rule");
        assert_eq!(io.target, "io");
        assert_eq!(io.query, "postgres");
        assert_eq!(io.threshold, (1u64 << 20) as f64);

        let cpu = parse_rule("trend:cpu:postgres>20", 0).expect("a cpu trend rule");
        assert_eq!(cpu.target, "cpu");
        assert_eq!(cpu.query, "postgres");
        assert_eq!(cpu.threshold, 20.0);

        // The query keeps its own colons: a filter term has them too.
        let scoped = parse_rule("trend:io:user:postgres>1M", 0).expect("a scoped io rule");
        assert_eq!(scoped.target, "io");
        assert_eq!(scoped.query, "user:postgres");

        // `rss` may be said out loud, and means what it always did.
        assert_eq!(parse_rule("trend:rss:nginx>1M", 0).unwrap().query, "nginx");

        // A series with nothing to measure is a rule that waits for ever.
        let err = parse_rule("trend:io:>1M", 0).unwrap_err();
        assert!(err.contains("needs a query"), "{err}");

        // ...and a word that is not a series is still an ordinary query, so
        // the spelling that has always worked goes on working.
        assert_eq!(parse_rule("trend:chrome>1M", 0).unwrap().query, "chrome");
    }

    /// A rule that cannot measure anything never fires, so a typo in the
    /// target used to spend the whole timeout and then report success.
    #[test]
    fn a_misspelled_pressure_resource_is_rejected_rather_than_inert() {
        assert!(parse_rule("psi:oi>20", 0).is_err());
        assert!(parse_rule("psi:memory.ful>20", 0).is_err());
        for ok in ["psi>20", "psi:io>20", "psi:mem.full>20", "psi:CPU.some>1"] {
            assert!(parse_rule(ok, 0).is_ok(), "{ok} was refused");
        }
    }

    #[test]
    fn a_failure_is_distinguishable_from_a_match() {
        // `1` is a match. A watch that could not run must not look like one.
        assert_ne!(FAILED, MATCHED);
        assert_ne!(FAILED, Outcome::TimedOut.exit_code());
    }

    #[test]
    fn exit_codes_make_the_shell_useful() {
        // `crabmon --watch ... && page-someone` has to work.
        assert_eq!(Outcome::Matched(vec![]).exit_code(), 1);
        assert_eq!(Outcome::TimedOut.exit_code(), 0);
    }

    #[test]
    fn a_rule_expression_names_its_kind_its_target_and_its_threshold() {
        let r = parse_rule("cpu>=90", 0).unwrap();
        assert_eq!(r.kind, AlertKind::Cpu);
        assert_eq!(r.threshold, 90.0);
        assert!(!r.below);

        let r = parse_rule("psi:io>20", 5).unwrap();
        assert_eq!(r.kind, AlertKind::Psi);
        assert_eq!(r.target, "io");
        assert_eq!(r.for_secs, 5);

        let r = parse_rule("disk:/var>=95", 0).unwrap();
        assert_eq!(r.target, "/var");

        // Spacing and case are what someone actually types.
        let r = parse_rule(" TEMP >= 80 ", 0).unwrap();
        assert_eq!(r.kind, AlertKind::Temp);
        assert_eq!(r.threshold, 80.0);
    }

    #[test]
    fn under_a_threshold_is_how_you_wait_for_something_to_stop() {
        let r = parse_rule("proc:nginx<1", 0).unwrap();
        assert_eq!(r.kind, AlertKind::Proc);
        assert_eq!(r.query, "nginx");
        assert_eq!(r.threshold, 1.0);
        assert!(r.below, "`<` is what makes this process-absence alerting");
        assert!(r.target.is_empty(), "a proc rule's target is its query");
    }

    /// The reason the operator is found from the right: a `proc` target is a
    /// filter query, and a filter query contains comparisons of its own.
    #[test]
    fn a_query_containing_a_comparison_is_not_split_on_its_own_operator() {
        let r = parse_rule("proc:cpu>5<1", 0).unwrap();
        assert_eq!(r.query, "cpu>5", "the query keeps its own comparison");
        assert_eq!(r.threshold, 1.0);
        assert!(r.below);

        // ...and it is a query that actually parses, so the rule is not inert.
        let f = crate::filter::parse(&r.query).unwrap();
        assert!(f.matches(&ProcRow { cpu: 50.0, ..Default::default() }));
    }

    #[test]
    fn a_throughput_threshold_takes_the_suffixes_a_rate_is_written_with() {
        assert_eq!(parse_rule("net>=1M", 0).unwrap().threshold, 1_048_576.0);
        assert_eq!(parse_rule("net>100K", 0).unwrap().threshold, 102_400.0);
        // Percentages and counts stay plain numbers.
        assert_eq!(parse_rule("cpu>=90", 0).unwrap().threshold, 90.0);
    }

    #[test]
    fn a_rule_that_cannot_be_meant_is_refused_rather_than_waiting_for_ever() {
        // Every one of these used to be a watch that blocked until its
        // timeout and then reported, truthfully, that nothing had happened.
        assert!(parse_rule("", 0).unwrap_err().contains("empty"));
        assert!(parse_rule("cpu", 0).unwrap_err().contains("no comparison"));
        assert!(parse_rule("nonsense>1", 0).unwrap_err().contains("unknown kind"));
        assert!(parse_rule("cpu>abc", 0).unwrap_err().contains("bad number"));
        assert!(parse_rule("net>huge", 0).unwrap_err().contains("bad size"));
        assert!(parse_rule("proc<1", 0).unwrap_err().contains("needs a query"));
        assert!(parse_rule("proc:re:[unclosed<1", 0).unwrap_err().contains("bad query"));
        assert!(parse_rule("cpu>nan", 0).unwrap_err().contains("bad number"));
    }

    /// An alert rule compares `>=` or `<` and nothing else. Accepting `<=`
    /// and quietly treating it as `<` would make `--watch-rule` fire on a
    /// threshold the identical `[[alert]]` would not.
    #[test]
    fn an_operator_no_alert_rule_has_is_refused_rather_than_approximated() {
        let err = parse_rule("cpu<=90", 0).unwrap_err();
        assert!(err.contains("use <"), "{err}");
    }

    #[test]
    fn cpu_mode_and_vm_rules_check_their_target_and_take_rate_suffixes() {
        let r = parse_rule("cputime:steal>=20", 0).unwrap();
        assert_eq!((r.kind, r.target.as_str(), r.threshold), (AlertKind::Cputime, "steal", 20.0));
        let err = parse_rule("cputime:stael>20", 0).unwrap_err();
        assert!(err.contains("steal") && err.contains("iowait"), "{err}");
        let r = parse_rule("vm:swapin>1M", 0).unwrap();
        assert_eq!(r.threshold, (1 << 20) as f64);
        assert_eq!(parse_rule("vm:majfault>200", 0).unwrap().threshold, 200.0);
        assert!(parse_rule("vm:swpin>1M", 0).unwrap_err().contains("swapin"));
    }

    #[test]
    fn tcp_rules_check_their_figure_and_container_rules_take_any_name() {
        assert_eq!(parse_rule("tcp:listendrops>0", 0).unwrap().target, "listendrops");
        assert_eq!(parse_rule("tcp>2", 0).unwrap().kind, AlertKind::Tcp);
        assert!(parse_rule("tcp:retrains>2", 0).unwrap_err().contains("closewait"));
        assert_eq!(parse_rule("throttle:api>=25", 0).unwrap().target, "api");
        assert_eq!(parse_rule("oomkill:api>=1", 0).unwrap().kind, AlertKind::OomKill);
    }

    #[test]
    fn the_unknown_kind_error_lists_what_is_accepted() {
        let err = parse_rule("cpuu>1", 0).unwrap_err();
        for kind in ALL_ALERT_KINDS {
            assert!(err.contains(kind.key_name()), "{} missing from {err}", kind.key_name());
        }
    }

    /// Every kind an `[[alert]]` can use is one `--watch-rule` can wait for.
    /// Two lists of the same thing is how one of them goes stale.
    #[test]
    fn every_alert_kind_can_be_watched_for() {
        for kind in ALL_ALERT_KINDS {
            let expr = match kind {
                AlertKind::Proc => "proc:nginx>=1".to_string(),
                AlertKind::Net => format!("{}>=1M", kind.key_name()),
                _ => format!("{}>=1", kind.key_name()),
            };
            let rule = parse_rule(&expr, 0).unwrap_or_else(|e| panic!("{expr}: {e}"));
            assert_eq!(rule.kind, kind);
        }
    }

    #[test]
    fn a_rule_watch_fires_on_the_thing_it_describes() {
        let hot = Snapshot {
            cpu: CpuSample { per_core: vec![95.0], freq_mhz: vec![] },
            ..Default::default()
        };
        let cool = Snapshot {
            cpu: CpuSample { per_core: vec![5.0], freq_mhz: vec![] },
            ..Default::default()
        };
        let mut e = AlertEngine::new(vec![parse_rule("cpu>=90", 0).unwrap()]);
        assert!(e.evaluate(&cool, 0).is_empty());
        assert!(!e.evaluate(&hot, 1).is_empty());
    }

    #[test]
    fn the_sampling_interval_never_goes_below_what_the_sampler_needs() {
        assert_eq!(interval(1), Duration::from_millis(crate::MIN_REFRESH_MS));
        assert_eq!(interval(5_000), Duration::from_millis(5_000));
    }
}
