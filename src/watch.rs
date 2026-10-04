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

use crate::alerts::{AlertKind, AlertRule, ALL_ALERT_KINDS};

/// Exit code when the watch condition was met. 0 means it never was.
pub const MATCHED: i32 = 1;

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
pub fn process_rule(query: &str, for_secs: u64) -> AlertRule {
    AlertRule {
        name: format!("watch {query}"),
        kind: AlertKind::Proc,
        threshold: 1.0,
        for_secs,
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
/// ```
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
        AlertKind::Net => crate::format::parse_size(value)
            .map(|v| v as f64)
            .ok_or_else(|| format!("bad size: {value}"))?,
        _ => value.parse::<f64>().map_err(|_| format!("bad number: {value}"))?,
    };
    if !threshold.is_finite() {
        return Err(format!("bad number: {value}"));
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
        for_secs,
        below,
        target: if kind == AlertKind::Proc { String::new() } else { target.to_string() },
        query: if kind == AlertKind::Proc { target.to_string() } else { String::new() },
        command: String::new(),
        command_clear: String::new(),
    })
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
