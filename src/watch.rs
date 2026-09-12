//! `--watch`: wait for a condition, headless.
//!
//! The alert engine already knows how to hold a condition for a while before
//! believing it, and the filter language already knows how to describe a set of
//! processes. `--watch` is the two of them pointed at a shell script: block
//! until something matches, print what matched, and exit non-zero so `&&` and
//! `||` work.

use std::time::{Duration, Instant};

use crate::filter::Filter;
use crate::metrics::Snapshot;

/// Exit code when the watch condition was met. 0 means it never was.
pub const MATCHED: i32 = 1;

/// How a watch ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The query held for long enough. Carries the matching rows.
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

/// Tracks how long a query has matched without a break.
///
/// A condition that flickers in and out must not accumulate credit across the
/// gaps: `--watch-for 30` has to mean thirty *consecutive* seconds, or a
/// machine that is briefly busy once a minute would trip it.
#[derive(Debug, Default)]
pub struct Holding {
    since: Option<Instant>,
}

impl Holding {
    /// Feed one sample. Returns the matching rows once the query has held for
    /// `for_secs`, and `None` until then.
    pub fn observe(
        &mut self,
        snap: &Snapshot,
        filter: &Filter,
        for_secs: u64,
        now: Instant,
    ) -> Option<Vec<crate::metrics::ProcRow>> {
        let matches: Vec<_> = snap.procs.iter().filter(|p| filter.matches(p)).cloned().collect();
        if matches.is_empty() {
            self.since = None;
            return None;
        }
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since) < Duration::from_secs(for_secs) {
            return None;
        }
        Some(matches)
    }

    /// Seconds the condition has held, for a progress message.
    pub fn held_for(&self, now: Instant) -> u64 {
        self.since.map(|s| now.duration_since(s).as_secs()).unwrap_or(0)
    }
}

/// Whether a watch should give up. `timeout_secs == 0` waits forever.
pub fn timed_out(started: Instant, timeout_secs: u64, now: Instant) -> bool {
    timeout_secs > 0 && now.duration_since(started) >= Duration::from_secs(timeout_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ProcRow;

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

    /// A fixed timeline, so the assertions are about the logic and not about
    /// how long the test itself took.
    fn clock() -> impl Fn(u64) -> Instant {
        let base = Instant::now();
        move |secs| base + Duration::from_secs(secs)
    }

    #[test]
    fn a_query_that_never_matches_never_fires() {
        let at = clock();
        let f = crate::filter::parse("state:D").unwrap();
        let mut h = Holding::default();
        assert!(h.observe(&snap(&['S', 'R']), &f, 0, at(0)).is_none());
        assert!(h.observe(&snap(&['S', 'R']), &f, 0, at(60)).is_none());
    }

    #[test]
    fn a_zero_hold_fires_on_the_first_matching_sample() {
        let at = clock();
        let f = crate::filter::parse("state:D").unwrap();
        let mut h = Holding::default();
        let hit = h.observe(&snap(&['S', 'D']), &f, 0, at(0)).expect("should fire");
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].pid, 2);
    }

    #[test]
    fn the_hold_must_be_unbroken() {
        // A machine that stalls for a second every minute must not satisfy
        // `--watch-for 30` after half an hour of flickering.
        let at = clock();
        let f = crate::filter::parse("state:D").unwrap();
        let mut h = Holding::default();
        assert!(h.observe(&snap(&['D']), &f, 30, at(0)).is_none());
        assert!(h.observe(&snap(&['D']), &f, 30, at(20)).is_none());
        assert!(h.observe(&snap(&['S']), &f, 30, at(21)).is_none(), "the streak breaks");
        assert!(h.observe(&snap(&['D']), &f, 30, at(40)).is_none(), "the clock restarted at 40");
        assert!(h.observe(&snap(&['D']), &f, 30, at(69)).is_none(), "one second short");
        assert!(h.observe(&snap(&['D']), &f, 30, at(70)).is_some(), "thirty unbroken seconds");
    }

    #[test]
    fn a_timeout_of_zero_waits_forever() {
        let at = clock();
        let start = at(0);
        assert!(!timed_out(start, 0, at(86_400)));
        assert!(!timed_out(start, 60, at(59)));
        assert!(timed_out(start, 60, at(60)));
    }

    #[test]
    fn exit_codes_make_the_shell_useful() {
        // `crabmon --watch ... && page-someone` has to work.
        assert_eq!(Outcome::Matched(vec![]).exit_code(), 1);
        assert_eq!(Outcome::TimedOut.exit_code(), 0);
    }

    #[test]
    fn the_hold_duration_is_reportable_while_waiting() {
        let at = clock();
        let f = crate::filter::parse("state:D").unwrap();
        let mut h = Holding::default();
        h.observe(&snap(&['D']), &f, 30, at(0));
        assert_eq!(h.held_for(at(10)), 10);
    }
}
