//! Alert silencing: maintenance windows from the config, and silences set
//! from the interface.
//!
//! A silenced rule is still evaluated, still drawn and still recorded — the
//! flight recorder keeps its evidence — but its hooks and notifications are
//! held back. When the silence ends the engine catches up: a rule still
//! firing pages then, one that fired and recovered unseen says nothing. That
//! is the difference between a silence and switching the rule off: the
//! deploy that crash-loops a service for ten minutes pages nobody, and the
//! one that leaves it broken pages the moment the window closes.
//!
//! All times are UTC, like every other timestamp crabmon writes. A local
//! wall-clock window would need the machine's time-zone database, and a
//! window that means a different hour on each host of a fleet is not a
//! maintenance window anybody scheduled.

use serde::{Deserialize, Serialize};

/// A `[[silence]]` table: a scheduled window in which some rules on some
/// hosts page nobody.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SilenceWindow {
    /// What the window is for, shown wherever the silence is: "nightly
    /// backup" explains a quiet pager where "silenced" does not.
    pub comment: String,
    /// Rules by name; empty means every rule.
    pub rules: Vec<String>,
    /// Hosts by fleet target or hostname; empty means every host.
    pub hosts: Vec<String>,
    /// A one-off window, `2026-11-20T02:00Z` to `2026-11-20T04:00Z`.
    pub start: String,
    pub end: String,
    /// Or one every day, `02:00-03:30`; it may cross midnight.
    pub daily: String,
}

/// A silence set from the interface: one rule, or every rule, until then.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Muted {
    pub rule: Option<String>,
    pub until_unix: u64,
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`), so a date parses without a calendar crate.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `HH:MM` or `HH:MM:SS` as seconds into the day.
fn time_of_day(s: &str) -> Option<u64> {
    let mut parts = s.split(':');
    let h: u64 = parts.next()?.parse().ok()?;
    let m: u64 = parts.next()?.parse().ok()?;
    let sec: u64 = match parts.next() {
        Some(p) => p.parse().ok()?,
        None => 0,
    };
    if parts.next().is_some() || h > 23 || m > 59 || sec > 59 {
        return None;
    }
    Some(h * 3600 + m * 60 + sec)
}

/// `2026-11-20T02:00Z`, `2026-11-20T02:00:30Z` or `2026-11-20 02:00` as Unix
/// seconds, UTC. A trailing `Z` is optional — the time is UTC either way —
/// and any other offset is refused rather than quietly ignored.
pub fn parse_utc(s: &str) -> Option<u64> {
    let s = s.trim().strip_suffix('Z').unwrap_or(s.trim());
    let (date, time) = s.split_once(['T', ' '])?;
    if time.contains(['+', '-']) {
        return None;
    }
    let mut d = date.split('-');
    let (y, m, day): (i64, i64, i64) =
        (d.next()?.parse().ok()?, d.next()?.parse().ok()?, d.next()?.parse().ok()?);
    if d.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&day) {
        return None;
    }
    let base = u64::try_from(days_from_civil(y, m, day) * 86_400).ok()?;
    Some(base + time_of_day(time)?)
}

/// `HH:MM-HH:MM` as a start and end in seconds into the day.
fn parse_daily(s: &str) -> Option<(u64, u64)> {
    let (a, b) = s.trim().split_once('-')?;
    let (a, b) = (time_of_day(a.trim())?, time_of_day(b.trim())?);
    (a != b).then_some((a, b))
}

/// `HH:MM` UTC of a Unix time, for saying when a silence ends.
pub fn clock(unix: u64) -> String {
    let secs = unix % 86_400;
    format!("{:02}:{:02} UTC", secs / 3600, secs % 3600 / 60)
}

impl SilenceWindow {
    /// What the window is called in messages.
    fn label(&self) -> String {
        match self.comment.is_empty() {
            true => "maintenance window".to_string(),
            false => format!("maintenance window \"{}\"", self.comment),
        }
    }

    /// The end of the window around `at_unix`, or `None` when it is not
    /// open then.
    pub fn open_until(&self, at_unix: u64) -> Option<u64> {
        if !self.daily.is_empty() {
            let (from, to) = parse_daily(&self.daily)?;
            let day = at_unix - at_unix % 86_400;
            let now = at_unix % 86_400;
            return match from < to {
                true => (from <= now && now < to).then_some(day + to),
                // Across midnight: open from `from` to the end of the day,
                // and from midnight to `to`.
                false if now >= from => Some(day + 86_400 + to),
                false => (now < to).then_some(day + to),
            };
        }
        let (start, end) = (parse_utc(&self.start)?, parse_utc(&self.end)?);
        (start <= at_unix && at_unix < end).then_some(end)
    }

    fn names(&self, rule: &str, hosts: &[&str]) -> bool {
        (self.rules.is_empty() || self.rules.iter().any(|r| r == rule))
            && (self.hosts.is_empty()
                || self.hosts.iter().any(|h| hosts.iter().any(|x| !x.is_empty() && x == h)))
    }
}

/// Refuse a window that could never open, so a typo fails at start rather
/// than at three in the morning, with the pager going off.
pub fn validate(windows: &[SilenceWindow]) -> Result<(), String> {
    for (i, w) in windows.iter().enumerate() {
        let which = match w.comment.is_empty() {
            true => format!("[[silence]] #{}", i + 1),
            false => format!("[[silence]] \"{}\"", w.comment),
        };
        match (w.daily.is_empty(), w.start.is_empty() && w.end.is_empty()) {
            (false, true) => {
                parse_daily(&w.daily).ok_or_else(|| {
                    format!("{which}: daily must be HH:MM-HH:MM in UTC, got {:?}", w.daily)
                })?;
            }
            (true, false) => {
                let start = parse_utc(&w.start).ok_or_else(|| {
                    format!(
                        "{which}: start must be like 2026-11-20T02:00Z (UTC), got {:?}",
                        w.start
                    )
                })?;
                let end = parse_utc(&w.end).ok_or_else(|| {
                    format!("{which}: end must be like 2026-11-20T04:00Z (UTC), got {:?}", w.end)
                })?;
                if end <= start {
                    return Err(format!("{which}: end is not after start"));
                }
            }
            (true, true) => return Err(format!("{which}: needs start and end, or daily")),
            (false, false) => return Err(format!("{which}: has both daily and start/end")),
        }
    }
    Ok(())
}

/// Why `rule` is silenced at `at_unix` on a host known by any of `hosts`,
/// or `None` when it is not. Silences from the interface are checked first,
/// because they are the ones somebody set just now on purpose.
pub fn covering(
    windows: &[SilenceWindow],
    muted: &[Muted],
    rule: &str,
    hosts: &[&str],
    at_unix: u64,
) -> Option<String> {
    if let Some(m) = muted
        .iter()
        .filter(|m| m.rule.as_deref().is_none_or(|r| r == rule) && at_unix < m.until_unix)
        .max_by_key(|m| m.until_unix)
    {
        return Some(format!("silenced until {}", clock(m.until_unix)));
    }
    windows
        .iter()
        .filter(|w| w.names(rule, hosts))
        .filter_map(|w| w.open_until(at_unix).map(|until| (w, until)))
        .max_by_key(|(_, until)| *until)
        .map(|(w, until)| format!("{} until {}", w.label(), clock(until)))
}

/// What the `S` prompt was asked: how long, and which rule (none = every
/// rule), or to lift silences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SilenceRequest {
    For { secs: u64, rule: Option<String> },
    Lift { rule: Option<String> },
}

/// `1h`, `30m cpu-saturated`, `off`, `off cpu-saturated`. The rule is the
/// rest of the line, so a name with spaces in it still works, and must be
/// one of `rules`.
pub fn parse_request(input: &str, rules: &[String]) -> Result<SilenceRequest, String> {
    let input = input.trim();
    let (head, rest) = match input.split_once(char::is_whitespace) {
        Some((h, r)) => (h, Some(r.trim()).filter(|r| !r.is_empty())),
        None => (input, None),
    };
    let rule = match rest {
        None => None,
        Some(name) if rules.iter().any(|r| r == name) => Some(name.to_string()),
        Some(name) => return Err(format!("no rule named {name:?}")),
    };
    if head.eq_ignore_ascii_case("off") {
        return Ok(SilenceRequest::Lift { rule });
    }
    match crate::format::parse_duration(head) {
        Some(secs) if secs > 0 => Ok(SilenceRequest::For { secs, rule }),
        _ => Err(format!("expected a duration like 1h or 30m, or off: {head:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_times_parse_and_offsets_are_refused() {
        assert_eq!(parse_utc("1970-01-01T00:00Z"), Some(0));
        assert_eq!(parse_utc("2026-11-20T02:00Z"), Some(1_795_140_000));
        assert_eq!(parse_utc("2026-11-20 02:00:30"), Some(1_795_140_030));
        assert_eq!(parse_utc("2024-02-29T00:00Z"), Some(1_709_164_800));
        assert_eq!(parse_utc("2026-11-20T02:00+03:00"), None, "only UTC");
        assert_eq!(parse_utc("2026-13-01T00:00Z"), None);
        assert_eq!(parse_utc("2026-11-20T25:00Z"), None);
        assert_eq!(parse_utc("tomorrow"), None);
    }

    fn daily(spec: &str) -> SilenceWindow {
        SilenceWindow { daily: spec.into(), ..Default::default() }
    }

    #[test]
    fn a_daily_window_opens_every_day_and_may_cross_midnight() {
        let day = 1_795_132_800; // 2026-11-20T00:00Z
        let w = daily("02:00-03:30");
        assert_eq!(w.open_until(day + 3600), None);
        assert_eq!(w.open_until(day + 2 * 3600), Some(day + 3 * 3600 + 1800));
        assert_eq!(w.open_until(day + 3 * 3600 + 1800), None, "the end is not inside");
        let night = daily("23:00-01:00");
        assert_eq!(night.open_until(day + 23 * 3600 + 60), Some(day + 86_400 + 3600));
        assert_eq!(night.open_until(day + 1800), Some(day + 3600));
        assert_eq!(night.open_until(day + 12 * 3600), None);
    }

    #[test]
    fn windows_that_cannot_open_are_refused_by_name() {
        assert!(validate(&[daily("02:00-03:30")]).is_ok());
        let bad = |w: SilenceWindow| validate(&[w]).unwrap_err();
        assert!(bad(daily("2am-3am")).contains("HH:MM-HH:MM"));
        assert!(bad(SilenceWindow::default()).contains("needs start and end, or daily"));
        let mut w = SilenceWindow {
            comment: "deploy".into(),
            start: "2026-11-20T04:00Z".into(),
            end: "2026-11-20T02:00Z".into(),
            ..Default::default()
        };
        assert!(bad(w.clone()).contains("\"deploy\": end is not after start"));
        w.daily = "02:00-03:00".into();
        assert!(bad(w).contains("both"));
    }

    #[test]
    fn a_silence_covers_its_rules_hosts_and_time_only() {
        let at = parse_utc("2026-11-20T02:30Z").unwrap();
        let w = SilenceWindow {
            comment: "backup".into(),
            rules: vec!["disk-busy".into()],
            hosts: vec!["db-1".into()],
            daily: "02:00-03:00".into(),
            ..Default::default()
        };
        let cover = |rule: &str, hosts: &[&str], at: u64| {
            covering(std::slice::from_ref(&w), &[], rule, hosts, at)
        };
        assert_eq!(
            cover("disk-busy", &["db-1"], at).as_deref(),
            Some("maintenance window \"backup\" until 03:00 UTC")
        );
        assert_eq!(cover("disk-busy", &["web-1", "db-1.internal"], at), None);
        assert_eq!(cover("cpu", &["db-1"], at), None);
        assert_eq!(cover("disk-busy", &["db-1"], at + 3600), None);

        // One set from the interface wins, and covers every rule when it
        // names none.
        let muted = [Muted { rule: None, until_unix: at + 600 }];
        assert_eq!(
            covering(&[], &muted, "cpu", &["web-1"], at).as_deref(),
            Some("silenced until 02:40 UTC")
        );
        assert_eq!(covering(&[], &muted, "cpu", &[], at + 600), None, "over at its end");
    }

    #[test]
    fn the_prompt_takes_a_duration_a_rule_or_off() {
        let rules = vec!["cpu-saturated".to_string(), "disk full".to_string()];
        assert_eq!(parse_request("1h", &rules), Ok(SilenceRequest::For { secs: 3600, rule: None }));
        assert_eq!(
            parse_request("30m disk full", &rules),
            Ok(SilenceRequest::For { secs: 1800, rule: Some("disk full".into()) })
        );
        assert_eq!(parse_request("off", &rules), Ok(SilenceRequest::Lift { rule: None }));
        assert!(parse_request("1h nope", &rules).unwrap_err().contains("no rule named"));
        assert!(parse_request("0", &rules).is_err());
        assert!(parse_request("soon", &rules).is_err());
    }
}
