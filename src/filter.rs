//! The process filter language.
//!
//! Whitespace-separated terms, all ANDed, each optionally negated with `!`.
//! A bare `|` between terms is OR, which binds looser than the implicit AND,
//! so `a b | c` is `(a AND b) OR c`:
//!
//! ```text
//! firefox            name contains "firefox"
//! !kworker           name does not contain "kworker"
//! user:vlad          user name contains "vlad"
//! pid:1234           exact pid       ppid:1  exact parent
//! state:R            process state letter
//! service:sshd       systemd unit contains "sshd"
//! container:abc123   container id contains "abc123"
//! cmd:--headless     substring of the full command line
//! re:^chrom(e|ium)$  regex over the name
//! cpu>5  mem>100M    numeric comparisons (> >= < <= =)
//! virt>2G  thr>50    virtual size, thread count
//! nice<0   fd>1000   scheduling priority, open descriptors
//! time>2h            run time, in s/m/h/d
//! nginx | apache     either
//! ```
//!
//! OR is spelled with a standalone `|` token rather than by splitting on the
//! character, because `re:^chrom(e|ium)$` contains one and splitting it would
//! turn a working regex into two broken queries.

use regex::Regex;

use crate::format::{parse_duration, parse_size};
use crate::metrics::ProcRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

impl Cmp {
    fn test(self, a: f64, b: f64) -> bool {
        match self {
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Eq => (a - b).abs() < f64::EPSILON,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Term {
    Name(String),
    Cmd(String),
    User(String),
    Pid(u32),
    Ppid(u32),
    State(char),
    /// systemd unit; `service:-` matches processes with no unit at all.
    Service(String),
    /// Container id; `container:-` matches processes outside any container.
    Container(String),
    Regex(Regex),
    Cpu(Cmp, f64),
    Mem(Cmp, f64),
    Io(Cmp, f64),
    Virt(Cmp, f64),
    /// Thread count. A row that does not report one — a thread itself — never
    /// matches, rather than counting as zero threads.
    Threads(Cmp, f64),
    Nice(Cmp, f64),
    /// Seconds since the process started.
    Time(Cmp, f64),
    /// Open file descriptors. Needs `[procs] fds`; without it no row reports a
    /// count and the term matches nothing, which is the honest answer.
    Fds(Cmp, f64),
    /// Descriptors as a percentage of the process's own soft limit.
    FdPercent(Cmp, f64),
}

#[derive(Debug, Clone)]
pub struct FilterTerm {
    pub negated: bool,
    pub term: Term,
}

/// One alternative: terms that must all hold.
#[derive(Debug, Clone, Default)]
pub struct Conjunction {
    pub terms: Vec<FilterTerm>,
}

impl Conjunction {
    fn matches(&self, p: &ProcRow) -> bool {
        self.terms.iter().all(|t| t.term.matches(p) != t.negated)
    }
}

/// A disjunction of conjunctions: `a b | c` is `(a AND b) OR c`. One group is
/// the shape every query had before `|` existed.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub groups: Vec<Conjunction>,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.groups.iter().all(|g| g.terms.is_empty())
    }

    pub fn matches(&self, p: &ProcRow) -> bool {
        // No groups is the empty query, which matches everything — not the
        // empty disjunction, which would match nothing and blank the table.
        self.groups.is_empty() || self.groups.iter().any(|g| g.matches(p))
    }
}

fn split_cmp(s: &str) -> Option<(&str, Cmp, &str)> {
    for (token, cmp) in
        [(">=", Cmp::Ge), ("<=", Cmp::Le), (">", Cmp::Gt), ("<", Cmp::Lt), ("=", Cmp::Eq)]
    {
        if let Some(idx) = s.find(token) {
            return Some((&s[..idx], cmp, &s[idx + token.len()..]));
        }
    }
    None
}

/// Match an optional grouping field. `-` is how the grouped view labels the
/// rows that have no value, so it has to mean the same thing here — otherwise
/// drilling into the "-" group would filter to nothing.
fn field_matches(field: Option<&str>, want: &str) -> bool {
    // "-" means *absent*, and only absent. Letting it fall through to the
    // substring arm made `service:-` match every unit with a hyphen in its
    // name — `systemd-journald.service`, `user-1000.slice`, `docker-<id>.scope`
    // — so drilling into the "no unit" group listed most of the machine, and
    // `!service:-` hid it.
    if want == "-" {
        return field.is_none_or(|v| v.trim().is_empty());
    }
    match field {
        Some(v) => want == v.to_lowercase() || v.to_lowercase().contains(want),
        None => false,
    }
}

impl Term {
    fn matches(&self, p: &ProcRow) -> bool {
        match self {
            Term::Name(n) => p.name.to_lowercase().contains(n),
            Term::Cmd(c) => p.cmd.to_lowercase().contains(c),
            Term::User(u) => {
                p.user.as_deref().map(|s| s.to_lowercase().contains(u)).unwrap_or(false)
            }
            Term::Pid(pid) => p.pid == *pid,
            Term::Ppid(pid) => p.ppid == Some(*pid),
            Term::State(s) => p.state.eq_ignore_ascii_case(s),
            Term::Service(v) => field_matches(p.service.as_deref(), v),
            Term::Container(v) => field_matches(p.container.as_deref(), v),
            Term::Regex(re) => re.is_match(&p.name),
            Term::Cpu(c, v) => c.test(p.cpu as f64, *v),
            Term::Mem(c, v) => c.test(p.mem as f64, *v),
            Term::Io(c, v) => c.test(p.io_bps(), *v),
            Term::Virt(c, v) => c.test(p.virt as f64, *v),
            // An absent value is unknown, not zero: a thread has no thread
            // count and a process whose fd table is another user's is not a
            // process holding no files. Comparing against 0 would sweep every
            // one of them into `thr<2` and `fd<100`.
            Term::Threads(c, v) => p.threads.is_some_and(|t| c.test(t as f64, *v)),
            Term::Nice(c, v) => p.nice.is_some_and(|n| c.test(n as f64, *v)),
            Term::Time(c, v) => c.test(p.run_time as f64, *v),
            Term::Fds(c, v) => p.fds.is_some_and(|n| c.test(n as f64, *v)),
            Term::FdPercent(c, v) => p.fd_ratio().is_some_and(|r| c.test(r * 100.0, *v)),
        }
    }
}

/// Parse a query. Returns the first syntax error so the UI can flag it.
pub fn parse(query: &str) -> Result<Filter, String> {
    let mut groups: Vec<Conjunction> = Vec::new();
    let mut current = Conjunction::default();
    let mut pending_or = false;

    for raw in query.split_whitespace() {
        if raw == "|" || raw == "||" {
            if current.terms.is_empty() {
                // `| a` and `a | | b` are typos, and treating them as no-ops
                // would silently widen the query to everything.
                return Err("`|` needs a term on both sides".into());
            }
            groups.push(std::mem::take(&mut current));
            pending_or = true;
            continue;
        }
        let (negated, body) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw),
        };
        if body.is_empty() {
            continue;
        }
        let term = parse_term(body)?;
        current.terms.push(FilterTerm { negated, term });
        pending_or = false;
    }

    if pending_or {
        return Err("`|` needs a term on both sides".into());
    }
    if !current.terms.is_empty() {
        groups.push(current);
    }
    Ok(Filter { groups })
}

fn parse_term(body: &str) -> Result<Term, String> {
    if let Some(rest) = body.strip_prefix("re:") {
        return Regex::new(rest).map(Term::Regex).map_err(|e| format!("bad regex: {e}"));
    }
    if let Some(rest) = body.strip_prefix("user:") {
        return Ok(Term::User(rest.to_lowercase()));
    }
    if let Some(rest) = body.strip_prefix("cmd:") {
        return Ok(Term::Cmd(rest.to_lowercase()));
    }
    if let Some(rest) = body.strip_prefix("pid:") {
        return rest.parse().map(Term::Pid).map_err(|_| format!("bad pid: {rest}"));
    }
    if let Some(rest) = body.strip_prefix("ppid:") {
        return rest.parse().map(Term::Ppid).map_err(|_| format!("bad ppid: {rest}"));
    }
    if let Some(rest) = body.strip_prefix("service:") {
        return Ok(Term::Service(rest.to_lowercase()));
    }
    if let Some(rest) = body.strip_prefix("container:") {
        return Ok(Term::Container(rest.to_lowercase()));
    }
    if let Some(rest) = body.strip_prefix("state:") {
        return rest
            .chars()
            .next()
            .map(Term::State)
            .ok_or_else(|| "state: needs a letter".to_string());
    }

    if let Some((field, cmp, value)) = split_cmp(body) {
        let field = field.to_ascii_lowercase();
        return match field.as_str() {
            "cpu" => value
                .parse::<f64>()
                .map(|v| Term::Cpu(cmp, v))
                .map_err(|_| format!("bad number: {value}")),
            "mem" | "rss" => parse_size(value)
                .map(|v| Term::Mem(cmp, v as f64))
                .ok_or_else(|| format!("bad size: {value}")),
            "io" | "disk" => parse_size(value)
                .map(|v| Term::Io(cmp, v as f64))
                .ok_or_else(|| format!("bad size: {value}")),
            "virt" | "vsz" => parse_size(value)
                .map(|v| Term::Virt(cmp, v as f64))
                .ok_or_else(|| format!("bad size: {value}")),
            "thr" | "threads" => value
                .parse::<f64>()
                .map(|v| Term::Threads(cmp, v))
                .map_err(|_| format!("bad number: {value}")),
            "ni" | "nice" => value
                .parse::<f64>()
                .map(|v| Term::Nice(cmp, v))
                .map_err(|_| format!("bad number: {value}")),
            "time" | "age" => parse_duration(value)
                .map(|v| Term::Time(cmp, v as f64))
                .ok_or_else(|| format!("bad duration: {value}")),
            "fd" | "fds" => value
                .parse::<f64>()
                .map(|v| Term::Fds(cmp, v))
                .map_err(|_| format!("bad number: {value}")),
            "fd%" | "fdpct" => value
                .parse::<f64>()
                .map(|v| Term::FdPercent(cmp, v))
                .map_err(|_| format!("bad number: {value}")),
            // Not a known field — fall through and treat the whole token as a
            // name substring, so searching for "a=b" still works.
            _ => Ok(Term::Name(body.to_lowercase())),
        };
    }

    Ok(Term::Name(body.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pid: u32, name: &str, cpu: f32, mem: u64) -> ProcRow {
        ProcRow {
            pid,
            ppid: Some(1),
            name: name.into(),
            cpu,
            mem,
            state: 'S',
            user: Some("vlad".into()),
            cmd: format!("/usr/bin/{name} --flag"),
            ..ProcRow::default()
        }
    }

    #[test]
    fn a_bare_word_is_a_case_insensitive_name_substring() {
        let f = parse("FIRE").unwrap();
        assert!(f.matches(&p(1, "firefox", 0.0, 0)));
        assert!(!f.matches(&p(2, "chrome", 0.0, 0)));
    }

    #[test]
    fn an_empty_query_matches_everything() {
        let f = parse("   ").unwrap();
        assert!(f.is_empty());
        assert!(f.matches(&p(1, "anything", 0.0, 0)));
    }

    #[test]
    fn terms_are_anded_together() {
        let f = parse("chrome cpu>10").unwrap();
        assert!(f.matches(&p(1, "chrome", 50.0, 0)));
        assert!(!f.matches(&p(2, "chrome", 1.0, 0)));
        assert!(!f.matches(&p(3, "firefox", 50.0, 0)));
    }

    #[test]
    fn negation_excludes_matches() {
        let f = parse("!kworker").unwrap();
        assert!(f.matches(&p(1, "firefox", 0.0, 0)));
        assert!(!f.matches(&p(2, "kworker/3:1", 0.0, 0)));
    }

    #[test]
    fn grouping_fields_are_filterable_so_a_group_can_be_drilled_into() {
        let mut p = p(1, "dockerd", 0.0, 0);
        p.service = Some("docker.service".into());
        p.container = Some("abc123def456".into());
        assert!(parse("service:docker.service").unwrap().matches(&p));
        assert!(parse("service:DOCKER").unwrap().matches(&p));
        assert!(parse("container:abc123def456").unwrap().matches(&p));
        assert!(!parse("service:sshd").unwrap().matches(&p));

        // `-` is what the grouped view calls "no value", so it has to filter
        // to the same set the group displayed.
        let bare = p2(2, "kworker");
        assert!(parse("service:-").unwrap().matches(&bare));
        assert!(parse("container:-").unwrap().matches(&bare));
        assert!(!parse("service:-").unwrap().matches(&p));
    }

    fn p2(pid: u32, name: &str) -> ProcRow {
        ProcRow { pid, name: name.into(), ..ProcRow::default() }
    }

    #[test]
    fn field_terms_target_the_right_columns() {
        assert!(parse("user:VLA").unwrap().matches(&p(1, "x", 0.0, 0)));
        assert!(parse("pid:42").unwrap().matches(&p(42, "x", 0.0, 0)));
        assert!(!parse("pid:42").unwrap().matches(&p(43, "x", 0.0, 0)));
        assert!(parse("ppid:1").unwrap().matches(&p(1, "x", 0.0, 0)));
        assert!(parse("state:s").unwrap().matches(&p(1, "x", 0.0, 0)));
        assert!(parse("cmd:--flag").unwrap().matches(&p(1, "x", 0.0, 0)));
    }

    #[test]
    fn size_comparisons_understand_suffixes() {
        let f = parse("mem>100M").unwrap();
        assert!(f.matches(&p(1, "x", 0.0, 200 * 1024 * 1024)));
        assert!(!f.matches(&p(2, "x", 0.0, 50 * 1024 * 1024)));
        assert!(parse("mem<=1G").unwrap().matches(&p(3, "x", 0.0, 1024 * 1024 * 1024)));
    }

    #[test]
    fn regex_terms_anchor_like_regexes() {
        let f = parse("re:^chrom(e|ium)$").unwrap();
        assert!(f.matches(&p(1, "chromium", 0.0, 0)));
        assert!(!f.matches(&p(2, "chromium-sandbox", 0.0, 0)));
    }

    #[test]
    fn a_broken_regex_is_reported_not_swallowed() {
        let err = parse("re:[unclosed").unwrap_err();
        assert!(err.contains("bad regex"), "{err}");
        assert!(parse("cpu>abc").is_err());
        assert!(parse("mem>huge").is_err());
    }

    #[test]
    fn unknown_fields_degrade_to_a_name_search() {
        // Would otherwise be a confusing hard error while typing.
        let f = parse("foo=bar").unwrap();
        assert!(f.matches(&p(1, "xfoo=barx", 0.0, 0)));
    }

    #[test]
    fn all_comparison_operators_work() {
        assert!(parse("cpu>=5").unwrap().matches(&p(1, "x", 5.0, 0)));
        assert!(parse("cpu<5").unwrap().matches(&p(1, "x", 4.0, 0)));
        assert!(parse("cpu<=5").unwrap().matches(&p(1, "x", 5.0, 0)));
        assert!(parse("cpu=5").unwrap().matches(&p(1, "x", 5.0, 0)));
        assert!(!parse("cpu>5").unwrap().matches(&p(1, "x", 5.0, 0)));
    }

    #[test]
    fn a_bare_pipe_is_or_and_binds_looser_than_the_implicit_and() {
        let f = parse("nginx cpu>10 | apache").unwrap();
        assert!(f.matches(&p(1, "nginx", 50.0, 0)), "left side, both terms");
        assert!(!f.matches(&p(2, "nginx", 1.0, 0)), "left side, only one term");
        assert!(f.matches(&p(3, "apache", 0.0, 0)), "right side alone");
        assert!(!f.matches(&p(4, "postgres", 99.0, 0)));
    }

    #[test]
    fn or_does_not_split_a_regex_that_contains_a_pipe() {
        // The reason `|` is only an operator as a standalone token: splitting
        // on the character turns this working query into two broken ones.
        let f = parse("re:^chrom(e|ium)$").unwrap();
        assert!(f.matches(&p(1, "chromium", 0.0, 0)));
        assert!(f.matches(&p(2, "chrome", 0.0, 0)));
        assert!(!f.matches(&p(3, "chromium-sandbox", 0.0, 0)));
        assert_eq!(f.groups.len(), 1, "one alternative, not two");
    }

    #[test]
    fn a_dangling_pipe_is_an_error_rather_than_a_query_that_matches_everything() {
        for bad in ["| firefox", "firefox |", "a | | b", "|"] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains('|'), "{bad}: {err}");
        }
        // `||` is the same operator, for anyone typing out of shell habit.
        assert_eq!(parse("a || b").unwrap().groups.len(), 2);
    }

    #[test]
    fn every_column_the_table_can_sort_on_can_also_be_filtered_on() {
        // The asymmetry this closes: `--sort nice` worked and `nice<0` did not.
        let mut row = p(1, "worker", 0.0, 0);
        row.virt = 4 * 1024 * 1024 * 1024;
        row.threads = Some(64);
        row.nice = Some(-5);
        row.run_time = 7_200;

        assert!(parse("virt>2G").unwrap().matches(&row));
        assert!(!parse("virt>8G").unwrap().matches(&row));
        assert!(parse("thr>=64").unwrap().matches(&row));
        assert!(parse("threads<100").unwrap().matches(&row));
        assert!(parse("nice<0").unwrap().matches(&row), "negatives parse as values, not operators");
        assert!(parse("ni=-5").unwrap().matches(&row));
        assert!(parse("time>1h").unwrap().matches(&row));
        assert!(!parse("time>3h").unwrap().matches(&row));
        assert!(parse("age>=7200").unwrap().matches(&row), "a bare number is seconds");
    }

    #[test]
    fn an_absent_value_never_satisfies_a_comparison() {
        // A thread reports no thread count and a process whose fd table
        // belongs to another user reports no descriptors. Reading either as
        // zero would sweep every one of them into `thr<2` and `fd<100`.
        let unknown = p2(1, "kworker/3:1");
        assert!(!parse("thr<2").unwrap().matches(&unknown));
        assert!(!parse("thr>2").unwrap().matches(&unknown));
        assert!(!parse("nice<1").unwrap().matches(&unknown));
        assert!(!parse("fd<10").unwrap().matches(&unknown));
        assert!(!parse("fd%>0").unwrap().matches(&unknown));
        // ...and the negation does match it, which is how you find them.
        assert!(parse("!thr>2").unwrap().matches(&unknown));
    }

    #[test]
    fn descriptors_can_be_filtered_by_count_or_by_share_of_the_limit() {
        // 4000 fds is unremarkable against a limit of a million and fatal
        // against the default 1024, so both spellings have to exist.
        let roomy = ProcRow { fds: Some(4_000), fd_limit: Some(1_048_576), ..p2(1, "envoy") };
        let doomed = ProcRow { fds: Some(1_000), fd_limit: Some(1_024), ..p2(2, "api") };

        assert!(parse("fd>2000").unwrap().matches(&roomy));
        assert!(!parse("fd>2000").unwrap().matches(&doomed));
        assert!(parse("fd%>90").unwrap().matches(&doomed), "97% of its own limit");
        assert!(!parse("fd%>90").unwrap().matches(&roomy));
    }

    #[test]
    fn a_bad_value_in_a_new_field_is_reported_like_any_other() {
        assert!(parse("time>yesterday").unwrap_err().contains("duration"));
        assert!(parse("virt>huge").unwrap_err().contains("size"));
        assert!(parse("thr>many").unwrap_err().contains("number"));
        assert!(parse("fd>lots").unwrap_err().contains("number"));
    }

    /// The grouped view labels rows with no unit "-", and drilling into that
    /// group filters on `service:-`. Falling through to the substring arm made
    /// it match every unit with a hyphen in its name, i.e. most of the machine.
    #[test]
    fn the_absent_value_sentinel_does_not_match_hyphenated_names() {
        let f = parse("service:-").unwrap();
        let with = |unit: Option<&str>| ProcRow {
            service: unit.map(String::from),
            ..ProcRow { pid: 1, name: "x".into(), ..Default::default() }
        };

        assert!(f.matches(&with(None)), "a process in no unit must match");
        assert!(f.matches(&with(Some(""))), "an empty unit is also no unit");
        for unit in ["systemd-journald.service", "user-1000.slice", "docker-abc.scope"] {
            assert!(!f.matches(&with(Some(unit))), "{unit} is not an absent unit");
        }
        assert!(!f.matches(&with(Some("nginx.service"))));

        // ...and the negation is the exact complement.
        let neg = parse("!service:-").unwrap();
        assert!(neg.matches(&with(Some("systemd-journald.service"))));
        assert!(!neg.matches(&with(None)));
    }
}
