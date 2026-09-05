//! The process filter language.
//!
//! Whitespace-separated terms, all ANDed, each optionally negated with `!`:
//!
//! ```text
//! firefox            name contains "firefox"
//! !kworker           name does not contain "kworker"
//! user:vlad          user name contains "vlad"
//! pid:1234           exact pid       ppid:1  exact parent
//! state:R            process state letter
//! cmd:--headless     substring of the full command line
//! re:^chrom(e|ium)$  regex over the name
//! cpu>5  mem>100M    numeric comparisons (> >= < <= =)
//! ```

use regex::Regex;

use crate::format::parse_size;
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
    Regex(Regex),
    Cpu(Cmp, f64),
    Mem(Cmp, f64),
    Io(Cmp, f64),
}

#[derive(Debug, Clone)]
pub struct FilterTerm {
    pub negated: bool,
    pub term: Term,
}

#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub terms: Vec<FilterTerm>,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn matches(&self, p: &ProcRow) -> bool {
        self.terms.iter().all(|t| t.term.matches(p) != t.negated)
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
            Term::Regex(re) => re.is_match(&p.name),
            Term::Cpu(c, v) => c.test(p.cpu as f64, *v),
            Term::Mem(c, v) => c.test(p.mem as f64, *v),
            Term::Io(c, v) => c.test(p.io_bps(), *v),
        }
    }
}

/// Parse a query. Returns the first syntax error so the UI can flag it.
pub fn parse(query: &str) -> Result<Filter, String> {
    let mut terms = Vec::new();
    for raw in query.split_whitespace() {
        let (negated, body) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw),
        };
        if body.is_empty() {
            continue;
        }
        let term = parse_term(body)?;
        terms.push(FilterTerm { negated, term });
    }
    Ok(Filter { terms })
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
}
