//! Hand-rolled argument parsing. Small enough to stay dependency-light and
//! fully testable without spawning a process.

use crate::export::ExportFormat;
use crate::sort::SortBy;
use crate::ui::Layout;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Every long flag the parser accepts. The docs tests check this against the
/// README, the man page and the shell completions.
pub const LONG_FLAGS: [&str; 28] = [
    "--refresh",
    "--sort",
    "--ascending",
    "--descending",
    "--filter",
    "--tree",
    "--layout",
    "--theme",
    "--no-color",
    "--no-mouse",
    "--once",
    "--format",
    "--top",
    "--config",
    "--group",
    "--record",
    "--replay",
    "--remote",
    "--remote-command",
    "--serve",
    "--stream",
    "--diff",
    "--watch",
    "--watch-for",
    "--watch-timeout",
    "--help",
    "--version",
    "--no-colour",
];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub refresh_ms: Option<u64>,
    pub sort: Option<SortBy>,
    pub sort_desc: Option<bool>,
    pub filter: Option<String>,
    pub theme: Option<String>,
    pub layout: Option<Layout>,
    pub tree: Option<bool>,
    pub no_color: bool,
    pub no_mouse: bool,
    /// Print one snapshot and exit instead of starting the TUI.
    pub once: bool,
    pub format: Option<ExportFormat>,
    pub top: Option<usize>,
    pub config: Option<String>,
    /// Record every sampled snapshot to this file, replacing any file there.
    pub record: Option<String>,
    /// Drive the UI from a recording instead of the live host.
    pub replay: Option<String>,
    /// Serve Prometheus metrics on this address, headless.
    pub serve: Option<String>,
    /// Write one JSON snapshot per line forever. What `--remote` runs on the
    /// far end, and usable on its own as a poor man's collector.
    pub stream: bool,
    /// Compare two snapshots, or the ends of one recording.
    pub diff: Option<(String, Option<String>)>,
    /// Headless: watch for processes matching this query.
    pub watch: Option<String>,
    /// How long the watch condition must hold before it counts.
    pub watch_for: Option<u64>,
    /// Give up waiting after this many seconds. 0 waits forever.
    pub watch_timeout: Option<u64>,
    /// Monitor another host over SSH.
    pub remote: Option<String>,
    /// Command run on the far end of `--remote`.
    pub remote_command: Option<String>,
    pub group: Option<crate::metrics::GroupBy>,
}

#[derive(Debug, PartialEq)]
pub enum Parsed {
    Run(Box<Args>),
    Help,
    Version,
}

pub fn help_text() -> String {
    format!(
        "crabmon {VERSION} — terminal system monitor

USAGE:
    crabmon [OPTIONS]

OPTIONS:
    -r, --refresh <MS>       Refresh interval, {min}-{max} ms
    -s, --sort <KEY>         pid|name|cpu|mem|virt|disk|time|user|state|threads|nice
    -a, --ascending          Sort ascending
    -d, --descending         Sort descending (the default)
    -f, --filter <QUERY>     Initial process filter, e.g. 'user:root cpu>5'
    -t, --tree               Start in process-tree view
    -l, --layout <NAME>      dashboard|processes|cpu|io
        --theme <NAME>       default|mono|nord|solarized|gruvbox
        --no-color           Disable colour (same as --theme mono)
        --no-mouse           Do not capture mouse events
    -1, --once               Print a single snapshot and exit
        --format <FMT>       Output format for --once: json|csv (default json)
    -n, --top <N>            Limit --once output to the top N processes
    -c, --config <PATH>      Use an alternate config file
    -g, --group <BY>         Group processes: none|service|container|user
        --record <PATH>      Record every sample to a JSONL file (replaces it)
        --replay <PATH>      Replay a recording instead of sampling this host
        --remote <TARGET>    Monitor TARGET over SSH (needs crabmon installed there)
        --remote-command <C> Command to run on the remote host
        --serve <ADDR>       Serve Prometheus metrics on ADDR, e.g. :9100
        --stream             Print a JSON snapshot per line forever
        --diff <A> [B]       Compare two snapshots, or one recording end to end
        --watch <QUERY>      Wait for processes matching QUERY, then exit 1
        --watch-for <SECS>   ...only if the match holds this long
        --watch-timeout <S>  ...and give up after this long (0 waits forever)
    -h, --help               Show this help
    -V, --version            Show version

KEYS (in the TUI, press ? for the full list):
    q            quit                 /      filter      1-9  saved filters
    j/k, arrows  move selection       Enter  process detail
    c m p n      sort by cpu/mem/pid/name    < >  cycle sort column
    s            reverse sort         T      tree view   H    show threads
    Tab          cycle layout         +/-    refresh faster/slower
    t            signal menu          r      renice      A    CPU affinity
    Space        tag for bulk action  G      group by    z    pause
    f/F          pin / unpin all      [ ]    scrub a recording
    y            copy cmd             L      action log
    P            export snapshot      ?      help
",
        min = crate::MIN_REFRESH_MS,
        max = crate::MAX_REFRESH_MS,
    )
}

pub fn version_text() -> String {
    format!("crabmon {VERSION}")
}

/// Parse an argv tail (without the program name).
pub fn parse<I, S>(argv: I) -> Result<Parsed, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = argv.into_iter().map(|s| s.as_ref().to_string()).collect();
    let mut out = Args::default();
    let mut i = 0;

    // `--opt=value` is accepted alongside `--opt value`.
    let take_value =
        |i: &mut usize, args: &[String], inline: Option<String>, flag: &str| match inline {
            Some(v) => Ok(v),
            None => {
                *i += 1;
                let v = args.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
                // `crabmon --serve --stream` used to read "--stream" as the
                // bind address: the exclusivity checks below then never saw
                // --stream at all, and the run died much later with an
                // unrelated "invalid port". Only crabmon's own long flags are
                // refused here — `--ssh-option -oBatchMode=yes` is a real and
                // legitimate value that begins with a dash.
                if LONG_FLAGS.contains(&v.as_str()) {
                    return Err(format!("{flag} needs a value, but {v} is a flag"));
                }
                Ok(v)
            }
        };

    while i < args.len() {
        let raw = args[i].clone();
        let (flag, inline) = match raw.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (raw.clone(), None),
        };

        match flag.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-V" | "--version" => return Ok(Parsed::Version),
            "-r" | "--refresh" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                let ms: u64 = v.parse().map_err(|_| format!("bad refresh interval: {v}"))?;
                out.refresh_ms = Some(ms.clamp(crate::MIN_REFRESH_MS, crate::MAX_REFRESH_MS));
            }
            "-s" | "--sort" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.sort = Some(SortBy::parse(&v).ok_or_else(|| format!("unknown sort key: {v}"))?);
            }
            "-a" | "--ascending" => out.sort_desc = Some(false),
            "-d" | "--descending" => out.sort_desc = Some(true),
            "-f" | "--filter" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                // An unparseable filter used to degrade to the empty filter,
                // which matches *everything* — so `--once --filter 'cpu>'`
                // printed the whole process table with exit 0. Its sibling
                // `--watch` has always rejected the same string.
                crate::filter::parse(&v).map_err(|e| format!("bad --filter: {e}"))?;
                out.filter = Some(v);
            }
            "-t" | "--tree" => out.tree = Some(true),
            "-l" | "--layout" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.layout = Some(Layout::parse(&v).ok_or_else(|| format!("unknown layout: {v}"))?);
            }
            "--theme" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                if crate::theme::Theme::preset(&v).is_none() {
                    return Err(format!("unknown theme: {v}"));
                }
                out.theme = Some(v);
            }
            "--no-color" | "--no-colour" => out.no_color = true,
            "--no-mouse" => out.no_mouse = true,
            "-1" | "--once" => out.once = true,
            "--format" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.format =
                    Some(ExportFormat::parse(&v).ok_or_else(|| format!("unknown format: {v}"))?);
            }
            "-n" | "--top" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.top = Some(v.parse().map_err(|_| format!("bad process count: {v}"))?);
            }
            "-c" | "--config" => out.config = Some(take_value(&mut i, &args, inline, &flag)?),
            "-g" | "--group" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.group = Some(
                    crate::metrics::GroupBy::parse(&v)
                        .ok_or_else(|| format!("unknown grouping: {v}"))?,
                );
            }
            "--record" => out.record = Some(take_value(&mut i, &args, inline, &flag)?),
            "--replay" => out.replay = Some(take_value(&mut i, &args, inline, &flag)?),
            "--remote" => out.remote = Some(take_value(&mut i, &args, inline, &flag)?),
            "--remote-command" => {
                out.remote_command = Some(take_value(&mut i, &args, inline, &flag)?)
            }
            "--serve" => out.serve = Some(take_value(&mut i, &args, inline, &flag)?),
            "--stream" => out.stream = true,
            "--diff" => {
                let a = take_value(&mut i, &args, inline, &flag)?;
                // A second path may follow; one on its own means "this
                // recording's first frame against its last".
                let b = args.get(i + 1).filter(|v| !v.starts_with('-')).cloned();
                if b.is_some() {
                    i += 1;
                }
                out.diff = Some((a, b));
            }
            "--watch" => out.watch = Some(take_value(&mut i, &args, inline, &flag)?),
            "--watch-for" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.watch_for = Some(v.parse().map_err(|_| format!("bad duration: {v}"))?);
            }
            "--watch-timeout" => {
                let v = take_value(&mut i, &args, inline, &flag)?;
                out.watch_timeout = Some(v.parse().map_err(|_| format!("bad duration: {v}"))?);
            }
            other => return Err(format!("unknown option: {other}\nTry 'crabmon --help'.")),
        }
        i += 1;
    }

    if out.no_color {
        out.theme = Some("mono".into());
    }
    // Sources are mutually exclusive: each replaces where snapshots come from.
    let sources = [out.replay.is_some(), out.remote.is_some()];
    if sources.iter().filter(|s| **s).count() > 1 {
        return Err("--replay and --remote cannot be combined".into());
    }
    if out.replay.is_some() && out.record.is_some() {
        return Err("--record and --replay cannot be combined".into());
    }
    // The headless modes all sample this host. Accepting `--remote` alongside
    // them used to print the *local* machine's snapshot with exit 0, which is a
    // worse failure than refusing: a script gets plausible data about the wrong
    // host and never finds out.
    for (name, on) in [
        ("--once", out.once),
        ("--serve", out.serve.is_some()),
        ("--stream", out.stream),
        // --watch calls live_source() exactly as the other three do; leaving it
        // out meant `--remote box --watch 'state:D'` paged on the *local*
        // machine's processes while naming the remote host.
        ("--watch", out.watch.is_some()),
    ] {
        if !on {
            continue;
        }
        if out.remote.is_some() {
            return Err(format!("{name} samples this host; it cannot be combined with --remote"));
        }
        if out.replay.is_some() {
            return Err(format!("{name} samples this host; it cannot be combined with --replay"));
        }
    }
    let headless =
        [out.once, out.serve.is_some(), out.stream, out.watch.is_some(), out.diff.is_some()];
    if headless.iter().filter(|s| **s).count() > 1 {
        return Err("--once, --serve, --stream, --watch and --diff are mutually exclusive".into());
    }
    Ok(Parsed::Run(Box::new(out)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(argv: &[&str]) -> Args {
        match parse(argv).unwrap() {
            Parsed::Run(a) => *a,
            other => panic!("expected Run, got {other:?}"),
        }
    }

    #[test]
    fn no_arguments_means_all_defaults() {
        assert_eq!(run(&[]), Args::default());
    }

    #[test]
    fn help_and_version_short_circuit() {
        assert_eq!(parse(["--help"]).unwrap(), Parsed::Help);
        assert_eq!(parse(["-h"]).unwrap(), Parsed::Help);
        assert_eq!(parse(["-V"]).unwrap(), Parsed::Version);
        // Even alongside other flags.
        assert_eq!(parse(["--tree", "--version"]).unwrap(), Parsed::Version);
    }

    #[test]
    fn values_may_be_separate_or_inline() {
        assert_eq!(run(&["--refresh", "1500"]).refresh_ms, Some(1500));
        assert_eq!(run(&["--refresh=1500"]).refresh_ms, Some(1500));
        assert_eq!(run(&["--filter=user:root cpu>5"]).filter.as_deref(), Some("user:root cpu>5"));
    }

    #[test]
    fn the_refresh_floor_is_applied_at_the_command_line_too() {
        assert_eq!(run(&["-r", "10"]).refresh_ms, Some(crate::MIN_REFRESH_MS));
        assert_eq!(run(&["-r", "99999"]).refresh_ms, Some(crate::MAX_REFRESH_MS));
    }

    #[test]
    fn sort_layout_theme_and_format_are_validated() {
        assert_eq!(run(&["-s", "mem"]).sort, Some(SortBy::Mem));
        assert_eq!(run(&["-l", "processes"]).layout, Some(Layout::Processes));
        assert_eq!(run(&["--theme", "nord"]).theme.as_deref(), Some("nord"));
        assert_eq!(run(&["--format", "csv"]).format, Some(ExportFormat::Csv));

        assert!(parse(["-s", "nope"]).is_err());
        assert!(parse(["-l", "nope"]).is_err());
        assert!(parse(["--theme", "nope"]).is_err());
        assert!(parse(["--format", "nope"]).is_err());
    }

    #[test]
    fn no_color_selects_the_mono_theme() {
        assert_eq!(run(&["--no-color"]).theme.as_deref(), Some("mono"));
    }

    #[test]
    fn missing_values_and_unknown_flags_are_errors() {
        assert!(parse(["--refresh"]).unwrap_err().contains("needs a value"));
        assert!(parse(["--wat"]).unwrap_err().contains("unknown option"));
        assert!(parse(["-r", "abc"]).is_err());
        assert!(parse(["-n", "abc"]).is_err());
    }

    #[test]
    fn headless_mode_combines_once_format_and_top() {
        let a = run(&["--once", "--format", "csv", "--top", "10"]);
        assert!(a.once);
        assert_eq!(a.format, Some(ExportFormat::Csv));
        assert_eq!(a.top, Some(10));
    }

    #[test]
    fn help_text_documents_every_long_flag_it_accepts() {
        let help = help_text();
        for flag in [
            "--refresh",
            "--sort",
            "--ascending",
            "--filter",
            "--tree",
            "--layout",
            "--theme",
            "--no-color",
            "--no-mouse",
            "--once",
            "--format",
            "--top",
            "--config",
            "--help",
            "--version",
        ] {
            assert!(help.contains(flag), "{flag} is undocumented");
        }
    }

    /// `--version` is what a bug report quotes; it has to name the build that
    /// is actually running, not a string frozen at the last manual edit.
    #[test]
    fn the_version_string_comes_from_the_crate_itself() {
        let text = version_text();
        assert_eq!(text, format!("crabmon {}", env!("CARGO_PKG_VERSION")));
        assert!(!text.ends_with(' '), "a trailing space breaks scripted parsing");
        assert!(text.starts_with("crabmon "), "{text}");
    }

    /// `--watch` samples this host exactly as `--once` does, so combining it
    /// with `--remote` used to page on the wrong machine and never say so.
    #[test]
    fn every_mode_that_samples_this_host_refuses_remote_and_replay() {
        for mode in
            [vec!["--once"], vec!["--serve", "9100"], vec!["--stream"], vec!["--watch", "state:D"]]
        {
            for elsewhere in [vec!["--remote", "box"], vec!["--replay", "r.jsonl"]] {
                let argv: Vec<&str> = mode.iter().chain(elsewhere.iter()).copied().collect();
                assert!(parse(&argv).is_err(), "{argv:?} was accepted");
            }
        }
    }

    /// An unparseable filter used to degrade to the empty filter, which matches
    /// everything: `--once --filter 'cpu>'` printed the whole process table
    /// with exit 0, while `--watch 'cpu>'` rejected the identical string.
    #[test]
    fn a_malformed_filter_is_refused_rather_than_matching_everything() {
        for bad in ["cpu>", "cpu>abc", "mem>>5"] {
            let err = parse(["--filter", bad]).unwrap_err();
            assert!(err.contains("--filter"), "{bad}: {err}");
        }
        assert_eq!(run(&["--filter", "cpu>5"]).filter.as_deref(), Some("cpu>5"));
        assert_eq!(run(&["--filter=cpu>5"]).filter.as_deref(), Some("cpu>5"));
    }

    /// A value-taking flag that swallows the next flag skips the mutual
    /// exclusion checks entirely and fails much later for an unrelated reason.
    #[test]
    fn a_flag_is_never_consumed_as_another_flags_value() {
        for argv in [
            ["--serve", "--stream"],
            ["--filter", "--once"],
            ["--record", "--once"],
            ["--config", "--tree"],
        ] {
            let err = parse(argv).unwrap_err();
            assert!(err.contains("is a flag"), "{argv:?} gave: {err}");
        }
        // A value that merely looks flag-ish is still a value.
        assert_eq!(run(&["--filter", "name:-x"]).filter.as_deref(), Some("name:-x"));
    }
}
