//! Terminal setup/teardown and the event loop.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind};
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crabmon::app::{Action, App};
use crabmon::cli::{self, Args, Parsed};
use crabmon::config::{self, Config};
use crabmon::export;
use crabmon::metrics::{MetricSource, SysinfoSource};
use crabmon::record::{FlightRecorder, ReplaySource};
use crabmon::remote::RemoteSource;
use crabmon::sampler::{Pacer, ThreadedSource};
use crabmon::supervisor;

fn main() -> Result<()> {
    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(Parsed::Help) => {
            print!("{}", cli::help_text());
            return Ok(());
        }
        Ok(Parsed::Version) => {
            println!("{}", cli::version_text());
            return Ok(());
        }
        Ok(Parsed::Run(a)) => *a,
        Err(e) => {
            eprintln!("crabmon: {e}");
            std::process::exit(2);
        }
    };

    let path = config_path(&args);
    let (loaded, cfg_error) = config::load_checked(&path);
    if let Some(e) = &cfg_error {
        // Loud, and on stderr, because the alternative is a monitor running
        // with none of the alert rules the file defines and no sign of it.
        eprintln!("crabmon: {}: {e}", path.display());
        eprintln!("crabmon: using built-in defaults; nothing from that file was applied");
    }
    let cfg = apply_args(loaded, &args);

    // Before everything else that samples: the whole question is what this
    // machine will let crabmon see, and the answer must not depend on having
    // got far enough to take a sample.
    if args.doctor {
        return print_doctor(&cfg, &args);
    }
    // The headless modes. `cli::parse` refuses to combine any of them with
    // `--remote` or `--replay`, so each one here samples this host.
    if let Some((a, b)) = &args.diff {
        return print_diff(a, b.as_deref(), &args);
    }
    if let Some(addr) = &args.serve {
        return serve_metrics(addr, &cfg, &args);
    }
    if args.stream {
        return stream_frames(&cfg, &args);
    }
    let hold = args.watch_for.unwrap_or(0);
    if let Some(query) = &args.watch {
        // Validated here rather than left to the engine: an unparseable query
        // became a rule that measures nothing, which never fires, so a typo
        // waited out the whole timeout and then exited 0 — "nothing went
        // wrong" — instead of saying the query was wrong.
        if let Err(e) = crabmon::filter::parse(query) {
            fail_watch(&format!("bad watch query: {e}"));
        }
        return watch_for(crabmon::watch::process_rule(query, hold), &cfg, &args)
            .or_else(|e| fail_watch(&e.to_string()));
    }
    if let Some(expr) = &args.watch_rule {
        // Not `?`: returning an error from `main` exits 1, which is also what
        // a *successful* watch exits with, so `--watch-rule 'psi:oi>20' ||
        // page-oncall` paged on a typo rather than reporting it.
        let rule = match crabmon::watch::parse_rule(expr, hold) {
            Ok(rule) => rule,
            Err(e) => fail_watch(&e),
        };
        return watch_for(rule, &cfg, &args).or_else(|e| fail_watch(&e.to_string()));
    }
    if args.once {
        let mut source = live_source(&cfg);
        return print_once(&cfg, &args, &mut source);
    }

    let source = build_source(&cfg, &args)?;
    run_tui(cfg, source, &args, path, cfg_error.is_none())
}

fn live_source(cfg: &Config) -> SysinfoSource {
    SysinfoSource::new(cfg.virtual_iface_prefixes.clone(), cfg.gpu.enabled && cfg.gpu.nvidia_smi)
        .with_fds(cfg.procs.fds)
        .with_pss(cfg.procs.pss)
}

/// Where snapshots come from: a recording, another host, or this one.
///
/// Everything but a replay is sampled on a worker thread. A replay is already
/// instant and needs `seek`/`peek` for the scrub keys, which only make sense
/// synchronously.
fn build_source(cfg: &Config, args: &Args) -> Result<Box<dyn MetricSource>> {
    if let Some(path) = &args.replay {
        let source =
            ReplaySource::open(std::path::Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("crabmon: replaying {} frames from {path}", source.len());
        return Ok(Box::new(source));
    }
    if let Some(spec) = &args.remote {
        let targets = crabmon::fleet::parse_targets(spec);
        let one = |target: &str| -> Box<dyn MetricSource> {
            let remote = match (&args.remote_command, cfg.remote.stream) {
                (Some(_), _) => RemoteSource::new(target, args.remote_command.clone(), Vec::new()),
                (None, true) => RemoteSource::streaming(target, cfg.refresh_ms, Vec::new()),
                (None, false) => RemoteSource::new(target, None, Vec::new()),
            };
            Box::new(ThreadedSource::new(Box::new(remote)))
        };
        return match targets.len() {
            0 => Err(anyhow::anyhow!("--remote names no hosts")),
            // One host behaves exactly as it always has: no fleet table, no
            // extra layout in the Tab cycle, no wrapper in the way.
            1 => Ok(one(&targets[0])),
            _ => {
                eprintln!("crabmon: watching {} hosts: {}", targets.len(), targets.join(", "));
                let sources = targets.iter().map(|t| (t.clone(), one(t))).collect();
                Ok(Box::new(crabmon::fleet::FleetSource::new(sources)))
            }
        };
    }
    Ok(Box::new(ThreadedSource::new(Box::new(live_source(cfg)))))
}

/// Open whatever `--record`, `--record-rotate` and `[record]` between them
/// ask for, and announce it on stderr.
///
/// stderr because `--stream`'s stdout is a frame per line that something else
/// is parsing, and the same call serves every headless mode.
fn open_recordings(cfg: &Config, args: &Args) -> Result<crabmon::record::Recordings> {
    let recordings = crabmon::record::Recordings::open(
        args.record.as_deref(),
        args.record_rotate.as_deref(),
        &cfg.record,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    for note in recordings.notices() {
        eprintln!("crabmon: {note}");
    }
    Ok(recordings)
}

/// `--stream`: one JSON snapshot per line, forever. This is what a streaming
/// `--remote` runs on the far end. Transport frames retain the complete
/// inventory: trimming would turn a quiet service into a false absence alert.
///
/// With `--alerts` the same rules the TUI evaluates run here too, so a host
/// streaming to a collector also pages and writes flight recordings. Hook
/// output and recording notices go to stderr, which keeps stdout a clean
/// stream of frames for whatever is reading it.
///
/// `--record` and `--record-rotate` write here too. The frame on stdout stays
/// untrimmed — a collector needs the full inventory — while the one going to
/// disk is trimmed by `[record]` like any other recording, so a long-running
/// stream does not fill the disk to say what it has already said.
fn stream_frames(cfg: &Config, args: &Args) -> Result<()> {
    let mut source = live_source(cfg);
    source.snapshot(Duration::ZERO);
    // The interval the rates are divided by has to be the one that actually
    // passed, not the one that was asked for — see `Pacer`.
    let mut pacer = Pacer::new(Duration::from_millis(cfg.refresh_ms));
    let mut sup = args.alerts.then(|| supervisor::Supervisor::new(cfg));
    let mut recordings = open_recordings(cfg, args)?;
    let mut out = io::stdout().lock();
    loop {
        let dt = pacer.wait();
        let snap = source.snapshot(dt);
        for note in recordings.write(&snap) {
            eprintln!("crabmon: {note}");
        }
        if let Some(sup) = sup.as_mut() {
            report(sup.observe(&snap));
        }
        let line = serde_json::to_string(&snap)?;
        // A broken pipe means the reader went away; that is a normal exit here,
        // not a crash to report.
        if out.write_all(line.as_bytes()).is_err()
            || out.write_all(b"\n").is_err()
            || out.flush().is_err()
        {
            if let Some(note) = recordings.summary() {
                eprintln!("crabmon: {note}");
            }
            return Ok(());
        }
    }
}

/// `--doctor`: what this host lets crabmon measure.
///
/// Exits non-zero when something the config actually asks for cannot work
/// here, so it is usable as a check before a deploy rather than only as
/// something to read. A missing GPU is not a failure; a `kind = "fd"` rule on
/// a host whose fd tables are unreadable is, because that rule will never
/// fire and a rule that never fires looks exactly like a quiet machine.
fn print_doctor(cfg: &Config, args: &Args) -> Result<()> {
    // The file this run actually loaded, which `--config` may have redirected:
    // a report naming a different file than the one in force is worse than one
    // that says nothing about config at all.
    let report = crabmon::doctor::run(cfg, &config_path(args));
    let text = match args.format {
        Some(export::ExportFormat::Json) => report.to_json(),
        _ => report.to_text(),
    };
    let mut out = io::stdout().lock();
    out.write_all(text.as_bytes())?;
    out.flush()?;
    if report.has_problems() {
        std::process::exit(1);
    }
    Ok(())
}

/// `--diff`: compare two snapshots, or the two ends of one recording.
fn print_diff(a: &str, b: Option<&str>, args: &Args) -> Result<()> {
    let (before, after) = match b {
        Some(b) => (
            crabmon::diff::load(std::path::Path::new(a)).map_err(|e| anyhow::anyhow!("{e}"))?,
            crabmon::diff::load(std::path::Path::new(b)).map_err(|e| anyhow::anyhow!("{e}"))?,
        ),
        // One path: the file's own first frame against its last.
        None => {
            crabmon::diff::load_ends(std::path::Path::new(a)).map_err(|e| anyhow::anyhow!("{e}"))?
        }
    };
    let diff = crabmon::diff::compare(&before, &after);
    let text = match args.format {
        Some(export::ExportFormat::Json) => diff.to_json(),
        _ => diff.to_text(),
    };
    print!("{text}");
    io::stdout().flush()?;
    Ok(())
}

/// `--watch`: block until a condition holds, then say so and exit non-zero.
///
/// Both forms are alert rules — `--watch <query>` is "at least one process
/// matches", `--watch-rule` is every other kind — so a condition behaves the
/// same here, in `[[alert]]` and in `--serve --alerts`. The exit code is what
/// a shell is waiting on: 1 when it happened, 0 when it did not.
fn watch_for(rule: crabmon::alerts::AlertRule, cfg: &Config, args: &Args) -> Result<()> {
    // The rows to print on a match. A `proc` rule has a query to show them
    // from; the other kinds are about the machine as a whole and print no
    // process list at all.
    //
    // Not `filter::parse(&rule.query)` unconditionally: an empty query parses
    // into the empty filter, which matches *everything*, so `--watch-rule
    // 'cpu>=90'` printed the entire process table instead of nothing.
    let rows_of = match rule.query.is_empty() {
        true => None,
        false => crabmon::filter::parse(&rule.query).ok(),
    };
    let timeout = args.watch_timeout.unwrap_or(0);
    let interval = crabmon::watch::interval(cfg.refresh_ms);

    let mut source = live_source(cfg);
    source.snapshot(Duration::ZERO);
    let started = Instant::now();
    let mut pacer = Pacer::new(interval);
    let mut engine = crabmon::alerts::AlertEngine::new(vec![rule]);
    // A watch is the one headless mode that is *waiting* for something to go
    // wrong, so a recording of the run-up to it is the evidence the condition
    // exists to collect: `--watch 'state:D' --record-rotate /var/log/crabmon`
    // leaves the minutes before the stall on disk.
    let mut recordings = open_recordings(cfg, args)?;

    let (outcome, fired) = loop {
        let dt = pacer.wait();
        let snap = source.snapshot(dt);
        for note in recordings.write(&snap) {
            eprintln!("crabmon: {note}");
        }
        let now = Instant::now();
        // The engine's hold time is in seconds since the watch started, which
        // is the same clock `--watch-for` is quoted in.
        let active = engine.evaluate(&snap, now.duration_since(started).as_secs());
        if let Some(alert) = active.into_iter().next() {
            let rows = match &rows_of {
                Some(f) => snap.procs.iter().filter(|p| f.matches(p)).cloned().collect(),
                None => Vec::new(),
            };
            break (crabmon::watch::Outcome::Matched(rows), Some(alert));
        }
        if crabmon::watch::timed_out(started, timeout, now) {
            break (crabmon::watch::Outcome::TimedOut, None);
        }
    };

    if let crabmon::watch::Outcome::Matched(rows) = &outcome {
        let format = args
            .format
            .or_else(|| export::ExportFormat::parse(&cfg.export.format))
            .unwrap_or(export::ExportFormat::Json);
        let mut out = io::stdout().lock();
        // Processes when there are processes to show, which is what makes
        // `--watch nginx | jq` work. A rule about the machine — or a `below`
        // rule, which fires precisely because nothing matches — has no rows,
        // and an empty snapshot full of zeroes would be a worse answer than
        // the condition that actually fired.
        let text = match (rows.is_empty(), &fired) {
            (false, _) => {
                let snap = crabmon::Snapshot { procs: rows.clone(), ..Default::default() };
                export::render(&snap, format)
            }
            (true, Some(alert)) => match format {
                export::ExportFormat::Json => {
                    serde_json::to_string_pretty(&serde_json::json!({
                        "rule": alert.name,
                        "message": alert.message,
                        "value": alert.value,
                        "threshold": alert.threshold,
                    }))? + "\n"
                }
                export::ExportFormat::Csv => {
                    format!(
                        "rule,value,threshold\n{},{},{}\n",
                        alert.name, alert.value, alert.threshold
                    )
                }
                export::ExportFormat::Text => {
                    format!(
                        "{}\n{} against a threshold of {}\n",
                        alert.message, alert.value, alert.threshold
                    )
                }
            },
            (true, None) => String::new(),
        };
        out.write_all(text.as_bytes())?;
        out.flush()?;
    }
    // Before `exit`, which runs no destructors: every frame is flushed as it
    // is written, so the recording is already complete, but the line saying
    // where it is would otherwise never be printed.
    if let Some(note) = recordings.summary() {
        eprintln!("crabmon: {note}");
    }
    std::process::exit(outcome.exit_code());
}

/// Headless Prometheus exporter.
///
/// With `--alerts` the rules are evaluated on every scrape and exported
/// alongside the metrics, and their hooks and flight recordings run from here.
/// Evaluating on the scrape rather than on a timer of its own is deliberate:
/// it is the only moment a fresh sample exists, and a second sampler would
/// double this process's cost to measure the same machine twice.
fn serve_metrics(addr: &str, cfg: &Config, args: &Args) -> Result<()> {
    let alerts = args.alerts;
    if let Some(spec) = &args.remote {
        return serve_fleet(addr, cfg, args, spec);
    }
    let mut source = live_source(cfg);
    let interval = Duration::from_millis(cfg.refresh_ms);
    // Prime the counters so the first scrape carries real rates.
    source.snapshot(Duration::ZERO);
    std::thread::sleep(interval);
    let top = cfg.serve.top_procs;
    let mut sup = alerts.then(|| supervisor::Supervisor::new(cfg));
    if alerts {
        eprintln!("crabmon: evaluating {} alert rules on every scrape", cfg.alerts.len());
    }
    // Recording an exporter records what it *measured*, so the cadence is the
    // scrape's rather than `refresh_ms`. That is the honest timeline to keep:
    // a recording of samples nobody took would not line up with the series
    // the same process published.
    let mut recordings = open_recordings(cfg, args)?;
    if !recordings.is_empty() {
        eprintln!("crabmon: recording one frame per scrape");
    }
    // `serve` measures the gap between scrapes and hands it over; the rates are
    // deltas over that interval, not over the configured refresh.
    crabmon::serve::serve_rendered(addr, cfg.serve_token(), move |elapsed| {
        let snap = source.snapshot(elapsed);
        for note in recordings.write(&snap) {
            eprintln!("crabmon: {note}");
        }
        let states = match sup.as_mut() {
            None => Vec::new(),
            Some(sup) => {
                report(sup.observe(&snap));
                sup.alerts.states(&snap)
            }
        };
        crabmon::serve::render_with_alerts(&snap, top, &states)
    })?;
    Ok(())
}

/// Headless exporter for a fleet.
///
/// crabmon could already watch several hosts over SSH, and could already
/// export what it measured; the two could not meet, so a fleet had a
/// dashboard and no metrics. Every series carries `host="..."`, and each
/// host gets its own alert engine — rules are about *a* machine, and one
/// engine fed three hosts' snapshots in turn would hold one `fired` flag
/// between them and page about whichever host sampled last.
fn serve_fleet(addr: &str, cfg: &Config, args: &Args, spec: &str) -> Result<()> {
    let alerts = args.alerts;
    let targets = crabmon::fleet::parse_targets(spec);
    if targets.is_empty() {
        return Err(anyhow::anyhow!("--remote names no hosts"));
    }
    let top = cfg.serve.top_procs;
    eprintln!("crabmon: exporting {} hosts: {}", targets.len(), targets.join(", "));
    if alerts {
        eprintln!("crabmon: evaluating {} alert rules per host on every scrape", cfg.alerts.len());
    }

    // One source and one supervisor per host. The sources are threaded, so a
    // host that is down costs a scrape nothing but its own missing series.
    // `Send`, because the sampling closure runs on whichever connection
    // thread is serving the scrape.
    // One source, one supervisor and one set of recordings per host. A fleet
    // has a timeline per machine, so a single file would interleave them into
    // a recording that replays as one machine behaving impossibly; each host
    // gets its own name instead.
    // Distinct filesystem names for the targets, decided over the whole list:
    // two that reduce to the same thing would otherwise share a recording and
    // truncate each other's, and one that reduces to nothing would write
    // among the flight dumps.
    let labels = crabmon::record::unique_labels(&targets);
    let mut hosts: Vec<Host> = Vec::with_capacity(targets.len());
    for (target, label) in targets.iter().zip(&labels) {
        let remote = match cfg.remote.stream {
            true => RemoteSource::streaming(target, cfg.refresh_ms, Vec::new()),
            false => RemoteSource::new(target, None, Vec::new()),
        };
        let recordings = crabmon::record::Recordings::for_host(
            args.record.as_deref(),
            args.record_rotate.as_deref(),
            &cfg.record,
            label,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        for note in recordings.notices() {
            eprintln!("crabmon: {target}: {note}");
        }
        hosts.push(Host {
            name: target.clone(),
            source: ThreadedSource::new(Box::new(remote)),
            sup: alerts.then(|| supervisor::Supervisor::new(cfg).for_host(target)),
            freshness: supervisor::RemoteFreshness::new(Duration::from_millis(cfg.refresh_ms)),
            recordings,
        });
    }

    crabmon::serve::serve_rendered(addr, cfg.serve_token(), move |elapsed| {
        let parts: Vec<(String, String)> = hosts
            .iter_mut()
            .map(|host| {
                let snap = host.source.snapshot(elapsed);
                let (fresh, reachable) = host.freshness.update(
                    &snap,
                    host.source.frame_id(),
                    host.source.error().is_none(),
                    Instant::now(),
                    host.sup.as_mut().map(|s| &mut s.alerts),
                );
                // Only a frame that actually arrived: re-recording the last
                // good one on every scrape would fill the file with an
                // outage drawn as measurement.
                if fresh {
                    for note in host.recordings.write(&snap) {
                        eprintln!("crabmon: {}: {note}", host.name);
                    }
                }
                let states = match host.sup.as_mut() {
                    None => Vec::new(),
                    Some(sup) => {
                        if fresh {
                            report(sup.observe(&snap));
                        }
                        if reachable { sup.alerts.states(&snap) } else { Vec::new() }
                    }
                };
                // A host that cannot be reached is not up, whatever its
                // last good frame still says.
                (host.name.clone(), crabmon::serve::render_host(&snap, top, &states, reachable))
            })
            .collect();
        crabmon::serve::merge_hosts(&parts)
    })?;
    Ok(())
}

/// One machine of a fleet export: where its frames come from, what its rules
/// make of them, and where they are recorded.
struct Host {
    name: String,
    source: ThreadedSource,
    sup: Option<supervisor::Supervisor>,
    freshness: supervisor::RemoteFreshness,
    recordings: crabmon::record::Recordings,
}

/// Print what a headless alert pass produced, and run its hooks.
///
/// stderr, not stdout: `--stream`'s stdout is a frame per line that something
/// else is parsing, and a recording notice in the middle of it would break the
/// reader on the far end of an SSH pipe.
fn report(obs: crabmon::supervisor::Observation) {
    for note in obs.notes {
        eprintln!("crabmon: {note}");
    }
    for name in obs.active.iter().map(|a| &a.message) {
        eprintln!("crabmon: {name}");
    }
    for err in supervisor::spawn_hooks(&obs.commands) {
        eprintln!("crabmon: {err}");
    }
}

/// Report a watch that could not be run, and exit with a code no match uses.
fn fail_watch(message: &str) -> ! {
    eprintln!("crabmon: {message}");
    std::process::exit(crabmon::watch::FAILED)
}

fn config_path(args: &Args) -> std::path::PathBuf {
    match &args.config {
        Some(path) => std::path::PathBuf::from(path),
        None => config::config_path(),
    }
}

/// Command-line flags win over the config file for this run.
fn apply_args(mut cfg: Config, args: &Args) -> Config {
    if let Some(ms) = args.refresh_ms {
        cfg.refresh_ms = ms;
    }
    if let Some(s) = args.sort {
        cfg.sort_by = s;
    }
    if let Some(d) = args.sort_desc {
        cfg.sort_desc = d;
    }
    if let Some(f) = &args.filter {
        cfg.filter = f.clone();
    }
    if let Some(t) = &args.theme {
        cfg.theme = t.clone();
    }
    if let Some(c) = &args.columns {
        cfg.procs.columns = c.clone();
    }
    if let Some(l) = args.layout {
        cfg.layout = l;
    }
    if let Some(t) = args.tree {
        cfg.tree = t;
    }
    if let Some(n) = args.top {
        cfg.export.top_n = n;
    }
    if let Some(g) = args.group {
        cfg.group_by = g;
    }
    cfg.sanitize()
}

/// `--once`: sample twice so CPU percentages are real, then print and exit.
fn print_once(cfg: &Config, args: &Args, source: &mut SysinfoSource) -> Result<()> {
    let interval = Duration::from_millis(cfg.refresh_ms.max(crabmon::MIN_REFRESH_MS));
    source.snapshot(Duration::from_secs(0));
    // Not `interval`: the priming sample costs real time, and the rates in the
    // printed snapshot are counter deltas over the gap between the two reads.
    let mut pacer = Pacer::new(interval);
    let dt = pacer.wait();
    let mut snap = source.snapshot(dt);

    // A filter from the config file reaches here unvalidated; silently
    // degrading to "match everything" is the failure a script cannot see.
    let filter =
        crabmon::filter::parse(&cfg.filter).map_err(|e| anyhow::anyhow!("bad filter: {e}"))?;
    snap.procs.retain(|p| filter.matches(p));
    crabmon::sort::sort_rows(&mut snap.procs, cfg.sort_by, cfg.sort_desc);
    let snap = export::trim_procs(&snap, cfg.export.top_n);

    let format = args
        .format
        .or_else(|| export::ExportFormat::parse(&cfg.export.format))
        .unwrap_or(export::ExportFormat::Json);
    let mut out = io::stdout().lock();
    // The text table is laid out for whoever is reading it: the terminal's
    // width when there is a terminal, and a fixed width when the output is
    // going into a file or a pipe, so a dump is not reflowed by the window
    // that happened to produce it.
    let text = match format {
        export::ExportFormat::Text => export::to_text_with(
            &snap,
            &cfg.procs.columns,
            crossterm::terminal::size().map(|(w, _)| w.max(60)).unwrap_or(export::TEXT_WIDTH),
        ),
        other => export::render(&snap, other),
    };
    out.write_all(text.as_bytes())?;
    out.flush()?;
    Ok(())
}

fn run_tui(
    cfg: Config,
    source: Box<dyn MetricSource>,
    args: &Args,
    config_path: std::path::PathBuf,
    config_ok: bool,
) -> Result<()> {
    let mouse = !args.no_mouse;
    // What the file itself says about the filter, before `apply_args` folded a
    // one-off `--filter` over it. Restored before `persist()` below.
    let file_filter = crabmon::config::load_from(&config_path).filter;
    install_panic_hook(mouse);
    let stop = install_signal_handler();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    if mouse {
        execute!(stdout, EnableMouseCapture)?;
    }
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let mut app = App::new(cfg, source);
    // A fleet opens on the fleet table: the first question is which machine,
    // and the dashboard of whichever host happened to be listed first is not
    // an answer to it.
    if !app.fleet().is_empty() && args.layout.is_none() {
        app.cfg.layout = crabmon::ui::Layout::Fleet;
    }
    // A config file that could not be parsed is never written back over:
    // saving would replace the user's file — typo and all the settings
    // around it — with the defaults crabmon fell back to.
    app.config_path = config_ok.then_some(config_path);
    if !config_ok {
        app.set_status("config could not be parsed; defaults in use and not saving");
    }
    app.audit_path = audit_path(&app.cfg);

    // The same resolution every headless mode does, through the same type:
    // an explicit `--record` file, plus a rotating set from
    // `--record-rotate` or `[record] rotate_frames`.
    let mut recordings = match crabmon::record::Recordings::open(
        args.record.as_deref(),
        args.record_rotate.as_deref(),
        &app.cfg.record,
    ) {
        Ok(r) => r,
        Err(e) => {
            restore(mouse);
            return Err(anyhow::anyhow!("{e}"));
        }
    };
    for note in recordings.notices() {
        eprintln!("crabmon: {note}");
    }
    // The flight recorder writes into the working directory by default, the
    // same place `P` exports to, so a dump lands somewhere the user can find.
    let flight_dir = match app.cfg.record.flight_dir.is_empty() {
        true => std::path::PathBuf::from("."),
        false => std::path::PathBuf::from(&app.cfg.record.flight_dir),
    };
    let mut flight = FlightRecorder::new(
        &flight_dir,
        app.cfg.record.flight,
        app.cfg.record.flight_after,
        app.cfg.record.limits(),
    );

    let res = event_loop(&mut terminal, &mut app, &stop, &mut recordings, flight.as_mut());

    // Always restore the terminal, even if the loop failed.
    restore(mouse);
    // The filter is session state: it comes from `/` or from a one-off
    // `--filter`, and persisting it silently poisoned every later run — a
    // config left holding `filter = "..."` makes `crabmon --once` print an
    // empty process list with exit 0, and the TUI open on an empty table with
    // no visible cause. Everything else the user changed in the TUI still
    // persists; only what the file itself said about the filter is written.
    app.cfg.filter = file_filter;
    app.persist();
    if let Some(note) = recordings.summary() {
        eprintln!("crabmon: {note}");
    }
    res
}

/// Where the action log is written, or `None` when it is disabled.
fn audit_path(cfg: &Config) -> Option<std::path::PathBuf> {
    if !cfg.audit.enabled {
        return None;
    }
    Some(if cfg.audit.path.is_empty() {
        crabmon::audit::default_path()
    } else {
        std::path::PathBuf::from(&cfg.audit.path)
    })
}

/// Longest the event loop will block for input. Bounds how late a transient
/// status message can be to expire, and is what a paused UI waits out.
const MAX_POLL: Duration = Duration::from_millis(250);

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    stop: &StopFlag,
    recordings: &mut crabmon::record::Recordings,
    mut flight: Option<&mut FlightRecorder>,
) -> Result<()> {
    let mut last_tick = Instant::now();
    loop {
        if stop.triggered() {
            break;
        }
        terminal.draw(|f| crabmon::ui::draw(f, app))?;

        // Wake up in time for the next metric refresh, but at least often
        // enough that a transient status message can expire on screen.
        //
        // The "no tick is due and the deadline has passed" arm is what pausing
        // (`z`) and replay scrubbing land in — `App::scrub` pauses too — and
        // there `last_tick` stops advancing, so the countdown is expired
        // forever. Falling through to a zero timeout made `event::poll` return
        // instantly and turned the loop into a redraw spin: a pegged core and
        // several MB/s of escape sequences at the terminal, for a UI that is by
        // definition not changing. A paused screen has nothing to do until a
        // key arrives, so wait out the cap.
        let timeout = match app.refresh.checked_sub(last_tick.elapsed()) {
            Some(remaining) => remaining.min(MAX_POLL),
            None if app.due() => Duration::ZERO,
            None => MAX_POLL,
        };

        if event::poll(timeout)? {
            let action = match event::read()? {
                // Windows sends both press and release; only act on press.
                Event::Key(key) if key.kind != KeyEventKind::Release => app.on_key(key),
                Event::Mouse(m) => app.on_mouse(m),
                _ => Action::None,
            };
            match action {
                Action::Quit => break,
                Action::Export => app.export_now(),
                Action::None => {}
            }
        }

        if stop.triggered() {
            break;
        }
        if app.due() {
            app.tick();
            last_tick = Instant::now();
            // On screen rather than on stderr: stderr is behind the alternate
            // screen and nobody would ever see it.
            for note in recordings.write(&app.snap) {
                app.set_status(note);
            }
            // Hooks and flight recordings. `App::tick` has already evaluated
            // the rules; this drains what that produced, through the same
            // code `--serve --alerts` runs, so an incident is handled
            // identically whether or not anyone is watching.
            for note in app.drain_alert_side_effects(flight.as_deref_mut()) {
                app.set_status(note);
            }
        }
    }
    Ok(())
}

fn restore(mouse: bool) {
    let mut stdout = io::stdout();
    if mouse {
        let _ = execute!(stdout, DisableMouseCapture);
    }
    let _ = execute!(stdout, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

/// Without this, any panic leaves the user in raw mode on the alternate screen.
fn install_panic_hook(mouse: bool) {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore(mouse);
        default(info);
    }));
}

#[derive(Clone)]
struct StopFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl StopFlag {
    fn triggered(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// How long the watchdog gives the event loop to notice the stop flag before
/// killing the process outright.
#[cfg(unix)]
const SHUTDOWN_GRACE: Duration = Duration::from_millis(750);

/// SIGTERM exits through the normal path so the config is saved and the
/// terminal restored.
///
/// SIGHUP is deliberately *not* handled: it means the terminal is already gone,
/// and `crossterm::event::poll` never returns once its pty is hung up — it
/// spins internally, so the loop would never reach the flag check and the
/// process would burn a core forever. Leaving SIGHUP at its default action lets
/// the kernel end it immediately.
///
/// A watchdog covers the same wedge for SIGTERM: if the loop is stuck inside
/// crossterm, the process still dies instead of ignoring the signal.
#[cfg(unix)]
fn install_signal_handler() -> StopFlag {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _ = signal_hook::flag::register(signal_hook::consts::SIGTERM, flag.clone());

    let watch = flag.clone();
    std::thread::spawn(move || {
        while !watch.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(SHUTDOWN_GRACE);
        // Still here: the event loop is wedged, which means writing to the
        // terminal is exactly what is stuck. Restoring it would block on the
        // same fd, and `process::exit` would deadlock trying to flush a stdout
        // whose lock the main thread is holding mid-write. `_exit` skips all
        // cleanup, which is the only thing that can still make progress.
        //
        // SAFETY: `_exit` only ends the process; it touches no Rust state.
        unsafe { libc::_exit(0) };
    });

    StopFlag(flag)
}

#[cfg(not(unix))]
fn install_signal_handler() -> StopFlag {
    StopFlag(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
}
