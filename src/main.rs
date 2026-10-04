//! Terminal setup/teardown and the event loop.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind};
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crabmon::app::{Action, App};
use crabmon::cli::{self, Args, Parsed};
use crabmon::config::{self, Config};
use crabmon::export;
use crabmon::metrics::{MetricSource, SysinfoSource};
use crabmon::record::{trim_frame, FlightRecorder, Recorder, ReplaySource};
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
    let cfg = apply_args(config::load_from(&path), &args);

    // The headless modes. `cli::parse` refuses to combine any of them with
    // `--remote` or `--replay`, so each one here samples this host.
    if let Some((a, b)) = &args.diff {
        return print_diff(a, b.as_deref(), &args);
    }
    if let Some(addr) = &args.serve {
        return serve_metrics(addr, &cfg, args.alerts);
    }
    if args.stream {
        return stream_frames(&cfg, args.alerts);
    }
    let hold = args.watch_for.unwrap_or(0);
    if let Some(query) = &args.watch {
        return watch_for(crabmon::watch::process_rule(query, hold), &cfg, &args);
    }
    if let Some(expr) = &args.watch_rule {
        let rule = crabmon::watch::parse_rule(expr, hold).map_err(|e| anyhow::anyhow!("{e}"))?;
        return watch_for(rule, &cfg, &args);
    }
    if args.once {
        let mut source = live_source(&cfg);
        return print_once(&cfg, &args, &mut source);
    }

    let source = build_source(&cfg, &args)?;
    run_tui(cfg, source, &args, path)
}

fn live_source(cfg: &Config) -> SysinfoSource {
    SysinfoSource::new(cfg.virtual_iface_prefixes.clone(), cfg.gpu.enabled && cfg.gpu.nvidia_smi)
        .with_fds(cfg.procs.fds)
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

/// `--stream`: one JSON snapshot per line, forever. This is what a streaming
/// `--remote` runs on the far end, so frames are trimmed the same way a
/// recording is — a full process list every second over SSH is unusable.
///
/// With `--alerts` the same rules the TUI evaluates run here too, so a host
/// streaming to a collector also pages and writes flight recordings. Hook
/// output and recording notices go to stderr, which keeps stdout a clean
/// stream of frames for whatever is reading it.
fn stream_frames(cfg: &Config, alerts: bool) -> Result<()> {
    let mut source = live_source(cfg);
    source.snapshot(Duration::ZERO);
    // The interval the rates are divided by has to be the one that actually
    // passed, not the one that was asked for — see `Pacer`.
    let mut pacer = Pacer::new(Duration::from_millis(cfg.refresh_ms));
    let mut sup = alerts.then(|| supervisor::Supervisor::new(cfg));
    let mut out = io::stdout().lock();
    loop {
        let dt = pacer.wait();
        let snap = source.snapshot(dt);
        if let Some(sup) = sup.as_mut() {
            report(sup.observe(&snap));
        }
        let frame =
            trim_frame(&snap, cfg.record.top_n, cfg.record.omit_paths, cfg.record.max_cmd_len);
        let line = serde_json::to_string(&frame)?;
        // A broken pipe means the reader went away; that is a normal exit here,
        // not a crash to report.
        if out.write_all(line.as_bytes()).is_err() || out.write_all(b"\n").is_err() {
            return Ok(());
        }
        if out.flush().is_err() {
            return Ok(());
        }
    }
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

    let (outcome, fired) = loop {
        let dt = pacer.wait();
        let snap = source.snapshot(dt);
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
            },
            (true, None) => String::new(),
        };
        out.write_all(text.as_bytes())?;
        out.flush()?;
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
fn serve_metrics(addr: &str, cfg: &Config, alerts: bool) -> Result<()> {
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
    // `serve` measures the gap between scrapes and hands it over; the rates are
    // deltas over that interval, not over the configured refresh.
    crabmon::serve::serve_rendered(addr, move |elapsed| {
        let snap = source.snapshot(elapsed);
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
    out.write_all(export::render(&snap, format).as_bytes())?;
    out.flush()?;
    Ok(())
}

fn run_tui(
    cfg: Config,
    source: Box<dyn MetricSource>,
    args: &Args,
    config_path: std::path::PathBuf,
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
    app.config_path = Some(config_path);
    app.audit_path = audit_path(&app.cfg);

    let mut recorder = match &args.record {
        Some(path) => match Recorder::create(std::path::Path::new(path)) {
            Ok(r) => Some(r.with_limits(
                app.cfg.record.top_n,
                app.cfg.record.omit_paths,
                app.cfg.record.max_cmd_len,
            )),
            Err(e) => {
                restore(mouse);
                return Err(anyhow::anyhow!("cannot record to {path}: {e}"));
            }
        },
        None => None,
    };
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
    let res = event_loop(&mut terminal, &mut app, &stop, recorder.as_mut(), flight.as_mut());

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
    if let Some(rec) = &recorder {
        eprintln!("crabmon: wrote {} frames to {}", rec.frames(), rec.path().display());
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
    mut recorder: Option<&mut Recorder>,
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
            if let Some(rec) = recorder.as_deref_mut() {
                if let Err(e) = rec.write(&app.snap) {
                    app.set_status(format!("recording failed: {e}"));
                }
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
