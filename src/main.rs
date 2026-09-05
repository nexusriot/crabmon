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
use crabmon::record::{Recorder, ReplaySource};
use crabmon::remote::RemoteSource;

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

    // `--serve` and `--once` are headless; both need the live host.
    if let Some(addr) = &args.serve {
        return serve_metrics(addr, &cfg);
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
}

/// Where snapshots come from: a recording, another host, or this one.
fn build_source(cfg: &Config, args: &Args) -> Result<Box<dyn MetricSource>> {
    if let Some(path) = &args.replay {
        let source =
            ReplaySource::open(std::path::Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("crabmon: replaying {} frames from {path}", source.len());
        return Ok(Box::new(source));
    }
    if let Some(target) = &args.remote {
        return Ok(Box::new(RemoteSource::new(target, args.remote_command.clone(), Vec::new())));
    }
    Ok(Box::new(live_source(cfg)))
}

/// Headless Prometheus exporter.
fn serve_metrics(addr: &str, cfg: &Config) -> Result<()> {
    let mut source = live_source(cfg);
    let interval = Duration::from_millis(cfg.refresh_ms);
    // Prime the counters so the first scrape carries real rates.
    source.snapshot(Duration::ZERO);
    std::thread::sleep(interval);
    let top = cfg.serve.top_procs;
    crabmon::serve::serve(addr, top, move || source.snapshot(interval))?;
    Ok(())
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
    std::thread::sleep(interval);
    let mut snap = source.snapshot(interval);

    let filter = crabmon::filter::parse(&cfg.filter).unwrap_or_default();
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
    let res = event_loop(&mut terminal, &mut app, &stop, recorder.as_mut());

    // Always restore the terminal, even if the loop failed.
    restore(mouse);
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

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    stop: &StopFlag,
    mut recorder: Option<&mut Recorder>,
) -> Result<()> {
    let mut last_draw = Instant::now();
    loop {
        if stop.triggered() {
            break;
        }
        terminal.draw(|f| crabmon::ui::draw(f, app))?;

        // Wake up in time for the next metric refresh, but at least often
        // enough that a transient status message can expire on screen.
        let timeout = app
            .refresh
            .checked_sub(last_draw.elapsed())
            .unwrap_or(Duration::ZERO)
            .min(Duration::from_millis(250));

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
            last_draw = Instant::now();
            if let Some(rec) = recorder.as_deref_mut() {
                if let Err(e) = rec.write(&app.snap) {
                    app.set_status(format!("recording failed: {e}"));
                }
            }
            run_alert_commands(app);
        }
    }
    Ok(())
}

/// Alert hooks are user-configured shell commands, spawned detached.
fn run_alert_commands(app: &mut App) {
    for cmd in app.alerts.take_commands() {
        let spawned = std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if let Err(e) = spawned {
            app.set_status(format!("alert command failed: {e}"));
        }
    }
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
