//! Monitoring another machine over SSH.
//!
//! `crabmon --remote host` runs `crabmon --once` on the far end and parses its
//! JSON, so the whole UI works against a remote host with no daemon, no open
//! port and no protocol of its own. The refresh interval is bounded by the SSH
//! round trip, not by the sampler.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

use crate::metrics::{MetricSource, Snapshot};

/// What the far end runs when it is only asked for a single frame.
pub const ONE_SHOT_COMMAND: &str = "crabmon --once --refresh 200";

/// How long a one-shot fetch may take before it is killed.
///
/// `ConnectTimeout` only bounds setting the connection up. Once SSH is
/// connected, a command that never returns — a `statvfs` on a wedged NFS mount
/// on the far end, or a `--remote-command` that streams instead of exiting —
/// used to hang forever, and the first sample is taken synchronously, so the
/// interface never appeared and there was nothing to press `q` in.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// ...and when it can stream. One SSH session and one process start for the
/// whole session instead of one per sample, which is what made `--remote` feel
/// like a slideshow: every frame paid a TCP handshake, a key exchange and a
/// full crabmon startup.
pub fn stream_command(refresh_ms: u64) -> String {
    format!("crabmon --stream --refresh {refresh_ms}")
}

/// Whether an error from the far end means "this crabmon is too old to stream"
/// rather than something worth reporting.
///
/// Falling back on *any* failure would paper over a genuinely broken host; this
/// only matches the argument parser refusing the flag.
pub fn is_unsupported_flag(stderr: &str) -> bool {
    let s = stderr.to_lowercase();
    (s.contains("unknown option") || s.contains("unrecognized") || s.contains("unrecognised"))
        && s.contains("stream")
}

/// A frame stream from one long-lived SSH session.
struct Stream {
    child: Child,
    frames: Receiver<Result<Snapshot, String>>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        // The reader thread ends when the pipe closes, which killing the child
        // does; without this an abandoned ssh keeps running until the far end
        // notices.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct RemoteSource {
    target: String,
    /// The command run on the far end; overridable for a non-standard path.
    remote_command: String,
    /// Set when the command was chosen by us and may be swapped for the
    /// one-shot form. An explicit `--remote-command` is never second-guessed.
    stream_wanted: bool,
    ssh_options: Vec<String>,
    stream: Option<Stream>,
    last_good: Option<Snapshot>,
    pub last_error: Option<String>,
}

/// Collect a child's output, killing it if it outlives `timeout`.
///
/// `Command::output()` waits forever, which is the wrong thing when the answer
/// is being read into a full-screen interface.
pub fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output, String> {
    // Drain both pipes on threads: a child that fills one and blocks would
    // never exit, so polling `try_wait` alone could not tell the difference
    // between "slow" and "deadlocked on a full pipe".
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_reader = std::thread::spawn(move || read_all(stdout));
    let err_reader = std::thread::spawn(move || read_all(stderr));

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => return Err(format!("ssh: {e}")),
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("no answer in {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    Ok(std::process::Output {
        status,
        stdout: out_reader.join().unwrap_or_default(),
        stderr: err_reader.join().unwrap_or_default(),
    })
}

fn read_all(pipe: Option<impl std::io::Read>) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut buf);
    }
    buf
}

/// Build the argv for the SSH invocation.
pub fn ssh_args(target: &str, options: &[String], remote_command: &str) -> Vec<String> {
    let mut args: Vec<String> = vec![
        // Fail fast rather than hanging the UI on an unreachable host, and
        // never stop to ask about host keys in a full-screen TUI.
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=5".into(),
    ];
    args.extend(options.iter().cloned());
    args.push(target.to_string());
    args.push(remote_command.to_string());
    args
}

impl RemoteSource {
    pub fn new(target: &str, remote_command: Option<String>, ssh_options: Vec<String>) -> Self {
        RemoteSource {
            target: target.to_string(),
            remote_command: remote_command.unwrap_or_else(|| ONE_SHOT_COMMAND.to_string()),
            stream_wanted: false,
            ssh_options,
            stream: None,
            last_good: None,
            last_error: None,
        }
    }

    /// Hold one SSH session open and read frames from it. Falls back to a
    /// process per sample if the far end does not understand `--stream`.
    pub fn streaming(target: &str, refresh_ms: u64, ssh_options: Vec<String>) -> Self {
        RemoteSource {
            remote_command: stream_command(refresh_ms),
            stream_wanted: true,
            ..RemoteSource::new(target, None, ssh_options)
        }
    }

    pub fn command(&self) -> &str {
        &self.remote_command
    }

    /// Start (or restart) the SSH session behind a stream.
    fn open_stream(&mut self) -> Result<(), String> {
        let args = ssh_args(&self.target, &self.ssh_options, &self.remote_command);
        let mut child = Command::new("ssh")
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ssh: {e}"))?;
        let stdout = child.stdout.take().ok_or("ssh produced no stdout")?;
        let stderr = child.stderr.take();
        let (tx, rx) = mpsc::channel();

        // stderr is drained on its own thread: a full pipe would otherwise
        // block the far end mid-frame and stall the stream silently.
        let err_tx = tx.clone();
        std::thread::spawn(move || {
            let Some(stderr) = stderr else { return };
            let mut first: Option<String> = None;
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if first.is_none() && !line.trim().is_empty() {
                    first = Some(line.trim().to_string());
                }
            }
            if let Some(msg) = first {
                let _ = err_tx.send(Err(msg));
            }
        });

        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                let frame = serde_json::from_str::<Snapshot>(&line)
                    .map_err(|e| format!("unparseable frame: {e}"));
                if tx.send(frame).is_err() {
                    return; // the source was dropped
                }
            }
        });

        self.stream = Some(Stream { child, frames: rx });
        Ok(())
    }

    /// Newest frame the stream has produced, or `None` if it has produced none
    /// since the last call. Never blocks: a stream that is behind shows the
    /// previous frame rather than freezing the interface.
    fn drain_stream(&mut self) -> Option<Snapshot> {
        let mut newest = None;
        let mut dead = false;
        if let Some(stream) = &self.stream {
            loop {
                match stream.frames.try_recv() {
                    Ok(Ok(snap)) => newest = Some(snap),
                    Ok(Err(e)) => {
                        // A stale crabmon on the far end: drop to one frame per
                        // sample rather than showing nothing at all.
                        if self.stream_wanted && is_unsupported_flag(&e) {
                            self.remote_command = ONE_SHOT_COMMAND.to_string();
                            self.stream_wanted = false;
                        } else {
                            self.last_error = Some(format!("{}: {e}", self.target));
                        }
                        dead = true;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        dead = true;
                        break;
                    }
                }
            }
        }
        if dead {
            self.stream = None;
        }
        newest
    }

    /// One fetch. Separated from `snapshot` so the error path is testable.
    pub fn fetch(&mut self) -> Result<Snapshot, String> {
        let args = ssh_args(&self.target, &self.ssh_options, &self.remote_command);
        let child = Command::new("ssh")
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ssh: {e}"))?;
        let out =
            wait_with_timeout(child, FETCH_TIMEOUT).map_err(|e| format!("{}: {e}", self.target))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let first = err.lines().next().unwrap_or("ssh failed").trim();
            return Err(format!("{}: {first}", self.target));
        }
        parse_remote_output(&out.stdout).map_err(|e| format!("{}: {e}", self.target))
    }
}

/// Parse a remote `crabmon --once` response.
pub fn parse_remote_output(stdout: &[u8]) -> Result<Snapshot, String> {
    let text = String::from_utf8_lossy(stdout);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("no output — is crabmon installed on the remote host?".into());
    }
    serde_json::from_str::<Snapshot>(trimmed).map_err(|e| {
        if trimmed.starts_with("pid,") {
            "remote returned CSV; it needs --format json".to_string()
        } else {
            format!("unparseable response: {e}")
        }
    })
}

impl MetricSource for RemoteSource {
    fn snapshot(&mut self, _dt: Duration) -> Snapshot {
        if self.stream_wanted {
            if self.stream.is_none() {
                if let Err(e) = self.open_stream() {
                    self.last_error = Some(format!("{}: {e}", self.target));
                    return self.last_good.clone().unwrap_or_default();
                }
            }
            if let Some(snap) = self.drain_stream() {
                self.last_error = None;
                self.last_good = Some(snap.clone());
                return snap;
            }
            // Nothing new yet — the first frame is still in flight, or the far
            // end is slower than we are asking. Keep what is on screen.
            if self.stream.is_some() && self.last_good.is_some() {
                return self.last_good.clone().unwrap_or_default();
            }
            if self.stream.is_some() {
                return Snapshot::default();
            }
        }
        match self.fetch() {
            Ok(snap) => {
                self.last_error = None;
                self.last_good = Some(snap.clone());
                snap
            }
            Err(e) => {
                self.last_error = Some(e);
                // Hold the last good frame so a transient network blip does not
                // blank every panel.
                self.last_good.clone().unwrap_or_default()
            }
        }
    }

    fn label(&self) -> Option<String> {
        let how = if self.stream_wanted { "streaming" } else { "remote" };
        Some(match &self.last_error {
            Some(e) => format!("remote {} UNREACHABLE — {e}", self.target),
            None => format!("{how} {}", self.target),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_args_fail_fast_instead_of_hanging_the_ui() {
        let args = ssh_args("host", &[], "crabmon --once");
        assert!(args.windows(2).any(|w| w == ["-o", "BatchMode=yes"]));
        assert!(args.windows(2).any(|w| w == ["-o", "ConnectTimeout=5"]));
        assert_eq!(args[args.len() - 2], "host");
        assert_eq!(args[args.len() - 1], "crabmon --once");
    }

    #[test]
    fn extra_ssh_options_are_passed_before_the_target() {
        let args = ssh_args("user@host", &["-p".into(), "2222".into()], "crabmon --once");
        let p = args.iter().position(|a| a == "-p").unwrap();
        assert_eq!(args[p + 1], "2222");
        assert!(p < args.iter().position(|a| a == "user@host").unwrap());
    }

    #[test]
    fn a_valid_response_parses_into_a_snapshot() {
        let snap = Snapshot { taken_at_unix: 42, ..Default::default() };
        let json = serde_json::to_string(&snap).unwrap();
        assert_eq!(parse_remote_output(json.as_bytes()).unwrap().taken_at_unix, 42);
    }

    #[test]
    fn an_empty_response_says_what_is_probably_wrong() {
        let err = parse_remote_output(b"").unwrap_err();
        assert!(err.contains("is crabmon installed"), "{err}");
    }

    #[test]
    fn a_csv_response_names_the_actual_mistake() {
        let err = parse_remote_output(b"pid,ppid,name,cpu_percent\n1,,init,0.0\n").unwrap_err();
        assert!(err.contains("--format json"), "{err}");
    }

    #[test]
    fn garbage_is_reported_rather_than_silently_dropped() {
        assert!(parse_remote_output(b"bash: crabmon: command not found").is_err());
    }

    #[test]
    fn a_command_that_never_returns_is_killed_rather_than_waited_on_forever() {
        // The first sample is synchronous, so a far end that hangs used to mean
        // the interface never appeared and there was nothing to press `q` in.
        let child = Command::new("sleep")
            .arg("300")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        let err = wait_with_timeout(child, Duration::from_millis(300)).unwrap_err();
        assert!(err.contains("no answer"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(5), "took {:?}", start.elapsed());
    }

    #[test]
    fn a_command_that_answers_in_time_is_collected_whole() {
        let child = Command::new("sh")
            .args(["-c", "printf hello; printf oops >&2"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let out = wait_with_timeout(child, Duration::from_secs(5)).unwrap();
        assert!(out.status.success());
        assert_eq!(out.stdout, b"hello");
        assert_eq!(out.stderr, b"oops");
    }

    #[test]
    fn a_child_that_writes_more_than_a_pipe_holds_still_completes() {
        // Polling `try_wait` without draining the pipes would deadlock here:
        // the child blocks writing, so it never exits, so the wait never ends.
        let child = Command::new("sh")
            .args(["-c", "head -c 400000 /dev/zero | tr '\\0' 'x'"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let out = wait_with_timeout(child, Duration::from_secs(10)).unwrap();
        assert_eq!(out.stdout.len(), 400_000);
    }

    #[test]
    fn a_failed_fetch_holds_the_previous_frame_instead_of_blanking_the_ui() {
        let mut src = RemoteSource::new("nonexistent.invalid", None, vec![]);
        src.last_good = Some(Snapshot { taken_at_unix: 99, ..Default::default() });
        let snap = src.snapshot(Duration::ZERO);
        assert_eq!(snap.taken_at_unix, 99, "the stale frame is kept");
        assert!(src.last_error.is_some());
        let label = src.label().unwrap();
        assert!(label.contains("nonexistent.invalid"), "{label}");
        assert!(label.contains("UNREACHABLE"), "a blank UI must explain itself: {label}");
    }

    #[test]
    fn the_default_remote_command_asks_for_json_at_a_usable_interval() {
        let src = RemoteSource::new("host", None, vec![]);
        assert!(src.remote_command.contains("--once"));
        assert!(src.remote_command.contains("--refresh 200"));
    }

    #[test]
    fn a_streaming_source_runs_one_long_lived_command() {
        let src = RemoteSource::streaming("host", 1000, vec![]);
        assert_eq!(src.command(), "crabmon --stream --refresh 1000");
        assert!(src.label().unwrap().starts_with("streaming host"));
    }

    #[test]
    fn an_explicit_remote_command_is_never_second_guessed() {
        let src = RemoteSource::new("host", Some("/opt/crabmon --once".into()), vec![]);
        assert_eq!(src.command(), "/opt/crabmon --once");
        assert!(!src.stream_wanted, "an explicit command must not be swapped out");
    }

    #[test]
    fn only_the_parser_refusing_the_flag_triggers_a_fallback() {
        assert!(is_unsupported_flag("crabmon: unknown option: --stream"));
        assert!(is_unsupported_flag("CRABMON: UNRECOGNIZED OPTION --STREAM"));
        // Everything else is a real failure and must be reported, not hidden.
        assert!(!is_unsupported_flag("ssh: connect to host x port 22: No route to host"));
        assert!(!is_unsupported_flag("bash: crabmon: command not found"));
        assert!(!is_unsupported_flag("crabmon: unknown option: --nonsense"));
    }

    #[test]
    fn a_dead_stream_falls_back_to_one_shot_when_the_far_end_is_old() {
        let mut src = RemoteSource::streaming("host", 800, vec![]);
        let (tx, rx) = mpsc::channel();
        tx.send(Err("crabmon: unknown option: --stream".to_string())).unwrap();
        drop(tx);
        // A child we control, so the test never touches the network.
        let child = Command::new("true").spawn().unwrap();
        src.stream = Some(Stream { child, frames: rx });

        assert!(src.drain_stream().is_none());
        assert!(!src.stream_wanted, "it should stop asking for a stream");
        assert_eq!(src.command(), ONE_SHOT_COMMAND);
        assert!(src.last_error.is_none(), "an expected fallback is not an error");
    }

    #[test]
    fn a_real_stream_error_is_reported_rather_than_swallowed() {
        let mut src = RemoteSource::streaming("host", 800, vec![]);
        let (tx, rx) = mpsc::channel();
        tx.send(Err("ssh: Permission denied (publickey)".to_string())).unwrap();
        drop(tx);
        let child = Command::new("true").spawn().unwrap();
        src.stream = Some(Stream { child, frames: rx });

        src.drain_stream();
        let err = src.last_error.clone().expect("the failure must surface");
        assert!(err.contains("Permission denied"), "{err}");
    }

    #[test]
    fn the_newest_frame_wins_when_the_stream_runs_ahead() {
        // Falling behind must not queue up stale frames and replay them slowly.
        let mut src = RemoteSource::streaming("host", 800, vec![]);
        let (tx, rx) = mpsc::channel();
        for n in 1..=3u64 {
            tx.send(Ok(Snapshot { taken_at_unix: n, ..Default::default() })).unwrap();
        }
        let child = Command::new("sleep").arg("30").spawn().unwrap();
        src.stream = Some(Stream { child, frames: rx });

        assert_eq!(src.drain_stream().unwrap().taken_at_unix, 3);
        assert!(src.drain_stream().is_none(), "nothing new yet");
        drop(tx);
    }
}
