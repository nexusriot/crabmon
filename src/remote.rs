//! Monitoring another machine over SSH.
//!
//! `crabmon --remote host` runs `crabmon --once` on the far end and parses its
//! JSON, so the whole UI works against a remote host with no daemon, no open
//! port and no protocol of its own. The refresh interval is bounded by the SSH
//! round trip, not by the sampler.

use std::process::Command;
use std::time::Duration;

use crate::metrics::{MetricSource, Snapshot};

pub struct RemoteSource {
    target: String,
    /// The command run on the far end; overridable for a non-standard path.
    remote_command: String,
    ssh_options: Vec<String>,
    last_good: Option<Snapshot>,
    pub last_error: Option<String>,
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
            remote_command: remote_command
                .unwrap_or_else(|| "crabmon --once --refresh 200".to_string()),
            ssh_options,
            last_good: None,
            last_error: None,
        }
    }

    /// One fetch. Separated from `snapshot` so the error path is testable.
    pub fn fetch(&mut self) -> Result<Snapshot, String> {
        let args = ssh_args(&self.target, &self.ssh_options, &self.remote_command);
        let out = Command::new("ssh").args(&args).output().map_err(|e| format!("ssh: {e}"))?;
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
        Some(match &self.last_error {
            Some(e) => format!("remote {} UNREACHABLE — {e}", self.target),
            None => format!("remote {}", self.target),
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
}
