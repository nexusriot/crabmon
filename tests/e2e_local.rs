//! Runs the real binary on this machine. Child workloads, configs and hook
//! markers are private to each test; SSH is a controlled local transport.
#![cfg(target_os = "linux")]

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{FromRawFd, RawFd};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_crabmon");

struct Workload(Child);
impl Drop for Workload {
    fn drop(&mut self) {
        // Every workload owns its process group, including transport helpers.
        unsafe {
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("crabmon-e2e-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn file(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn spawn(command: &mut Command) -> Workload {
    // SAFETY: only an async-signal-safe syscall between fork and exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Workload(command.stdin(Stdio::null()).stderr(Stdio::null()).spawn().unwrap())
}

#[test]
fn stream_transports_every_process_and_the_full_command_despite_recording_limits() {
    let dir = Scratch::new("stream");
    let cfg = dir.file("config.toml");
    fs::write(&cfg, "[record]\ntop_n = 1\nomit_paths = true\nmax_cmd_len = 8\n").unwrap();
    let mut workloads = Vec::new();
    for _ in 0..105 {
        workloads.push(spawn(Command::new("sleep").arg("30")));
    }
    let long_name = format!("crabmon-long-{}", "x".repeat(600));
    workloads.push(spawn(Command::new("bash").args([
        "-c",
        "exec -a \"$1\" sleep 30",
        "test",
        &long_name,
    ])));
    let mut stream = spawn(
        Command::new(BIN)
            .args(["--stream", "--refresh", "200", "--config"])
            .arg(&cfg)
            .stdout(Stdio::piped()),
    );
    let reader = stream.0.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    let line = rx.recv_timeout(Duration::from_secs(15)).expect("stream's first frame");
    let frame: crabmon::Snapshot = serde_json::from_str(&line).unwrap();
    for child in &workloads {
        assert!(
            frame.procs.iter().any(|p| p.pid == child.0.id()),
            "transport dropped PID {}",
            child.0.id()
        );
    }
    let row = frame.procs.iter().find(|p| p.pid == workloads.last().unwrap().0.id()).unwrap();
    assert!(row.cmd.contains(&long_name), "transport truncated command: {}", row.cmd);
    assert!(!row.exe.is_empty(), "recording omit_paths affected transport");
}

fn scrape(addr: &str) -> Option<String> {
    let mut stream =
        TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(200)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    stream.write_all(b"GET /metrics HTTP/1.0\r\n\r\n").ok()?;
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    (!response.is_empty()).then_some(response)
}
fn wait_until(mut test: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !test() {
        assert!(Instant::now() < deadline, "timed out waiting for binary");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn fleet_exporter_does_not_page_on_connecting_stale_or_failed_hosts() {
    let dir = Scratch::new("fleet");
    let marker = dir.file("paged");
    let release = dir.file("release");
    let frame = dir.file("frame");
    fs::write(&frame, "{\"taken_at_unix\":1000,\"host\":{\"hostname\":\"fixture\"},\"cpu\":{\"per_core\":[99]}}\n").unwrap();
    let ssh = dir.file("ssh");
    // Every request is a real SSH source, reader thread and worker. A failed
    // host never produces a frame; the slow host releases exactly one at a time.
    fs::write(
        &ssh,
        format!(
            r#"#!/bin/sh
if [ "$5" = broken ]; then echo unreachable >&2; exit 1; fi
while [ ! -f '{release}' ]; do sleep 0.05; done
cat '{frame}'
while [ ! -f '{next}' ]; do sleep 0.05; done
cat '{frame}'
sleep 30
"#,
            release = release.display(),
            frame = frame.display(),
            next = dir.file("next").display()
        ),
    )
    .unwrap();
    fs::set_permissions(&ssh, fs::Permissions::from_mode(0o755)).unwrap();
    let cfg = dir.file("config.toml");
    fs::write(
        &cfg,
        format!(
            r#"refresh_ms = 200
[[alert]]
name = "absent"
kind = "proc"
query = "nginx"
threshold = 1
below = true
for_secs = 1
command = "touch {}"
"#,
            marker.display()
        ),
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    drop(listener);
    let path = format!("{}:{}", dir.0.display(), std::env::var("PATH").unwrap_or_default());
    let _exporter = spawn(
        Command::new(BIN)
            .args(["--serve", &addr, "--remote", "slow,broken", "--alerts", "--config"])
            .arg(&cfg)
            .env("PATH", path)
            .env_remove("CRABMON_SERVE_TOKEN")
            .current_dir(&dir.0)
            .stdout(Stdio::null()),
    );
    wait_until(|| scrape(&addr).is_some());
    for _ in 0..12 {
        let text = scrape(&addr).unwrap();
        assert!(text.contains("crabmon_up{host=\"slow\"} 0"), "{text}");
        assert!(!marker.exists(), "blank startup sample triggered absence hook");
        std::thread::sleep(Duration::from_millis(100));
    }
    fs::write(&release, "go").unwrap();
    wait_until(|| scrape(&addr).unwrap().contains("crabmon_up{host=\"slow\"} 1"));
    // Repeated scrapes must not turn the one delivered frame into a hold.
    let until = Instant::now() + Duration::from_millis(1400);
    while Instant::now() < until {
        let _ = scrape(&addr).unwrap();
        assert!(!marker.exists(), "duplicates completed an absence hold");
        std::thread::sleep(Duration::from_millis(80));
    }
    // Once expired, both availability and the hold must reset. A new frame
    // after the gap starts a new hold instead of immediately running a hook.
    wait_until(|| scrape(&addr).unwrap().contains("crabmon_up{host=\"slow\"} 0"));
    fs::write(dir.file("next"), "go").unwrap();
    wait_until(|| scrape(&addr).unwrap().contains("crabmon_up{host=\"slow\"} 1"));
    assert!(!marker.exists(), "reconnection inherited the pre-outage hold");
}

#[test]
fn replay_in_a_real_terminal_never_executes_alert_hooks_or_writes_flight_dumps() {
    let dir = Scratch::new("replay");
    let marker = dir.file("hook-ran");
    let cfg = dir.file("config.toml");
    fs::write(
        &cfg,
        format!(
            r#"refresh_ms = 200
[record]
flight = 2
flight_after = 1
flight_dir = "{}"
[[alert]]
name = "recorded-cpu"
kind = "cpu"
threshold = 1
command = "touch {}"
"#,
            dir.0.display(),
            marker.display()
        ),
    )
    .unwrap();
    let recording = dir.file("incident.jsonl");
    let frame = "{\"taken_at_unix\":1000,\"cpu\":{\"per_core\":[99]}}\n";
    fs::write(&recording, frame.repeat(10)).unwrap();
    let (mut master, slave) = open_pty();
    let input = slave.try_clone().unwrap();
    let output = slave.try_clone().unwrap();
    let mut command = Command::new(BIN);
    command
        .args(["--replay"])
        .arg(&recording)
        .arg("--config")
        .arg(&cfg)
        .arg("--no-mouse")
        .env("TERM", "xterm-256color")
        .current_dir(&dir.0)
        .stdin(Stdio::from(input))
        .stdout(Stdio::from(output))
        .stderr(Stdio::from(slave));
    // SAFETY: async-signal-safe terminal/session setup on the child's stdin.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut app = Workload(command.spawn().unwrap());
    let mut reader = master.try_clone().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n) = reader.read(&mut buf) {
            if n == 0 {
                break;
            }
            let _ = tx.send(String::from_utf8_lossy(&buf[..n]).to_string());
        }
    });
    // Wait until the recorded alert is actually drawn, proving the TUI
    // sampled the replay and evaluated the configured rule.
    let mut screen = String::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !screen.contains("recorded-cpu") {
        assert!(Instant::now() < deadline, "replay alert never appeared: {screen}");
        screen.push_str(&rx.recv_timeout(Duration::from_secs(2)).expect("TUI output"));
    }
    master.write_all(b"q").unwrap();
    wait_until(|| app.0.try_wait().unwrap().is_some());
    assert!(app.0.wait().unwrap().success());
    assert!(!marker.exists(), "replay executed its hook");
    assert!(
        !fs::read_dir(&dir.0)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().starts_with("crabmon-recorded-cpu")),
        "replay wrote a flight dump"
    );
}

fn open_pty() -> (File, File) {
    let mut master: RawFd = -1;
    let mut slave: RawFd = -1;
    let size = libc::winsize { ws_row: 40, ws_col: 120, ws_xpixel: 0, ws_ypixel: 0 };
    // SAFETY: valid output fd pointers, default termios, fixed window size.
    let rc = unsafe {
        libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null(), &size)
    };
    assert_eq!(rc, 0, "openpty: {}", std::io::Error::last_os_error());
    // SAFETY: successful openpty transferred two fresh owned descriptors.
    unsafe { (File::from_raw_fd(master), File::from_raw_fd(slave)) }
}
