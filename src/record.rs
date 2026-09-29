//! Recording snapshots to disk and replaying them.
//!
//! A recording is JSON Lines: one serialised `Snapshot` per line. Because the
//! UI only ever reads a `Snapshot`, replaying one exercises every panel exactly
//! as the live source would — which is what makes "send me a recording of the
//! slowdown" a workable bug report.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::metrics::{MetricSource, Snapshot};

/// Appends snapshots to a JSONL file.
pub struct Recorder {
    path: PathBuf,
    writer: BufWriter<File>,
    frames: usize,
    top_n: usize,
    omit_paths: bool,
    max_cmd_len: usize,
}

impl Recorder {
    pub fn create(path: &Path) -> std::io::Result<Recorder> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
        Ok(Recorder {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            frames: 0,
            top_n: 100,
            omit_paths: false,
            max_cmd_len: 200,
        })
    }

    /// Bound each frame. `top_n == 0` keeps every process, which on a busy
    /// machine means roughly 1.5 MB per frame.
    pub fn with_limits(mut self, top_n: usize, omit_paths: bool, max_cmd_len: usize) -> Recorder {
        self.top_n = top_n;
        self.omit_paths = omit_paths;
        self.max_cmd_len = max_cmd_len;
        self
    }

    /// Write one frame. Flushed immediately so a recording survives a kill.
    pub fn write(&mut self, snap: &Snapshot) -> std::io::Result<()> {
        let trimmed = trim_frame(snap, self.top_n, self.omit_paths, self.max_cmd_len);
        self.write_trimmed(&trimmed)
    }

    /// Write a frame that has already been trimmed, for the flight recorder,
    /// which trims once on the way into its ring rather than once per frame
    /// per dump.
    pub fn write_trimmed(&mut self, snap: &Snapshot) -> std::io::Result<()> {
        let line = serde_json::to_string(snap)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.frames += 1;
        Ok(())
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// How a frame budget is spelled everywhere a recording is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub top_n: usize,
    pub omit_paths: bool,
    pub max_cmd_len: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { top_n: 100, omit_paths: false, max_cmd_len: 200 }
    }
}

/// A rolling window of recent frames, written out when something goes wrong.
///
/// `--record` only helps if you started it before the incident, which is the
/// one thing nobody does. The flight recorder keeps the last `pre` frames in
/// memory at all times and, when an alert fires, writes them plus the next
/// `post` frames to a file — so the recording of the slowdown exists because
/// the slowdown happened, not because someone predicted it.
///
/// Frames are trimmed on the way *into* the ring, not on the way out: a
/// ten-minute window of untrimmed frames is well over a gigabyte of live
/// memory in a process whose job is to make memory pressure visible.
pub struct FlightRecorder {
    ring: VecDeque<Snapshot>,
    pre: usize,
    post: usize,
    dir: PathBuf,
    limits: Limits,
    active: Option<Dump>,
}

struct Dump {
    recorder: Recorder,
    /// Frames still to be written after the trigger.
    remaining: usize,
}

impl FlightRecorder {
    /// `pre == 0` disables the recorder entirely, which is the default: a
    /// monitor that writes files to disk unasked is not what anyone installed.
    pub fn new(dir: &Path, pre: usize, post: usize, limits: Limits) -> Option<FlightRecorder> {
        (pre > 0).then(|| FlightRecorder {
            ring: VecDeque::with_capacity(pre.min(1024)),
            pre,
            post,
            dir: dir.to_path_buf(),
            limits,
            active: None,
        })
    }

    /// Offer a frame. Returns the path of a dump that this frame completed.
    pub fn push(&mut self, snap: &Snapshot) -> Option<PathBuf> {
        let trimmed =
            trim_frame(snap, self.limits.top_n, self.limits.omit_paths, self.limits.max_cmd_len);

        let mut finished = None;
        if let Some(dump) = &mut self.active {
            // Already trimmed, so this writes the frame as it stands rather
            // than trimming it a second time.
            let _ = dump.recorder.write_trimmed(&trimmed);
            dump.remaining = dump.remaining.saturating_sub(1);
            if dump.remaining == 0 {
                finished = self.active.take().map(|d| d.recorder.path().to_path_buf());
            }
        }

        while self.ring.len() >= self.pre {
            self.ring.pop_front();
        }
        self.ring.push_back(trimmed);
        finished
    }

    /// Start a dump: the whole ring, then the next `post` frames.
    ///
    /// A trigger while a dump is already running extends that one instead of
    /// opening a second file. A flapping rule would otherwise leave a file per
    /// refresh, each one mostly the same frames.
    pub fn trigger(&mut self, reason: &str, at_unix: u64) -> std::io::Result<Option<PathBuf>> {
        if let Some(dump) = &mut self.active {
            dump.remaining = dump.remaining.max(self.post);
            return Ok(None);
        }
        let path = self.dir.join(dump_name(reason, at_unix));
        let mut recorder = Recorder::create(&path)?.with_limits(
            self.limits.top_n,
            self.limits.omit_paths,
            self.limits.max_cmd_len,
        );
        for frame in &self.ring {
            recorder.write_trimmed(frame)?;
        }
        // `post == 0` still leaves a usable file: the context up to and
        // including the frame that tripped the rule.
        if self.post == 0 {
            return Ok(Some(path));
        }
        self.active = Some(Dump { recorder, remaining: self.post });
        Ok(Some(path))
    }

    pub fn is_dumping(&self) -> bool {
        self.active.is_some()
    }

    pub fn buffered(&self) -> usize {
        self.ring.len()
    }
}

/// `crabmon-<rule>-<unix>.jsonl`, with the rule name reduced to something a
/// filesystem will take — an alert may legitimately be called `disk / full`.
pub fn dump_name(reason: &str, at_unix: u64) -> String {
    let safe: String = reason
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let safe = safe.trim_matches('-');
    let safe = if safe.is_empty() { "alert" } else { safe };
    format!(
        "crabmon-{}-{at_unix}.jsonl",
        crate::format::truncate_fit(safe, 40).replace('\u{2026}', "")
    )
}

/// Replays a recording as a `MetricSource`.
pub struct ReplaySource {
    frames: Vec<Snapshot>,
    position: usize,
    label: String,
    /// When false, `snapshot` holds position instead of advancing.
    pub advancing: bool,
}

/// Reduce a snapshot to what is worth keeping in a recording: the busiest
/// processes, optionally without their paths.
pub fn trim_frame(snap: &Snapshot, top_n: usize, omit_paths: bool, max_cmd_len: usize) -> Snapshot {
    let mut out = snap.clone();
    if top_n > 0 && out.procs.len() > top_n {
        // Rank real processes ahead of threads. Ranking by CPU alone fills the
        // budget with a busy process's threads, and since the UI hides threads
        // by default the replay would then look almost empty.
        out.procs.sort_by(|a, b| {
            b.threads
                .is_some()
                .cmp(&a.threads.is_some())
                .then_with(|| b.cpu.partial_cmp(&a.cpu).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.mem.cmp(&a.mem))
        });
        out.procs.truncate(top_n);
    }
    for p in &mut out.procs {
        if omit_paths {
            p.cmd.clear();
            p.exe.clear();
            p.cwd.clear();
        } else if max_cmd_len > 0 {
            p.cmd = crate::format::truncate_fit(&p.cmd, max_cmd_len);
            p.cwd = crate::format::truncate_fit(&p.cwd, max_cmd_len);
        }
    }
    out
}

/// Parse JSONL, skipping blank lines. A truncated final line (the recorder was
/// killed mid-write) is dropped rather than failing the whole file.
pub fn parse_jsonl(content: &str) -> Result<Vec<Snapshot>, String> {
    let mut frames = Vec::new();
    let total = content.lines().count();
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Snapshot>(line) {
            Ok(snap) => frames.push(snap),
            Err(e) => {
                if i + 1 == total {
                    break; // truncated tail
                }
                return Err(format!("line {}: {e}", i + 1));
            }
        }
    }
    if frames.is_empty() {
        return Err("no frames in recording".into());
    }
    Ok(frames)
}

impl ReplaySource {
    pub fn open(path: &Path) -> Result<ReplaySource, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut content = String::new();
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(|e| e.to_string())? > 0 {
            content.push_str(&line);
            line.clear();
        }
        let frames = parse_jsonl(&content)?;
        Ok(ReplaySource {
            frames,
            position: 0,
            label: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            advancing: true,
        })
    }

    pub fn from_frames(frames: Vec<Snapshot>, label: impl Into<String>) -> ReplaySource {
        ReplaySource { frames, position: 0, label: label.into(), advancing: true }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Move by `delta` frames, clamped to the recording.
    pub fn step(&mut self, delta: isize) {
        let last = self.frames.len().saturating_sub(1);
        let next = (self.position as isize + delta).clamp(0, last as isize);
        self.position = next as usize;
    }

    /// Wall-clock span of the recording, from the frames' own timestamps.
    pub fn duration_secs(&self) -> u64 {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) => b.taken_at_unix.saturating_sub(a.taken_at_unix),
            _ => 0,
        }
    }

    fn current(&self) -> Snapshot {
        // `seek`/`step` already use saturating arithmetic; this did not, so an
        // empty frame list underflowed to usize::MAX and indexed out of bounds.
        // `from_frames` is public and accepts any Vec.
        match self.frames.get(self.position.min(self.frames.len().saturating_sub(1))) {
            Some(f) => f.clone(),
            None => Snapshot::default(),
        }
    }
}

impl MetricSource for ReplaySource {
    fn snapshot(&mut self, _dt: Duration) -> Snapshot {
        let snap = self.current();
        // Advance *after* reading, so the first call returns frame 0 and the
        // final frame is held rather than wrapping around.
        if self.advancing && self.position + 1 < self.frames.len() {
            self.position += 1;
        }
        snap
    }

    fn timeline(&self) -> Option<(usize, usize)> {
        Some((self.position, self.frames.len()))
    }

    fn seek(&mut self, position: usize) {
        self.position = position.min(self.frames.len().saturating_sub(1));
    }

    fn peek(&self) -> Option<Snapshot> {
        Some(self.current())
    }

    fn label(&self) -> Option<String> {
        Some(format!("replay {}", self.label))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, MemSample, ProcRow};

    fn frame(n: u32) -> Snapshot {
        Snapshot {
            cpu: CpuSample { per_core: vec![n as f32], freq_mhz: vec![1000] },
            mem: MemSample { total: 1024, used: n as u64, ..Default::default() },
            procs: vec![ProcRow { pid: n, name: format!("p{n}"), ..Default::default() }],
            taken_at_unix: 1_700_000_000 + n as u64,
            ..Default::default()
        }
    }

    fn tmpfile(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("crabmon-rec-{tag}-{}.jsonl", std::process::id()))
    }

    fn flight_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("crabmon-flight-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn dumps_in(dir: &Path) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> =
            std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        out.sort();
        out
    }

    #[test]
    fn a_dump_contains_the_frames_from_before_the_alert() {
        // The whole point: `--record` only helps if you started it before the
        // incident, and nobody does.
        let dir = flight_dir("before");
        let mut fr = FlightRecorder::new(&dir, 4, 2, Limits::default()).unwrap();
        for n in 0..10 {
            assert_eq!(fr.push(&frame(n)), None, "nothing finished yet");
        }
        assert_eq!(fr.buffered(), 4, "the ring holds the last four");

        let path = fr.trigger("cpu-saturated", 1_700_000_042).unwrap().unwrap();
        let written = parse_jsonl(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written.len(), 4, "the ring is written out immediately");
        let pids: Vec<u32> = written.iter().map(|f| f.procs[0].pid).collect();
        assert_eq!(pids, vec![6, 7, 8, 9], "the four frames leading up to the trigger");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_dump_runs_on_past_the_alert_and_then_closes() {
        let dir = flight_dir("after");
        let mut fr = FlightRecorder::new(&dir, 2, 3, Limits::default()).unwrap();
        fr.push(&frame(1));
        fr.push(&frame(2));
        let path = fr.trigger("disk", 1_700_000_000).unwrap().unwrap();
        assert!(fr.is_dumping());

        assert_eq!(fr.push(&frame(3)), None);
        assert_eq!(fr.push(&frame(4)), None);
        // The frame that completes the budget is the one that reports the path.
        assert_eq!(fr.push(&frame(5)), Some(path.clone()));
        assert!(!fr.is_dumping());

        // ...and later frames do not reopen it.
        assert_eq!(fr.push(&frame(6)), None);

        let written = parse_jsonl(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let pids: Vec<u32> = written.iter().map(|f| f.procs[0].pid).collect();
        assert_eq!(pids, vec![1, 2, 3, 4, 5], "two of context, then three of aftermath");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_flapping_rule_extends_one_dump_rather_than_opening_a_file_per_frame() {
        let dir = flight_dir("flap");
        let mut fr = FlightRecorder::new(&dir, 2, 2, Limits::default()).unwrap();
        fr.push(&frame(1));
        assert!(fr.trigger("flappy", 1).unwrap().is_some(), "the first one opens a file");
        for n in 2..8 {
            fr.push(&frame(n));
            // A second trigger extends the window and announces nothing.
            assert_eq!(fr.trigger("flappy", n as u64).unwrap(), None);
        }
        assert_eq!(dumps_in(&dir).len(), 1, "one incident, one file");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_dump_with_no_aftermath_budget_still_leaves_a_usable_file() {
        let dir = flight_dir("nopost");
        let mut fr = FlightRecorder::new(&dir, 3, 0, Limits::default()).unwrap();
        for n in 0..5 {
            fr.push(&frame(n));
        }
        let path = fr.trigger("now", 7).unwrap().unwrap();
        assert!(!fr.is_dumping(), "nothing left to wait for");
        assert_eq!(parse_jsonl(&std::fs::read_to_string(&path).unwrap()).unwrap().len(), 3);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_ring_holds_its_frames_trimmed_so_ten_minutes_is_not_a_gigabyte() {
        // Untrimmed, a busy machine's frame is ~1.5 MB; a window of them is
        // well over a gigabyte of live memory in the process whose job is to
        // make memory pressure visible.
        let dir = flight_dir("trim");
        let mut fr =
            FlightRecorder::new(&dir, 2, 0, Limits { top_n: 3, omit_paths: false, max_cmd_len: 8 })
                .unwrap();
        let mut big = frame(1);
        big.procs = (0..50)
            .map(|i| ProcRow {
                pid: i,
                cpu: i as f32,
                threads: Some(1),
                cmd: "/usr/bin/something --with --many --flags".into(),
                ..Default::default()
            })
            .collect();
        fr.push(&big);
        let path = fr.trigger("big", 1).unwrap().unwrap();
        let written = parse_jsonl(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written[0].procs.len(), 3, "trimmed on the way in, not on the way out");
        assert!(written[0].procs[0].cmd.chars().count() <= 8);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zero_frame_window_switches_the_recorder_off_entirely() {
        // The default: a monitor that writes files to disk unasked is not what
        // anyone installed.
        assert!(FlightRecorder::new(Path::new("."), 0, 30, Limits::default()).is_none());
    }

    #[test]
    fn a_rule_name_that_is_not_a_filename_still_produces_one() {
        // Alert names are free text: `disk / full` and `cpu>90%` are both
        // legal, and neither is a path component.
        assert_eq!(dump_name("cpu-saturated", 1700), "crabmon-cpu-saturated-1700.jsonl");
        assert_eq!(dump_name("disk / full", 1700), "crabmon-disk---full-1700.jsonl");
        assert!(!dump_name("../../etc/passwd", 1).contains('/'));
        assert!(!dump_name("", 1).is_empty());
        assert!(dump_name("", 1).starts_with("crabmon-alert-"));
        assert!(dump_name(&"x".repeat(200), 1).len() < 80, "a name is not a filesystem limit");
    }

    #[test]
    fn frames_are_trimmed_to_the_busiest_processes() {
        let mut snap = frame(0);
        snap.procs = (0..500)
            .map(|i| ProcRow {
                pid: i,
                cpu: i as f32,
                cmd: "/usr/bin/something --with --flags".into(),
                exe: "/usr/bin/something".into(),
                ..Default::default()
            })
            .collect();

        let trimmed = trim_frame(&snap, 10, false, 0);
        assert_eq!(trimmed.procs.len(), 10);
        assert_eq!(trimmed.procs[0].pid, 499, "the busiest process is kept");
        assert!(!trimmed.procs[0].cmd.is_empty());

        let stripped = trim_frame(&snap, 10, true, 0);
        assert!(stripped.procs.iter().all(|p| p.cmd.is_empty() && p.exe.is_empty()));

        // A full-size frame is the thing that made recordings unusable.
        let full = serde_json::to_string(&snap).unwrap().len();
        let small = serde_json::to_string(&trim_frame(&snap, 10, true, 0)).unwrap().len();
        assert!(small * 20 < full, "trimming must actually shrink the frame");
    }

    #[test]
    fn trimming_keeps_processes_in_preference_to_threads() {
        // Otherwise a busy Electron app's threads fill the whole budget and the
        // replay shows an almost empty process list.
        let mut snap = frame(0);
        snap.procs = (0..50)
            .map(|i| ProcRow {
                pid: i,
                cpu: 100.0, // threads are the busiest rows
                threads: None,
                ..Default::default()
            })
            .chain((100..110).map(|i| ProcRow {
                pid: i,
                cpu: 1.0,
                threads: Some(4),
                ..Default::default()
            }))
            .collect();

        let trimmed = trim_frame(&snap, 10, false, 0);
        assert!(
            trimmed.procs.iter().all(|p| p.threads.is_some()),
            "real processes must win the budget"
        );
    }

    #[test]
    fn oversized_command_lines_are_truncated_not_dropped() {
        // A Chromium renderer's argv runs to several kilobytes; a hundred of
        // them was 99% of every recorded frame.
        let snap = Snapshot {
            procs: vec![ProcRow { pid: 1, cmd: "--flag=value ".repeat(500), ..Default::default() }],
            ..Default::default()
        };
        let trimmed = trim_frame(&snap, 0, false, 200);
        assert_eq!(trimmed.procs[0].cmd.chars().count(), 200);
        assert!(trimmed.procs[0].cmd.ends_with('…'), "truncation should be visible");

        // Still readable enough to identify the process.
        assert!(trimmed.procs[0].cmd.starts_with("--flag=value"));
        // And 0 means keep it whole.
        assert_eq!(trim_frame(&snap, 0, false, 0).procs[0].cmd.len(), snap.procs[0].cmd.len());
    }

    #[test]
    fn empty_fields_are_omitted_from_the_wire_format() {
        let snap = Snapshot {
            procs: vec![ProcRow { pid: 1, name: "x".into(), ..Default::default() }],
            ..Default::default()
        };
        let json = serde_json::to_string(&snap).unwrap();
        assert!(!json.contains("\"cmd\""), "empty strings must not be written: {json}");
        assert!(!json.contains("\"service\""), "None must not be written: {json}");
        // ...and they still come back as their defaults.
        let back: Snapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.procs[0].cmd, "");
        assert_eq!(back.procs[0].service, None);
    }

    #[test]
    fn a_recording_round_trips_through_the_file() {
        let path = tmpfile("roundtrip");
        let mut rec = Recorder::create(&path).unwrap();
        for n in 0..5 {
            rec.write(&frame(n)).unwrap();
        }
        assert_eq!(rec.frames(), 5);

        let replay = ReplaySource::open(&path).unwrap();
        assert_eq!(replay.len(), 5);
        assert_eq!(replay.duration_secs(), 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_field_survives_the_round_trip() {
        // Replay is only useful if the recording carries the whole snapshot.
        let path = tmpfile("fields");
        let mut original = frame(7);
        original.host.hostname = "testbox".into();
        original.psi.io = Some(crate::metrics::psi::Pressure {
            some: crate::metrics::psi::PressureLine { avg10: 2.55, ..Default::default() },
            full: None,
        });
        original.power.batteries.push(crate::metrics::power::Battery {
            name: "BAT0".into(),
            percent: 98.0,
            status: "Discharging".into(),
            power_w: Some(-12.0),
            ..Default::default()
        });
        original.procs[0].service = Some("docker.service".into());
        original.procs[0].nice = Some(5);

        let mut rec = Recorder::create(&path).unwrap();
        rec.write(&original).unwrap();
        let replay = ReplaySource::open(&path).unwrap();
        let back = &replay.frames[0];

        assert_eq!(back.host.hostname, "testbox");
        assert_eq!(back.psi.io.unwrap().some.avg10, 2.55);
        assert_eq!(back.power.batteries[0].power_w, Some(-12.0));
        assert_eq!(back.procs[0].service.as_deref(), Some("docker.service"));
        assert_eq!(back.procs[0].nice, Some(5));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_advances_one_frame_per_snapshot_then_holds_the_last() {
        let mut src = ReplaySource::from_frames((0..3).map(frame).collect(), "t");
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 0);
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 1);
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 2);
        // Holding beats wrapping: a recording that loops looks like live data.
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 2);
    }

    #[test]
    fn seeking_and_stepping_stay_inside_the_recording() {
        let mut src = ReplaySource::from_frames((0..5).map(frame).collect(), "t");
        src.seek(3);
        assert_eq!(src.position(), 3);
        src.seek(99);
        assert_eq!(src.position(), 4, "clamped to the last frame");
        src.step(-10);
        assert_eq!(src.position(), 0, "clamped to the first frame");
        src.step(2);
        assert_eq!(src.position(), 2);
    }

    #[test]
    fn a_paused_replay_holds_its_position() {
        let mut src = ReplaySource::from_frames((0..5).map(frame).collect(), "t");
        src.advancing = false;
        src.seek(2);
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 2);
        assert_eq!(src.snapshot(Duration::ZERO).procs[0].pid, 2);
        assert_eq!(src.position(), 2);
    }

    #[test]
    fn a_truncated_final_line_is_dropped_not_fatal() {
        // The recorder was killed mid-write; the frames before it are still good.
        let good = serde_json::to_string(&frame(1)).unwrap();
        let content = format!("{good}\n{good}\n{{\"cpu\":{{\"per_c");
        let frames = parse_jsonl(&content).unwrap();
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn corruption_in_the_middle_is_reported() {
        let good = serde_json::to_string(&frame(1)).unwrap();
        let err = parse_jsonl(&format!("{good}\nnot json\n{good}\n")).unwrap_err();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn an_empty_recording_is_an_error_rather_than_a_blank_ui() {
        assert!(parse_jsonl("").is_err());
        assert!(parse_jsonl("\n\n").is_err());
    }

    #[test]
    fn older_recordings_still_load_when_fields_are_added() {
        // `#[serde(default)]` on the snapshot types is what makes a recording
        // from an older crabmon readable by a newer one.
        let minimal = r#"{"taken_at_unix":123}"#;
        let frames = parse_jsonl(minimal).unwrap();
        assert_eq!(frames[0].taken_at_unix, 123);
        assert!(frames[0].procs.is_empty());

        // A recording made before disk saturation and descriptor counts
        // existed still replays; the new fields read as "not measured" rather
        // than as an idle disk and a process holding no files.
        let pre_0_7 = r#"{"taken_at_unix":1,"disks":[{"mount":"/","total":10,"used":5,"read_bps":2048.0}],"procs":[{"pid":1,"name":"init","threads":1}]}"#;
        let frames = parse_jsonl(pre_0_7).unwrap();
        let disk = &frames[0].disks[0];
        assert_eq!(disk.read_bps, 2048.0);
        assert_eq!(disk.util, None);
        assert_eq!(disk.await_ms, None);
        assert_eq!(frames[0].procs[0].fds, None);
        assert_eq!(frames[0].procs[0].fd_ratio(), None);
    }

    #[test]
    fn peeking_reads_the_current_frame_without_advancing() {
        let mut src = ReplaySource::from_frames((0..3).map(frame).collect(), "t");
        src.seek(1);
        assert_eq!(src.peek().unwrap().procs[0].pid, 1);
        assert_eq!(src.peek().unwrap().procs[0].pid, 1, "peeking must not move");
        assert_eq!(src.position(), 1);
    }

    #[test]
    fn a_replay_source_reports_its_timeline_and_a_live_one_does_not() {
        let src = ReplaySource::from_frames((0..4).map(frame).collect(), "run.jsonl");
        assert_eq!(src.timeline(), Some((0, 4)));
        assert_eq!(src.label().as_deref(), Some("replay run.jsonl"));
    }

    /// `trim_frame` is tested directly above, but the recorder is what carries
    /// the configured limits into it. A builder that set the wrong field would
    /// leave every one of those tests passing while `[record] top_n`,
    /// `omit_paths` and `max_cmd_len` did nothing at all.
    #[test]
    fn the_configured_limits_reach_the_frames_on_disk() {
        let path = tmpfile("limits");
        let mut busy = frame(0);
        busy.procs = (0..10)
            .map(|n| ProcRow {
                pid: n,
                name: format!("p{n}"),
                cpu: n as f32,
                threads: Some(1),
                cmd: "/usr/bin/something --with-a-very-long-argument-list".into(),
                exe: "/usr/bin/something".into(),
                cwd: "/home/vlad/workspace".into(),
                ..Default::default()
            })
            .collect();

        let mut rec = Recorder::create(&path).unwrap().with_limits(3, false, 12);
        assert_eq!(rec.path(), path);
        rec.write(&busy).unwrap();
        drop(rec);

        let replay = ReplaySource::open(&path).unwrap();
        let stored = replay.peek().unwrap();
        assert_eq!(stored.procs.len(), 3, "top_n did not reach the writer");
        assert_eq!(stored.procs[0].pid, 9, "the busiest process should be kept");
        assert!(stored.procs[0].cmd.chars().count() <= 12, "{:?}", stored.procs[0].cmd);
        assert!(!stored.procs[0].exe.is_empty(), "paths were not asked to be omitted");

        // ...and omitting paths wins over truncating them.
        let mut rec = Recorder::create(&path).unwrap().with_limits(0, true, 200);
        rec.write(&busy).unwrap();
        drop(rec);

        let stored = ReplaySource::open(&path).unwrap().peek().unwrap();
        assert_eq!(stored.procs.len(), 10, "top_n = 0 means keep everything");
        assert!(stored.procs.iter().all(|p| p.cmd.is_empty() && p.exe.is_empty()));
        let _ = std::fs::remove_file(&path);
    }

    /// `from_frames` is public and takes any Vec; `len() - 1` on an empty one
    /// underflowed to usize::MAX and indexed out of bounds on first use.
    #[test]
    fn an_empty_replay_source_does_not_panic_on_first_use() {
        let mut src = ReplaySource::from_frames(Vec::new(), "empty");
        assert!(src.is_empty());
        let snap = src.snapshot(Duration::from_millis(100));
        assert!(snap.procs.is_empty());
        assert!(src.peek().is_some());
        src.seek(5);
        let _ = src.snapshot(Duration::from_millis(100));
    }
}
