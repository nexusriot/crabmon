//! Kernel events: OOM kills, crashes and IO errors, from `/dev/kmsg`.
//!
//! "Which process got OOM-killed at three in the morning" is the question a
//! system monitor is asked after the fact, and it was the one thing crabmon
//! could not answer at all: the dashboard shows memory *now*, and the process
//! that was killed is by definition no longer in the table. The kernel says
//! exactly what happened and when, and nothing was reading it.
//!
//! Only the lines worth waking someone for are kept — the kernel log is
//! mostly device enumeration — and only their classification and text, so a
//! recording carries them without carrying a megabyte of ring buffer.
//!
//! `/dev/kmsg` is root-only on most distributions (`kernel.dmesg_restrict=1`),
//! so the panel has to degrade to saying why rather than to showing nothing,
//! which reads as "nothing has gone wrong".

use serde::{Deserialize, Serialize};

/// What a kernel line is about. Only the kinds worth interrupting someone
/// for; everything else is dropped at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    /// The OOM killer chose a victim.
    OomKill,
    /// A process died on a bad address.
    Segfault,
    /// A block device or filesystem reported a failure.
    IoError,
    /// The kernel itself hit something it did not expect.
    Oops,
}

impl EventKind {
    pub fn key_name(self) -> &'static str {
        match self {
            EventKind::OomKill => "oom",
            EventKind::Segfault => "segfault",
            EventKind::IoError => "io-error",
            EventKind::Oops => "oops",
        }
    }

    /// How the panel labels it, short enough for a narrow side column.
    pub fn label(self) -> &'static str {
        match self {
            EventKind::OomKill => "OOM",
            EventKind::Segfault => "SEGV",
            EventKind::IoError => "I/O",
            EventKind::Oops => "OOPS",
        }
    }
}

/// Every kind, so the docs can be checked against the code.
pub const ALL_EVENT_KINDS: [EventKind; 4] =
    [EventKind::OomKill, EventKind::Segfault, EventKind::IoError, EventKind::Oops];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelEvent {
    pub kind: EventKind,
    /// The kernel's sequence number, which is what makes "have I seen this
    /// one" answerable across reads.
    pub seq: u64,
    /// Microseconds since boot, as the kernel stamps it.
    pub monotonic_us: u64,
    /// The message, trimmed of its metadata.
    pub text: String,
    /// The process the line is about, where it names one.
    pub pid: Option<u32>,
    pub process: Option<String>,
}

impl KernelEvent {
    /// Seconds before `uptime_secs` that this happened, for "11 minutes ago".
    pub fn age_secs(&self, uptime_secs: u64) -> u64 {
        uptime_secs.saturating_sub(self.monotonic_us / 1_000_000)
    }
}

/// Events kept per sample, and in a recording.
///
/// A machine with a failing disk can emit thousands of IO errors a second.
/// Keeping all of them would put the ring buffer in every frame of a
/// recording; the question being asked is "is this happening", which the
/// newest handful answers.
pub const MAX_EVENTS: usize = 64;

/// Classify one kernel message, or `None` if it is not worth reporting.
///
/// Matching on text is unavoidable — the kernel has no structured form of
/// "this is an OOM kill" — so the patterns are the ones that have been
/// spelled the same way for a very long time, and anything unrecognised is
/// simply not an event rather than a guess.
pub fn classify(message: &str) -> Option<EventKind> {
    let m = message.to_ascii_lowercase();
    // "Out of memory: Killed process 4242 (firefox)" and the cgroup-aware
    // "oom-kill:constraint=...,task=firefox,pid=4242".
    if m.contains("out of memory: kill") || m.starts_with("oom-kill:") || m.contains("oom_reaper:")
    {
        return Some(EventKind::OomKill);
    }
    if m.contains("segfault at") || m.contains("general protection fault") {
        return Some(EventKind::Segfault);
    }
    if m.contains("kernel panic")
        || m.contains("oops:")
        || m.contains("bug: kernel")
        || m.contains("call trace:")
    {
        return Some(EventKind::Oops);
    }
    if m.contains("i/o error")
        || m.contains("ext4-fs error")
        || m.contains("xfs ") && m.contains("error")
        || m.contains("critical medium error")
        || m.contains("critical target error")
        || m.contains("ata bus error")
    {
        return Some(EventKind::IoError);
    }
    None
}

/// Pull the process name and pid out of a message that names one.
///
/// Three spellings cover everything the kernel emits here:
/// `Killed process 4242 (firefox)`, `task=firefox,pid=4242`, and
/// `firefox[4242]: segfault at …`.
pub fn subject(message: &str) -> (Option<u32>, Option<String>) {
    // `task=name,pid=1234`
    if let Some(rest) = message.split("task=").nth(1) {
        let name = rest.split(',').next().unwrap_or("").trim().to_string();
        let pid = message
            .split("pid=")
            .nth(1)
            .and_then(|r| r.trim_start().split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|d| d.parse().ok());
        if !name.is_empty() {
            return (pid, Some(name));
        }
    }
    // `Killed process 4242 (firefox)`
    if let Some(rest) = message.split("process ").nth(1) {
        let mut it = rest.splitn(2, ' ');
        let pid: Option<u32> = it.next().and_then(|d| d.trim().parse().ok());
        let name = it
            .next()
            .and_then(|r| r.trim().strip_prefix('('))
            .and_then(|r| r.split(')').next())
            .map(|n| n.to_string());
        if pid.is_some() {
            return (pid, name);
        }
    }
    // `firefox[4242]: segfault at 0`
    if let Some(open) = message.find('[') {
        let name = message[..open].trim();
        let digits: String =
            message[open + 1..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if !name.is_empty() && !digits.is_empty() && !name.contains(' ') {
            return (digits.parse().ok(), Some(name.to_string()));
        }
    }
    (None, None)
}

/// Parse one `/dev/kmsg` record: `prio,seq,timestamp_us,flags;message`.
///
/// Returns `None` for a record that is malformed or not worth reporting,
/// which is the overwhelming majority of a kernel log.
pub fn parse_record(record: &str) -> Option<KernelEvent> {
    let (header, rest) = record.split_once(';')?;
    let mut fields = header.split(',');
    let _priority: u32 = fields.next()?.trim().parse().ok()?;
    let seq: u64 = fields.next()?.trim().parse().ok()?;
    let monotonic_us: u64 = fields.next()?.trim().parse().ok()?;
    // Continuation lines (`\n key=value`) carry structured detail nothing
    // here reads.
    let text = rest.split('\n').next().unwrap_or(rest).trim().to_string();
    let kind = classify(&text)?;
    let (pid, process) = subject(&text);
    Some(KernelEvent { kind, seq, monotonic_us, text, pid, process })
}

/// Why the kernel log could not be read, phrased for someone looking at an
/// empty panel.
pub fn unreadable_reason(err: &std::io::Error) -> String {
    match err.kind() {
        std::io::ErrorKind::PermissionDenied => {
            // Short enough to fit a side panel at its narrowest.
            "needs root or dmesg_restrict=0".to_string()
        }
        std::io::ErrorKind::NotFound => "no /dev/kmsg on this kernel".to_string(),
        _ => err.to_string(),
    }
}

/// Follows `/dev/kmsg`, keeping the events worth reporting.
///
/// Opening the device starts at the oldest record the ring still holds, which
/// is what makes an OOM kill from twenty minutes ago visible the moment
/// crabmon starts — the case this exists for.
pub struct KmsgReader {
    #[cfg(target_os = "linux")]
    file: Option<std::fs::File>,
    /// Why it cannot be read, when it cannot.
    pub error: Option<String>,
    /// Newest-last, bounded by `MAX_EVENTS`.
    events: Vec<KernelEvent>,
    /// Highest sequence number already taken, so a record is never counted
    /// twice if the device hands it over again.
    last_seq: Option<u64>,
    /// Events seen since the last `take_new`, for the alert rules.
    new_since: Vec<KernelEvent>,
}

/// Records read in one drain. Opening the device replays the whole ring, and
/// a sample is not the place to spend an unbounded number of syscalls.
const MAX_RECORDS_PER_DRAIN: usize = 8192;

impl Default for KmsgReader {
    fn default() -> Self {
        KmsgReader::closed()
    }
}

impl KmsgReader {
    /// A reader that will never produce anything, for platforms without a
    /// kernel log and for tests.
    pub fn closed() -> KmsgReader {
        KmsgReader {
            #[cfg(target_os = "linux")]
            file: None,
            error: None,
            events: Vec::new(),
            last_seq: None,
            new_since: Vec::new(),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn open() -> KmsgReader {
        use std::os::unix::fs::OpenOptionsExt;

        let mut reader = KmsgReader::closed();
        match std::fs::OpenOptions::new()
            .read(true)
            // Non-blocking: a drained device must return rather than parking
            // the sampler until the next kernel message, which on a healthy
            // machine could be hours.
            .custom_flags(libc::O_NONBLOCK)
            .open("/dev/kmsg")
        {
            Ok(file) => reader.file = Some(file),
            Err(e) => reader.error = Some(unreadable_reason(&e)),
        }
        reader
    }

    #[cfg(not(target_os = "linux"))]
    pub fn open() -> KmsgReader {
        let mut reader = KmsgReader::closed();
        reader.error = Some("kernel events are Linux-only".into());
        reader
    }

    pub fn is_available(&self) -> bool {
        self.error.is_none()
    }

    /// Everything kept, newest last.
    pub fn events(&self) -> &[KernelEvent] {
        &self.events
    }

    /// Events seen since this was last called. The alert rules ask this
    /// question — "has anything been killed *since the last sample*" — which
    /// the accumulated list cannot answer.
    pub fn take_new(&mut self) -> Vec<KernelEvent> {
        std::mem::take(&mut self.new_since)
    }

    /// Accept a record as though it had been read from the device. The IO and
    /// the bookkeeping are separated so the latter can be driven from a test.
    pub fn accept(&mut self, record: &str) {
        let Some(event) = parse_record(record) else { return };
        if self.last_seq.is_some_and(|seen| event.seq <= seen) {
            return;
        }
        self.last_seq = Some(event.seq);
        self.new_since.push(event.clone());
        self.events.push(event);
        let overflow = self.events.len().saturating_sub(MAX_EVENTS);
        if overflow > 0 {
            self.events.drain(..overflow);
        }
        let overflow = self.new_since.len().saturating_sub(MAX_EVENTS);
        if overflow > 0 {
            self.new_since.drain(..overflow);
        }
    }

    /// Read whatever the kernel has said since the last call.
    #[cfg(target_os = "linux")]
    pub fn drain(&mut self) {
        use std::io::Read;

        let Some(mut file) = self.file.take() else { return };
        // One record per read, as the device defines it. The file is taken
        // out of `self` for the loop so the records can be handed straight
        // to `accept`, and put back unless the device itself failed.
        let mut buf = [0u8; 8192];
        let mut records: Vec<String> = Vec::new();
        let mut keep = true;
        for _ in 0..MAX_RECORDS_PER_DRAIN {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => records.push(String::from_utf8_lossy(&buf[..n]).into_owned()),
                // Caught up.
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                // The records we had not read yet were overwritten. That is
                // not a failure: the next read resumes at the oldest one the
                // ring still holds.
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => continue,
                Err(e) => {
                    self.error = Some(unreadable_reason(&e));
                    keep = false;
                    break;
                }
            }
        }
        if keep {
            self.file = Some(file);
        }
        for record in records {
            self.accept(&record);
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn drain(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real lines, as the kernels that emit them spell them.
    const OOM_CLASSIC: &str = "4,1234,1234567890,-;Out of memory: Killed process 4242 (firefox) \
         total-vm:8000000kB, anon-rss:4000000kB";
    const OOM_CGROUP: &str = "4,1235,1234567891,-;oom-kill:constraint=CONSTRAINT_MEMCG,\
         nodemask=(null),cpuset=/,mems_allowed=0,oom_memcg=/docker/abc,task=postgres,pid=991";
    const SEGV: &str =
        "6,1236,1234567892,-;firefox[4242]: segfault at 0 ip 00007f0 sp 00007ff error 4";
    const IO_ERR: &str = "3,1237,1234567893,-;blk_update_request: I/O error, dev sda, sector 123";
    const BORING: &str = "6,1238,1234567894,-;usb 1-4: new high-speed USB device number 7";

    #[test]
    fn the_lines_worth_waking_someone_for_are_recognised() {
        assert_eq!(parse_record(OOM_CLASSIC).unwrap().kind, EventKind::OomKill);
        assert_eq!(parse_record(OOM_CGROUP).unwrap().kind, EventKind::OomKill);
        assert_eq!(parse_record(SEGV).unwrap().kind, EventKind::Segfault);
        assert_eq!(parse_record(IO_ERR).unwrap().kind, EventKind::IoError);
    }

    /// A kernel log is mostly device enumeration. Reporting it as events
    /// would bury the one line that matters.
    #[test]
    fn ordinary_kernel_chatter_is_not_an_event() {
        assert!(parse_record(BORING).is_none());
        assert!(classify("EXT4-fs (sda1): mounted filesystem with ordered data mode").is_none());
        assert!(classify("systemd[1]: Started Daily apt download activities.").is_none());
    }

    /// The victim's name is the entire point of the line.
    #[test]
    fn an_oom_kill_names_its_victim_in_either_spelling() {
        let classic = parse_record(OOM_CLASSIC).unwrap();
        assert_eq!(classic.pid, Some(4242));
        assert_eq!(classic.process.as_deref(), Some("firefox"));

        let cgroup = parse_record(OOM_CGROUP).unwrap();
        assert_eq!(cgroup.pid, Some(991));
        assert_eq!(cgroup.process.as_deref(), Some("postgres"));

        let segv = parse_record(SEGV).unwrap();
        assert_eq!(segv.pid, Some(4242));
        assert_eq!(segv.process.as_deref(), Some("firefox"));
    }

    #[test]
    fn a_malformed_record_is_dropped_rather_than_guessed_at() {
        assert!(parse_record("").is_none());
        assert!(parse_record("no semicolon here").is_none());
        assert!(parse_record("x,y,z;Out of memory: Killed process 1 (init)").is_none());
    }

    #[test]
    fn a_record_is_never_counted_twice() {
        let mut r = KmsgReader::closed();
        r.accept(OOM_CLASSIC);
        r.accept(OOM_CLASSIC);
        assert_eq!(r.events().len(), 1);
        r.accept(OOM_CGROUP);
        assert_eq!(r.events().len(), 2);
    }

    /// A failing disk can emit thousands of these a second, and every one
    /// would otherwise land in every frame of a recording.
    #[test]
    fn the_kept_list_is_bounded() {
        let mut r = KmsgReader::closed();
        for seq in 0..(MAX_EVENTS * 3) {
            r.accept(&format!("3,{seq},{seq}000,-;blk_update_request: I/O error, dev sda"));
        }
        assert_eq!(r.events().len(), MAX_EVENTS);
        assert_eq!(r.take_new().len(), MAX_EVENTS);
        assert!(r.take_new().is_empty(), "the new-since list is drained by reading it");
    }

    #[test]
    fn new_events_are_only_new_once() {
        let mut r = KmsgReader::closed();
        r.accept(OOM_CLASSIC);
        assert_eq!(r.take_new().len(), 1);
        assert!(r.take_new().is_empty());
        r.accept(SEGV);
        assert_eq!(r.take_new().len(), 1);
    }

    #[test]
    fn an_age_is_measured_against_the_hosts_uptime() {
        let e = parse_record(OOM_CLASSIC).unwrap();
        // Stamped at 1234.56 s since boot, on a host up for 2000 s.
        assert_eq!(e.age_secs(2000), 2000 - 1234);
        // A stamp in the future — a recording replayed on another host —
        // saturates rather than wrapping.
        assert_eq!(e.age_secs(0), 0);
    }

    #[test]
    fn an_unreadable_device_explains_itself() {
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert!(unreadable_reason(&denied).contains("dmesg_restrict"));
        let missing = std::io::Error::from(std::io::ErrorKind::NotFound);
        assert!(unreadable_reason(&missing).contains("/dev/kmsg"));
    }

    /// Opening the real device must never panic, whatever it answers.
    #[test]
    fn opening_the_real_device_is_safe() {
        let mut r = KmsgReader::open();
        r.drain();
        // Either it worked, or it said why. Never both, never neither.
        assert_eq!(r.is_available(), r.error.is_none());
    }
}
