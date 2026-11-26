//! Network traffic per process, from the kernel's per-socket TCP counters.
//!
//! The kernel accounts bytes to interfaces and, with `[containers]`, to
//! network namespaces. It does not account them to processes: there is no
//! `/proc/<pid>/net/bytes`, and `/proc/<pid>/net/dev` is the *namespace's*
//! table, identical for every process in it. The only per-process figure the
//! kernel will give without eBPF is the one on each TCP socket
//! (`tcpi_bytes_sent` and `tcpi_bytes_received`), which `ss -tinp` prints
//! beside the socket's owner.
//!
//! This reads that. It has limits, and they are the reason it is opt-in
//! (`[procs] net`):
//!
//! - **TCP only.** UDP sockets carry no byte counters, so a DNS-heavy or QUIC
//!   process reads as quiet. QUIC (HTTP/3) traffic is invisible.
//! - **Live sockets only.** The counter dies with the socket, so bytes moved
//!   by a connection that opened and closed between two samples are lost. A
//!   process making many short requests is under-counted; one with a
//!   long-lived connection is exact.
//! - **Other users' processes need root.** `ss` shows who owns a socket only
//!   for sockets it may inspect.
//! - **A cost per sample**: one `ss` run, which is a netlink dump of every
//!   TCP socket on the machine.

use std::collections::HashMap;

/// One TCP socket's lifetime counters and its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketBytes {
    /// The socket's inode, which is what identifies it between samples.
    pub inode: u64,
    /// The owning process. Sockets inherited by a child have several; the
    /// first is credited, so a forked server's bytes are counted once.
    pub pid: Option<u32>,
    pub sent: u64,
    pub received: u64,
}

fn after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(key)? + key.len();
    Some(&line[at..])
}

fn leading_number(s: &str) -> Option<u64> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    s[..end].parse().ok()
}

/// Parse `ss -tinHep` output.
///
/// Each socket is two lines: the address line, carrying `users:(...)` and
/// `ino:`, then an indented line of `tcp_info` carrying `bytes_sent:` and
/// `bytes_received:`. A socket missing either counter is skipped — an older
/// kernel, or a listener, which has nothing to count.
pub fn parse_ss(output: &str) -> Vec<SocketBytes> {
    let mut out = Vec::new();
    let mut head: Option<(u64, Option<u32>)> = None;
    for line in output.lines() {
        let indented = line.starts_with(char::is_whitespace);
        if !indented {
            // A new socket. Anything left over from the last one is dropped:
            // it never got its counters.
            // Inode 0 is a socket with no file behind it (TIME-WAIT, mostly):
            // nothing to key it by between samples, and no owner either.
            head = after(line, "ino:").and_then(leading_number).filter(|i| *i > 0).map(|ino| {
                let pid = after(line, "pid=").and_then(leading_number).map(|p| p as u32);
                (ino, pid)
            });
            continue;
        }
        let Some((inode, pid)) = head.take() else { continue };
        // `bytes_sent:` must not match inside `data_segs_out`-style keys, so
        // anchor on the preceding space.
        let counter = |key: &str| {
            after(line, &format!(" {key}:")).or_else(|| after(line, &format!("\t{key}:")))
        };
        let (Some(sent), Some(received)) = (
            counter("bytes_sent").and_then(leading_number),
            counter("bytes_received").and_then(leading_number),
        ) else {
            continue;
        };
        out.push(SocketBytes { inode, pid, sent, received });
    }
    out
}

/// What one process moved in the last interval, in bytes per second.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rates {
    pub rx_bps: f64,
    pub tx_bps: f64,
}

impl Rates {
    pub fn total(self) -> f64 {
        self.rx_bps + self.tx_bps
    }
}

/// Turns cumulative per-socket counters into per-process rates.
///
/// Keyed by socket rather than by process: a process opening and closing
/// connections would otherwise see its total go *down* when a busy one
/// closed, and the difference would be read as a negative rate. By socket,
/// a closed one simply stops contributing.
#[derive(Debug, Default)]
pub struct NetTracker {
    prev: HashMap<u64, (u64, u64)>,
    primed: bool,
}

impl NetTracker {
    /// Feed one sample. The first call only learns the baseline and returns
    /// nothing, because "bytes since the socket opened" is not a rate.
    pub fn update(&mut self, sockets: &[SocketBytes], secs: f64) -> HashMap<u32, Rates> {
        let mut out: HashMap<u32, Rates> = HashMap::new();
        let secs = secs.max(0.001);
        let mut next = HashMap::with_capacity(sockets.len());
        for s in sockets {
            next.insert(s.inode, (s.sent, s.received));
            if !self.primed {
                continue;
            }
            let Some(pid) = s.pid else { continue };
            // A socket first seen this interval moved everything it has
            // moved since it opened, which is within the interval: the whole
            // counter is the delta. Where it is known, the difference.
            let (ps, pr) = self.prev.get(&s.inode).copied().unwrap_or((0, 0));
            let (ds, dr) = (s.sent.saturating_sub(ps), s.received.saturating_sub(pr));
            let e = out.entry(pid).or_default();
            e.tx_bps += ds as f64 / secs;
            e.rx_bps += dr as f64 / secs;
        }
        self.prev = next;
        self.primed = true;
        out
    }
}

/// Whether `ss` is on this machine, so the UI can say "not available here"
/// rather than draw a column of dashes.
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("ss")
            .arg("-V")
            .output()
            .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Run `ss` and parse it. `None` where it cannot be run.
pub fn read() -> Option<Vec<SocketBytes>> {
    #[cfg(target_os = "linux")]
    {
        let out = std::process::Command::new("ss")
            .args(["-t", "-i", "-n", "-H", "-e", "-p"])
            .output()
            .ok()?;
        out.status.success().then(|| parse_ss(&String::from_utf8_lossy(&out.stdout)))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `ss -tinHep` output: two lines a socket, the second indented.
    const SS: &str = "\
ESTAB 0 0 10.0.0.5:49218 146.71.85.157:18661 users:((\"i2pd\",pid=812,fd=34)) uid:127 ino:1711162 sk:55 cgroup:/system.slice/i2pd.service <->
\t cubic wscale:14,10 rto:422 rtt:221.14/23.56 mss:1380 cwnd:10 bytes_sent:15393 bytes_acked:15394 bytes_received:1234 segs_out:43 segs_in:42 data_segs_out:39
ESTAB 0 0 127.0.0.1:18840 127.0.0.1:44946 users:((\"kompressor\",pid=900,fd=7),(\"kompressor\",pid=901,fd=7)) uid:126 ino:1716156 sk:56 <->
\t ts sack cubic bytes_sent:20500 bytes_acked:20500 bytes_received:1828668 segs_out:37
ESTAB 0 0 10.0.0.5:22 10.0.0.9:51000 uid:0 ino:42 sk:9 <->
\t cubic bytes_sent:100 bytes_received:200 segs_out:1
";

    #[test]
    fn ss_output_yields_a_socket_with_its_owner_and_counters() {
        let s = parse_ss(SS);
        assert_eq!(s.len(), 3);
        assert_eq!(
            s[0],
            SocketBytes { inode: 1_711_162, pid: Some(812), sent: 15_393, received: 1_234 }
        );
        // Shared by two processes: the first is credited, once.
        assert_eq!(s[1].pid, Some(900));
        assert_eq!(s[1].received, 1_828_668);
        // No `users:` — somebody else's socket, and not ours to name.
        assert_eq!(s[2].pid, None);
    }

    #[test]
    fn a_socket_with_no_inode_cannot_be_tracked_and_is_skipped() {
        let text = "TIME-WAIT 0 0 a b ino:0\n\t cubic bytes_sent:1 bytes_received:2\n";
        assert!(parse_ss(text).is_empty());
    }

    #[test]
    fn a_socket_without_counters_is_skipped_not_zeroed() {
        let listener = "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:* users:((\"sshd\",pid=1,fd=3)) ino:5\n\
                        \t cubic rto:200 mss:536\n";
        assert!(parse_ss(listener).is_empty());
        assert!(parse_ss("").is_empty());
        assert!(parse_ss("\t orphan bytes_sent:1 bytes_received:2\n").is_empty());
    }

    /// `bytes_sent` must not be satisfied by a longer key ending the same way.
    #[test]
    fn a_counter_is_matched_by_its_whole_name() {
        let text = "ESTAB 0 0 a b users:((\"x\",pid=2,fd=3)) ino:7\n\
                    \t data_bytes_sent:999 bytes_sent:10 bytes_received:20\n";
        let s = parse_ss(text);
        assert_eq!(s[0].sent, 10);
    }

    fn sock(inode: u64, pid: u32, sent: u64, received: u64) -> SocketBytes {
        SocketBytes { inode, pid: Some(pid), sent, received }
    }

    #[test]
    fn the_first_sample_is_a_baseline_not_a_rate() {
        let mut t = NetTracker::default();
        assert!(t.update(&[sock(1, 10, 1_000_000, 5_000_000)], 1.0).is_empty());
    }

    #[test]
    fn rates_are_the_growth_of_each_socket_over_the_interval() {
        let mut t = NetTracker::default();
        t.update(&[sock(1, 10, 1000, 2000), sock(2, 10, 0, 0), sock(3, 20, 500, 500)], 1.0);
        let r =
            t.update(&[sock(1, 10, 3000, 2500), sock(2, 10, 100, 0), sock(3, 20, 500, 500)], 2.0);
        // pid 10 owns two sockets: sent (2000 + 100) / 2s, received 500 / 2s.
        assert_eq!(r[&10], Rates { rx_bps: 250.0, tx_bps: 1050.0 });
        assert_eq!(r[&10].total(), 1300.0);
        // An idle socket still reports a zero rate rather than going missing.
        assert_eq!(r[&20], Rates::default());
    }

    /// A busy socket closing must not read as negative traffic.
    #[test]
    fn a_closed_socket_stops_contributing_instead_of_going_negative() {
        let mut t = NetTracker::default();
        t.update(&[sock(1, 10, 9_000_000, 9_000_000), sock(2, 10, 0, 0)], 1.0);
        let r = t.update(&[sock(2, 10, 10, 20)], 1.0);
        assert_eq!(r[&10], Rates { rx_bps: 20.0, tx_bps: 10.0 });
    }

    /// A connection opened and used within the interval is all new traffic.
    #[test]
    fn a_new_socket_counts_everything_it_has_moved() {
        let mut t = NetTracker::default();
        t.update(&[], 1.0);
        let r = t.update(&[sock(9, 5, 4_000, 0)], 2.0);
        assert_eq!(r[&5].tx_bps, 2_000.0);
    }

    /// A socket whose owner `ss` could not name belongs to nobody we show.
    #[test]
    fn unowned_sockets_are_not_credited_to_anyone() {
        let mut t = NetTracker::default();
        let anon = SocketBytes { inode: 1, pid: None, sent: 10, received: 10 };
        t.update(std::slice::from_ref(&anon), 1.0);
        let r = t.update(&[SocketBytes { sent: 5000, ..anon }], 1.0);
        assert!(r.is_empty());
    }

    /// Reading the real machine must not panic whatever `ss` says or whether
    /// it exists.
    #[test]
    fn reading_this_machine_is_safe() {
        let _ = is_available();
        if let Some(socks) = read() {
            // Every socket kept has an inode to be tracked by.
            assert!(socks.iter().all(|s| s.inode > 0));
        }
    }
}
