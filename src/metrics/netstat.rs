//! TCP health, from the kernel's protocol counters.
//!
//! Throughput says how much moved; it cannot say that a quarter of it was
//! sent twice, that an accept queue is full and the kernel is dropping new
//! connections on the floor, or that a service is leaking sockets in
//! CLOSE_WAIT. All three are counters the kernel already keeps:
//!
//! - `/proc/net/snmp`, the `Tcp:` MIB: segments out, retransmitted, resets,
//!   failed connection attempts and the number established.
//! - `/proc/net/netstat`, the `TcpExt:` block: `ListenOverflows` and
//!   `ListenDrops` — connections refused because a listener's accept queue
//!   was full, which is what "nginx is dropping connections" looks like from
//!   below, and which no throughput chart will ever show.
//! - `/proc/net/sockstat`: sockets in TIME_WAIT and orphaned, for free.
//!
//! CLOSE_WAIT is not counted anywhere, so it is the one figure that reads the
//! socket tables themselves, and only while there are few enough sockets for
//! that to be cheap.
//!
//! These files describe the network namespace crabmon runs in: the host's
//! when it runs on the host, a container's when it runs inside one.

use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

/// Above this many sockets the CLOSE_WAIT count is skipped rather than
/// reading megabytes of socket table every sample.
pub const MAX_SOCKETS_SCANNED: u64 = 50_000;

/// The cumulative counters the rates are computed from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TcpCounters {
    pub out_segs: u64,
    pub retrans_segs: u64,
    pub in_errs: u64,
    pub estab_resets: u64,
    pub attempt_fails: u64,
    pub curr_estab: u64,
    /// `TcpExt`, which some kernels or namespaces do not expose.
    pub listen_drops: Option<u64>,
    pub listen_overflows: Option<u64>,
}

/// TCP over the last interval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TcpHealth {
    /// Connections currently ESTABLISHED or CLOSE_WAIT, as the MIB counts it.
    pub established: u64,
    pub time_wait: Option<u64>,
    pub orphans: Option<u64>,
    /// Sockets the peer has closed and this side has not: a steadily rising
    /// figure is a service that forgets to close what it accepted. `None`
    /// when there were too many sockets to count cheaply.
    pub close_wait: Option<u64>,
    /// Retransmitted segments as a percentage of segments sent. `None` when
    /// nothing was sent, which is not the same as nothing lost.
    pub retrans_percent: Option<f64>,
    /// Established connections reset, per second.
    pub resets_ps: f64,
    /// Connection attempts that failed, per second.
    pub attempt_fails_ps: f64,
    /// Incoming connections dropped at a listener, per second — a full accept
    /// queue (`ListenOverflows`) among other causes.
    pub listen_drops_ps: Option<f64>,
    /// Segments received in error, per second.
    pub in_errs_ps: f64,
}

/// The names a `tcp` rule's target may use. An empty target is `retrans`.
pub const TCP_SERIES: [&str; 7] =
    ["retrans", "listendrops", "resets", "failed", "closewait", "timewait", "estab"];

impl TcpHealth {
    /// The figure a `tcp` rule's target names, with its unit.
    pub fn get(&self, series: &str) -> Option<(f64, &'static str)> {
        match series.trim().to_ascii_lowercase().as_str() {
            "" | "retrans" => self.retrans_percent.map(|v| (v, "%")),
            "listendrops" => self.listen_drops_ps.map(|v| (v, "/s")),
            "resets" => Some((self.resets_ps, "/s")),
            "failed" => Some((self.attempt_fails_ps, "/s")),
            "closewait" => self.close_wait.map(|v| (v as f64, "")),
            "timewait" => self.time_wait.map(|v| (v as f64, "")),
            "estab" => Some((self.established as f64, "")),
            _ => None,
        }
    }

    pub fn between(
        prev: &TcpCounters,
        cur: &TcpCounters,
        secs: f64,
        sockstat: Sockstat,
        close_wait: Option<u64>,
    ) -> Option<TcpHealth> {
        if secs <= 0.0 {
            return None;
        }
        let r = |a: u64, b: u64| super::rate(a, b, secs);
        let sent = cur.out_segs.checked_sub(prev.out_segs);
        let resent = cur.retrans_segs.checked_sub(prev.retrans_segs);
        let retrans_percent = match (sent, resent) {
            (Some(s), Some(r)) if s > 0 => Some((r as f64 * 100.0 / s as f64).min(100.0)),
            _ => None,
        };
        let listen_drops_ps = match (prev.listen_drops, cur.listen_drops) {
            (Some(a), Some(b)) => Some(r(a, b)),
            _ => None,
        };
        Some(TcpHealth {
            established: cur.curr_estab,
            time_wait: sockstat.time_wait,
            orphans: sockstat.orphans,
            close_wait,
            retrans_percent,
            resets_ps: r(prev.estab_resets, cur.estab_resets),
            attempt_fails_ps: r(prev.attempt_fails, cur.attempt_fails),
            listen_drops_ps,
            in_errs_ps: r(prev.in_errs, cur.in_errs),
        })
    }
}

/// `Prefix: Name Name ...` followed by `Prefix: value value ...`, the shape of
/// both `/proc/net/snmp` and `/proc/net/netstat`. Negative values (`MaxConn
/// -1`) are skipped: none of the counters read here is ever negative.
pub fn parse_mib(content: &str, prefix: &str) -> HashMap<String, u64> {
    let mut lines = content.lines().filter(|l| l.starts_with(prefix));
    let mut out = HashMap::new();
    while let (Some(names), Some(values)) = (lines.next(), lines.next()) {
        let names = names[prefix.len()..].split_whitespace();
        let values = values[prefix.len()..].split_whitespace();
        for (n, v) in names.zip(values) {
            if let Ok(v) = v.parse::<u64>() {
                out.insert(n.to_string(), v);
            }
        }
    }
    out
}

pub fn parse_counters(snmp: &str, netstat: Option<&str>) -> Option<TcpCounters> {
    let tcp = parse_mib(snmp, "Tcp:");
    let ext = netstat.map(|n| parse_mib(n, "TcpExt:")).unwrap_or_default();
    let get = |k: &str| tcp.get(k).copied();
    Some(TcpCounters {
        out_segs: get("OutSegs")?,
        retrans_segs: get("RetransSegs")?,
        in_errs: get("InErrs").unwrap_or(0),
        estab_resets: get("EstabResets").unwrap_or(0),
        attempt_fails: get("AttemptFails").unwrap_or(0),
        curr_estab: get("CurrEstab").unwrap_or(0),
        listen_drops: ext.get("ListenDrops").copied(),
        listen_overflows: ext.get("ListenOverflows").copied(),
    })
}

/// What `/proc/net/sockstat` says about TCP.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sockstat {
    pub in_use: Option<u64>,
    pub orphans: Option<u64>,
    pub time_wait: Option<u64>,
}

/// `TCP: inuse 5 orphan 0 tw 2 alloc 7 mem 1`. `inuse` there is IPv4 only, so
/// `TCP6: inuse` from `sockstat6` is added by the caller.
pub fn parse_sockstat(content: &str) -> Sockstat {
    let mut out = Sockstat::default();
    let Some(line) = content.lines().find(|l| l.starts_with("TCP:")) else { return out };
    let fields: Vec<&str> = line.split_whitespace().skip(1).collect();
    for pair in fields.chunks(2) {
        let [key, value] = pair else { continue };
        let Ok(v) = value.parse() else { continue };
        match *key {
            "inuse" => out.in_use = Some(v),
            "orphan" => out.orphans = Some(v),
            "tw" => out.time_wait = Some(v),
            _ => {}
        }
    }
    out
}

/// Sockets in CLOSE_WAIT (state `08`) in one `/proc/net/tcp` table.
pub fn count_close_wait(table: &str) -> u64 {
    table.lines().skip(1).filter(|l| l.split_whitespace().nth(3) == Some("08")).count() as u64
}

/// Keeps the previous counters so each sample is a rate.
#[derive(Debug, Default)]
pub struct TcpReader {
    prev: Option<TcpCounters>,
}

impl TcpReader {
    pub fn new() -> Self {
        Self { prev: read_counters() }
    }

    pub fn sample(&mut self, secs: f64) -> Option<TcpHealth> {
        let cur = read_counters();
        let out = match (&self.prev, &cur) {
            (Some(p), Some(c)) => {
                let sockstat = read_sockstat();
                TcpHealth::between(p, c, secs, sockstat, read_close_wait(sockstat))
            }
            _ => None,
        };
        if cur.is_some() {
            self.prev = cur;
        }
        out
    }
}

fn read_counters() -> Option<TcpCounters> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let snmp = fs::read_to_string("/proc/net/snmp").ok()?;
    let netstat = fs::read_to_string("/proc/net/netstat").ok();
    parse_counters(&snmp, netstat.as_deref())
}

fn read_sockstat() -> Sockstat {
    let mut s =
        fs::read_to_string("/proc/net/sockstat").map(|c| parse_sockstat(&c)).unwrap_or_default();
    // IPv6 sockets are counted separately; TIME_WAIT is shared and already in.
    if let (Some(v4), Ok(c)) = (s.in_use, fs::read_to_string("/proc/net/sockstat6")) {
        let v6 = c
            .lines()
            .find(|l| l.starts_with("TCP6:"))
            .and_then(|l| l.split_whitespace().nth(2))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        s.in_use = Some(v4 + v6);
    }
    s
}

fn read_close_wait(sockstat: Sockstat) -> Option<u64> {
    // Without a socket count there is no knowing what the read would cost.
    if sockstat.in_use? + sockstat.time_wait.unwrap_or(0) > MAX_SOCKETS_SCANNED {
        return None;
    }
    let mut n = 0;
    for file in ["/proc/net/tcp", "/proc/net/tcp6"] {
        if let Ok(t) = fs::read_to_string(file) {
            n += count_close_wait(&t);
        }
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed from /proc/net/snmp and /proc/net/netstat on the development
    // machine; the column order is the kernel's.
    const SNMP: &str = "Ip: Forwarding DefaultTTL
Ip: 1 64
Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts InCsumErrors
Tcp: 1 200 120000 -1 1500 300 40 25 87 900000 1000000 2000 3 400 0
Udp: InDatagrams NoPorts
Udp: 10 2
";
    const NETSTAT: &str = "TcpExt: SyncookiesSent ListenOverflows ListenDrops TCPTimeouts
TcpExt: 0 12 15 99
IpExt: InNoRoutes
IpExt: 0
";

    #[test]
    fn the_mib_is_read_by_name_not_by_position() {
        let c = parse_counters(SNMP, Some(NETSTAT)).unwrap();
        assert_eq!(c.out_segs, 1_000_000);
        assert_eq!(c.retrans_segs, 2000);
        assert_eq!(c.curr_estab, 87);
        assert_eq!(c.estab_resets, 25);
        assert_eq!(c.listen_drops, Some(15));
        assert_eq!(c.listen_overflows, Some(12));
        // MaxConn is -1 and must not break the parse of what follows it.
        assert_eq!(parse_mib(SNMP, "Tcp:").get("MaxConn"), None);
        assert_eq!(parse_counters(SNMP, None).unwrap().listen_drops, None);
        assert_eq!(parse_counters("Udp: a\nUdp: 1\n", None), None);
    }

    #[test]
    fn retransmits_are_a_share_of_what_was_sent_in_the_interval() {
        let a = parse_counters(SNMP, Some(NETSTAT)).unwrap();
        let b = TcpCounters {
            out_segs: a.out_segs + 1000,
            retrans_segs: a.retrans_segs + 25,
            estab_resets: a.estab_resets + 4,
            listen_drops: Some(35),
            ..a
        };
        let ss = Sockstat { in_use: Some(100), orphans: Some(1), time_wait: Some(40) };
        let h = TcpHealth::between(&a, &b, 2.0, ss, Some(3)).unwrap();
        assert_eq!(h.retrans_percent, Some(2.5));
        assert_eq!(h.resets_ps, 2.0);
        assert_eq!(h.listen_drops_ps, Some(10.0));
        assert_eq!(h.time_wait, Some(40));
        assert_eq!(h.get("closewait"), Some((3.0, "")));
        assert_eq!(h.get(""), Some((2.5, "%")));
        assert_eq!(h.get("nonsense"), None);
        // Nothing sent: no percentage, rather than 0% that reads as healthy.
        let idle = TcpHealth::between(&a, &a, 2.0, ss, None).unwrap();
        assert_eq!(idle.retrans_percent, None);
        assert_eq!(idle.get("retrans"), None);
    }

    #[test]
    fn sockstat_and_close_wait_are_read_from_their_own_shapes() {
        let s = parse_sockstat(
            "sockets: used 300\nTCP: inuse 5 orphan 1 tw 22 alloc 7 mem 1\nUDP: inuse 3\n",
        );
        assert_eq!(s, Sockstat { in_use: Some(5), orphans: Some(1), time_wait: Some(22) });
        let table = "  sl  local_address rem_address   st tx_queue
   0: 0100007F:0277 00000000:0000 0A 00000000:00000000
   1: 0100007F:9C40 0100007F:1F90 08 00000000:00000000
   2: 0100007F:9C41 0100007F:1F90 01 00000000:00000000
   3: 0100007F:9C42 0100007F:1F90 08 00000000:00000000
";
        assert_eq!(count_close_wait(table), 2);
    }

    #[test]
    fn every_documented_target_is_one_a_rule_can_measure() {
        let h = TcpHealth {
            retrans_percent: Some(1.0),
            listen_drops_ps: Some(0.0),
            close_wait: Some(0),
            time_wait: Some(0),
            ..Default::default()
        };
        for t in TCP_SERIES {
            assert!(h.get(t).is_some(), "{t}");
        }
    }
}
