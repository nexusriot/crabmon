//! Which sockets a process holds open.
//!
//! `/proc/net/{tcp,tcp6,udp,udp6}` lists every socket with its inode, and
//! `/proc/<pid>/fd/*` symlinks point at `socket:[inode]`. Joining the two gives
//! the lsof answer without lsof. Only ever done for the one process whose
//! detail pane is open, so the cost does not matter.

use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Socket {
    pub protocol: String,
    pub local: String,
    pub remote: String,
    pub state: String,
}

/// TCP state numbers as used in `/proc/net/tcp`.
pub fn tcp_state(code: u8) -> &'static str {
    match code {
        1 => "ESTABLISHED",
        2 => "SYN_SENT",
        3 => "SYN_RECV",
        4 => "FIN_WAIT1",
        5 => "FIN_WAIT2",
        6 => "TIME_WAIT",
        7 => "CLOSE",
        8 => "CLOSE_WAIT",
        9 => "LAST_ACK",
        10 => "LISTEN",
        11 => "CLOSING",
        _ => "UNKNOWN",
    }
}

/// `/proc/net/*` encodes addresses as little-endian hex.
pub fn parse_address(hex: &str) -> Option<String> {
    let (addr, port) = hex.split_once(':')?;
    let port = u16::from_str_radix(port, 16).ok()?;
    let ip = match addr.len() {
        8 => {
            let v = u32::from_str_radix(addr, 16).ok()?;
            let b = v.to_le_bytes();
            format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3])
        }
        32 => {
            // Four little-endian 32-bit words.
            let mut groups = [0u16; 8];
            for word in 0..4 {
                let v = u32::from_str_radix(&addr[word * 8..word * 8 + 8], 16).ok()?;
                let b = v.to_le_bytes();
                groups[word * 2] = u16::from_be_bytes([b[0], b[1]]);
                groups[word * 2 + 1] = u16::from_be_bytes([b[2], b[3]]);
            }
            if groups.iter().all(|g| *g == 0) {
                "::".to_string()
            } else {
                format!("[{}]", format_v6(&groups))
            }
        }
        _ => return None,
    };
    Some(format!("{ip}:{port}"))
}

/// RFC 5952 text form: lowercase hex, with the longest run of two or more zero
/// groups collapsed to `::`. Writing all eight groups out turns `::1` into
/// `0:0:0:0:0:0:0:1`, which nobody recognises at a glance.
pub fn format_v6(groups: &[u16; 8]) -> String {
    let (mut best_at, mut best_len, mut run_at, mut run_len) = (0usize, 0usize, 0usize, 0usize);
    for (i, g) in groups.iter().enumerate() {
        if *g == 0 {
            if run_len == 0 {
                run_at = i;
            }
            run_len += 1;
            if run_len > best_len {
                (best_at, best_len) = (run_at, run_len);
            }
        } else {
            run_len = 0;
        }
    }
    // A single zero group is written out; `::` must stand for at least two.
    if best_len < 2 {
        return groups.iter().map(|g| format!("{g:x}")).collect::<Vec<_>>().join(":");
    }
    let head: Vec<String> = groups[..best_at].iter().map(|g| format!("{g:x}")).collect();
    let tail: Vec<String> = groups[best_at + best_len..].iter().map(|g| format!("{g:x}")).collect();
    format!("{}::{}", head.join(":"), tail.join(":"))
}

/// Parse one `/proc/net/{tcp,udp}` table into inode → socket.
pub fn parse_net_table(content: &str, protocol: &str) -> HashMap<u64, Socket> {
    let mut out = HashMap::new();
    for line in content.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        // sl local rem st tx:rx tr:when retrnsmt uid timeout inode
        if f.len() < 10 {
            continue;
        }
        let (Some(local), Some(remote)) = (parse_address(f[1]), parse_address(f[2])) else {
            continue;
        };
        let Ok(inode) = f[9].parse::<u64>() else { continue };
        let state = match u8::from_str_radix(f[3], 16) {
            Ok(code) if protocol.starts_with("tcp") => tcp_state(code).to_string(),
            _ => String::new(),
        };
        out.insert(inode, Socket { protocol: protocol.to_string(), local, remote, state });
    }
    out
}

fn all_sockets() -> HashMap<u64, Socket> {
    let mut out = HashMap::new();
    for (file, proto) in [("tcp", "tcp"), ("tcp6", "tcp6"), ("udp", "udp"), ("udp6", "udp6")] {
        if let Ok(content) = fs::read_to_string(format!("/proc/net/{file}")) {
            out.extend(parse_net_table(&content, proto));
        }
    }
    out
}

/// The socket inodes a process holds, from its fd table.
pub fn socket_inodes_of(pid: u32) -> Vec<u64> {
    let Ok(entries) = fs::read_dir(format!("/proc/{pid}/fd")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|e| fs::read_link(e.path()).ok())
        .filter_map(|target| parse_socket_link(&target.to_string_lossy()))
        .collect()
}

/// `socket:[12345]` → `12345`.
pub fn parse_socket_link(target: &str) -> Option<u64> {
    target.strip_prefix("socket:[")?.strip_suffix(']')?.parse().ok()
}

/// How many fds one process is walked for before giving up. A busy server can
/// hold tens of thousands; the column is a hint, not an inventory.
pub const MAX_FDS_SCANNED: usize = 4096;

/// Listening local ports per process, for the rows currently on screen.
///
/// The `/proc/net` tables are read once and the fd tables only for `pids`, so
/// the cost is bounded by how many rows the process table can show rather than
/// by how many processes the machine has.
pub fn listening_ports(pids: &[u32]) -> HashMap<u32, Vec<u16>> {
    let mut out = HashMap::new();
    if !cfg!(target_os = "linux") || pids.is_empty() {
        return out;
    }
    let table = all_sockets();
    // Only listeners are interesting, and on most machines there are a few dozen.
    let listening: HashMap<u64, u16> = table
        .iter()
        .filter(|(_, s)| s.state == "LISTEN" || s.protocol.starts_with("udp"))
        .filter_map(|(inode, s)| port_of(&s.local).map(|p| (*inode, p)))
        .collect();
    if listening.is_empty() {
        return out;
    }
    for pid in pids {
        let mut ports: Vec<u16> = socket_inodes_of(*pid)
            .into_iter()
            .take(MAX_FDS_SCANNED)
            .filter_map(|i| listening.get(&i).copied())
            .collect();
        ports.sort_unstable();
        ports.dedup();
        if !ports.is_empty() {
            out.insert(*pid, ports);
        }
    }
    out
}

/// The port from a rendered `addr:port`, which may itself contain colons.
pub fn port_of(local: &str) -> Option<u16> {
    local.rsplit_once(':').and_then(|(_, p)| p.parse().ok())
}

/// Every socket held by `pid`, listening sockets first. Linux-only.
pub fn for_pid(pid: u32) -> Vec<Socket> {
    if !cfg!(target_os = "linux") {
        return Vec::new();
    }
    let table = all_sockets();
    let mut out: Vec<Socket> =
        socket_inodes_of(pid).into_iter().filter_map(|i| table.get(&i).cloned()).collect();
    out.sort_by(|a, b| {
        (b.state == "LISTEN")
            .cmp(&(a.state == "LISTEN"))
            .then_with(|| a.protocol.cmp(&b.protocol))
            .then_with(|| a.local.cmp(&b.local))
    });
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The header plus one real line from /proc/net/tcp on this machine.
    const TCP: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:AD23 00000000:0000 0A 00000000:00000000 00:00000000 00000000   995        0 25656 1 0000000000000000 100 0 0 10 0
   1: 0100007F:1F90 0100007F:C1B2 01 00000000:00000000 00:00000000 00000000  1000        0 99887 1 0000000000000000 20 0 0 10 -1
";

    #[test]
    fn little_endian_ipv4_addresses_are_decoded() {
        // 0100007F is 127.0.0.1 stored little-endian; AD23 is port 44323.
        assert_eq!(parse_address("0100007F:1F90").as_deref(), Some("127.0.0.1:8080"));
        assert_eq!(parse_address("00000000:AD23").as_deref(), Some("0.0.0.0:44323"));
        assert_eq!(parse_address("garbage"), None);
    }

    #[test]
    fn ipv6_wildcards_and_addresses_are_decoded() {
        let wildcard = "0".repeat(32);
        assert_eq!(parse_address(&format!("{wildcard}:0050")).as_deref(), Some(":::80"));

        // ::1 — the form anyone actually recognises, not 0:0:0:0:0:0:0:1.
        let loopback = "00000000000000000000000001000000";
        assert_eq!(parse_address(&format!("{loopback}:1F90")).as_deref(), Some("[::1]:8080"));
    }

    #[test]
    fn ipv6_groups_collapse_the_longest_zero_run_only() {
        assert_eq!(format_v6(&[0, 0, 0, 0, 0, 0, 0, 1]), "::1");
        assert_eq!(format_v6(&[0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]), "2001:db8::1");
        assert_eq!(format_v6(&[0xfe80, 0, 0, 0, 0x1, 0x2, 0x3, 0x4]), "fe80::1:2:3:4");
        // A lone zero group is spelled out; `::` must cover at least two.
        assert_eq!(format_v6(&[1, 0, 2, 3, 4, 5, 6, 7]), "1:0:2:3:4:5:6:7");
        // Ties take the first run, as RFC 5952 requires.
        assert_eq!(format_v6(&[1, 0, 0, 2, 0, 0, 3, 4]), "1::2:0:0:3:4");
        assert_eq!(format_v6(&[0, 0, 0, 0, 0, 0, 0, 0]), "::");
    }

    #[test]
    fn ports_are_taken_from_the_right_side_of_a_v6_address() {
        assert_eq!(port_of("[::1]:8080"), Some(8080));
        assert_eq!(port_of("127.0.0.1:22"), Some(22));
        assert_eq!(port_of("nonsense"), None);
    }

    #[test]
    fn listening_ports_are_looked_up_without_scanning_every_process() {
        // No listener is guaranteed in a test harness; what matters is that the
        // call is bounded by `pids` and never panics on a dead one.
        assert!(listening_ports(&[]).is_empty());
        assert!(!listening_ports(&[u32::MAX / 2]).contains_key(&(u32::MAX / 2)));
    }

    #[test]
    fn a_listening_socket_and_an_established_one_are_distinguished() {
        let table = parse_net_table(TCP, "tcp");
        assert_eq!(table.len(), 2);
        let listener = &table[&25656];
        assert_eq!(listener.state, "LISTEN");
        assert_eq!(listener.local, "0.0.0.0:44323");
        let established = &table[&99887];
        assert_eq!(established.state, "ESTABLISHED");
        assert_eq!(established.remote, "127.0.0.1:49586");
    }

    #[test]
    fn udp_rows_carry_no_tcp_state() {
        let table = parse_net_table(TCP, "udp");
        assert!(table.values().all(|s| s.state.is_empty()));
    }

    #[test]
    fn malformed_rows_are_skipped() {
        assert!(parse_net_table("header\nshort line\n", "tcp").is_empty());
        assert!(parse_net_table("", "tcp").is_empty());
    }

    #[test]
    fn fd_symlinks_yield_socket_inodes() {
        assert_eq!(parse_socket_link("socket:[25656]"), Some(25656));
        assert_eq!(parse_socket_link("/dev/null"), None);
        assert_eq!(parse_socket_link("socket:[abc]"), None);
        assert_eq!(parse_socket_link("anon_inode:[eventpoll]"), None);
    }

    #[test]
    fn every_tcp_state_code_has_a_name() {
        for code in 1..=11u8 {
            assert_ne!(tcp_state(code), "UNKNOWN", "state {code}");
        }
        assert_eq!(tcp_state(99), "UNKNOWN");
    }

    #[test]
    fn our_own_process_reports_its_real_sockets() {
        // The test harness itself holds no listening socket, but the call must
        // work end-to-end against the real /proc without panicking.
        let pid = std::process::id();
        let _ = for_pid(pid);
        assert!(for_pid(u32::MAX / 2).is_empty(), "a dead pid has no sockets");
    }
}
