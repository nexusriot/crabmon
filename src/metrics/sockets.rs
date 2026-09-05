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
            let mut groups = Vec::with_capacity(8);
            for word in 0..4 {
                let v = u32::from_str_radix(&addr[word * 8..word * 8 + 8], 16).ok()?;
                let b = v.to_le_bytes();
                groups.push(u16::from_be_bytes([b[0], b[1]]));
                groups.push(u16::from_be_bytes([b[2], b[3]]));
            }
            if groups.iter().all(|g| *g == 0) {
                "::".to_string()
            } else {
                format!(
                    "[{}]",
                    groups.iter().map(|g| format!("{g:x}")).collect::<Vec<_>>().join(":")
                )
            }
        }
        _ => return None,
    };
    Some(format!("{ip}:{port}"))
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

        let loopback = "00000000000000000000000001000000";
        let rendered = parse_address(&format!("{loopback}:1F90")).unwrap();
        assert!(rendered.starts_with('['), "{rendered}");
        assert!(rendered.ends_with(":8080"), "{rendered}");
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
