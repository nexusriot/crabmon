//! Network throughput per network namespace — which is to say, per container.
//!
//! Per-*process* network bytes is the thing a process monitor is asked for
//! most often and the thing it cannot honestly provide: the kernel does not
//! account bytes to a PID, and getting there means eBPF, `NETLINK_SOCK_DIAG`
//! or packet capture, all of which need privileges crabmon has promised not to
//! ask for.
//!
//! What the kernel *does* account, and to a readable file, is bytes per
//! network namespace: `/proc/<pid>/net/dev` reports the interfaces of the
//! namespace that process is in, not the host's. Every process in a container
//! shares one namespace, so the same counters that cannot say "nginx sent 4
//! MB" can say "this container sent 4 MB" — which is usually the question
//! anyway, and is the one the grouped view is already organised around.
//!
//! The cost is one readlink per process (cached, since a process does not
//! change namespace after it starts) and one file read per *namespace* per
//! sample, not per process.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// One namespace's traffic over the last interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NetNamespace {
    /// The namespace's inode number, which is what `net:[4026532001]` names
    /// and what makes two processes comparable.
    pub id: u64,
    /// The container these processes belong to, when they are in one. A
    /// namespace with no container is a `ip netns` namespace, a sandboxed
    /// service, or the host itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// True for the namespace crabmon itself is running in. Its traffic is
    /// already in `Snapshot::nets`; the flag is here so it is not counted as
    /// another container's.
    pub host: bool,
    pub rx_bps: f64,
    pub tx_bps: f64,
    pub rx_total: u64,
    pub tx_total: u64,
    /// Processes crabmon can see in this namespace.
    pub procs: usize,
}

/// The namespace id a process is in, from `/proc/<pid>/ns/net`.
pub fn id_of(pid: u32) -> Option<u64> {
    parse_ns_link(&read_link(&format!("/proc/{pid}/ns/net"))?)
}

/// `net:[4026532001]` → `4026532001`.
pub fn parse_ns_link(target: &str) -> Option<u64> {
    let inner = target.trim().strip_prefix("net:[")?.strip_suffix(']')?;
    inner.parse().ok()
}

/// Cumulative received and transmitted bytes across a namespace's interfaces.
///
/// Loopback is excluded for the same reason the dashboard excludes it from the
/// aggregate: traffic a container sends to itself is not traffic it put on a
/// network, and counting it makes a busy local socket look like a busy uplink.
pub fn parse_net_dev(content: &str) -> (u64, u64) {
    let mut rx = 0u64;
    let mut tx = 0u64;
    for line in content.lines() {
        // `  eth0: 1234 56 0 0 0 0 0 0 4321 65 ...`, after two header lines.
        let Some((name, rest)) = line.split_once(':') else { continue };
        let name = name.trim();
        if name.is_empty() || name == "lo" {
            continue;
        }
        let f: Vec<&str> = rest.split_whitespace().collect();
        // Receive bytes is the first column, transmit bytes the ninth.
        if f.len() < 9 {
            continue;
        }
        let (Ok(r), Ok(t)) = (f[0].parse::<u64>(), f[8].parse::<u64>()) else {
            continue;
        };
        rx = rx.saturating_add(r);
        tx = tx.saturating_add(t);
    }
    (rx, tx)
}

/// Read one namespace's counters through a process that lives in it.
pub fn counters_of(pid: u32) -> Option<(u64, u64)> {
    Some(parse_net_dev(&read_file(&format!("/proc/{pid}/net/dev"))?))
}

/// Cumulative per-namespace counters between samples, so rates can be derived.
#[derive(Debug, Default)]
pub struct NetnsRates {
    prev: HashMap<u64, (u64, u64)>,
}

/// What `collect` needs to know about one process: which namespace it is in,
/// and what container it belongs to.
#[derive(Debug, Clone, Copy)]
pub struct Member<'a> {
    pub pid: u32,
    pub netns: Option<u64>,
    pub container: Option<&'a str>,
}

impl NetnsRates {
    /// One row per namespace crabmon can see a process in.
    ///
    /// `read` is the counter reader, injected so the aggregation can be tested
    /// without a container to hand. The representative process for a namespace
    /// is its lowest PID, which is stable between samples as long as that
    /// process lives — and when it does not, the namespace is simply read
    /// through the next one, with no effect on the counters, since they belong
    /// to the namespace rather than to the process.
    pub fn collect(
        &mut self,
        members: &[Member<'_>],
        host_ns: Option<u64>,
        secs: f64,
        read: impl Fn(u32) -> Option<(u64, u64)>,
    ) -> Vec<NetNamespace> {
        // Lowest PID per namespace, every container seen in it, and how many
        // processes it holds.
        let mut groups: HashMap<u64, (u32, HashSet<&str>, usize)> = HashMap::new();
        for m in members {
            let Some(ns) = m.netns else { continue };
            let entry = groups.entry(ns).or_insert((m.pid, HashSet::new(), 0));
            entry.0 = entry.0.min(m.pid);
            if let Some(c) = m.container {
                entry.1.insert(c);
            }
            entry.2 += 1;
        }

        let mut out: Vec<NetNamespace> = groups
            .into_iter()
            .filter_map(|(id, (pid, containers, procs))| {
                let host = Some(id) == host_ns;
                // A namespace belongs to a container only when every process
                // in it that names one names the *same* one.
                //
                // Taking whichever member came first put a container's id on
                // the host's own namespace, because a container started with
                // `--network host`, or a `docker-proxy`, lives in a container
                // cgroup while sharing the host's network. A Kubernetes pod is
                // the other direction: several containers deliberately share
                // one namespace, and none of them owns the traffic. Both are
                // "no single container", and the namespace is named by its
                // inode instead.
                let container = match (host, containers.len()) {
                    (false, 1) => containers.iter().next().map(|c| c.to_string()),
                    _ => None,
                };
                let (rx_total, tx_total) = read(pid)?;
                let (rx_bps, tx_bps) = match self.prev.get(&id) {
                    Some(&(prx, ptx)) => {
                        (super::rate(prx, rx_total, secs), super::rate(ptx, tx_total, secs))
                    }
                    None => (0.0, 0.0),
                };
                Some(NetNamespace {
                    id,
                    container,
                    host,
                    rx_bps,
                    tx_bps,
                    rx_total,
                    tx_total,
                    procs,
                })
            })
            .collect();

        // Only namespaces still present keep their previous counters, so the
        // map cannot grow for the life of the process.
        self.prev = out.iter().map(|n| (n.id, (n.rx_total, n.tx_total))).collect();
        // Busiest first, and by id for the idle ones so the panel does not
        // reshuffle every refresh.
        out.sort_by(|a, b| {
            (b.rx_bps + b.tx_bps)
                .partial_cmp(&(a.rx_bps + a.tx_bps))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        out
    }
}

#[cfg(target_os = "linux")]
fn read_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

#[cfg(not(target_os = "linux"))]
fn read_file(_path: &str) -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn read_link(path: &str) -> Option<String> {
    std::fs::read_link(path).ok().map(|p| p.to_string_lossy().to_string())
}

#[cfg(not(target_os = "linux"))]
fn read_link(_path: &str) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim from a container, trimmed to three interfaces.
    const NET_DEV: &str = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo:  131072     512    0    0    0     0          0         0   131072     512    0    0    0     0       0          0
  eth0: 4194304    3000    0    0    0     0          0         0  1048576    2000    1    0    0     0       0          0
  eth1:  524288     400    0    0    0     0          0         0   262144     300    0    0    0     0       0          0
";

    #[test]
    fn namespace_links_parse_to_their_inode() {
        assert_eq!(parse_ns_link("net:[4026532001]"), Some(4_026_532_001));
        assert_eq!(parse_ns_link(" net:[123] "), Some(123));
        // Anything else is not a network namespace.
        assert_eq!(parse_ns_link("mnt:[4026531840]"), None);
        assert_eq!(parse_ns_link("net:[]"), None);
        assert_eq!(parse_ns_link("net:[abc]"), None);
        assert_eq!(parse_ns_link(""), None);
    }

    #[test]
    fn counters_sum_every_interface_except_loopback() {
        // Traffic a container sends to itself is not traffic it put on a
        // network; counting `lo` makes a busy local socket look like an uplink.
        let (rx, tx) = parse_net_dev(NET_DEV);
        assert_eq!(rx, 4_194_304 + 524_288);
        assert_eq!(tx, 1_048_576 + 262_144);
    }

    #[test]
    fn a_malformed_table_yields_nothing_rather_than_a_wrong_total() {
        assert_eq!(parse_net_dev(""), (0, 0));
        assert_eq!(parse_net_dev("nonsense\nmore nonsense\n"), (0, 0));
        // A short line is skipped rather than read with the wrong columns.
        assert_eq!(parse_net_dev("  eth0: 100 2 0\n"), (0, 0));
    }

    fn member<'a>(pid: u32, ns: u64, container: Option<&'a str>) -> Member<'a> {
        Member { pid, netns: Some(ns), container }
    }

    #[test]
    fn processes_sharing_a_namespace_are_one_row_not_several() {
        // Every process in a container reports the *same* counters, because
        // they belong to the namespace. Summing per process would multiply a
        // container's traffic by however many processes it happens to run.
        let members = [
            member(100, 42, Some("abc123")),
            member(101, 42, Some("abc123")),
            member(102, 42, Some("abc123")),
        ];
        let mut rates = NetnsRates::default();
        let read = |_pid| Some((1_000_000u64, 500_000u64));

        rates.collect(&members, None, 1.0, read);
        let out = rates.collect(&members, None, 1.0, |_| Some((2_000_000, 1_500_000)));

        assert_eq!(out.len(), 1, "one namespace, one row");
        assert_eq!(out[0].procs, 3);
        assert_eq!(out[0].container.as_deref(), Some("abc123"));
        assert_eq!(out[0].rx_bps, 1_000_000.0);
        assert_eq!(out[0].tx_bps, 1_000_000.0);
    }

    #[test]
    fn the_first_sample_has_no_rate_to_report() {
        // There is no rate without two readings, and 0 is the honest answer
        // rather than the whole counter divided by one interval.
        let mut rates = NetnsRates::default();
        let out = rates.collect(&[member(1, 7, None)], None, 1.0, |_| Some((9_999, 9_999)));
        assert_eq!(out[0].rx_bps, 0.0);
        assert_eq!(out[0].rx_total, 9_999);
    }

    #[test]
    fn the_namespace_crabmon_runs_in_is_marked_so_it_is_not_counted_twice() {
        // The host's traffic is already in `Snapshot::nets`.
        let members = [member(1, 7, None), member(900, 42, Some("abc"))];
        let mut rates = NetnsRates::default();
        let out = rates.collect(&members, Some(7), 1.0, |_| Some((0, 0)));
        assert_eq!(out.iter().filter(|n| n.host).count(), 1);
        assert!(out.iter().find(|n| n.id == 7).unwrap().host);
        assert!(!out.iter().find(|n| n.id == 42).unwrap().host);
    }

    /// Taking whichever process came first put a container's id on the
    /// *host's* namespace, because a container started with `--network host`,
    /// and `docker-proxy`, both live in a container cgroup while sharing the
    /// host's network.
    #[test]
    fn the_host_namespace_is_never_labelled_with_a_container_that_shares_it() {
        let members = [
            member(1, 7, Some("047d0a3cd35b")),
            member(2, 7, None),
            member(900, 42, Some("abc123")),
        ];
        let mut rates = NetnsRates::default();
        let out = rates.collect(&members, Some(7), 1.0, |_| Some((0, 0)));

        let host = out.iter().find(|n| n.id == 7).unwrap();
        assert!(host.host);
        assert_eq!(host.container, None, "the host's namespace is the host's");
        assert_eq!(out.iter().find(|n| n.id == 42).unwrap().container.as_deref(), Some("abc123"));
    }

    /// A Kubernetes pod is several containers sharing one namespace on
    /// purpose. None of them owns the traffic, so none of them is named.
    #[test]
    fn a_namespace_shared_by_several_containers_is_not_attributed_to_one_of_them() {
        let members = [member(10, 42, Some("app")), member(11, 42, Some("sidecar"))];
        let mut rates = NetnsRates::default();
        let out = rates.collect(&members, None, 1.0, |_| Some((0, 0)));
        assert_eq!(out[0].container, None, "neither container owns it");
        assert_eq!(out[0].procs, 2);
    }

    #[test]
    fn a_namespace_read_through_a_process_that_exited_is_dropped_not_zeroed() {
        // A row of zeroes would draw an idle container; no row says the
        // namespace went away, which it did.
        let mut rates = NetnsRates::default();
        let out = rates.collect(&[member(1, 7, None)], None, 1.0, |_| None);
        assert!(out.is_empty());
    }

    #[test]
    fn processes_with_no_namespace_are_skipped_rather_than_pooled() {
        // A platform with no `/proc/<pid>/ns/net`, or a process whose link is
        // not readable. Pooling them under one key would invent a namespace.
        let members = [Member { pid: 1, netns: None, container: None }, member(2, 9, None)];
        let mut rates = NetnsRates::default();
        let out = rates.collect(&members, None, 1.0, |_| Some((5, 5)));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 9);
    }

    #[test]
    fn namespaces_that_disappear_do_not_keep_their_counters_for_ever() {
        let mut rates = NetnsRates::default();
        rates.collect(&[member(1, 7, None), member(2, 8, None)], None, 1.0, |_| Some((10, 10)));
        assert_eq!(rates.prev.len(), 2);
        rates.collect(&[member(1, 7, None)], None, 1.0, |_| Some((20, 20)));
        assert_eq!(rates.prev.len(), 1, "the dead namespace is forgotten");
    }

    #[test]
    fn the_busiest_namespace_is_listed_first() {
        let members = [member(1, 7, None), member(2, 8, None), member(3, 9, None)];
        let mut rates = NetnsRates::default();
        let totals = |pid: u32| Some((pid as u64 * 1000, 0u64));
        rates.collect(&members, None, 1.0, |_| Some((0, 0)));
        let out = rates.collect(&members, None, 1.0, totals);
        assert_eq!(out.iter().map(|n| n.id).collect::<Vec<_>>(), vec![9, 8, 7]);
    }
}
