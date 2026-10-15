//! A Prometheus exporter.
//!
//! `crabmon --serve :9100` samples on the configured interval and answers
//! `/metrics` over HTTP. The server is hand-rolled against `TcpListener`
//! because the whole contract is one route, one method and a text body — a
//! framework would be more dependency than feature.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::metrics::Snapshot;

/// How long a client gets to send its request line, and to take its response.
/// Generous for anything that actually means to scrape, short enough that a
/// connection which opens and says nothing is not a resource.
pub const CLIENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Connections handled at once. Past this, callers get 503 rather than an
/// unbounded pile of threads: the point is to stay answerable under abuse, not
/// to serve everybody.
pub const MAX_CONNECTIONS: usize = 16;

/// Longest request line the exporter will read before giving up on it. A URL
/// Prometheus actually sends is well under 200 bytes; this is only here so a
/// client that never sends a newline cannot make the exporter buffer without
/// bound.
pub const MAX_REQUEST_LINE: u64 = 8 * 1024;

/// Combine several hosts' exposition text into one scrape, labelling each
/// host's samples with `host="..."`.
///
/// crabmon can already watch a fleet over SSH and can already export what it
/// measures; the two could not meet, so a fleet had a dashboard and no
/// metrics. Merging here rather than threading a label through every
/// `metric` call keeps the single-host path exactly as it was.
///
/// Families are grouped and their `HELP`/`TYPE` lines emitted once: Prometheus
/// rejects a repeated `HELP` for a family, which would lose the whole scrape
/// rather than one host's worth of it.
pub fn merge_hosts(parts: &[(String, String)]) -> String {
    // Insertion-ordered so the output is stable rather than hash-ordered.
    let mut order: Vec<String> = Vec::new();
    let mut meta: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut samples: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for (host, body) in parts {
        for line in body.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) =
                line.strip_prefix("# HELP ").or_else(|| line.strip_prefix("# TYPE "))
            {
                let family = rest.split_whitespace().next().unwrap_or_default().to_string();
                if !meta.contains_key(&family) {
                    order.push(family.clone());
                }
                let entry = meta.entry(family).or_default();
                if !entry.iter().any(|l| l == line) {
                    entry.push(line.to_string());
                }
                continue;
            }
            let family = line.split(['{', ' ']).next().unwrap_or_default().to_string();
            if !meta.contains_key(&family) {
                order.push(family.clone());
                meta.entry(family.clone()).or_default();
            }
            samples.entry(family).or_default().push(labelled(line, host));
        }
    }

    let mut out = String::new();
    for family in order {
        for line in meta.get(&family).into_iter().flatten() {
            out.push_str(line);
            out.push('\n');
        }
        for line in samples.get(&family).into_iter().flatten() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Add `host="..."` to one sample line, whether or not it already has labels.
fn labelled(line: &str, host: &str) -> String {
    let host = format!("host=\"{}\"", escape_label(host));
    match line.find('{') {
        // `name{a="b"} 1` → `name{host="h",a="b"} 1`
        Some(at) => format!("{}{{{host},{}", &line[..at], &line[at + 1..]),
        // `name 1` → `name{host="h"} 1`
        None => match line.split_once(' ') {
            Some((name, value)) => format!("{name}{{{host}}} {value}"),
            None => line.to_string(),
        },
    }
}

/// Escape a Prometheus label value.
pub fn escape_label(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

/// Emit one metric family: a HELP/TYPE pair and its samples.
///
/// Samples that repeat a label set are dropped, keeping the first. Prometheus
/// treats two samples with identical labels in one family as a duplicate
/// series and rejects the **whole scrape** — the same all-or-nothing failure
/// that repeating a HELP line used to cause here, and from the same cause: a
/// label built from something that is not actually unique. Two identical GPUs
/// are the obvious case, since a card's name is `"<vendor> <driver>"` from
/// sysfs or whatever `nvidia-smi` calls the model, and a box with a matched
/// pair reports the same string twice. Two NVMe drives presenting the same
/// sensor label do it too.
///
/// Losing the second of a duplicated pair is a real loss, but it is strictly
/// better than losing every series in the scrape, which is what happened
/// before: a dual-GPU host exported nothing at all.
fn metric(out: &mut String, name: &str, help: &str, kind: &str, samples: &[(String, f64)]) {
    if samples.is_empty() {
        return;
    }
    out.push_str(&format!("# HELP crabmon_{name} {help}\n"));
    out.push_str(&format!("# TYPE crabmon_{name} {kind}\n"));
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (labels, value) in samples {
        if !seen.insert(labels.as_str()) {
            continue;
        }
        if labels.is_empty() {
            out.push_str(&format!("crabmon_{name} {value}\n"));
        } else {
            out.push_str(&format!("crabmon_{name}{{{labels}}} {value}\n"));
        }
    }
}

fn plain(value: f64) -> Vec<(String, f64)> {
    vec![(String::new(), value)]
}

/// Render a snapshot in the Prometheus text exposition format.
///
/// `top_procs` limits the per-process series, because one series per process on
/// a 2000-task machine would dwarf everything else in the scrape.
pub fn render(snap: &Snapshot, top_procs: usize) -> String {
    render_with_alerts(snap, top_procs, &[])
}

/// As `render`, plus the state of the alert rules being evaluated.
///
/// crabmon knows things Prometheus does not: pressure-stall averages, the
/// busiest device's utilisation, how close a process is to *its own*
/// descriptor limit, whether a named process has stopped existing. Those were
/// alertable in the TUI and invisible to anything scraping it, so the rules
/// most worth paging on were the ones that could not reach a pager.
///
/// One series per rule, 1 while it is firing and 0 while it is not — the shape
/// that makes `crabmon_alert_active == 1` a usable Prometheus expression —
/// alongside the measurement and the threshold it was compared against, so an
/// alert can be shown with the number that caused it.
pub fn render_with_alerts(
    snap: &Snapshot,
    top_procs: usize,
    alerts: &[crate::alerts::RuleState],
) -> String {
    render_host(snap, top_procs, alerts, true)
}

/// As `render_with_alerts`, saying whether the host these numbers came from
/// could actually be reached.
///
/// For a single host the answer is always yes — the exporter is the process
/// being scraped. For a fleet it is not: a machine that is down still gets a
/// series, and `crabmon_up{host="db-1"} 1` on a host nobody can reach reads
/// as "db-1 is fine" when the frame behind it is the last good one, or
/// zeroes. `up` is the one series whose whole job is to answer that.
pub fn render_host(
    snap: &Snapshot,
    top_procs: usize,
    alerts: &[crate::alerts::RuleState],
    reachable: bool,
) -> String {
    let mut out = String::new();

    if !alerts.is_empty() {
        let label = |r: &crate::alerts::RuleState| {
            format!("rule=\"{}\",kind=\"{}\"", escape_label(&r.name), r.kind.key_name())
        };
        metric(
            &mut out,
            "alert_active",
            "1 while a rule is over its threshold and past its hold time.",
            "gauge",
            &alerts.iter().map(|r| (label(r), r.active as u8 as f64)).collect::<Vec<_>>(),
        );
        metric(
            &mut out,
            "alert_value",
            "The measurement a rule was last evaluated against.",
            "gauge",
            &alerts.iter().filter_map(|r| r.value.map(|v| (label(r), v))).collect::<Vec<_>>(),
        );
        metric(
            &mut out,
            "alert_threshold",
            "The threshold a rule compares against.",
            "gauge",
            &alerts.iter().map(|r| (label(r), r.threshold)).collect::<Vec<_>>(),
        );
    }

    metric(
        &mut out,
        "up",
        "1 when this host's metrics are a live sample of it.",
        "gauge",
        &plain(reachable as u8 as f64),
    );
    metric(
        &mut out,
        "cpu_usage_percent",
        "Average CPU usage across all cores.",
        "gauge",
        &plain(snap.cpu.avg()),
    );
    metric(
        &mut out,
        "cpu_core_usage_percent",
        "Per-core CPU usage.",
        "gauge",
        &snap
            .cpu
            .per_core
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("core=\"{i}\""), *v as f64))
            .collect::<Vec<_>>(),
    );
    metric(&mut out, "load1", "One-minute load average.", "gauge", &plain(snap.host.load[0]));
    metric(
        &mut out,
        "uptime_seconds",
        "Host uptime.",
        "gauge",
        &plain(snap.host.uptime_secs as f64),
    );

    metric(&mut out, "memory_total_bytes", "Total RAM.", "gauge", &plain(snap.mem.total as f64));
    metric(&mut out, "memory_used_bytes", "Used RAM.", "gauge", &plain(snap.mem.used as f64));
    metric(
        &mut out,
        "memory_available_bytes",
        "Available RAM.",
        "gauge",
        &plain(snap.mem.available as f64),
    );
    metric(
        &mut out,
        "memory_cache_bytes",
        "Buffers plus reclaimable cache.",
        "gauge",
        &plain(snap.mem.detail.buff_cache() as f64),
    );
    metric(
        &mut out,
        "swap_total_bytes",
        "Total swap.",
        "gauge",
        &plain(snap.mem.swap_total as f64),
    );
    metric(&mut out, "swap_used_bytes", "Used swap.", "gauge", &plain(snap.mem.swap_used as f64));

    // Per-container and per-unit accounting, from each cgroup's own files.
    // The limit is the series that matters: a container at 98% of a 512 MB
    // limit is invisible in the host's memory gauge.
    let cg_label =
        |c: &crate::metrics::cgroup::CgroupStat| format!("name=\"{}\"", escape_label(&c.name));
    metric(
        &mut out,
        "cgroup_memory_bytes",
        "Memory charged to a container or unit's own cgroup.",
        "gauge",
        &snap
            .cgroups
            .iter()
            .filter_map(|c| c.mem_current.map(|m| (cg_label(c), m as f64)))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "cgroup_memory_limit_bytes",
        "A cgroup's memory limit, where it has one.",
        "gauge",
        &snap
            .cgroups
            .iter()
            .filter_map(|c| c.mem_max.map(|m| (cg_label(c), m as f64)))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "cgroup_cpu_cores",
        "CPU used by a cgroup over the scrape interval, in cores.",
        "gauge",
        &snap
            .cgroups
            .iter()
            .filter_map(|c| c.cpu_cores.map(|v| (cg_label(c), v)))
            .collect::<Vec<_>>(),
    );

    // Kernel events as a gauge per kind: `crabmon_kernel_events{kind="oom"}`
    // is the shape that makes "page me if anything was OOM-killed" a
    // Prometheus expression. Absent entirely where the log is unreadable,
    // because exporting 0 there would claim the machine is healthy.
    if snap.kernel_unreadable.is_none() {
        metric(
            &mut out,
            "kernel_events",
            "Kernel events in the sampler's recent window, by kind.",
            "gauge",
            &crate::metrics::kmsg::ALL_EVENT_KINDS
                .iter()
                .map(|k| (format!("kind=\"{}\"", k.key_name()), snap.kernel_count(*k) as f64))
                .collect::<Vec<_>>(),
        );
    }

    metric(
        &mut out,
        "processes",
        "Number of tasks reported by the sampler.",
        "gauge",
        &plain(snap.procs.len() as f64),
    );

    let disk_label = |d: &crate::metrics::DiskRow| format!("mount=\"{}\"", escape_label(&d.mount));
    metric(
        &mut out,
        "disk_total_bytes",
        "Filesystem size.",
        "gauge",
        &snap.disks.iter().map(|d| (disk_label(d), d.total as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_used_bytes",
        "Filesystem usage.",
        "gauge",
        &snap.disks.iter().map(|d| (disk_label(d), d.used as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_inodes_total",
        "Inodes on the filesystem.",
        "gauge",
        &snap
            .disks
            .iter()
            .filter(|d| d.inodes_total > 0)
            .map(|d| (disk_label(d), d.inodes_total as f64))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_inodes_used",
        "Inodes in use.",
        "gauge",
        &snap
            .disks
            .iter()
            .filter(|d| d.inodes_total > 0)
            .map(|d| (disk_label(d), d.inodes_used as f64))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_inodes_used_ratio",
        "Inode usage, 0..1.",
        "gauge",
        &snap
            .disks
            .iter()
            .filter(|d| d.inodes_total > 0)
            .map(|d| (disk_label(d), d.inode_ratio()))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_read_bytes_per_second",
        "Disk read throughput.",
        "gauge",
        &snap.disks.iter().map(|d| (disk_label(d), d.read_bps)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "disk_write_bytes_per_second",
        "Disk write throughput.",
        "gauge",
        &snap.disks.iter().map(|d| (disk_label(d), d.write_bps)).collect::<Vec<_>>(),
    );

    let iface = |n: &crate::metrics::NetIface| format!("interface=\"{}\"", escape_label(&n.name));
    metric(
        &mut out,
        "network_receive_bytes_per_second",
        "Interface receive rate.",
        "gauge",
        &snap.nets.iter().map(|n| (iface(n), n.rx_bps)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "network_transmit_bytes_per_second",
        "Interface transmit rate.",
        "gauge",
        &snap.nets.iter().map(|n| (iface(n), n.tx_bps)).collect::<Vec<_>>(),
    );

    metric(
        &mut out,
        "network_receive_errors_total",
        "Receive errors since boot.",
        "counter",
        &snap.nets.iter().map(|n| (iface(n), n.errors_rx as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "network_transmit_errors_total",
        "Transmit errors since boot.",
        "counter",
        &snap.nets.iter().map(|n| (iface(n), n.errors_tx as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "network_receive_bytes_total",
        "Bytes received since boot.",
        "counter",
        &snap.nets.iter().map(|n| (iface(n), n.rx_total as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "network_transmit_bytes_total",
        "Bytes transmitted since boot.",
        "counter",
        &snap.nets.iter().map(|n| (iface(n), n.tx_total as f64)).collect::<Vec<_>>(),
    );

    metric(
        &mut out,
        "temperature_celsius",
        "Sensor temperature.",
        "gauge",
        &snap
            .sensors
            .iter()
            .map(|s| (format!("sensor=\"{}\"", escape_label(&s.label)), s.temp as f64))
            .collect::<Vec<_>>(),
    );

    // One HELP/TYPE pair per metric family, with the resources as labelled
    // samples underneath it. Emitting them inside the loop gave three HELP lines
    // for `crabmon_pressure_some_ratio`, and Prometheus rejects the *entire*
    // scrape on a repeated HELP — so every other series here was dropped too.
    let mut some_samples: Vec<(String, f64)> = Vec::new();
    let mut full_samples: Vec<(String, f64)> = Vec::new();
    for (name, pressure) in
        [("cpu", snap.psi.cpu), ("memory", snap.psi.memory), ("io", snap.psi.io)]
    {
        if let Some(p) = pressure {
            some_samples.push((format!("resource=\"{name}\""), p.some.avg10 / 100.0));
            if let Some(full) = p.full {
                full_samples.push((format!("resource=\"{name}\""), full.avg10 / 100.0));
            }
        }
    }
    metric(
        &mut out,
        "pressure_some_ratio",
        "Share of time with at least one task stalled, 10s average.",
        "gauge",
        &some_samples,
    );
    metric(
        &mut out,
        "pressure_full_ratio",
        "Share of time with every task stalled, 10s average.",
        "gauge",
        &full_samples,
    );

    metric(
        &mut out,
        "battery_percent",
        "Battery charge.",
        "gauge",
        &snap
            .power
            .batteries
            .iter()
            .map(|b| (format!("battery=\"{}\"", escape_label(&b.name)), b.percent))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "battery_watts",
        "Battery power flow; negative while discharging.",
        "gauge",
        &snap
            .power
            .batteries
            .iter()
            .filter_map(|b| {
                b.power_w.map(|w| (format!("battery=\"{}\"", escape_label(&b.name)), w))
            })
            .collect::<Vec<_>>(),
    );
    if let Some(w) = snap.power.rapl_watts {
        metric(&mut out, "package_watts", "CPU package power draw.", "gauge", &plain(w));
    }

    let gpu_label = |g: &crate::metrics::GpuInfo| format!("gpu=\"{}\"", escape_label(&g.name));
    metric(
        &mut out,
        "gpu_busy_percent",
        "GPU utilisation.",
        "gauge",
        &snap
            .gpus
            .iter()
            .filter_map(|g| g.busy_percent.map(|b| (gpu_label(g), b as f64)))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "gpu_memory_used_bytes",
        "GPU memory in use.",
        "gauge",
        &snap
            .gpus
            .iter()
            .filter_map(|g| g.vram_used.map(|v| (gpu_label(g), v as f64)))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "gpu_memory_total_bytes",
        "GPU memory installed.",
        "gauge",
        &snap
            .gpus
            .iter()
            .filter_map(|g| g.vram_total.map(|v| (gpu_label(g), v as f64)))
            .collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "gpu_temperature_celsius",
        "GPU temperature.",
        "gauge",
        &snap
            .gpus
            .iter()
            .filter_map(|g| g.temp_c.map(|t| (gpu_label(g), t as f64)))
            .collect::<Vec<_>>(),
    );

    // Per-namespace traffic, which is per container. The kernel accounts
    // bytes to a namespace rather than to a PID, so this is the finest
    // honest granularity available without eBPF — and the host's own
    // namespace is left out, because `network_*_bytes_per_second` above
    // already carries it per interface.
    let ns_label = |n: &crate::metrics::NetNamespace| match &n.container {
        Some(c) => format!("container=\"{}\"", escape_label(c)),
        // A namespace with no container is still worth a series — `ip netns`,
        // a sandboxed unit — and its inode is the only stable name it has.
        None => format!("container=\"\",netns=\"{}\"", n.id),
    };
    let containers: Vec<&crate::metrics::NetNamespace> =
        snap.netns.iter().filter(|n| !n.host).collect();
    metric(
        &mut out,
        "container_receive_bytes_per_second",
        "Receive rate of a container's network namespace.",
        "gauge",
        &containers.iter().map(|n| (ns_label(n), n.rx_bps)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "container_transmit_bytes_per_second",
        "Transmit rate of a container's network namespace.",
        "gauge",
        &containers.iter().map(|n| (ns_label(n), n.tx_bps)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "container_receive_bytes_total",
        "Bytes received in a container's network namespace since it was created.",
        "counter",
        &containers.iter().map(|n| (ns_label(n), n.rx_total as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "container_transmit_bytes_total",
        "Bytes transmitted in a container's network namespace since it was created.",
        "counter",
        &containers.iter().map(|n| (ns_label(n), n.tx_total as f64)).collect::<Vec<_>>(),
    );
    metric(
        &mut out,
        "container_processes",
        "Processes crabmon can see in a container's network namespace.",
        "gauge",
        &containers.iter().map(|n| (ns_label(n), n.procs as f64)).collect::<Vec<_>>(),
    );

    // The cgroup the exporter itself runs in. On a container host this is the
    // only view of the limit the process is actually up against; the host
    // totals above say nothing about it.
    if let Some(cg) = &snap.cgroup {
        let label = format!("path=\"{}\"", escape_label(&cg.path));
        if let Some(used) = cg.mem_current {
            metric(
                &mut out,
                "cgroup_memory_used_bytes",
                "Memory charged to this process's cgroup.",
                "gauge",
                &[(label.clone(), used as f64)],
            );
        }
        if let Some(max) = cg.mem_max {
            metric(
                &mut out,
                "cgroup_memory_limit_bytes",
                "This cgroup's memory limit.",
                "gauge",
                &[(label.clone(), max as f64)],
            );
        }
        if let Some(quota) = cg.cpu_quota_cores {
            metric(
                &mut out,
                "cgroup_cpu_quota_cores",
                "This cgroup's CPU quota, in cores.",
                "gauge",
                &[(label, quota)],
            );
        }
    }

    // Utilisation is the disk figure a byte counter cannot carry, and the one
    // an alert is usually written against.
    let utilised: Vec<(String, f64)> = snap
        .disks
        .iter()
        .filter_map(|d| d.util.map(|u| (format!("mount=\"{}\"", escape_label(&d.mount)), u)))
        .collect();
    if !utilised.is_empty() {
        metric(
            &mut out,
            "disk_utilisation_ratio",
            "Fraction of the interval this device had IO in flight.",
            "gauge",
            &utilised,
        );
    }
    let latency: Vec<(String, f64)> = snap
        .disks
        .iter()
        .filter_map(|d| {
            d.await_ms.map(|ms| (format!("mount=\"{}\"", escape_label(&d.mount)), ms / 1000.0))
        })
        .collect();
    if !latency.is_empty() {
        metric(
            &mut out,
            "disk_request_seconds",
            "Mean time from queueing to completion, per request.",
            "gauge",
            &latency,
        );
    }

    if top_procs > 0 {
        let proc_label = |p: &crate::metrics::ProcRow| {
            format!("pid=\"{}\",name=\"{}\"", p.pid, escape_label(&p.name))
        };
        // `snap.procs` arrives in the sampler's hash order, so taking the first
        // N exported an arbitrary handful — on a 2400-process box, every series
        // was an idle kernel thread and not one of the busiest tasks appeared.
        // `--once` and the recorder both sort before truncating; so must this.
        let mut ranked: Vec<&crate::metrics::ProcRow> = snap.procs.iter().collect();
        ranked.sort_by(|a, b| {
            b.cpu
                .partial_cmp(&a.cpu)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.mem.cmp(&a.mem))
                .then_with(|| a.pid.cmp(&b.pid))
        });
        let top: Vec<_> = ranked.into_iter().take(top_procs).collect();
        metric(
            &mut out,
            "process_cpu_percent",
            "Per-process CPU usage.",
            "gauge",
            &top.iter().map(|p| (proc_label(p), p.cpu as f64)).collect::<Vec<_>>(),
        );
        metric(
            &mut out,
            "process_memory_bytes",
            "Per-process resident memory.",
            "gauge",
            &top.iter().map(|p| (proc_label(p), p.mem as f64)).collect::<Vec<_>>(),
        );
        // Only ever populated when `[procs] fds` is on, and a process whose fd
        // table belongs to another user reports nothing rather than zero — so
        // the series are emitted only for the rows that actually have a count.
        let fds: Vec<(String, f64)> =
            top.iter().filter_map(|p| p.fds.map(|n| (proc_label(p), n as f64))).collect();
        if !fds.is_empty() {
            metric(
                &mut out,
                "process_open_fds",
                "Per-process open file descriptors.",
                "gauge",
                &fds,
            );
            let limits: Vec<(String, f64)> =
                top.iter().filter_map(|p| p.fd_limit.map(|n| (proc_label(p), n as f64))).collect();
            metric(
                &mut out,
                "process_max_fds",
                "Per-process soft limit on open file descriptors.",
                "gauge",
                &limits,
            );
        }
    }

    out
}

/// The first line of an HTTP request: `GET /metrics HTTP/1.1`.
pub fn parse_request_path(request_line: &str) -> Option<(&str, &str)> {
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?;
    let path = parts.next()?;
    Some((method, path.split('?').next().unwrap_or(path)))
}

pub fn http_response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Compare two secrets without leaking their contents through how long it
/// takes.
///
/// A naive `==` returns on the first differing byte, which over enough
/// requests tells a caller how much of a token it has guessed. The exporter
/// is reachable from the network by design, so the comparison is the obvious
/// place for that to matter.
pub fn secret_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    // The length is not a secret — the lengths of the two are compared in
    // constant time anyway by folding it into the difference.
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

/// Whether a request's `Authorization` header satisfies `required`.
///
/// `Bearer <token>` is what every scraper already knows how to send —
/// Prometheus has `bearer_token` and `bearer_token_file` in its scrape
/// config — so no new vocabulary is invented here.
pub fn authorized(header: Option<&str>, required: Option<&str>) -> bool {
    let Some(want) = required else { return true };
    let Some(given) = header else { return false };
    let given = given.trim();
    let token = given
        .strip_prefix("Bearer ")
        .or_else(|| given.strip_prefix("bearer "))
        .unwrap_or(given)
        .trim();
    secret_eq(token, want)
}

/// Route a request to its body. Split out so the routing is testable without a socket.
///
/// `token` is the secret a request must present, when one is configured.
/// `/healthz` stays open whatever the setting: it carries nothing about the
/// machine, and a liveness probe that needs a credential is a liveness probe
/// that gets switched off.
pub fn route(
    method: &str,
    path: &str,
    auth: Option<&str>,
    token: Option<&str>,
    body: impl FnOnce() -> String,
) -> String {
    if method != "GET" {
        return http_response("405 Method Not Allowed", "text/plain", "only GET is supported\n");
    }
    if path != "/healthz" && !authorized(auth, token) {
        let mut response = http_response("401 Unauthorized", "text/plain", "unauthorized\n");
        // The header a client needs to be told what to send back.
        response = response.replace(
            "Content-Type:",
            "WWW-Authenticate: Bearer realm=\"crabmon\"\r\nContent-Type:",
        );
        return response;
    }
    match path {
        "/metrics" => http_response("200 OK", "text/plain; version=0.0.4", &body()),
        "/" => http_response(
            "200 OK",
            "text/html",
            "<html><head><title>crabmon</title></head><body><a href=\"/metrics\">metrics</a></body></html>",
        ),
        "/healthz" => http_response("200 OK", "text/plain", "ok\n"),
        _ => http_response("404 Not Found", "text/plain", "not found\n"),
    }
}

/// Headers read before a request is answered. A scraper sends a handful;
/// this only exists so a client cannot make the exporter read for ever.
pub const MAX_HEADERS: usize = 64;

/// A client connection, whatever it arrived over.
///
/// A Unix socket is the natural companion to the bearer token: no port, no
/// network, and the filesystem does the access control — which is what a
/// local scraper on a machine that should not be listening actually wants.
pub trait Conn: Read + Write + Send + Sized + 'static {
    fn set_timeouts(&self, timeout: Duration);
    /// A second handle on the same connection, for bounding the read without
    /// giving up the write.
    fn try_clone_conn(&self) -> Option<Self>;
    /// Read through a shared reference. Both socket types implement `Read`
    /// for `&Self`; naming it here saves every caller a higher-ranked bound.
    fn read_shared(&self, buf: &mut [u8]) -> std::io::Result<usize>;
}

impl Conn for TcpStream {
    fn set_timeouts(&self, timeout: Duration) {
        let _ = self.set_read_timeout(Some(timeout));
        let _ = self.set_write_timeout(Some(timeout));
    }
    fn try_clone_conn(&self) -> Option<Self> {
        self.try_clone().ok()
    }
    fn read_shared(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        (&mut { self }).read(buf)
    }
}

#[cfg(unix)]
impl Conn for std::os::unix::net::UnixStream {
    fn set_timeouts(&self, timeout: Duration) {
        let _ = self.set_read_timeout(Some(timeout));
        let _ = self.set_write_timeout(Some(timeout));
    }
    fn try_clone_conn(&self) -> Option<Self> {
        self.try_clone().ok()
    }
    fn read_shared(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        (&mut { self }).read(buf)
    }
}

fn handle<S: Conn>(mut stream: S, token: Option<&str>, body: impl FnOnce() -> String) {
    // Both directions: a client that stops reading mid-response would otherwise
    // block the write just as effectively as one that never sends a request.
    stream.set_timeouts(CLIENT_TIMEOUT);
    let mut line = String::new();
    let peer = stream.try_clone_conn();
    // `.take(MAX_REQUEST_LINE)`: an unbounded `read_line` grows a `String`
    // until a newline arrives, and the read timeout only bounds how long a
    // *single* read may block — a client that keeps sending never trips it. On
    // loopback that is hundreds of megabytes of resident memory per connection
    // in the process whose whole job is to make memory pressure visible, and
    // `MAX_CONNECTIONS` of them at once.
    let reader =
        ReadRef(peer.as_ref().unwrap_or(&stream)).take(MAX_REQUEST_LINE * MAX_HEADERS as u64);
    let mut reader = BufReader::new(reader);
    if reader.read_line(&mut line).is_err() {
        return;
    }
    // Headers are only read when something is going to look at one: an
    // exporter with no token configured has no reason to read past the
    // request line, and reading it would only give a client more to say.
    let mut auth: Option<String> = None;
    if token.is_some() {
        for _ in 0..MAX_HEADERS {
            let mut header = String::new();
            match reader.read_line(&mut header) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => return,
            }
            if header.trim().is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                if name.trim().eq_ignore_ascii_case("authorization") {
                    auth = Some(value.trim().to_string());
                }
            }
        }
    }
    // A line that filled the budget without a newline is not a request line.
    let response = match parse_request_path(&line) {
        _ if line.len() as u64 >= MAX_REQUEST_LINE && !line.ends_with('\n') => {
            http_response("431 Request Header Fields Too Large", "text/plain", "too long\n")
        }
        Some((method, path)) => route(method, path, auth.as_deref(), token, body),
        None => http_response("400 Bad Request", "text/plain", "malformed request\n"),
    };
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// `&S` as a reader, so the request can be read through a borrowed handle
/// while the response is written through the owned one.
struct ReadRef<'a, S>(&'a S);

impl<S: Conn> Read for ReadRef<'_, S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read_shared(buf)
    }
}

fn reject<S: Conn>(mut stream: S) {
    stream.set_timeouts(CLIENT_TIMEOUT);
    let _ = stream
        .write_all(http_response("503 Service Unavailable", "text/plain", "busy\n").as_bytes());
}

/// Normalise an address to something bindable.
///
/// `:9100` is the conventional spelling for "every interface" and keeps meaning
/// that. A bare `9100` binds to loopback instead: process metrics name every
/// running command and its arguments, and the lazy spelling should not be the
/// one that publishes them to the network by accident.
pub fn normalise_addr(addr: &str) -> String {
    let addr = addr.trim();
    if let Some(port) = addr.strip_prefix(':') {
        return format!("0.0.0.0:{port}");
    }
    if !addr.contains(':') {
        return format!("127.0.0.1:{addr}");
    }
    addr.to_string()
}

/// Whether a bound address is reachable from off the machine, for the warning
/// printed at startup.
pub fn is_public(bind: &str) -> bool {
    bind.starts_with("0.0.0.0:") || bind.starts_with("[::]:") || bind.starts_with("*:")
}

/// Serve until the process is killed, re-sampling for each scrape.
///
/// `sample` is handed the real time since the previous scrape. Rates are
/// deltas of monotonic counters divided by that interval, so passing a fixed
/// configured interval instead — which is what this used to do — multiplied
/// every rate series by `scrape_interval / refresh_ms`.
///
/// Each connection is handled on its own thread. Doing the socket IO on the
/// accept loop meant one client that connected and never sent a request stopped
/// the exporter answering anybody; a timeout alone only bounds how long that
/// lasts. Sampling stays behind a mutex, so concurrent scrapes still see
/// consistent counter deltas rather than racing each other's `prev` state.
pub fn serve(
    addr: &str,
    top_procs: usize,
    mut sample: impl FnMut(Duration) -> Snapshot + Send + 'static,
) -> std::io::Result<()> {
    serve_rendered(addr, None, move |elapsed| render(&sample(elapsed), top_procs))
}

/// As `serve`, but the caller renders the body.
///
/// `--serve --alerts` evaluates its rules on the sample it just took and
/// exports their state with it, which means the thing that turns a snapshot
/// into a response needs the snapshot, not just the numbers in it.
pub fn serve_rendered(
    addr: &str,
    token: Option<String>,
    render_body: impl FnMut(Duration) -> String + Send + 'static,
) -> std::io::Result<()> {
    if let Some(path) = unix_socket_path(addr) {
        return serve_unix(path, token, render_body);
    }
    let bind = normalise_addr(addr);
    let listener = TcpListener::bind(&bind)?;
    eprintln!("crabmon: serving metrics on http://{bind}/metrics");
    match (&token, is_public(&bind)) {
        (Some(_), _) => eprintln!("crabmon: scrapes need an Authorization: Bearer header"),
        (None, true) => eprintln!(
            "crabmon: {bind} is reachable from the network with no token set; \
             per-process metrics include command lines"
        ),
        (None, false) => {}
    }
    accept_loop(listener.incoming(), token, render_body)
}

/// Whether an address names a Unix socket rather than a host and port.
///
/// `unix:/run/crabmon.sock` is explicit; a bare path is accepted too,
/// because anything starting with `/` or `./` is not a port and guessing
/// right is better than an error about an unparseable address.
pub fn unix_socket_path(addr: &str) -> Option<&str> {
    let addr = addr.trim();
    if let Some(path) = addr.strip_prefix("unix:") {
        return (!path.is_empty()).then_some(path);
    }
    (addr.starts_with('/') || addr.starts_with("./")).then_some(addr)
}

/// Serve over a Unix socket: no port, no network, and the filesystem does
/// the access control.
#[cfg(unix)]
fn serve_unix(
    path: &str,
    token: Option<String>,
    render_body: impl FnMut(Duration) -> String + Send + 'static,
) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;

    // A socket left behind by a killed process would otherwise make every
    // later start fail with "address in use". Removed only when nothing is
    // actually listening on it, so two exporters cannot steal each other's.
    if std::path::Path::new(path).exists() && std::os::unix::net::UnixStream::connect(path).is_err()
    {
        let _ = std::fs::remove_file(path);
    }
    let listener = UnixListener::bind(path)?;
    // Not the umask's business: a scrape carries every command line on the
    // machine, and a permissive umask would hand that to everyone.
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660));
    eprintln!("crabmon: serving metrics on unix:{path}");
    if token.is_some() {
        eprintln!("crabmon: scrapes need an Authorization: Bearer header");
    }
    accept_loop(listener.incoming(), token, render_body)
}

#[cfg(not(unix))]
fn serve_unix(
    _path: &str,
    _token: Option<String>,
    _render_body: impl FnMut(Duration) -> String + Send + 'static,
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unix sockets are not available on this platform",
    ))
}

/// The accept loop, over whatever kind of socket the listener is.
fn accept_loop<S: Conn>(
    incoming: impl Iterator<Item = std::io::Result<S>>,
    token: Option<String>,
    render_body: impl FnMut(Duration) -> String + Send + 'static,
) -> std::io::Result<()> {
    // Shared rather than cloned per connection: it never changes, and a
    // secret copied once per request is a secret in more places than it
    // needs to be.
    let token = token.map(std::sync::Arc::<str>::from);

    let sampler = Arc::new(Mutex::new((render_body, std::time::Instant::now())));
    let live = Arc::new(AtomicUsize::new(0));

    for stream in incoming {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("crabmon: accept failed: {e}");
                continue;
            }
        };
        if live.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
            reject(stream);
            continue;
        }
        live.fetch_add(1, Ordering::Relaxed);
        let sampler = Arc::clone(&sampler);
        // Released by `Drop`, not by a statement at the end of the thread
        // body: a panic anywhere inside `handle` — or in the sampling closure
        // it calls — skipped that statement and leaked the slot for good.
        // Sixteen of those and the exporter answers 503 to everything, for
        // the rest of its life, while still accepting connections.
        //
        // Dropping the closure also covers a thread that could not be spawned
        // at all, since `spawn` takes it by value.
        let slot = ConnectionSlot(Arc::clone(&live));
        let token = token.clone();
        let _ = std::thread::Builder::new().name("crabmon-http".into()).spawn(move || {
            let _slot = slot;
            handle(stream, token.as_deref(), || {
                let mut guard = match sampler.lock() {
                    Ok(g) => g,
                    // A panic in a previous handler must not take the exporter
                    // down with it; the sampler state is still usable.
                    Err(poisoned) => poisoned.into_inner(),
                };
                let (render_body, last) = &mut *guard;
                let elapsed = last.elapsed();
                *last = std::time::Instant::now();
                render_body(elapsed)
            });
        });
    }
    Ok(())
}

/// One live connection's share of `MAX_CONNECTIONS`, given back when dropped.
struct ConnectionSlot(Arc<AtomicUsize>);

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CpuSample, DiskRow, MemSample, NetIface, ProcRow};

    fn snap() -> Snapshot {
        Snapshot {
            cpu: CpuSample { per_core: vec![10.0, 90.0], freq_mhz: vec![800, 3800] },
            mem: MemSample { total: 1000, used: 400, available: 600, ..Default::default() },
            procs: vec![
                ProcRow { pid: 1, name: "init".into(), cpu: 1.0, mem: 100, ..Default::default() },
                ProcRow {
                    pid: 2,
                    name: "we\"ird".into(),
                    cpu: 2.0,
                    mem: 200,
                    ..Default::default()
                },
            ],
            disks: vec![DiskRow {
                mount: "/".into(),
                total: 100,
                used: 50,
                inodes_total: 1000,
                inodes_used: 250,
                ..Default::default()
            }],
            nets: vec![NetIface { name: "eth0".into(), rx_bps: 1024.0, ..Default::default() }],
            ..Default::default()
        }
    }

    /// `crabmon_up{host="db-1"} 1` for a machine nobody can reach reads as
    /// "db-1 is fine", which is the opposite of what happened.
    #[test]
    fn an_unreachable_host_in_a_fleet_is_not_up() {
        let down = render_host(&Snapshot::default(), 0, &[], false);
        assert!(down.contains("crabmon_up 0"), "{down}");
        let live = render_host(&snap(), 0, &[], true);
        assert!(live.contains("crabmon_up 1"), "{live}");
    }

    /// A fleet had a dashboard and no metrics: crabmon could watch several
    /// hosts over SSH and could export what it measured, and the two could
    /// not meet.
    #[test]
    fn several_hosts_merge_into_one_scrape_with_a_host_label() {
        let a = render(&snap(), 2);
        let b = render(&snap(), 2);
        let merged = merge_hosts(&[("web-1".into(), a), ("db-1".into(), b)]);

        assert!(merged.contains("crabmon_up{host=\"web-1\"} 1"), "{merged}");
        assert!(merged.contains("crabmon_up{host=\"db-1\"} 1"), "{merged}");
        // Labels a series already had survive alongside it.
        assert!(merged.contains("crabmon_cpu_core_usage_percent{host=\"web-1\",core=\"0\"}"));

        // A repeated HELP line loses the *whole* scrape, not one host's.
        for family in ["crabmon_up", "crabmon_cpu_usage_percent"] {
            let helps =
                merged.lines().filter(|l| l.starts_with(&format!("# HELP {family} "))).count();
            assert_eq!(helps, 1, "{family} declared {helps} times:\n{merged}");
        }
    }

    /// Every sample of a family has to sit under that family's own
    /// HELP/TYPE pair, not be scattered through the scrape.
    #[test]
    fn merged_families_stay_grouped() {
        let merged =
            merge_hosts(&[("a".into(), render(&snap(), 1)), ("b".into(), render(&snap(), 1))]);
        let mut seen: Vec<&str> = Vec::new();
        for line in merged.lines() {
            if line.starts_with('#') {
                continue;
            }
            let family = line.split(['{', ' ']).next().unwrap_or_default();
            if seen.last() != Some(&family) {
                assert!(!seen.contains(&family), "{family} appears in two places:\n{merged}");
                seen.push(family);
            }
        }
    }

    #[test]
    fn a_host_name_is_escaped_like_any_other_label() {
        let merged = merge_hosts(&[("we\"ird".into(), "crabmon_up 1\n".into())]);
        assert!(merged.contains("host=\"we\\\"ird\""), "{merged}");
    }

    #[test]
    fn the_output_is_valid_exposition_format() {
        let text = render(&snap(), 10);
        for line in text.lines() {
            if line.starts_with('#') {
                assert!(
                    line.starts_with("# HELP ") || line.starts_with("# TYPE "),
                    "bad comment: {line}"
                );
                continue;
            }
            let value = line.rsplit(' ').next().unwrap();
            assert!(value.parse::<f64>().is_ok(), "unparseable sample: {line}");
        }
        // Every metric must be preceded by its HELP and TYPE.
        assert_eq!(
            text.matches("# HELP ").count(),
            text.matches("# TYPE ").count(),
            "HELP and TYPE must be paired"
        );
    }

    /// Prometheus rejects an entire scrape that repeats a label set within a
    /// family, so one duplicated series took every other metric with it. A box
    /// with two identical GPUs — the ordinary ML rig — exported nothing at all.
    #[test]
    fn a_repeated_label_set_does_not_take_the_whole_scrape_down_with_it() {
        let twins = |name: &str, busy: f32| crate::metrics::GpuInfo {
            name: name.into(),
            vendor: "NVIDIA".into(),
            busy_percent: Some(busy),
            vram_used: Some(1024),
            vram_total: Some(2048),
            ..Default::default()
        };
        let snap = Snapshot {
            gpus: vec![
                twins("NVIDIA GeForce RTX 4090", 10.0),
                twins("NVIDIA GeForce RTX 4090", 90.0),
            ],
            // Two NVMe drives can present the same sensor label too.
            sensors: vec![
                crate::metrics::Sensor {
                    label: "nvme Composite".into(),
                    temp: 40.0,
                    critical: None,
                },
                crate::metrics::Sensor {
                    label: "nvme Composite".into(),
                    temp: 55.0,
                    critical: None,
                },
            ],
            ..snap()
        };
        let text = render(&snap, 10);

        for family in ["crabmon_gpu_busy_percent", "crabmon_temperature_celsius"] {
            let mut labels: Vec<&str> = text
                .lines()
                .filter(|l| l.starts_with(&format!("{family}{{")))
                .map(|l| l.split_once(' ').map(|(k, _)| k).unwrap_or(l))
                .collect();
            let before = labels.len();
            assert!(before > 0, "{family} is missing entirely:\n{text}");
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), before, "{family} emits a duplicate series:\n{text}");
        }

        // ...and the rest of the scrape is still there, which is the point.
        assert!(text.contains("crabmon_up 1"), "{text}");
        assert!(text.contains("crabmon_memory_total_bytes 1000"), "{text}");
    }

    /// Deduplication must keep the first sample rather than silently dropping
    /// a family down to nothing, and must not touch distinct label sets.
    #[test]
    fn deduplication_keeps_the_first_sample_and_leaves_distinct_ones_alone() {
        let mut out = String::new();
        metric(
            &mut out,
            "thing",
            "help",
            "gauge",
            &[("a=\"1\"".into(), 1.0), ("a=\"2\"".into(), 2.0), ("a=\"1\"".into(), 99.0)],
        );
        assert!(out.contains("crabmon_thing{a=\"1\"} 1\n"), "{out}");
        assert!(out.contains("crabmon_thing{a=\"2\"} 2\n"), "{out}");
        assert!(!out.contains("99"), "the repeat should be dropped, not emitted:\n{out}");
        assert_eq!(out.matches("# HELP ").count(), 1);
    }

    /// Per-container network, which is what the kernel will actually account.
    /// The host's own namespace is excluded: its traffic is already exported
    /// per interface, and a second copy of it labelled as a container would be
    /// double counting in anyone's dashboard.
    #[test]
    fn container_network_is_exported_and_the_host_namespace_is_not() {
        use crate::metrics::NetNamespace;

        let snap = Snapshot {
            netns: vec![
                NetNamespace {
                    id: 4_026_531_992,
                    host: true,
                    rx_bps: 999_999.0,
                    tx_bps: 999_999.0,
                    ..Default::default()
                },
                NetNamespace {
                    id: 4_026_532_001,
                    container: Some("abc123".into()),
                    rx_bps: 2048.0,
                    tx_bps: 1024.0,
                    rx_total: 50_000,
                    tx_total: 25_000,
                    procs: 4,
                    ..Default::default()
                },
                // A namespace with no container: `ip netns`, or a sandboxed
                // unit. Its inode is the only name it has.
                NetNamespace { id: 4_026_532_002, rx_bps: 16.0, ..Default::default() },
            ],
            ..snap()
        };
        let text = render(&snap, 10);

        assert!(
            text.contains("crabmon_container_receive_bytes_per_second{container=\"abc123\"} 2048"),
            "{text}"
        );
        assert!(text.contains("crabmon_container_processes{container=\"abc123\"} 4"), "{text}");
        assert!(
            text.contains("netns=\"4026532002\""),
            "an unnamed namespace still needs a series:\n{text}"
        );
        assert!(!text.contains("999999"), "the host namespace must not be exported again:\n{text}");
    }

    #[test]
    fn a_machine_with_no_containers_exports_no_container_series() {
        // An empty family with only HELP and TYPE is noise in every scrape.
        let text = render(&snap(), 10);
        assert!(!text.contains("crabmon_container_"), "{text}");
    }

    /// crabmon measures things Prometheus cannot: pressure-stall averages, the
    /// busiest device's utilisation, how close a process is to *its own*
    /// descriptor limit. All of it was alertable in the TUI and invisible to
    /// anything scraping it, so the rules most worth paging on were the ones
    /// that could not reach a pager.
    #[test]
    fn alert_rules_are_exported_with_their_state_and_the_value_that_caused_it() {
        use crate::alerts::{AlertKind, RuleState};

        let states = vec![
            RuleState {
                name: "cpu-saturated".into(),
                kind: AlertKind::Cpu,
                threshold: 90.0,
                value: Some(97.5),
                active: true,
            },
            RuleState {
                name: "nginx gone".into(),
                kind: AlertKind::Proc,
                threshold: 1.0,
                value: Some(3.0),
                active: false,
            },
        ];
        let text = render_with_alerts(&snap(), 10, &states);

        assert!(
            text.contains("crabmon_alert_active{rule=\"cpu-saturated\",kind=\"cpu\"} 1"),
            "{text}"
        );
        // A rule that is *not* firing still needs a series: a gauge that only
        // appears once something is wrong cannot be alerted on, and a
        // dashboard built on it has nothing to draw until the incident.
        assert!(
            text.contains("crabmon_alert_active{rule=\"nginx gone\",kind=\"proc\"} 0"),
            "{text}"
        );
        assert!(
            text.contains("crabmon_alert_value{rule=\"cpu-saturated\",kind=\"cpu\"} 97.5"),
            "{text}"
        );
        assert!(
            text.contains("crabmon_alert_threshold{rule=\"cpu-saturated\",kind=\"cpu\"} 90"),
            "{text}"
        );
    }

    /// A rule measuring something this machine does not report — a `gpu` rule
    /// with no GPU — has no value. Exporting 0 would read as "idle", which is
    /// a different claim from "not measured".
    #[test]
    fn a_rule_with_nothing_to_measure_exports_no_value_rather_than_zero() {
        use crate::alerts::{AlertKind, RuleState};

        let states = vec![RuleState {
            name: "gpu-hot".into(),
            kind: AlertKind::Gpu,
            threshold: 90.0,
            value: None,
            active: false,
        }];
        let text = render_with_alerts(&snap(), 0, &states);
        assert!(text.contains("crabmon_alert_active{rule=\"gpu-hot\""), "{text}");
        assert!(!text.contains("crabmon_alert_value{rule=\"gpu-hot\""), "{text}");
        // The threshold is still a fact about the rule.
        assert!(text.contains("crabmon_alert_threshold{rule=\"gpu-hot\""), "{text}");
    }

    #[test]
    fn a_run_with_no_alerts_exports_no_alert_families() {
        assert!(!render(&snap(), 10).contains("crabmon_alert_"));
    }

    #[test]
    fn core_metrics_are_present_with_the_expected_values() {
        let text = render(&snap(), 10);
        assert!(text.contains("crabmon_up 1"), "{text}");
        assert!(text.contains("crabmon_cpu_usage_percent 50"), "{text}");
        assert!(text.contains("crabmon_cpu_core_usage_percent{core=\"1\"} 90"), "{text}");
        assert!(text.contains("crabmon_memory_used_bytes 400"), "{text}");
        assert!(text.contains("crabmon_disk_used_bytes{mount=\"/\"} 50"), "{text}");
        assert!(text.contains("crabmon_disk_inodes_used_ratio{mount=\"/\"} 0.25"), "{text}");
        assert!(text.contains("crabmon_network_receive_bytes_per_second{interface=\"eth0\"} 1024"));
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label(r#"a"b\c"#), r#"a\"b\\c"#);
        let text = render(&snap(), 10);
        assert!(text.contains(r#"name="we\"ird""#), "{text}");
    }

    #[test]
    fn per_process_series_are_capped() {
        let mut s = snap();
        s.procs = (0..500).map(|i| ProcRow { pid: i, ..Default::default() }).collect();
        let text = render(&s, 5);
        assert_eq!(text.matches("crabmon_process_cpu_percent{").count(), 5);
        // Zero means "do not export per-process series at all".
        assert!(!render(&s, 0).contains("crabmon_process_cpu_percent"));
    }

    /// Exporting zero where the log cannot be read would claim the machine
    /// is healthy on every host with `dmesg_restrict=1`.
    #[test]
    fn kernel_events_are_exported_but_not_invented() {
        let mut snap = snap();
        snap.kernel = vec![crate::metrics::kmsg::parse_record(
            "4,1,1000000,-;Out of memory: Killed process 1 (init)",
        )
        .unwrap()];
        let out = render(&snap, 10);
        assert!(out.contains("crabmon_kernel_events{kind=\"oom\"} 1"), "{out}");
        assert!(out.contains("crabmon_kernel_events{kind=\"segfault\"} 0"), "{out}");

        snap.kernel.clear();
        snap.kernel_unreadable = Some("needs root".into());
        let out = render(&snap, 10);
        assert!(!out.contains("kernel_events"), "an unreadable log reported as healthy:\n{out}");
    }

    #[test]
    fn absent_hardware_produces_no_series_rather_than_zeros() {
        let text = render(&Snapshot::default(), 0);
        assert!(!text.contains("battery_percent"), "no battery must mean no series");
        assert!(!text.contains("gpu_busy_percent"));
        assert!(!text.contains("package_watts"));
        assert!(text.contains("crabmon_up 1"), "but the exporter is still up");
    }

    #[test]
    fn pressure_is_exported_as_a_ratio_not_a_percentage() {
        let mut s = snap();
        s.psi.io = Some(crate::metrics::psi::Pressure {
            some: crate::metrics::psi::PressureLine { avg10: 2.55, ..Default::default() },
            full: None,
        });
        let text = render(&s, 0);
        assert!(text.contains("crabmon_pressure_some_ratio{resource=\"io\"} 0.0255"), "{text}");
    }

    /// No port, no network, and the filesystem does the access control.
    #[cfg(unix)]
    #[test]
    fn metrics_can_be_served_over_a_unix_socket() {
        use std::io::Read;
        use std::os::unix::net::UnixStream;

        let dir = std::env::temp_dir().join(format!("crabmon-sock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.sock");
        let addr = format!("unix:{}", path.display());

        std::thread::spawn({
            let addr = addr.clone();
            move || {
                let _ = serve_rendered(&addr, None, |_| "socket body\n".into());
            }
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while UnixStream::connect(&path).is_err() {
            assert!(std::time::Instant::now() < deadline, "the socket never appeared");
            std::thread::sleep(Duration::from_millis(10));
        }

        let mut c = UnixStream::connect(&path).expect("connect");
        c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        c.write_all(b"GET /metrics HTTP/1.1\r\n\r\n").unwrap();
        let mut response = String::new();
        let _ = c.read_to_string(&mut response);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("socket body"), "{response}");

        // A scrape carries every command line on the machine; a permissive
        // umask must not hand that to everyone.
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode & 0o007, 0, "the socket is world-accessible: {mode:o}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A socket left behind by a killed exporter must not stop the next one
    /// starting, and must not let it steal a live one.
    #[cfg(unix)]
    #[test]
    fn a_stale_socket_file_is_replaced_but_a_live_one_is_not() {
        use std::os::unix::net::UnixListener;

        let dir = std::env::temp_dir().join(format!("crabmon-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Stale: a plain file where a socket used to be.
        let stale = dir.join("stale.sock");
        std::fs::write(&stale, "").unwrap();
        let addr = format!("unix:{}", stale.display());
        std::thread::spawn({
            let addr = addr.clone();
            move || {
                let _ = serve_rendered(&addr, None, |_| String::new());
            }
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::os::unix::net::UnixStream::connect(&stale).is_err() {
            assert!(std::time::Instant::now() < deadline, "a stale socket was not replaced");
            std::thread::sleep(Duration::from_millis(10));
        }

        // Live: something is already listening, so binding must fail rather
        // than unlink it.
        let live = dir.join("live.sock");
        let _held = UnixListener::bind(&live).unwrap();
        let err = serve_rendered(&format!("unix:{}", live.display()), None, |_| String::new());
        assert!(err.is_err(), "a live socket was stolen");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_address_says_whether_it_is_a_socket_or_a_port() {
        assert_eq!(unix_socket_path("unix:/run/crabmon.sock"), Some("/run/crabmon.sock"));
        assert_eq!(unix_socket_path("/run/crabmon.sock"), Some("/run/crabmon.sock"));
        assert_eq!(unix_socket_path("./crabmon.sock"), Some("./crabmon.sock"));
        assert_eq!(unix_socket_path(":9100"), None);
        assert_eq!(unix_socket_path("9100"), None);
        assert_eq!(unix_socket_path("127.0.0.1:9100"), None);
        assert_eq!(unix_socket_path("unix:"), None);
    }

    /// The exporter publishes every command line on the machine. Binding it
    /// off loopback was one flag away and there was no way to ask for a
    /// password.
    #[test]
    fn a_configured_token_is_required_for_metrics() {
        let token = Some("s3cret");
        let deny = route("GET", "/metrics", None, token, || "body".into());
        assert!(deny.starts_with("HTTP/1.1 401"), "{deny}");
        assert!(deny.contains("WWW-Authenticate"), "a 401 has to say what to send: {deny}");
        assert!(!deny.contains("body"), "the body leaked past the check: {deny}");

        let wrong = route("GET", "/metrics", Some("Bearer nope"), token, || "body".into());
        assert!(wrong.starts_with("HTTP/1.1 401"), "{wrong}");

        let ok = route("GET", "/metrics", Some("Bearer s3cret"), token, || "body".into());
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
        assert!(ok.contains("body"));
    }

    /// A liveness probe that needs a credential is a liveness probe that
    /// gets switched off, and `/healthz` says nothing about the machine.
    #[test]
    fn health_stays_open_even_with_a_token_set() {
        let out = route("GET", "/healthz", None, Some("s3cret"), String::new);
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    }

    #[test]
    fn with_no_token_configured_everything_is_open_as_before() {
        let out = route("GET", "/metrics", None, None, || "body".into());
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    }

    #[test]
    fn the_header_is_accepted_in_the_spellings_clients_send() {
        for header in ["Bearer s3cret", "bearer s3cret", " Bearer  s3cret ", "s3cret"] {
            assert!(authorized(Some(header), Some("s3cret")), "{header} was refused");
        }
        for header in ["Bearer s3cre", "Bearer s3crett", "", "Basic s3cret"] {
            assert!(!authorized(Some(header), Some("s3cret")), "{header} was accepted");
        }
    }

    /// A comparison that returns on the first differing byte tells a caller
    /// how much of the token it has guessed.
    #[test]
    fn secrets_compare_without_short_circuiting() {
        assert!(secret_eq("abc", "abc"));
        assert!(!secret_eq("abc", "abd"));
        assert!(!secret_eq("abc", "abcd"));
        assert!(!secret_eq("", "a"));
        assert!(secret_eq("", ""));
    }

    /// End to end over a socket, because the header has to be *read* as well
    /// as checked, and the request line alone used to be all that was read.
    #[test]
    fn a_scrape_over_a_socket_must_carry_the_token() {
        use std::io::Read;
        use std::net::TcpStream;

        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap().to_string();
        drop(probe);

        let bound = addr.clone();
        std::thread::spawn(move || {
            let _ = serve_rendered(&bound, Some("s3cret".into()), |_| "metrics body\n".into());
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while TcpStream::connect(&addr).is_err() {
            assert!(std::time::Instant::now() < deadline, "exporter never came up");
            std::thread::sleep(Duration::from_millis(10));
        }

        let ask = |request: &str| -> String {
            let mut c = TcpStream::connect(&addr).expect("connect");
            c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            c.write_all(request.as_bytes()).unwrap();
            let mut response = String::new();
            let _ = c.read_to_string(&mut response);
            response
        };

        let anonymous = ask("GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(anonymous.starts_with("HTTP/1.1 401"), "{anonymous}");
        assert!(!anonymous.contains("metrics body"), "the body leaked: {anonymous}");

        let authed =
            ask("GET /metrics HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n\r\n");
        assert!(authed.starts_with("HTTP/1.1 200"), "{authed}");
        assert!(authed.contains("metrics body"), "{authed}");

        assert!(ask("GET /healthz HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn requests_are_routed_and_bad_methods_rejected() {
        assert_eq!(parse_request_path("GET /metrics HTTP/1.1"), Some(("GET", "/metrics")));
        assert_eq!(parse_request_path("GET /metrics?x=1 HTTP/1.1"), Some(("GET", "/metrics")));
        assert_eq!(parse_request_path(""), None);

        assert!(
            route("GET", "/metrics", None, None, || "body".into()).starts_with("HTTP/1.1 200 OK")
        );
        assert!(route("GET", "/healthz", None, None, String::new).contains("200 OK"));
        assert!(route("GET", "/nope", None, None, String::new).contains("404"));
        assert!(route("POST", "/metrics", None, None, String::new).contains("405"));
    }

    #[test]
    fn one_silent_client_cannot_stop_the_exporter_answering_others() {
        use std::io::Read;
        use std::net::TcpStream;

        // Bind ourselves so the test never guesses a port, then let `serve`
        // take the listener's address.
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap().to_string();
        drop(probe);

        let bound = addr.clone();
        std::thread::spawn(move || {
            let _ = serve(&bound, 0, |_| Snapshot::default());
        });
        // Wait for the listener rather than sleeping a guess.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while TcpStream::connect(&addr).is_err() {
            assert!(std::time::Instant::now() < deadline, "exporter never came up");
            std::thread::sleep(Duration::from_millis(10));
        }

        // Connections that open and say nothing. Before connections were
        // handled off the accept loop, one of these was enough to wedge the
        // exporter for everybody, forever.
        let _silent: Vec<TcpStream> =
            (0..4).filter_map(|_| TcpStream::connect(&addr).ok()).collect();
        std::thread::sleep(Duration::from_millis(50));

        let mut client = TcpStream::connect(&addr).expect("connect");
        client.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        client.write_all(b"GET /healthz HTTP/1.1\r\n\r\n").unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).expect("the exporter was blocked");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    }

    /// A client that sends and sends without ever writing a newline. The read
    /// timeout does not bound this: it limits how long one read may *block*,
    /// and a sender that keeps the socket busy never blocks it. Unbounded,
    /// A handler that panics must give its connection slot back. It used to
    /// decrement the counter with a statement at the end of the thread body,
    /// which a panic skipped: `MAX_CONNECTIONS` panics and the exporter
    /// answered 503 to everything for the rest of its life.
    #[test]
    fn a_panicking_scrape_does_not_cost_the_exporter_a_connection_slot() {
        use std::io::Read;
        use std::net::TcpStream;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap().to_string();
        drop(probe);

        // Panic on every scrape but the last round, so the slots have to come
        // back for the final request to be answered at all.
        let scrapes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&scrapes);
        let bound = addr.clone();
        std::thread::spawn(move || {
            let _ = serve_rendered(&bound, None, move |_| {
                if counter.fetch_add(1, Ordering::SeqCst) < MAX_CONNECTIONS * 2 {
                    panic!("scrape blew up");
                }
                "ok\n".to_string()
            });
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while TcpStream::connect(&addr).is_err() {
            assert!(std::time::Instant::now() < deadline, "exporter never came up");
            std::thread::sleep(Duration::from_millis(10));
        }

        let scrape = |addr: &str| -> String {
            let mut c = TcpStream::connect(addr).expect("connect");
            c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            c.write_all(b"GET /metrics HTTP/1.1\r\n\r\n").unwrap();
            let mut response = String::new();
            let _ = c.read_to_string(&mut response);
            response
        };

        for _ in 0..MAX_CONNECTIONS * 2 {
            scrape(&addr);
        }
        // Give the last panicking handler time to unwind before asking again.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let response = scrape(&addr);
            if response.starts_with("HTTP/1.1 200 OK") {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the exporter never recovered its connection slots: {response}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// `read_line` grows a `String` for as long as the client cares to type.
    #[test]
    fn a_client_that_never_sends_a_newline_is_cut_off_rather_than_buffered() {
        use std::net::TcpStream;

        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap().to_string();
        drop(probe);

        let bound = addr.clone();
        std::thread::spawn(move || {
            let _ = serve(&bound, 0, |_| Snapshot::default());
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while TcpStream::connect(&addr).is_err() {
            assert!(std::time::Instant::now() < deadline, "exporter never came up");
            std::thread::sleep(Duration::from_millis(10));
        }

        let mut flood = TcpStream::connect(&addr).expect("connect");
        flood.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
        let chunk = vec![b'A'; 64 * 1024];
        let mut sent = 0usize;
        // Far past the cap; the exporter should stop reading long before this.
        while sent < 4 * 1024 * 1024 {
            if flood.write_all(&chunk).is_err() {
                break; // the exporter answered and closed, which is the point
            }
            sent += chunk.len();
        }

        // ...and it is still answering everybody else.
        let mut client = TcpStream::connect(&addr).expect("connect");
        client.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        client.write_all(b"GET /healthz HTTP/1.1\r\n\r\n").unwrap();
        let mut response = String::new();
        std::io::Read::read_to_string(&mut client, &mut response).expect("the exporter was busy");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    }

    #[test]
    fn responses_carry_an_accurate_content_length() {
        let body = "hello\n";
        let response = http_response("200 OK", "text/plain", body);
        assert!(response.contains(&format!("Content-Length: {}", body.len())));
        assert!(response.ends_with(body));
    }

    #[test]
    fn addresses_are_normalised_to_something_bindable() {
        // `:port` keeps its conventional "all interfaces" meaning...
        assert_eq!(normalise_addr(":9100"), "0.0.0.0:9100");
        assert_eq!(normalise_addr(" :9100 "), "0.0.0.0:9100");
        // ...but a bare port stays on loopback: per-process series carry
        // command lines, and the shortest spelling should not publish them.
        assert_eq!(normalise_addr("9100"), "127.0.0.1:9100");
        assert_eq!(normalise_addr("127.0.0.1:9100"), "127.0.0.1:9100");
        assert_eq!(normalise_addr("0.0.0.0:9100"), "0.0.0.0:9100");

        assert!(is_public("0.0.0.0:9100"));
        assert!(is_public("[::]:9100"));
        assert!(!is_public("127.0.0.1:9100"));
    }

    #[test]
    fn a_scrape_carries_the_series_every_panel_draws() {
        let mut s = snap();
        s.cgroup = Some(crate::metrics::CgroupInfo {
            path: "/docker/abc".into(),
            containerized: true,
            mem_current: Some(100),
            mem_max: Some(200),
            cpu_quota_cores: Some(1.5),
        });
        s.gpus = vec![crate::metrics::GpuInfo {
            name: "amdgpu".into(),
            busy_percent: Some(42.0),
            vram_used: Some(1024),
            vram_total: Some(4096),
            temp_c: Some(55.0),
            ..Default::default()
        }];
        s.nets[0].errors_rx = 7;
        let text = render(&s, 0);
        for needle in [
            "crabmon_cgroup_memory_used_bytes",
            "crabmon_cgroup_memory_limit_bytes",
            "crabmon_cgroup_cpu_quota_cores",
            "crabmon_gpu_memory_used_bytes",
            "crabmon_gpu_memory_total_bytes",
            "crabmon_gpu_temperature_celsius",
            "crabmon_disk_inodes_total",
            "crabmon_network_receive_errors_total",
            "crabmon_network_receive_bytes_total",
        ] {
            assert!(text.contains(needle), "{needle} missing from the scrape");
        }
        // A host with none of that hardware still exports no empty series.
        assert!(!render(&Snapshot::default(), 0).contains("cgroup_memory"));
    }

    /// Prometheus aborts an entire scrape on a repeated HELP line for one
    /// metric name. Emitting the pair inside the per-resource loop gave three,
    /// so on any host with /proc/pressure not one crabmon series was ingested.
    #[test]
    fn each_metric_family_declares_itself_exactly_once() {
        use crate::metrics::psi::{Pressure, PressureLine, PsiSample};
        let line = PressureLine { avg10: 1.0, ..Default::default() };
        let p = Pressure { some: line, full: Some(line) };
        let mut snap = snap();
        snap.psi = PsiSample { cpu: Some(p), memory: Some(p), io: Some(p) };

        let out = render(&snap, 5);
        let mut help: Vec<&str> = out
            .lines()
            .filter_map(|l| l.strip_prefix("# HELP "))
            .map(|l| l.split(' ').next().unwrap_or(""))
            .collect();
        let before = help.len();
        help.sort_unstable();
        help.dedup();
        assert_eq!(help.len(), before, "a metric family declared HELP more than once");

        // All three resources still reach the scrape, as labelled samples.
        for resource in ["cpu", "memory", "io"] {
            assert!(
                out.contains(&format!("crabmon_pressure_some_ratio{{resource=\"{resource}\"}}")),
                "{resource} is missing from the scrape"
            );
        }
    }

    /// `snap.procs` arrives in the sampler's hash order, so taking the first N
    /// exported an arbitrary handful of idle tasks and none of the busiest.
    #[test]
    fn the_exported_processes_are_the_busiest_ones() {
        let mut snap = snap();
        snap.procs = (0..50)
            .map(|i| ProcRow {
                pid: 1000 + i,
                name: format!("p{i}"),
                cpu: i as f32,
                ..Default::default()
            })
            .collect();
        snap.procs.rotate_left(17); // whatever order the map handed back

        let out = render(&snap, 3);
        let exported: Vec<&str> =
            out.lines().filter(|l| l.starts_with("crabmon_process_cpu_percent{")).collect();
        assert_eq!(exported.len(), 3);
        for (i, name) in ["p49", "p48", "p47"].iter().enumerate() {
            assert!(exported[i].contains(&format!("name=\"{name}\"")), "{:?}", exported[i]);
        }
    }
}
