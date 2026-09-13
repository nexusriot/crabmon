//! A Prometheus exporter.
//!
//! `crabmon --serve :9100` samples on the configured interval and answers
//! `/metrics` over HTTP. The server is hand-rolled against `TcpListener`
//! because the whole contract is one route, one method and a text body — a
//! framework would be more dependency than feature.

use std::io::{BufRead, BufReader, Write};
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

/// Escape a Prometheus label value.
pub fn escape_label(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

fn metric(out: &mut String, name: &str, help: &str, kind: &str, samples: &[(String, f64)]) {
    if samples.is_empty() {
        return;
    }
    out.push_str(&format!("# HELP crabmon_{name} {help}\n"));
    out.push_str(&format!("# TYPE crabmon_{name} {kind}\n"));
    for (labels, value) in samples {
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
    let mut out = String::new();

    metric(&mut out, "up", "Always 1; the exporter is answering.", "gauge", &plain(1.0));
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

/// Route a request to its body. Split out so the routing is testable without a socket.
pub fn route(method: &str, path: &str, body: impl FnOnce() -> String) -> String {
    if method != "GET" {
        return http_response("405 Method Not Allowed", "text/plain", "only GET is supported\n");
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

fn handle(mut stream: TcpStream, body: impl FnOnce() -> String) {
    // Both directions: a client that stops reading mid-response would otherwise
    // block the write just as effectively as one that never sends a request.
    let _ = stream.set_read_timeout(Some(CLIENT_TIMEOUT));
    let _ = stream.set_write_timeout(Some(CLIENT_TIMEOUT));
    let mut line = String::new();
    let peer = stream.try_clone();
    if BufReader::new(peer.as_ref().unwrap_or(&stream)).read_line(&mut line).is_err() {
        return;
    }
    let response = match parse_request_path(&line) {
        Some((method, path)) => route(method, path, body),
        None => http_response("400 Bad Request", "text/plain", "malformed request\n"),
    };
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn reject(mut stream: TcpStream) {
    let _ = stream.set_write_timeout(Some(CLIENT_TIMEOUT));
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
    sample: impl FnMut(Duration) -> Snapshot + Send + 'static,
) -> std::io::Result<()> {
    let bind = normalise_addr(addr);
    let listener = TcpListener::bind(&bind)?;
    eprintln!("crabmon: serving metrics on http://{bind}/metrics");
    if is_public(&bind) {
        eprintln!(
            "crabmon: {bind} is reachable from the network; \
             per-process metrics include command lines"
        );
    }

    let sampler = Arc::new(Mutex::new((sample, std::time::Instant::now())));
    let live = Arc::new(AtomicUsize::new(0));

    for stream in listener.incoming() {
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
        let counter = Arc::clone(&live);
        let spawned = std::thread::Builder::new().name("crabmon-http".into()).spawn(move || {
            handle(stream, || {
                let mut guard = match sampler.lock() {
                    Ok(g) => g,
                    // A panic in a previous handler must not take the exporter
                    // down with it; the sampler state is still usable.
                    Err(poisoned) => poisoned.into_inner(),
                };
                let (sample, last) = &mut *guard;
                let elapsed = last.elapsed();
                *last = std::time::Instant::now();
                let snap = sample(elapsed);
                render(&snap, top_procs)
            });
            counter.fetch_sub(1, Ordering::Relaxed);
        });
        if spawned.is_err() {
            live.fetch_sub(1, Ordering::Relaxed);
        }
    }
    Ok(())
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

    #[test]
    fn requests_are_routed_and_bad_methods_rejected() {
        assert_eq!(parse_request_path("GET /metrics HTTP/1.1"), Some(("GET", "/metrics")));
        assert_eq!(parse_request_path("GET /metrics?x=1 HTTP/1.1"), Some(("GET", "/metrics")));
        assert_eq!(parse_request_path(""), None);

        assert!(route("GET", "/metrics", || "body".into()).starts_with("HTTP/1.1 200 OK"));
        assert!(route("GET", "/healthz", String::new).contains("200 OK"));
        assert!(route("GET", "/nope", String::new).contains("404"));
        assert!(route("POST", "/metrics", String::new).contains("405"));
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
