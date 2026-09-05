//! A Prometheus exporter.
//!
//! `crabmon --serve :9100` samples on the configured interval and answers
//! `/metrics` over HTTP. The server is hand-rolled against `TcpListener`
//! because the whole contract is one route, one method and a text body — a
//! framework would be more dependency than feature.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

use crate::metrics::Snapshot;

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
        "temperature_celsius",
        "Sensor temperature.",
        "gauge",
        &snap
            .sensors
            .iter()
            .map(|s| (format!("sensor=\"{}\"", escape_label(&s.label)), s.temp as f64))
            .collect::<Vec<_>>(),
    );

    for (name, pressure) in
        [("cpu", snap.psi.cpu), ("memory", snap.psi.memory), ("io", snap.psi.io)]
    {
        if let Some(p) = pressure {
            metric(
                &mut out,
                "pressure_some_ratio",
                "Share of time with at least one task stalled, 10s average.",
                "gauge",
                &[(format!("resource=\"{name}\""), p.some.avg10 / 100.0)],
            );
            if let Some(full) = p.full {
                metric(
                    &mut out,
                    "pressure_full_ratio",
                    "Share of time with every task stalled, 10s average.",
                    "gauge",
                    &[(format!("resource=\"{name}\""), full.avg10 / 100.0)],
                );
            }
        }
    }

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

    metric(
        &mut out,
        "gpu_busy_percent",
        "GPU utilisation.",
        "gauge",
        &snap
            .gpus
            .iter()
            .filter_map(|g| {
                g.busy_percent.map(|b| (format!("gpu=\"{}\"", escape_label(&g.name)), b as f64))
            })
            .collect::<Vec<_>>(),
    );

    if top_procs > 0 {
        let proc_label = |p: &crate::metrics::ProcRow| {
            format!("pid=\"{}\",name=\"{}\"", p.pid, escape_label(&p.name))
        };
        let top: Vec<_> = snap.procs.iter().take(top_procs).collect();
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

/// Normalise `:9100`, `9100` and `0.0.0.0:9100` to a bindable address.
pub fn normalise_addr(addr: &str) -> String {
    let addr = addr.trim();
    if let Some(port) = addr.strip_prefix(':') {
        return format!("0.0.0.0:{port}");
    }
    if !addr.contains(':') {
        return format!("0.0.0.0:{addr}");
    }
    addr.to_string()
}

/// Serve until the process is killed, re-sampling for each scrape.
pub fn serve(
    addr: &str,
    top_procs: usize,
    mut sample: impl FnMut() -> Snapshot,
) -> std::io::Result<()> {
    let bind = normalise_addr(addr);
    let listener = TcpListener::bind(&bind)?;
    eprintln!("crabmon: serving metrics on http://{bind}/metrics");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let snap = sample();
                handle(stream, || render(&snap, top_procs));
            }
            Err(e) => eprintln!("crabmon: accept failed: {e}"),
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
    fn responses_carry_an_accurate_content_length() {
        let body = "hello\n";
        let response = http_response("200 OK", "text/plain", body);
        assert!(response.contains(&format!("Content-Length: {}", body.len())));
        assert!(response.ends_with(body));
    }

    #[test]
    fn addresses_are_normalised_to_something_bindable() {
        assert_eq!(normalise_addr(":9100"), "0.0.0.0:9100");
        assert_eq!(normalise_addr("9100"), "0.0.0.0:9100");
        assert_eq!(normalise_addr("127.0.0.1:9100"), "127.0.0.1:9100");
        assert_eq!(normalise_addr(" :9100 "), "0.0.0.0:9100");
    }
}
