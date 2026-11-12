//! What a container is *called*, from the files its runtime keeps.
//!
//! The cgroup path only carries an id, so the grouped view used to read
//! `3f9a1c2b4d5e` where everybody thinks `postgres`. Names are read from the
//! files every runtime already writes, and, failing those, asked of the
//! runtime's API socket when this user can already connect to it. An id
//! nothing names stays an id.
//!
//! - Docker: `<root>/containers/<id>/config.v2.json`, `"Name": "/postgres"`.
//!   Root-only on most hosts; rootless Docker keeps it under `$HOME`.
//! - Podman (and anything else on containers/storage):
//!   `overlay-containers/containers.json`, an array of `{id, names}`.
//!   Rootless Podman's copy is the user's own, so it always resolves.
//! - Kubernetes (containerd, CRI-O): the kubelet's
//!   `/var/log/containers/<pod>_<namespace>_<container>-<id>.log` symlinks,
//!   whose directory is world-readable, so pods resolve without root too.
//! - The Docker Engine API (`/var/run/docker.sock`, rootless Docker's and
//!   Podman's compatible sockets): one read-only `GET /containers/json`.
//!   This is the path for the ordinary non-root member of the `docker`
//!   group, who can run `docker ps` but cannot read `/var/lib/docker`.
//!   Connecting needs exactly the permission `docker ps` does, so crabmon
//!   asks for nothing that user does not already have, and a socket it cannot
//!   open is skipped.
//!
//! The id stays the identity everywhere — grouping, filters, alert targets,
//! exporter labels — and the name is what is drawn next to it. A rename or a
//! name reused by a new container cannot then merge two things into one.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// How often an id nothing names may trigger another look. A container whose
/// files crabmon cannot read must not cost a directory walk every sample.
const RESCAN_EVERY: Duration = Duration::from_secs(10);

/// Ids are shown, keyed and compared as their first twelve characters, which
/// is what `procgroup::container_of` produces and what `docker ps` prints.
const SHORT_ID: usize = 12;

/// Where each runtime's files live.
#[derive(Debug, Clone, Default)]
pub struct Roots {
    /// Directories holding one `<id>/config.v2.json` per container.
    pub docker: Vec<PathBuf>,
    /// `containers.json` files.
    pub podman: Vec<PathBuf>,
    /// Directories of kubelet log symlinks.
    pub k8s_logs: Vec<PathBuf>,
    /// Docker-Engine-compatible API sockets.
    pub sockets: Vec<PathBuf>,
}

impl Roots {
    /// The standard locations, rootful and the current user's rootless ones.
    pub fn system() -> Self {
        let home = dirs::home_dir();
        let under_home = |rel: &str| home.as_ref().map(|h| h.join(rel));
        Roots {
            docker: [
                Some(PathBuf::from("/var/lib/docker/containers")),
                under_home(".local/share/docker/containers"),
            ]
            .into_iter()
            .flatten()
            .collect(),
            podman: [
                Some(PathBuf::from(
                    "/var/lib/containers/storage/overlay-containers/containers.json",
                )),
                under_home(".local/share/containers/storage/overlay-containers/containers.json"),
            ]
            .into_iter()
            .flatten()
            .collect(),
            k8s_logs: vec![PathBuf::from("/var/log/containers")],
            sockets: {
                let runtime = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from);
                [
                    Some(PathBuf::from("/var/run/docker.sock")),
                    runtime.as_ref().map(|r| r.join("docker.sock")),
                    Some(PathBuf::from("/run/podman/podman.sock")),
                    runtime.as_ref().map(|r| r.join("podman/podman.sock")),
                ]
                .into_iter()
                .flatten()
                .collect()
            },
        }
    }
}

/// Container id → name, refreshed when an id turns up that it cannot name.
#[derive(Debug)]
pub struct ContainerNames {
    roots: Roots,
    names: HashMap<String, String>,
    last_scan: Option<Instant>,
}

impl Default for ContainerNames {
    fn default() -> Self {
        Self::new(Roots::system())
    }
}

impl ContainerNames {
    pub fn new(roots: Roots) -> Self {
        Self { roots, names: HashMap::new(), last_scan: None }
    }

    /// The name for a (short or full) id, if any runtime's files gave one.
    pub fn get(&self, id: &str) -> Option<&str> {
        self.names.get(short(id)).map(String::as_str)
    }

    /// Make sure every id in `live` that can be named is, and forget the ones
    /// no longer running so the map stays the size of what is on screen.
    pub fn resolve(&mut self, live: &HashSet<String>) {
        if !cfg!(target_os = "linux") {
            return;
        }
        let unknown = live.iter().any(|id| !self.names.contains_key(short(id)));
        let due = self.last_scan.is_none_or(|t| t.elapsed() >= RESCAN_EVERY);
        if unknown && due {
            self.scan();
            self.last_scan = Some(Instant::now());
        }
        let live_short: HashSet<&str> = live.iter().map(|id| short(id)).collect();
        self.names.retain(|id, _| live_short.contains(id.as_str()));
    }

    fn scan(&mut self) {
        for dir in &self.roots.docker {
            let Ok(entries) = fs::read_dir(dir) else { continue };
            for entry in entries.flatten() {
                let id = entry.file_name().to_string_lossy().to_string();
                if !is_container_id(&id) || self.names.contains_key(short(&id)) {
                    continue;
                }
                let config = entry.path().join("config.v2.json");
                if let Some(name) = fs::read_to_string(config).ok().and_then(|c| docker_name(&c)) {
                    self.names.insert(short(&id).to_string(), name);
                }
            }
        }
        for file in &self.roots.podman {
            let Ok(content) = fs::read_to_string(file) else { continue };
            for (id, name) in podman_names(&content) {
                self.names.entry(short(&id).to_string()).or_insert(name);
            }
        }
        // Last, and only while something is still unnamed: the files are
        // free, and a socket round trip is not.
        #[cfg(unix)]
        for sock in &self.roots.sockets {
            for (id, name) in api_names(sock) {
                self.names.entry(short(&id).to_string()).or_insert(name);
            }
        }
        for dir in &self.roots.k8s_logs {
            let Ok(entries) = fs::read_dir(dir) else { continue };
            for entry in entries.flatten() {
                if let Some((id, name)) = k8s_log_name(&entry.file_name().to_string_lossy()) {
                    self.names.entry(short(&id).to_string()).or_insert(name);
                }
            }
        }
    }
}

fn short(id: &str) -> &str {
    id.get(..SHORT_ID).unwrap_or(id)
}

fn is_container_id(s: &str) -> bool {
    s.len() >= SHORT_ID && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// `"Name": "/postgres"` out of a Docker `config.v2.json`.
pub fn docker_name(config: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(config).ok()?;
    let name = v.get("Name")?.as_str()?.trim_start_matches('/');
    (!name.is_empty()).then(|| name.to_string())
}

/// `(id, first name)` pairs out of a containers/storage `containers.json`.
pub fn podman_names(content: &str) -> Vec<(String, String)> {
    let Ok(serde_json::Value::Array(items)) = serde_json::from_str(content) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|c| {
            let id = c.get("id")?.as_str()?;
            let name = c.get("names")?.as_array()?.first()?.as_str()?;
            is_container_id(id).then(|| (id.to_string(), name.to_string()))
        })
        .collect()
}

/// How long one API request may take. The scan runs on the sampling thread,
/// so a wedged daemon must cost a bounded stall at most once per rescan.
#[cfg(unix)]
const API_TIMEOUT: Duration = Duration::from_millis(500);

/// `(id, name)` pairs from a Docker-Engine-compatible socket, or nothing when
/// it cannot be opened or does not answer sensibly.
#[cfg(unix)]
fn api_names(sock: &std::path::Path) -> Vec<(String, String)> {
    use std::io::{Read, Write};
    let Ok(mut stream) = std::os::unix::net::UnixStream::connect(sock) else {
        return Vec::new();
    };
    let _ = stream.set_read_timeout(Some(API_TIMEOUT));
    let _ = stream.set_write_timeout(Some(API_TIMEOUT));
    // HTTP/1.0, so the server closes when done and never chunks the body.
    if stream.write_all(b"GET /containers/json HTTP/1.0\r\nHost: localhost\r\n\r\n").is_err() {
        return Vec::new();
    }
    let mut response = Vec::new();
    // A container list is kilobytes; a megabyte bounds a misbehaving peer.
    let _ = stream.take(1 << 20).read_to_end(&mut response);
    api_response_names(&String::from_utf8_lossy(&response))
}

/// Parse an HTTP response carrying a `/containers/json` body.
pub fn api_response_names(response: &str) -> Vec<(String, String)> {
    let Some((head, body)) = response.split_once("\r\n\r\n") else { return Vec::new() };
    if head.split_whitespace().nth(1) != Some("200") {
        return Vec::new();
    }
    let Ok(serde_json::Value::Array(items)) = serde_json::from_str(body) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|c| {
            let id = c.get("Id")?.as_str()?;
            let name = c.get("Names")?.as_array()?.first()?.as_str()?.trim_start_matches('/');
            (is_container_id(id) && !name.is_empty()).then(|| (id.to_string(), name.to_string()))
        })
        .collect()
}

/// `(id, "pod/container")` out of a kubelet log symlink name,
/// `<pod>_<namespace>_<container>-<64 hex id>.log`.
///
/// None of the three names may contain `_`, and the id is the part after the
/// last `-`, so the split is unambiguous even though pod names are full of
/// dashes.
pub fn k8s_log_name(file: &str) -> Option<(String, String)> {
    let stem = file.strip_suffix(".log")?;
    let (rest, id) = stem.rsplit_once('-')?;
    if id.len() != 64 || !is_container_id(id) {
        return None;
    }
    let mut parts = rest.splitn(3, '_');
    let (pod, _namespace, container) = (parts.next()?, parts.next()?, parts.next()?);
    if pod.is_empty() || container.is_empty() {
        return None;
    }
    Some((id.to_string(), format!("{pod}/{container}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lays out a fake runtime directory.
    fn write(path: &std::path::Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    const ID: &str = "3f9a1c2b4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8";

    #[test]
    fn docker_names_lose_their_leading_slash() {
        assert_eq!(
            docker_name(r#"{"ID":"x","Name":"/postgres","State":{}}"#).as_deref(),
            Some("postgres")
        );
        assert_eq!(docker_name(r#"{"Name":"/"}"#), None);
        assert_eq!(docker_name("not json"), None);
    }

    #[test]
    fn podman_takes_the_first_name_and_skips_malformed_entries() {
        let json = format!(
            r#"[{{"id":"{ID}","names":["web","alias"]}},{{"id":"zz","names":["bad"]}},{{"id":"{ID}"}}]"#
        );
        assert_eq!(podman_names(&json), vec![(ID.to_string(), "web".to_string())]);
        assert!(podman_names("{}").is_empty());
    }

    #[test]
    fn the_engine_api_answer_is_read_only_when_it_is_a_200() {
        let ok = format!(
            "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n[{{\"Id\":\"{ID}\",\"Names\":[\"/clickhouse\"]}}]"
        );
        assert_eq!(api_response_names(&ok), vec![(ID.to_string(), "clickhouse".to_string())]);
        assert!(api_response_names("HTTP/1.0 403 Forbidden\r\n\r\n[]").is_empty());
        assert!(api_response_names("garbage").is_empty());
    }

    /// The whole round trip against a stand-in daemon on a real socket.
    /// Linux only: elsewhere `resolve` never connects, and the stand-in would
    /// wait for ever.
    #[cfg(target_os = "linux")]
    #[test]
    fn names_are_asked_of_the_socket_when_the_files_are_unreadable() {
        use std::io::{Read, Write};
        let dir = std::env::temp_dir().join(format!("crabmon-api-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("docker.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut req = [0u8; 256];
            let n = conn.read(&mut req).unwrap();
            assert!(String::from_utf8_lossy(&req[..n]).starts_with("GET /containers/json "));
            let body = format!(r#"[{{"Id":"{ID}","Names":["/clickhouse"]}}]"#);
            conn.write_all(format!("HTTP/1.0 200 OK\r\n\r\n{body}").as_bytes()).unwrap();
        });
        let mut names = ContainerNames::new(Roots { sockets: vec![sock], ..Default::default() });
        names.resolve(&[ID[..12].to_string()].into_iter().collect());
        server.join().unwrap();
        assert_eq!(names.get(ID), Some("clickhouse"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn kubelet_log_names_split_on_the_last_dash_not_the_first() {
        let file = format!("api-7d4b9c-x2k8p_prod_nginx-proxy-{ID}.log");
        assert_eq!(
            k8s_log_name(&file),
            Some((ID.to_string(), "api-7d4b9c-x2k8p/nginx-proxy".into()))
        );
        assert_eq!(
            k8s_log_name("api_prod_nginx-abc123.log"),
            None,
            "a short id is not a container"
        );
        assert_eq!(k8s_log_name(&format!("nounderscores-{ID}.log")), None);
        assert_eq!(k8s_log_name("syslog"), None);
    }

    #[test]
    fn names_come_from_every_runtime_and_unknown_ids_stay_unnamed() {
        let dir = std::env::temp_dir().join(format!("crabmon-containers-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let docker_id = ID;
        let podman_id = "a".repeat(64);
        let k8s_id = "b".repeat(64);
        write(
            &dir.join("docker").join(docker_id).join("config.v2.json"),
            r#"{"Name":"/postgres"}"#,
        );
        write(&dir.join("podman.json"), &format!(r#"[{{"id":"{podman_id}","names":["redis"]}}]"#));
        write(&dir.join("logs").join(format!("web-1_default_app-{k8s_id}.log")), "");

        let mut names = ContainerNames::new(Roots {
            docker: vec![dir.join("docker")],
            podman: vec![dir.join("podman.json")],
            k8s_logs: vec![dir.join("logs")],
            sockets: vec![dir.join("no-such.sock")],
        });
        let live: HashSet<String> =
            [&docker_id[..12], &podman_id[..12], &k8s_id[..12], "cccccccccccc"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        names.resolve(&live);
        if cfg!(target_os = "linux") {
            assert_eq!(names.get(docker_id), Some("postgres"), "a full id finds its short form");
            assert_eq!(names.get(&podman_id[..12]), Some("redis"));
            assert_eq!(names.get(&k8s_id[..12]), Some("web-1/app"));
        }
        assert_eq!(names.get("cccccccccccc"), None);

        // A container that has gone is forgotten.
        names.resolve(&HashSet::new());
        assert_eq!(names.get(docker_id), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
