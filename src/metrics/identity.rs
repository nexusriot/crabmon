//! What a process is *right now*, as opposed to what it was when it was first
//! seen.
//!
//! `exec` replaces the program a process is running without changing its PID,
//! its parent or its start time. The sampler underneath crabmon reads a
//! process's name only when it first encounters the PID, and treats an
//! unchanged start time as "same process, nothing to re-read" — so a process
//! that execs keeps the name, command line, executable and working directory
//! it had beforehand, for as long as it lives.
//!
//! That is not an exotic case. It is how a wrapper script hands off to the
//! real program, how a container entrypoint becomes the service, how `sudo`,
//! `nohup`, `ssh-agent` and every systemd unit with a shell in front of it
//! start. All of them were reported under the wrapper's name indefinitely:
//! the process table showed `sh`, `service:` grouping and `cmd:` filters
//! matched the wrapper's argv, and `--watch nginx` could never match an nginx
//! that had been exec'd into — a monitor quietly failing at the one thing it
//! was asked to do.
//!
//! `/proc/<pid>/comm` is one small file and always current, so it is read
//! every tick for every process; the heavier re-reads happen only for the
//! processes it shows have actually changed, which on any ordinary machine is
//! none of them.

/// The process's current name: `/proc/<pid>/comm`.
///
/// This is the same string the sampler calls a process's name — field 2 of
/// `/proc/<pid>/stat`, truncated by the kernel to 15 characters — so the two
/// compare directly, and a process that renamed itself with `prctl` is caught
/// alongside one that exec'd.
pub fn name(pid: u32) -> Option<String> {
    read_line(&format!("/proc/{pid}/comm"))
}

/// The full command line, NUL-separated in the file and space-separated here,
/// to match how the sampler joins it.
pub fn cmdline(pid: u32) -> Option<String> {
    let raw = read_file(&format!("/proc/{pid}/cmdline"))?;
    let joined = raw.split('\0').filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
    // A kernel thread has an empty cmdline; that is not a reason to blank a
    // command line that was read successfully before.
    (!joined.is_empty()).then_some(joined)
}

/// The resolved executable path, or `None` when the link cannot be read —
/// another user's process, or one that has already exited.
pub fn exe(pid: u32) -> Option<String> {
    read_link(&format!("/proc/{pid}/exe"))
}

pub fn cwd(pid: u32) -> Option<String> {
    read_link(&format!("/proc/{pid}/cwd"))
}

#[cfg(target_os = "linux")]
fn read_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

#[cfg(not(target_os = "linux"))]
fn read_file(_path: &str) -> Option<String> {
    None
}

fn read_line(path: &str) -> Option<String> {
    let text = read_file(path)?;
    let line = text.trim_end_matches('\n');
    (!line.is_empty()).then(|| line.to_string())
}

#[cfg(target_os = "linux")]
fn read_link(path: &str) -> Option<String> {
    std::fs::read_link(path).ok().map(|p| p.to_string_lossy().to_string())
}

#[cfg(not(target_os = "linux"))]
fn read_link(_path: &str) -> Option<String> {
    None
}

/// What a process became, cached so the correction is paid once.
///
/// The sampler's cached name never catches up — it is only read when a PID is
/// first seen — so a process that has exec'd goes on disagreeing with
/// `/proc/<pid>/comm` for the rest of its life. Without this, every one of
/// them would re-read its command line, executable and working directory on
/// every single tick, which is three extra reads a second each for a case
/// that, once seen, never changes again.
///
/// Keyed by PID *and* start time, like the cgroup and descriptor-limit caches,
/// so a recycled PID cannot inherit the previous process's identity. The comm
/// is part of the key too: a process that renames itself twice is corrected
/// twice.
#[derive(Debug, Default)]
pub struct ExecCache {
    seen: std::collections::HashMap<u32, Entry>,
}

#[derive(Debug, Clone)]
struct Entry {
    start_time: u64,
    comm: String,
    cmd: String,
    exe: String,
    cwd: String,
}

/// What a process is now running, as far as it can be established.
pub struct Identity {
    pub name: String,
    pub cmd: String,
    pub exe: String,
    pub cwd: String,
}

impl ExecCache {
    /// Reconcile the sampler's idea of a process with the kernel's.
    ///
    /// `was` is what the sampler reports — the name it recorded when it first
    /// met this PID, and the command line, executable and working directory it
    /// read at the same time. One small read says whether any of that is still
    /// true; the rest are only read when it is not.
    pub fn resolve(&mut self, pid: u32, start_time: u64, was: Identity) -> Identity {
        let Some(comm) = name(pid) else {
            // Unreadable: the process exited mid-sweep, or its `/proc` entry
            // is not ours to read. The sampler's copy is the best there is.
            return was;
        };
        if comm == was.name {
            self.seen.remove(&pid);
            return was;
        }
        if let Some(e) = self.seen.get(&pid) {
            if e.start_time == start_time && e.comm == comm {
                return Identity {
                    name: e.comm.clone(),
                    cmd: e.cmd.clone(),
                    exe: e.exe.clone(),
                    cwd: e.cwd.clone(),
                };
            }
        }
        let fresh = Identity {
            cmd: cmdline(pid).unwrap_or(was.cmd),
            exe: exe(pid).unwrap_or(was.exe),
            cwd: cwd(pid).unwrap_or(was.cwd),
            name: comm.clone(),
        };
        self.seen.insert(
            pid,
            Entry {
                start_time,
                comm,
                cmd: fresh.cmd.clone(),
                exe: fresh.exe.clone(),
                cwd: fresh.cwd.clone(),
            },
        );
        fresh
    }

    /// Drop entries for processes that have exited.
    pub fn retain_live(&mut self, live: &dyn Fn(u32) -> bool) {
        self.seen.retain(|pid, _| live(*pid));
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_can_read_its_own_identity() {
        let me = std::process::id();
        let name = name(me).expect("every live process has a comm");
        assert!(!name.is_empty());
        // The kernel truncates to 15 characters; nothing here may be longer.
        assert!(name.len() <= 15, "{name} is longer than TASK_COMM_LEN - 1");
        // ...and it is a prefix of the test binary's name, which is what makes
        // it comparable with the name the sampler reports.
        let argv0 = std::env::args().next().unwrap_or_default();
        let file = argv0.rsplit('/').next().unwrap_or_default();
        assert!(file.starts_with(&name), "comm {name} does not match argv[0] {file}");

        assert!(cmdline(me).is_some_and(|c| c.contains(file)));
        assert!(exe(me).is_some_and(|e| e.contains(file)));
        assert!(cwd(me).is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_that_is_not_there_reports_nothing_rather_than_an_empty_name() {
        // An empty string would be a name, and would overwrite a real one.
        let gone = u32::MAX;
        assert_eq!(name(gone), None);
        assert_eq!(cmdline(gone), None);
        assert_eq!(exe(gone), None);
        assert_eq!(cwd(gone), None);
    }

    /// A kernel thread has an empty `cmdline`. Reading that as "the command
    /// line is now blank" would wipe the one the sampler had.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_empty_command_line_is_absent_rather_than_blank() {
        // pid 2 is kthreadd on Linux; skip where it is not readable as one.
        if let Some(cmd) = cmdline(2) {
            assert!(!cmd.is_empty(), "an empty cmdline must not come back as Some(\"\")");
        }
    }

    /// A process whose name the sampler already has right costs one read and
    /// nothing else — which is what makes checking every process every tick
    /// affordable in the first place.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_that_has_not_changed_is_left_exactly_as_the_sampler_had_it() {
        let me = std::process::id();
        let comm = name(me).expect("comm");
        let mut cache = ExecCache::default();
        let got = cache.resolve(
            me,
            1_700_000_000,
            Identity {
                name: comm.clone(),
                cmd: "what the sampler read".into(),
                exe: "/sampler/exe".into(),
                cwd: "/sampler/cwd".into(),
            },
        );
        assert_eq!(got.name, comm);
        assert_eq!(got.cmd, "what the sampler read", "nothing should have been re-read");
        assert_eq!(got.exe, "/sampler/exe");
        assert_eq!(got.cwd, "/sampler/cwd");
        assert!(cache.is_empty(), "an unchanged process needs no cache entry");
    }

    /// The sampler's name never catches up, so the disagreement is permanent.
    /// Re-reading three files per tick for the rest of the process's life is
    /// what the cache exists to avoid.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_changed_process_is_re_read_once_and_then_remembered() {
        let me = std::process::id();
        let stale = || Identity {
            name: "sh".into(),
            cmd: "sh -c something".into(),
            exe: "/bin/sh".into(),
            cwd: "/".into(),
        };
        let mut cache = ExecCache::default();

        let first = cache.resolve(me, 1_700_000_000, stale());
        assert_ne!(first.name, "sh", "the kernel's answer must win");
        assert_ne!(first.cmd, "sh -c something", "the command line follows the name");
        assert_eq!(cache.len(), 1);

        let again = cache.resolve(me, 1_700_000_000, stale());
        assert_eq!(again.name, first.name);
        assert_eq!(again.cmd, first.cmd);
        assert_eq!(again.exe, first.exe);
        assert_eq!(cache.len(), 1, "the second look must not add an entry");
    }

    /// PIDs are recycled. A cached identity surviving onto an unrelated
    /// process would describe a program that is not running.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_recycled_pid_does_not_inherit_the_previous_process_identity() {
        let me = std::process::id();
        let stale = || Identity {
            name: "sh".into(),
            cmd: "sh -c something".into(),
            exe: "/bin/sh".into(),
            cwd: "/".into(),
        };
        let mut cache = ExecCache::default();
        cache.resolve(me, 1_700_000_000, stale());

        // Same PID, different process: the start time says so.
        let after = cache.resolve(me, 1_900_000_000, stale());
        assert_eq!(after.name, name(me).unwrap(), "it must be read again, not recalled");

        cache.retain_live(&|_| false);
        assert!(cache.is_empty(), "entries for dead processes must not accumulate");
    }
}
