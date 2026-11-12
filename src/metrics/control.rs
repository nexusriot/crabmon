//! Live process identity checks at the boundary where controls reach the OS.

use std::io;

#[cfg(target_os = "linux")]
pub fn start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // comm is parenthesised and may itself contain spaces or parentheses.
    stat.rsplit_once(')')?.1.split_whitespace().nth(19)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
pub fn start_ticks(_pid: u32) -> Option<u64> {
    None
}

/// `(start time in ticks, major faults)` out of one `/proc/<pid>/stat`, so
/// the identity check the sampler already makes per process also yields the
/// fault counter, at no extra read.
pub fn parse_stat_counters(stat: &str) -> Option<(u64, u64)> {
    // comm is parenthesised and may itself contain spaces or parentheses, so
    // fields are counted from after the *last* `)`: index 0 is the state
    // (field 3), majflt is field 12 and starttime field 22.
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    Some((fields.get(19)?.parse().ok()?, fields.get(9)?.parse().ok()?))
}

#[cfg(target_os = "linux")]
pub fn stat_counters(pid: u32) -> Option<(u64, u64)> {
    parse_stat_counters(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

#[cfg(not(target_os = "linux"))]
pub fn stat_counters(_pid: u32) -> Option<(u64, u64)> {
    None
}

fn gone() -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, "process is gone or its identity changed")
}

/// Pins signal delivery to a process incarnation. Other controls revalidate
/// immediately before their PID-based syscall; Linux has no pidfd equivalent
/// of setpriority, sched_setaffinity or ioprio_set.
pub struct LiveProcess {
    pid: u32,
    #[cfg(target_os = "linux")]
    ticks: u64,
    #[cfg(target_os = "linux")]
    fd: std::os::fd::OwnedFd,
    #[cfg(not(target_os = "linux"))]
    started: u64,
}

impl LiveProcess {
    pub fn open(pid: u32, started: u64, ticks: Option<u64>) -> io::Result<Self> {
        if pid == 0 || pid > i32::MAX as u32 {
            return Err(gone());
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::FromRawFd;
            // SAFETY: pidfd_open returns a new owned descriptor or -1.
            let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
            if raw < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: the successful syscall transferred this fresh fd to us.
            let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(raw as i32) };
            // Refuse identities without tick precision on Linux, including
            // processes that appeared during the sampler's refresh window.
            let current = start_ticks(pid).ok_or_else(gone)?;
            if ticks != Some(current) {
                return Err(gone());
            }
            let _ = started;
            let process = Self { pid, ticks: current, fd };
            process.check()?;
            Ok(process)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = ticks;
            let process = Self { pid, started };
            process.check()?;
            Ok(process)
        }
    }

    pub fn check(&self) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let mut poll =
                libc::pollfd { fd: self.fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: a valid one-element pollfd array, nonblocking poll.
            let rc = unsafe { libc::poll(&mut poll, 1, 0) };
            if rc != 0 || start_ticks(self.pid) != Some(self.ticks) {
                return Err(gone());
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            // A platform that cannot be read cannot vouch for the PID, and
            // acting on an unvouched PID is the one thing this guard exists
            // to refuse.
            let mut sys = sysinfo::System::new().map_err(|_| gone())?;
            let pid = sysinfo::Pid::from_u32(self.pid);
            sys.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                true,
                sysinfo::ProcessRefreshKind::nothing(),
            );
            if !sys.process(pid).is_some_and(|p| p.start_time() == self.started) {
                return Err(gone());
            }
        }
        Ok(())
    }

    pub fn apply<T>(&self, action: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
        self.check()?;
        action()
    }

    #[cfg(unix)]
    pub fn signal(&self, signal: nix::sys::signal::Signal) -> io::Result<()> {
        self.check()?;
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: valid pidfd, signal number, null siginfo, no flags.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    self.fd.as_raw_fd(),
                    signal as i32,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            };
            if rc < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
        }
        #[cfg(not(target_os = "linux"))]
        super::sysinfo_source::send_signal(self.pid, signal)
    }
}

#[cfg(test)]
mod stat_tests {
    use super::parse_stat_counters;

    #[test]
    fn fault_and_start_counters_survive_a_command_name_full_of_parentheses() {
        // Field 12 (majflt) is 37 and field 22 (starttime) is 8765; the comm
        // contains ") (" to make a naive split land on the wrong field.
        let stat = "4242 (evil) (name) S 1 4242 4242 0 -1 4194560 1500 0 37 0 \
                    10 5 0 0 20 0 1 0 8765 12345678 900 18446744073709551615";
        assert_eq!(parse_stat_counters(stat), Some((8765, 37)));
        assert_eq!(parse_stat_counters("4242 (short) S 1"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_reads_its_own_counters() {
        let (ticks, _) = super::stat_counters(std::process::id()).unwrap();
        assert_eq!(Some(ticks), super::start_ticks(std::process::id()));
    }
}
