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
            let mut sys = sysinfo::System::new();
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
