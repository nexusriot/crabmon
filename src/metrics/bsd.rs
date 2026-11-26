//! What `sysinfo` leaves blank or wrong on FreeBSD, read from `sysctl`.
//!
//! On a real FreeBSD 15 host `sysinfo` reports the kernel as `199506` (the
//! ABI revision, not a release), the OS as the whole `uname -v` line with its
//! newline, no CPU model and a clock of 0 MHz. The kernel has all four, under
//! names that have not changed in decades. They are asked once and cached:
//! none of them moves while the machine is up.

use std::sync::OnceLock;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BsdHost {
    /// `FreeBSD`. The release is the kernel's, as on Linux where the
    /// distribution and the kernel are two fields.
    pub os: Option<String>,
    /// `15.0-RELEASE-p12`.
    pub kernel: Option<String>,
    pub cpu_model: Option<String>,
    /// The CPU's nominal clock, in MHz.
    pub mhz: Option<u64>,
}

/// Silences standard error while it is alive, then puts it back.
///
/// On FreeBSD `sysinfo` reads each process's cwd and executable through
/// libprocstat, which `warn(3)`s — `crabmon: kinfo_getfile(): No such process`
/// — whenever a process exits between being listed and being read. That is a
/// process doing its job, not an error, and in a full-screen interface the
/// line lands in the middle of the table. The text is not the library's to
/// print, so for the length of a sample it goes nowhere.
///
/// A no-op anywhere but FreeBSD, where nothing prints. Anything another thread
/// writes to stderr during the sample is lost too; the sampler is the only
/// thread that writes there while it runs.
#[must_use = "stderr is restored when this is dropped"]
pub struct MuteStderr {
    #[cfg(unix)]
    saved: Option<libc::c_int>,
}

impl MuteStderr {
    pub fn new() -> Self {
        Self::when(cfg!(target_os = "freebsd"))
    }

    /// Mute if `on`; the platform test is the caller's, so it can be tested.
    pub fn when(on: bool) -> Self {
        #[cfg(unix)]
        {
            if !on {
                return Self { saved: None };
            }
            // SAFETY: plain fd calls on descriptors this function owns; on any
            // failure nothing is changed and nothing is muted.
            unsafe {
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
                if null < 0 {
                    return Self { saved: None };
                }
                let saved = libc::dup(2);
                if saved < 0 || libc::dup2(null, 2) < 0 {
                    if saved >= 0 {
                        libc::close(saved);
                    }
                    libc::close(null);
                    return Self { saved: None };
                }
                libc::close(null);
                Self { saved: Some(saved) }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = on;
            Self {}
        }
    }
}

impl Default for MuteStderr {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MuteStderr {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(saved) = self.saved.take() {
            // SAFETY: restores the descriptor `when` saved, then closes the copy.
            unsafe {
                libc::dup2(saved, 2);
                libc::close(saved);
            }
        }
    }
}

/// FreeBSD's `/etc/passwd` has two names for uid 0, `root` and `toor`, and a
/// uid lookup returns the last of them — so every root process read as `toor`.
/// Nobody logs in as `toor`; root is root.
pub fn user_name(uid: Option<u32>, name: String) -> String {
    if uid == Some(0) { "root".to_string() } else { name }
}

/// `sysinfo` takes a process's name from the last `/` in its title, and a
/// process that retitles itself with a path in it — `tmux: server
/// (/tmp/tmux-0/default) (tmux)` — is named `default) (tmux)`. Where the name
/// is cut that way and the executable is known, the executable's file name is
/// the honest one.
pub fn tidy_name(name: String, exe: &str) -> String {
    // A title cut mid-parenthesis has more closing brackets than opening ones.
    let (open, close) = (name.matches('(').count(), name.matches(')').count());
    if close <= open {
        return name;
    }
    match std::path::Path::new(exe).file_name().and_then(|f| f.to_str()) {
        Some(f) if !f.is_empty() => f.to_string(),
        _ => name,
    }
}

/// A process that retitled itself (`daemon: /usr/local/bin/x[1408] (daemon)`)
/// can come back from `sysinfo` with an empty argument vector, which left the
/// detail pane's Command line blank. The executable, then the name, are better
/// than nothing.
pub fn fallback_cmd(cmd: String, name: &str, exe: &str) -> String {
    if !cmd.trim().is_empty() {
        cmd
    } else if !exe.is_empty() {
        exe.to_string()
    } else {
        name.to_string()
    }
}

/// One line of `sysctl -n` output, without its newline. Empty is `None`.
pub fn clean(raw: &str) -> Option<String> {
    let t = raw.lines().next().unwrap_or("").trim();
    (!t.is_empty()).then(|| t.to_string())
}

pub fn build(release: Option<&str>, model: Option<&str>, clock: Option<&str>) -> BsdHost {
    let release = release.and_then(clean);
    BsdHost {
        os: release.as_ref().map(|_| "FreeBSD".to_string()),
        kernel: release,
        cpu_model: model.and_then(clean),
        mhz: clock.and_then(clean).and_then(|c| c.parse().ok()).filter(|m| *m > 0),
    }
}

#[cfg(target_os = "freebsd")]
fn sysctl(name: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl").args(["-n", name]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

/// The host's identity, read on first use. All `None` off FreeBSD, so callers
/// keep what `sysinfo` said.
pub fn host() -> &'static BsdHost {
    static HOST: OnceLock<BsdHost> = OnceLock::new();
    HOST.get_or_init(|| {
        #[cfg(target_os = "freebsd")]
        {
            // `dev.cpu.0.freq` is the live clock where cpufreq is loaded; the
            // nominal one is the fallback, and what a VM has.
            let clock = sysctl("dev.cpu.0.freq").or_else(|| sysctl("hw.clockrate"));
            build(
                sysctl("kern.osrelease").as_deref(),
                sysctl("hw.model").as_deref(),
                clock.as_deref(),
            )
        }
        #[cfg(not(target_os = "freebsd"))]
        {
            BsdHost::default()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysctl_output_loses_its_newline_and_empty_is_nothing() {
        assert_eq!(clean("15.0-RELEASE-p12\n").as_deref(), Some("15.0-RELEASE-p12"));
        assert_eq!(
            clean("  Intel(R) Core(TM) Ultra 7 155H \n").as_deref(),
            Some("Intel(R) Core(TM) Ultra 7 155H")
        );
        assert_eq!(clean("\n"), None);
        assert_eq!(clean(""), None);
        // Multi-line values keep the first line, which is the release.
        assert_eq!(clean("a\nb\n").as_deref(), Some("a"));
    }

    #[test]
    fn a_host_is_assembled_from_what_sysctl_gave() {
        let h = build(Some("15.0-RELEASE-p12\n"), Some("Intel\n"), Some("2995\n"));
        assert_eq!(h.os.as_deref(), Some("FreeBSD"));
        assert_eq!(h.kernel.as_deref(), Some("15.0-RELEASE-p12"));
        assert_eq!(h.cpu_model.as_deref(), Some("Intel"));
        assert_eq!(h.mhz, Some(2995));
    }

    #[test]
    fn a_missing_or_zero_figure_is_left_to_sysinfo() {
        assert_eq!(build(None, None, None), BsdHost::default());
        assert_eq!(build(None, None, Some("0\n")).mhz, None);
        assert_eq!(build(None, None, Some("fast")).mhz, None);
    }

    #[test]
    fn both_names_for_uid_zero_read_as_root() {
        assert_eq!(user_name(Some(0), "toor".into()), "root");
        assert_eq!(user_name(Some(0), "root".into()), "root");
        assert_eq!(user_name(Some(1001), "toor".into()), "toor");
        assert_eq!(user_name(None, "x".into()), "x");
    }

    #[test]
    fn a_title_cut_at_a_slash_falls_back_to_the_executable() {
        let t = |n: &str, e: &str| tidy_name(n.to_string(), e);
        assert_eq!(t("default)", "/usr/local/bin/tmux"), "tmux");
        assert_eq!(t("default) (tmux)", "/usr/local/bin/tmux"), "tmux");
        // Nothing to fall back to: keep what there is.
        assert_eq!(t("default)", ""), "default)");
        // Ordinary and balanced names are left alone.
        assert_eq!(
            t("sshd [listener] 0 of 10-100", "/usr/sbin/sshd"),
            "sshd [listener] 0 of 10-100"
        );
        assert_eq!(t("firefox (nightly)", "/opt/firefox"), "firefox (nightly)");
        assert_eq!(t("crabmon", "/x/crabmon"), "crabmon");
    }

    #[test]
    fn a_blank_command_line_falls_back_to_the_executable_then_the_name() {
        assert_eq!(fallback_cmd("a b".into(), "n", "/e"), "a b");
        assert_eq!(fallback_cmd("  ".into(), "n", "/usr/sbin/daemon"), "/usr/sbin/daemon");
        assert_eq!(fallback_cmd(String::new(), "beastied", ""), "beastied");
    }

    /// What the guard promises: while held, a write to fd 2 goes nowhere, and
    /// afterwards fd 2 is the file it was before. Done on a pipe standing in
    /// for stderr so the test can read what got through.
    #[cfg(unix)]
    #[test]
    fn stderr_is_silent_while_muted_and_restored_after() {
        use std::io::Read;
        use std::os::fd::FromRawFd;
        // SAFETY: the test owns every descriptor it creates, and puts fd 2 back.
        unsafe {
            let mut fds = [0 as libc::c_int; 2];
            assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
            let original = libc::dup(2);
            libc::dup2(fds[1], 2);

            let write = |msg: &[u8]| libc::write(2, msg.as_ptr().cast(), msg.len());
            write(b"before;");
            {
                let _quiet = MuteStderr::when(true);
                write(b"during;");
            }
            write(b"after;");
            {
                let _untouched = MuteStderr::when(false);
                write(b"unmuted;");
            }

            libc::dup2(original, 2);
            libc::close(original);
            libc::close(fds[1]);
            let mut got = String::new();
            std::fs::File::from_raw_fd(fds[0]).read_to_string(&mut got).unwrap();
            assert_eq!(got, "before;after;unmuted;");
        }
    }

    #[test]
    fn this_machine_answers_without_panicking() {
        let _ = host();
    }
}
