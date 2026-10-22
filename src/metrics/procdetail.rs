//! The per-process facts that are only worth reading for one process at a
//! time: OOM scores, the environment, and the cgroup path behind the service
//! and container labels.
//!
//! None of this belongs in `ProcRow`. A `ProcRow` is built for every process
//! on every sample, and reading four more `/proc` files per process per
//! refresh on a 2000-task machine to fill a pane that shows one of them is
//! the wrong trade. These are read when the detail pane opens, like the
//! socket list and the open files beside them.

/// How a process stands with the OOM killer.
///
/// "Why did the kernel pick *that* process" is the most common postmortem
/// question a system monitor is asked and the one nothing on the dashboard
/// could answer. `oom_score` is the kernel's own ranking — higher dies first
/// — and `oom_score_adj` is the thumb anyone has put on the scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OomScore {
    pub score: Option<i32>,
    pub adj: Option<i32>,
}

impl OomScore {
    pub fn is_empty(&self) -> bool {
        self.score.is_none() && self.adj.is_none()
    }

    /// `820 (adj +500)`, or `-` when the kernel says nothing.
    pub fn describe(&self) -> String {
        match (self.score, self.adj) {
            (None, None) => "-".into(),
            (Some(s), Some(0)) => s.to_string(),
            (Some(s), Some(a)) => format!("{s} (adj {a:+})"),
            (Some(s), None) => s.to_string(),
            (None, Some(a)) => format!("adj {a:+}"),
        }
    }

    /// Whether this process has been told to die first, or never to die.
    /// Worth saying out loud: an `oom_score_adj` of -1000 is why the OOM
    /// killer went past the obvious candidate and took something else.
    pub fn is_adjusted(&self) -> bool {
        !matches!(self.adj, None | Some(0))
    }
}

pub fn oom_score(pid: u32) -> OomScore {
    OomScore { score: read_int(pid, "oom_score"), adj: read_int(pid, "oom_score_adj") }
}

#[cfg(target_os = "linux")]
fn read_int(pid: u32, file: &str) -> Option<i32> {
    std::fs::read_to_string(format!("/proc/{pid}/{file}")).ok()?.trim().parse().ok()
}

#[cfg(not(target_os = "linux"))]
fn read_int(_pid: u32, _file: &str) -> Option<i32> {
    None
}

/// The cgroup a process is in, as the kernel spells it.
///
/// The process table already derives a unit and a container id from this, but
/// the derivation drops everything it did not recognise — and "which slice is
/// this actually in" is the question when the answer is neither.
pub fn cgroup_path(pid: u32) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
        super::cgroup::parse_proc_cgroup(&content)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

/// One environment variable as the pane shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvVar {
    pub name: String,
    pub value: String,
    /// The value was withheld because the name looks like a credential.
    pub hidden: bool,
}

/// Longest environment a process's pane will list.
///
/// A build system's environment can run to hundreds of entries, and the pane
/// scrolls but the `readlink`-shaped cost of reading does not need to.
pub const MAX_ENV: usize = 256;

/// Fragments that make a variable's *value* too dangerous to put on a screen
/// somebody might be sharing.
///
/// A monitor is a thing people screenshot into tickets and project onto a
/// wall in an incident call. `AWS_SECRET_ACCESS_KEY` sitting in a pane next
/// to a stuck process is a credential leak with a very ordinary cause, so the
/// name is shown — knowing the variable is *set* is often the whole question
/// — and the value is not.
const SECRET_HINTS: [&str; 9] = [
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "TOKEN",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "PRIVATE_KEY",
];

/// Names whose *ending* makes them a secret, for the ones the fragments above
/// cannot reach. `STRIPE_KEY`, `ENCRYPTION_KEY` and `SIGNING_KEY` are all
/// credentials and none of them contains `API_KEY` or `SECRET`.
const SECRET_SUFFIXES: [&str; 2] = ["_KEY", "_PWD"];

/// Whether a variable's value should be withheld.
///
/// `KEY` as a bare fragment is deliberately not a hint: names like
/// `KEYBOARD_LAYOUT`, `KEYMAP` and `XKB_DEFAULT_LAYOUT` are not secrets, and
/// masking half an environment teaches people to ignore the mask. `_KEY` as a
/// *suffix* is a different question — nothing is called `FOO_KEY` unless it
/// holds one — so the two are matched separately rather than by loosening the
/// fragment.
pub fn is_secret_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SECRET_HINTS.iter().any(|h| upper.contains(h))
        || SECRET_SUFFIXES.iter().any(|s| upper.ends_with(s))
}

/// Parse the NUL-separated block `/proc/<pid>/environ` holds.
pub fn parse_environ(raw: &str) -> Vec<EnvVar> {
    raw.split('\0')
        .filter(|e| !e.is_empty())
        .take(MAX_ENV)
        .map(|entry| {
            let (name, value) = match entry.split_once('=') {
                Some((n, v)) => (n.to_string(), v.to_string()),
                // A variable with no `=` is not something the kernel writes,
                // but `environ` is whatever the process put there.
                None => (entry.to_string(), String::new()),
            };
            let hidden = is_secret_name(&name) && !value.is_empty();
            EnvVar { name, value: if hidden { String::new() } else { value }, hidden }
        })
        .collect()
}

/// A process's environment, or `None` when it cannot be read — another user's
/// process, or one that exited while the pane was opening.
pub fn environment(pid: u32) -> Option<Vec<EnvVar>> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string(format!("/proc/{pid}/environ")).ok()?;
        Some(parse_environ(&raw))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_environment_block_splits_on_nuls() {
        let env = parse_environ("PATH=/usr/bin\0HOME=/root\0");
        assert_eq!(env.len(), 2);
        assert_eq!(env[0], EnvVar { name: "PATH".into(), value: "/usr/bin".into(), hidden: false });
        assert_eq!(env[1].name, "HOME");
    }

    /// A monitor is a thing people screenshot. The name is the useful half.
    #[test]
    fn credentials_keep_their_name_and_lose_their_value() {
        let env = parse_environ("AWS_SECRET_ACCESS_KEY=hunter2\0GITHUB_TOKEN=ghp_x\0PATH=/bin");
        assert!(env[0].hidden);
        assert_eq!(env[0].value, "", "a secret's value reached the screen");
        assert_eq!(env[0].name, "AWS_SECRET_ACCESS_KEY", "the name is worth knowing");
        assert!(env[1].hidden);
        assert!(!env[2].hidden);
        assert_eq!(env[2].value, "/bin");
    }

    /// Masking everything teaches people to ignore the mask.
    #[test]
    fn ordinary_variables_are_not_masked() {
        for name in [
            "PATH",
            "HOME",
            "KEYBOARD_LAYOUT",
            "XKB_DEFAULT_LAYOUT",
            "LANG",
            "TERM",
            // The `_KEY` suffix must not widen into a bare `KEY` fragment.
            "KEYMAP",
            "SSH_AUTH_SOCK",
            "GPG_KEY_ID",
        ] {
            assert!(!is_secret_name(name), "{name} was masked");
        }
        for name in [
            "AWS_SECRET_ACCESS_KEY",
            "DB_PASSWORD",
            "GITHUB_TOKEN",
            "MY_API_KEY",
            // Credentials the fragment list alone could not reach: nothing is
            // called `FOO_KEY` unless it holds one.
            "STRIPE_KEY",
            "ENCRYPTION_KEY",
            "SIGNING_KEY",
            "SSL_PASSPHRASE",
            "MYSQL_PWD",
        ] {
            assert!(is_secret_name(name), "{name} was not masked");
        }
    }

    #[test]
    fn a_very_long_environment_is_bounded() {
        let raw: String = (0..MAX_ENV * 2).map(|i| format!("V{i}=x\0")).collect();
        assert_eq!(parse_environ(&raw).len(), MAX_ENV);
    }

    #[test]
    fn an_entry_with_no_equals_is_kept_as_a_bare_name() {
        let env = parse_environ("WEIRD\0PATH=/bin");
        assert_eq!(env[0].name, "WEIRD");
        assert_eq!(env[0].value, "");
        assert!(!env[0].hidden, "a variable with no value has nothing to hide");
    }

    #[test]
    fn an_oom_score_reads_as_the_kernel_ranks_it() {
        assert_eq!(OomScore { score: Some(820), adj: Some(0) }.describe(), "820");
        assert_eq!(OomScore { score: Some(820), adj: Some(500) }.describe(), "820 (adj +500)");
        assert_eq!(OomScore { score: Some(0), adj: Some(-1000) }.describe(), "0 (adj -1000)");
        assert_eq!(OomScore::default().describe(), "-");
        assert!(OomScore { score: Some(1), adj: Some(-1000) }.is_adjusted());
        assert!(!OomScore { score: Some(1), adj: Some(0) }.is_adjusted());
    }

    /// Reading our own process must not panic, whatever the platform does.
    #[test]
    fn reading_this_process_is_safe() {
        let me = std::process::id();
        let _ = oom_score(me);
        let _ = cgroup_path(me);
        let env = environment(me);
        #[cfg(target_os = "linux")]
        {
            let env = env.expect("a process can read its own environment");
            assert!(env.iter().any(|v| v.name == "PATH"), "no PATH in our own environment");
        }
        let _ = env;
    }
}
