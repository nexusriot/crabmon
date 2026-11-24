//! What this machine will and will not let crabmon see.
//!
//! crabmon is careful never to report zero for something it could not read —
//! an unreadable fd table is `None`, a root-only `/dev/kmsg` says so rather
//! than showing an empty list, and a rule with nothing to measure is skipped
//! instead of firing. That is the right behaviour and it has one cost: the
//! only way to find out *why* a panel is empty is to go and look at it.
//!
//! `--doctor` asks the questions up front and answers them in one place. It
//! is also the honest answer to "should I turn `[procs] fds` on?", which
//! depends entirely on what this kernel and this user are allowed to read.

use std::fmt::Write as _;
use std::path::Path;

/// How well crabmon can see one thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Works, with nothing to qualify.
    Ok,
    /// Works, but reports less than it would elsewhere.
    Limited,
    /// Reports nothing at all here.
    Unavailable,
}

impl Status {
    pub fn key_name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Limited => "limited",
            Status::Unavailable => "unavailable",
        }
    }

    /// The marker the text report puts in front of a line.
    fn mark(self) -> char {
        match self {
            Status::Ok => '+',
            Status::Limited => '~',
            Status::Unavailable => '-',
        }
    }
}

/// One answer: what was asked, how it came out, and what to do about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// A stable identifier, so a script can match on it.
    pub id: &'static str,
    pub status: Status,
    /// What is true, in a sentence.
    pub detail: String,
    /// What to do about it, when there is something to do.
    pub remedy: String,
    /// Whether the active config requires this check to be available.
    pub required: bool,
}

impl Check {
    fn new(id: &'static str, status: Status, detail: impl Into<String>) -> Check {
        Check { id, status, detail: detail.into(), remedy: String::new(), required: false }
    }

    fn with_remedy(mut self, remedy: impl Into<String>) -> Check {
        self.remedy = remedy.into();
        self
    }

    fn with_required(mut self, required: bool) -> Check {
        self.required = required;
        self
    }
}

/// Everything `--doctor` found.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    /// Whether anything crabmon was *asked* to do cannot be done here.
    ///
    /// This is the exit code's question. A missing GPU is not a problem; a
    /// config that turns on `[procs] fds` where fd tables cannot be read is,
    /// because the column, the filters and the rules built on it will all
    /// quietly measure nothing.
    pub fn has_problems(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Unavailable && c.required)
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let width = self.checks.iter().map(|c| c.id.len()).max().unwrap_or(0);
        for check in &self.checks {
            let _ = writeln!(
                out,
                "{} {:width$}  {}",
                check.status.mark(),
                check.id,
                check.detail,
                width = width
            );
            if !check.remedy.is_empty() {
                let _ = writeln!(out, "  {:width$}  → {}", "", check.remedy, width = width);
            }
        }
        out
    }

    pub fn to_json(&self) -> String {
        let items: Vec<serde_json::Value> = self
            .checks
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "status": c.status.key_name(),
                    "detail": c.detail,
                    "remedy": c.remedy,
                })
            })
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "crabmon_version": crate::cli::VERSION,
            "problems": self.has_problems(),
            "checks": items,
        }))
        .unwrap_or_default()
            + "\n"
    }
}

/// Whether a path exists and this process can read it.
fn readable(path: &str) -> bool {
    std::fs::File::open(path).is_ok()
}

/// Run every check against this machine and this config.
///
/// The config matters: half of what makes a limitation a *problem* is having
/// asked for the thing that cannot work. `path` is the file it was loaded
/// from — the one `--config` names, when it names one — because a report that
/// says a different file parsed is worse than one that says nothing.
pub fn run(cfg: &crate::config::Config, path: &Path) -> Report {
    // Ordered as somebody reads them: the machine, then what it lets us
    // read, then what the config made of that.
    let checks = vec![
        platform(),
        privileges(),
        pressure(),
        kernel_events(),
        systemd_units(cfg),
        disk_health(cfg).with_required(cfg.smart.enabled),
        package_power(),
        cgroups(),
        gpu(cfg),
        descriptors(cfg).with_required(cfg.procs.fds),
        proportional_memory(cfg),
        process_network(cfg).with_required(cfg.procs.net),
        board(),
        backlight(),
        process_io(),
        config_file(cfg, path).with_required(true),
        alert_rules(cfg).with_required(true),
        notifications(cfg).with_required(!cfg.notify.is_empty()),
        silences(cfg).with_required(!cfg.silences.is_empty()),
    ];
    Report { checks }
}

fn platform() -> Check {
    let detail = format!("{} on {}", std::env::consts::OS, std::env::consts::ARCH,);
    match cfg!(target_os = "linux") {
        true => Check::new("platform", Status::Ok, detail),
        // Not a defect: the panels that need Linux are simply absent.
        false => Check::new(
            "platform",
            Status::Limited,
            format!("{detail}; per-disk IO, GPU, cgroup, pressure and fd counts are Linux-only"),
        ),
    }
}

#[cfg(unix)]
fn privileges() -> Check {
    // SAFETY: `geteuid` reads a property of this process and touches nothing.
    let uid = unsafe { libc::geteuid() };
    match uid {
        0 => Check::new("privileges", Status::Ok, "running as root; everything readable is read"),
        _ => Check::new(
            "privileges",
            Status::Limited,
            format!("running as uid {uid}; other users' processes report less"),
        ),
    }
}

#[cfg(not(unix))]
fn privileges() -> Check {
    Check::new("privileges", Status::Limited, "not a Unix host")
}

fn pressure() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("pressure", Status::Unavailable, "pressure stall needs Linux");
    }
    match readable("/proc/pressure/cpu") {
        true => Check::new("pressure", Status::Ok, "/proc/pressure is readable"),
        false => Check::new(
            "pressure",
            Status::Unavailable,
            "/proc/pressure is missing; the PSI panel will be hidden",
        )
        .with_remedy("needs Linux 4.20+ built with CONFIG_PSI=y, and psi=1 on some kernels"),
    }
}

fn kernel_events() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("kernel-events", Status::Unavailable, "/dev/kmsg needs Linux");
    }
    match readable("/dev/kmsg") {
        true => Check::new(
            "kernel-events",
            Status::Ok,
            "/dev/kmsg is readable; OOM kills, segfaults and IO errors are reported",
        ),
        false => Check::new(
            "kernel-events",
            Status::Unavailable,
            "/dev/kmsg cannot be read; the kernel panel will say so rather than show nothing",
        )
        .with_remedy(
            "run as root, or set kernel.dmesg_restrict=0 — \
             without it `kind = \"kernel\"` rules measure nothing",
        ),
    }
}

fn systemd_units(cfg: &crate::config::Config) -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("systemd-units", Status::Unavailable, "systemd needs Linux");
    }
    if !cfg.units.enabled {
        return Check::new("systemd-units", Status::Unavailable, "[units] enabled is off");
    }
    match crate::metrics::units::probe() {
        // Reading a unit's journal and restarting it are separate privileges
        // from listing units, and the ones somebody hitting "Access denied"
        // in the `u` popup wants explained.
        Ok(n) if is_root() => Check::new(
            "systemd-units",
            Status::Ok,
            format!(
                "systemctl show answers for {n} loaded services; failures, restart loops and \
                 their journals are reported, and `u` can restart units"
            ),
        ),
        Ok(n) => Check::new(
            "systemd-units",
            Status::Limited,
            format!(
                "systemctl show answers for {n} loaded services; failures and restart loops are \
                 reported"
            ),
        )
        .with_remedy(
            "other units' journals need root or the systemd-journal group, and `u` restarts \
             need root or a polkit rule allowing them",
        ),
        Err(e) => Check::new("systemd-units", Status::Unavailable, e).with_remedy(
            "without systemd there are no units to watch, and `kind = \"unit\"` rules measure nothing",
        ),
    }
}

fn disk_health(cfg: &crate::config::Config) -> Check {
    let Some(settings) = cfg.smart.settings() else {
        return Check::new("disk-health", Status::Unavailable, "[smart] enabled is off")
            .with_remedy(
                "set [smart] enabled = true to read SMART through smartctl (needs root, or \
                 `command = \"sudo -n smartctl\"` with a sudoers rule)",
            );
    };
    match crate::metrics::smart::probe(&settings) {
        Ok(n) => Check::new(
            "disk-health",
            Status::Ok,
            format!(
                "SMART read for {n} {}; failing ones are storage rows and `storage` rules",
                if n == 1 { "disk" } else { "disks" }
            ),
        ),
        Err(e) => {
            let remedy = smart_remedy(&e);
            Check::new("disk-health", Status::Unavailable, e).with_remedy(remedy)
        }
    }
}

/// What to do about a SMART failure depends on which one it is: telling root
/// on a QEMU guest to install smartmontools and run as root is advice that
/// cannot be followed.
fn smart_remedy(error: &str) -> &'static str {
    let e = error.to_ascii_lowercase();
    if e.contains("no smart support") || e.contains("smart is off") || e.contains("no disks") {
        "these disks have no SMART to read (virtual, or behind a USB bridge or RAID \
         controller); name real ones in [smart] devices, or turn [smart] off"
    } else if e.contains("permission denied")
        || e.contains("password")
        || e.contains("operation not permitted")
    {
        "reading SMART needs root: run as root, or set [smart] command to \
         `sudo -n smartctl` with a sudoers rule for it"
    } else if e.contains("no such file") || e.contains("not found") || e.contains("no json") {
        "install smartmontools 7.0 or later (`--json` is needed), or point [smart] command at it"
    } else {
        "check that `smartctl --json -i -H -A <disk>` works as the user crabmon runs as"
    }
}

fn is_root() -> bool {
    #[cfg(unix)]
    {
        // SAFETY: `geteuid` reads a property of this process and touches nothing.
        unsafe { libc::geteuid() == 0 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// `[[silence]]` windows, checked the way a start checks them: one that can
/// never open stops crabmon, so `--doctor` must not pass it.
fn silences(cfg: &crate::config::Config) -> Check {
    if cfg.silences.is_empty() {
        return Check::new("silences", Status::Ok, "no [[silence]] windows configured");
    }
    match crate::silence::validate(&cfg.silences) {
        Ok(()) => Check::new(
            "silences",
            Status::Ok,
            format!("{} [[silence]] windows, all in UTC", cfg.silences.len()),
        ),
        Err(e) => Check::new("silences", Status::Unavailable, e)
            .with_remedy("fix the window: crabmon refuses to start with it"),
    }
}

/// `[[notify]]` posts with curl, so a configured target on a machine
/// without it is a notification that will never arrive.
fn notifications(cfg: &crate::config::Config) -> Check {
    if cfg.notify.is_empty() {
        return Check::new("notifications", Status::Ok, "no [[notify]] targets configured");
    }
    if let Err(e) = crate::notify::validate(&cfg.notify) {
        return Check::new("notifications", Status::Unavailable, e);
    }
    let curl = std::process::Command::new("curl")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    match curl {
        true => Check::new(
            "notifications",
            Status::Ok,
            format!("{} [[notify]] targets; curl is installed", cfg.notify.len()),
        ),
        false => Check::new("notifications", Status::Unavailable, "curl is not installed")
            .with_remedy("install curl: [[notify]] posts with it"),
    }
}

fn package_power() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("package-power", Status::Unavailable, "RAPL needs Linux");
    }
    let root = "/sys/class/powercap/intel-rapl:0/energy_uj";
    if !Path::new(root).exists() {
        return Check::new(
            "package-power",
            Status::Unavailable,
            "no RAPL counters on this machine; battery figures still work",
        );
    }
    match readable(root) {
        true => Check::new("package-power", Status::Ok, "RAPL energy counters are readable"),
        false => Check::new(
            "package-power",
            Status::Limited,
            "RAPL counters exist but are not readable; package watts are omitted",
        )
        .with_remedy("root-only on most kernels since CVE-2020-8694; run as root to read them"),
    }
}

fn cgroups() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("cgroups", Status::Unavailable, "cgroups need Linux");
    }
    if Path::new("/sys/fs/cgroup/cgroup.controllers").exists() {
        return Check::new("cgroups", Status::Ok, "cgroup v2; per-container limits are readable");
    }
    if Path::new("/sys/fs/cgroup/memory").exists() {
        return Check::new("cgroups", Status::Limited, "cgroup v1; some per-group figures differ");
    }
    Check::new("cgroups", Status::Unavailable, "no cgroup filesystem; the cgroup panel is hidden")
}

fn gpu(cfg: &crate::config::Config) -> Check {
    if !cfg.gpu.enabled {
        return Check::new("gpu", Status::Limited, "switched off in [gpu] enabled");
    }
    // A Pi's GPU has a clock and nothing else, and only through the
    // firmware's tool: say so rather than reporting a healthy empty panel.
    if crate::metrics::socgpu::has_videocore(Path::new("/sys/class/drm")) && !which("vcgencmd") {
        return Check::new("gpu", Status::Limited, "a VideoCore GPU, but vcgencmd is not on PATH")
            .with_remedy("install libraspberrypi-bin to read its clock and reserved memory");
    }
    let drm = Path::new("/sys/class/drm").exists();
    let nvidia = which("nvidia-smi");
    match (drm, nvidia, cfg.gpu.nvidia_smi) {
        (_, true, true) => Check::new("gpu", Status::Ok, "nvidia-smi is on PATH and enabled"),
        (_, true, false) => Check::new("gpu", Status::Limited, "nvidia-smi is on PATH but off")
            .with_remedy("set [gpu] nvidia_smi = true to read NVIDIA cards (~50 ms a refresh)"),
        (true, false, _) => Check::new("gpu", Status::Ok, "/sys/class/drm is present"),
        (false, false, _) => {
            Check::new("gpu", Status::Unavailable, "no /sys/class/drm and no nvidia-smi")
        }
    }
}

/// Whether a command is somewhere on `PATH`.
fn which(cmd: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else { return false };
    path.split(':').any(|dir| Path::new(dir).join(cmd).exists())
}

/// The three per-process reads that are opt-in, checked against a process
/// that is *not* ours — which is the case that actually fails.
#[cfg(target_os = "linux")]
fn other_users_process() -> Option<u32> {
    // SAFETY: `geteuid` reads a property of this process.
    let me = unsafe { libc::geteuid() };
    let entries = std::fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Ok(pid) = name.parse::<u32>() else { continue };
        let Ok(meta) = std::fs::metadata(format!("/proc/{pid}")) else { continue };
        use std::os::unix::fs::MetadataExt;
        if meta.uid() != me {
            return Some(pid);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn other_users_process() -> Option<u32> {
    None
}

fn descriptors(cfg: &crate::config::Config) -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("fds", Status::Unavailable, "descriptor counts need Linux");
    }
    let asked = cfg.procs.fds;
    let Some(pid) = other_users_process() else {
        return Check::new("fds", Status::Ok, "every process on this machine is ours to read");
    };
    let readable_elsewhere = std::fs::read_dir(format!("/proc/{pid}/fd")).is_ok();
    match (readable_elsewhere, asked) {
        (true, _) => Check::new("fds", Status::Ok, "other users' fd tables are readable"),
        (false, true) => Check::new(
            "fds",
            Status::Unavailable,
            "[procs] fds is on, but other users' fd tables cannot be read: \
             their FD column, fd> filters and `kind = \"fd\"` rules see nothing",
        )
        .with_remedy("run as root, or accept that the count covers your own processes only"),
        (false, false) => Check::new(
            "fds",
            Status::Limited,
            "other users' fd tables are unreadable; [procs] fds is off anyway",
        ),
    }
}

fn proportional_memory(cfg: &crate::config::Config) -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("pss", Status::Unavailable, "smaps_rollup needs Linux");
    }
    if !Path::new("/proc/self/smaps_rollup").exists() {
        return Check::new(
            "pss",
            Status::Unavailable,
            "no smaps_rollup on this kernel; the PSS column cannot be filled",
        );
    }
    let ours = Status::Ok;
    let Some(pid) = other_users_process() else {
        return Check::new("pss", ours, "smaps_rollup is readable");
    };
    match readable(&format!("/proc/{pid}/smaps_rollup")) {
        true => Check::new("pss", Status::Ok, "other users' smaps_rollup is readable"),
        false if cfg.procs.pss => Check::new(
            "pss",
            Status::Limited,
            "[procs] pss is on, but only our own processes report PSS",
        ),
        false => Check::new("pss", Status::Limited, "PSS covers our own processes only"),
    }
}

fn process_network(cfg: &crate::config::Config) -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("process-net", Status::Unavailable, "per-process traffic needs Linux");
    }
    if !crate::metrics::procnet::is_available() {
        return Check::new(
            "process-net",
            Status::Unavailable,
            "ss is not installed; the NET column cannot be filled",
        )
        .with_remedy("install iproute2");
    }
    // SAFETY: `geteuid` reads a property of this process and touches nothing.
    #[cfg(unix)]
    let root = unsafe { libc::geteuid() } == 0;
    #[cfg(not(unix))]
    let root = false;
    match (root, cfg.procs.net) {
        (true, _) => Check::new("process-net", Status::Ok, "ss is available and sees every socket"),
        (false, true) => Check::new(
            "process-net",
            Status::Limited,
            "[procs] net is on, but only our own processes' sockets are visible",
        ),
        (false, false) => Check::new(
            "process-net",
            Status::Limited,
            "ss is available; other users' processes need root",
        ),
    }
}

/// The board's own state, where there is a board: a Raspberry Pi's throttle
/// word and the clock ceiling. Nothing on a PC, so it is not an error there.
fn board() -> Check {
    let soc = crate::metrics::soc::read();
    match (soc.throttle, soc.is_available()) {
        (Some(t), _) if t.count_now() > 0 => {
            Check::new("board", Status::Limited, format!("{} right now", t.now().join(", ")))
        }
        (Some(_), _) => Check::new("board", Status::Ok, "the firmware reports no throttling"),
        (None, true) => {
            Check::new("board", Status::Limited, "a board, but no throttle word to read")
        }
        (None, false) => {
            Check::new("board", Status::Unavailable, "no Raspberry Pi firmware to ask")
        }
    }
}

/// The screen backlight: whether there is one to read, and whether `b`/`B`
/// may move it. Not an error on a machine without a panel.
fn backlight() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("backlight", Status::Unavailable, "backlights need Linux");
    }
    let lights = crate::metrics::backlight::read();
    let Some(first) = lights.first() else {
        return Check::new("backlight", Status::Unavailable, "no /sys/class/backlight device");
    };
    let path = Path::new("/sys/class/backlight").join(&first.name).join("brightness");
    let writable = std::fs::OpenOptions::new().write(true).open(&path).is_ok();
    match writable {
        true => Check::new(
            "backlight",
            Status::Ok,
            format!("{} is readable and adjustable", first.name),
        ),
        false => Check::new(
            "backlight",
            Status::Limited,
            format!("{} is readable; b and B cannot adjust it", first.name),
        )
        .with_remedy("add yourself to the video group or install a udev rule for the backlight"),
    }
}

fn process_io() -> Check {
    if !cfg!(target_os = "linux") {
        return Check::new("process-io", Status::Limited, "per-process IO is read differently");
    }
    let Some(pid) = other_users_process() else {
        return Check::new("process-io", Status::Ok, "/proc/<pid>/io is readable");
    };
    match readable(&format!("/proc/{pid}/io")) {
        true => Check::new("process-io", Status::Ok, "other users' /proc/<pid>/io is readable"),
        false => Check::new(
            "process-io",
            Status::Limited,
            "other users' processes report zero disk IO, which is a permission, not a measurement",
        )
        .with_remedy("run as root to see their read and write rates"),
    }
}

fn config_file(cfg: &crate::config::Config, path: &Path) -> Check {
    if !path.exists() {
        return Check::new(
            "config",
            Status::Ok,
            format!("no file at {}; built-in defaults in use", path.display()),
        );
    }
    let (_, error) = crate::config::load_checked(&path.to_path_buf());
    match error {
        None => Check::new(
            "config",
            Status::Ok,
            format!(
                "{} parses; {} alert rule{}",
                path.display(),
                cfg.alerts.len(),
                if cfg.alerts.len() == 1 { "" } else { "s" }
            ),
        ),
        Some(e) => Check::new(
            "config",
            Status::Unavailable,
            format!("{} does not parse: {e}", path.display()),
        )
        .with_remedy("every setting in it is being ignored, alert rules included"),
    }
}

/// Rules that cannot measure anything here, which is the failure this whole
/// command exists to surface: a monitor whose alerts have quietly stopped
/// existing looks exactly like a monitor with nothing to report.
/// The mount points crabmon's disk panel lists — sysinfo's, which leave out
/// tmpfs and the other memory filesystems.
fn measured_mounts() -> Vec<String> {
    sysinfo::Disks::new_with_refreshed_list()
        .map(|d| d.iter().map(|d| d.mount_point().to_string_lossy().to_string()).collect())
        .unwrap_or_default()
}

fn alert_rules(cfg: &crate::config::Config) -> Check {
    use crate::alerts::AlertKind;
    let mut inert: Vec<&str> = Vec::new();
    let mounts = measured_mounts();
    for rule in &cfg.alerts {
        let needs = match rule.kind {
            AlertKind::Disk | AlertKind::FillIn
                if !rule.target.is_empty() && !mounts.contains(&rule.target) =>
            {
                Some("no measured filesystem is mounted there")
            }
            AlertKind::Fd if !cfg.procs.fds => Some("[procs] fds is off"),
            AlertKind::Psi if !readable("/proc/pressure/cpu") => Some("no pressure stall"),
            AlertKind::Kernel if !readable("/dev/kmsg") => Some("/dev/kmsg is unreadable"),
            AlertKind::Cputime if !readable("/proc/stat") => Some("/proc/stat is unreadable"),
            AlertKind::Vm if !readable("/proc/vmstat") => Some("/proc/vmstat is unreadable"),
            AlertKind::Tcp if !readable("/proc/net/snmp") => Some("/proc/net/snmp is unreadable"),
            AlertKind::Storage
                if crate::metrics::storage::read().is_empty() && !cfg.smart.enabled =>
            {
                Some("no md, ZFS or btrfs volumes, and [smart] is off")
            }
            AlertKind::Unit if !cfg.units.enabled => Some("[units] enabled is off"),
            AlertKind::Unit if !cfg!(target_os = "linux") => Some("systemd needs Linux"),
            AlertKind::Battery if crate::metrics::power::read_batteries().is_empty() => {
                Some("no battery in /sys/class/power_supply")
            }
            AlertKind::Soc if !crate::metrics::soc::read().is_available() => {
                Some("no board firmware to ask (a Raspberry Pi's throttle word)")
            }
            AlertKind::Anomaly if !crate::alerts::anomaly_series_is_known(rule.target.trim()) => {
                Some("no such series to compare against")
            }
            AlertKind::OomKill if rule.target.is_empty() && !readable("/proc/vmstat") => {
                Some("/proc/vmstat is unreadable")
            }
            _ => None,
        };
        if needs.is_some() {
            inert.push(&rule.name);
        }
    }
    match inert.is_empty() {
        true => Check::new(
            "alert-rules",
            Status::Ok,
            format!("all {} rules can measure something here", cfg.alerts.len()),
        ),
        false => Check::new(
            "alert-rules",
            Status::Unavailable,
            format!("these rules measure nothing on this host: {}", inert.join(", ")),
        )
        .with_remedy("a rule that measures nothing never fires, which reads as a quiet machine"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn smart_advice_follows_the_error() {
        assert!(
            smart_remedy("sda: no SMART support (a virtual or USB-bridged disk)")
                .contains("devices")
        );
        assert!(
            smart_remedy("nvme0: Smartctl open device: /dev/nvme0 failed: Permission denied")
                .contains("root")
        );
        assert!(smart_remedy("sudo: a password is required").contains("sudoers"));
        assert!(
            smart_remedy("smartctl: No such file or directory (os error 2)").contains("install")
        );
    }

    use super::*;
    use crate::config::Config;

    #[test]
    fn a_report_covers_every_check_and_renders_both_ways() {
        let report = run(&Config::default(), &crate::config::config_path());
        assert!(report.checks.len() >= 10, "{:?}", report.checks.len());

        let text = report.to_text();
        for check in &report.checks {
            assert!(text.contains(check.id), "the text report omits {}", check.id);
            assert!(!check.detail.is_empty(), "{} says nothing", check.id);
        }

        let json: serde_json::Value = serde_json::from_str(&report.to_json()).expect("valid JSON");
        assert_eq!(json["checks"].as_array().unwrap().len(), report.checks.len());
        assert!(json["crabmon_version"].is_string());
        assert!(json["problems"].is_boolean());
    }

    #[test]
    fn every_check_has_an_identifier_of_its_own() {
        let report = run(&Config::default(), &crate::config::config_path());
        let mut ids: Vec<&str> = report.checks.iter().map(|c| c.id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "two checks share an id");
    }

    /// A limitation nobody asked about is not a problem; one that makes a
    /// setting in the config do nothing is.
    #[test]
    fn only_an_unavailable_requirement_counts_as_a_problem() {
        let quiet = Report {
            checks: vec![
                Check::new("gpu", Status::Unavailable, "no GPU on this machine"),
                Check::new("kernel-events", Status::Unavailable, "/dev/kmsg is unreadable")
                    .with_remedy("run as root"),
                Check::new("pressure", Status::Unavailable, "no pressure stall")
                    .with_remedy("enable CONFIG_PSI"),
                Check::new("pss", Status::Limited, "only our own processes report PSS")
                    .with_required(true),
            ],
        };
        assert!(!quiet.has_problems());
        let json: serde_json::Value = serde_json::from_str(&quiet.to_json()).unwrap();
        assert_eq!(json["problems"], false);
        assert!(quiet.to_text().contains("run as root"));

        let loud = Report {
            checks: vec![
                Check::new("fds", Status::Unavailable, "asked for, cannot be read")
                    .with_required(true),
            ],
        };
        assert!(loud.has_problems());
        let json: serde_json::Value = serde_json::from_str(&loud.to_json()).unwrap();
        assert_eq!(json["problems"], true);
    }

    #[test]
    fn optional_host_limitations_do_not_fail_the_default_config() {
        let report = run(&Config::default(), Path::new("/nonexistent/crabmon.toml"));
        assert!(!report.has_problems(), "{}", report.to_text());
    }

    /// The one that matters most: a rule naming something this host cannot
    /// measure is a rule that never fires, and nothing else reports that.
    #[test]
    fn a_rule_that_cannot_measure_anything_here_is_named() {
        let mut cfg = Config { alerts: Vec::new(), ..Default::default() };
        cfg.procs.fds = false;
        cfg.alerts.push(crate::alerts::AlertRule {
            hysteresis: 0.0,
            name: "descriptors".into(),
            kind: crate::alerts::AlertKind::Fd,
            threshold: 90.0,
            for_secs: 0,
            over_secs: 0,
            below: false,
            target: String::new(),
            query: String::new(),
            command: String::new(),
            command_clear: String::new(),
        });
        let check = alert_rules(&cfg);
        assert_eq!(check.status, Status::Unavailable);
        assert!(check.detail.contains("descriptors"), "{}", check.detail);
        assert!(!check.remedy.is_empty());

        // ...and with the setting it needs, it is fine.
        cfg.procs.fds = true;
        assert_eq!(alert_rules(&cfg).status, Status::Ok);
    }
}
