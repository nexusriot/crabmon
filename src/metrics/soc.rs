//! Board-level health for ARM and RISC-V single-board computers: the
//! Raspberry Pi firmware's throttle word, the CPU's frequency ceiling and
//! governor, and what the board calls itself.
//!
//! A Pi that is undervolted does not crash. It quietly drops its clock, and
//! everything gets slower for a reason no process table shows: CPU use looks
//! fine, temperature looks fine, and the only evidence is a bit in a word the
//! firmware keeps. `vcgencmd get_throttled` is how people find out, usually
//! after an afternoon of looking elsewhere. A handheld like the uConsole runs
//! from a battery pack, so it is the case this exists for.
//!
//! Everything is read from sysfs and devicetree, with `vcgencmd` as the
//! fallback for the throttle word on kernels that do not expose it. Anything
//! absent is `None` rather than a zero: a machine with no firmware to ask has
//! not been found healthy.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The firmware's `get_throttled` word.
///
/// Bits 0-3 describe *now*; bits 16-19 are the same conditions *since boot*,
/// so a brownout that lasted a second is still visible long after.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThrottleFlags(pub u32);

/// `(bit, what it means)` for the conditions the firmware reports.
const CONDITIONS: [(u32, &str); 4] =
    [(0, "undervoltage"), (1, "frequency capped"), (2, "throttled"), (3, "soft temperature limit")];

/// How far above the "now" bit its "since boot" twin sits.
const SINCE_BOOT: u32 = 16;

impl ThrottleFlags {
    /// Conditions in effect at this moment.
    pub fn now(self) -> Vec<&'static str> {
        CONDITIONS.iter().filter(|(b, _)| self.0 & (1 << b) != 0).map(|(_, n)| *n).collect()
    }

    /// Conditions that happened since boot but are over.
    pub fn earlier(self) -> Vec<&'static str> {
        CONDITIONS
            .iter()
            .filter(|(b, _)| self.0 & (1 << (b + SINCE_BOOT)) != 0 && self.0 & (1 << b) == 0)
            .map(|(_, n)| *n)
            .collect()
    }

    pub fn undervoltage_now(self) -> bool {
        self.0 & 1 != 0
    }

    /// How many conditions are in effect, which is what an alert counts.
    pub fn count_now(self) -> usize {
        self.now().len()
    }

    /// How many happened at any point since boot, ended or not.
    pub fn count_since_boot(self) -> usize {
        CONDITIONS.iter().filter(|(b, _)| self.0 & (1 << (b + SINCE_BOOT)) != 0).count()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SocSample {
    /// `Raspberry Pi 4 Model B Rev 1.5`, from the devicetree.
    pub model: Option<String>,
    pub throttle: Option<ThrottleFlags>,
    /// `ondemand`, `schedutil`, `performance`...
    pub governor: Option<String>,
    /// The fastest any core is running right now, in kHz.
    pub cur_khz: Option<u64>,
    /// The fastest any core is allowed to run, in kHz.
    pub max_khz: Option<u64>,
}

impl SocSample {
    /// Whether this looks like a board worth a line of its own.
    ///
    /// A devicetree model means a board rather than a PC, which is what
    /// keeps every x86 laptop with `cpufreq` from growing a panel it has no
    /// use for. The throttle word alone is enough: only the Pi's firmware
    /// has one.
    pub fn is_available(&self) -> bool {
        self.throttle.is_some() || (self.model.is_some() && self.cur_khz.is_some())
    }

    /// Current clock as a share of the ceiling, 0..=1, where both are known.
    ///
    /// Under a cap this is how far down the machine has been pushed, which
    /// the percentage CPU gauge cannot say: a core pinned at 600 of 1800 MHz
    /// reads as quite idle.
    pub fn clock_ratio(&self) -> Option<f64> {
        match (self.cur_khz, self.max_khz) {
            (Some(c), Some(m)) if m > 0 => Some((c as f64 / m as f64).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

/// `throttled=0x50005`, `0x50005` or a bare `50005`, which is how sysfs
/// prints it — in hex either way.
pub fn parse_throttled(text: &str) -> Option<ThrottleFlags> {
    let t = text.trim();
    let t = t.strip_prefix("throttled=").unwrap_or(t).trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    u32::from_str_radix(t, 16).ok().map(ThrottleFlags)
}

fn read_trim(root: &Path, rel: &str) -> Option<String> {
    let text = fs::read_to_string(root.join(rel)).ok()?;
    // The devicetree pads its strings with a NUL.
    let text = text.trim_matches(|c: char| c.is_whitespace() || c == '\0').to_string();
    (!text.is_empty()).then_some(text)
}

fn read_khz(root: &Path, rel: &str) -> Option<u64> {
    read_trim(root, rel)?.parse().ok()
}

/// Where the Pi kernel puts the word, if it does.
const THROTTLED_SYSFS: &str = "sys/devices/platform/soc/soc:firmware/get_throttled";

/// Ask the firmware directly, for kernels without the sysfs file.
///
/// Only attempted where the mailbox device exists, so a PC never spends a
/// `fork` on a command it does not have every second.
fn vcgencmd(root: &Path) -> Option<ThrottleFlags> {
    if !root.join("dev/vcio").exists() {
        return None;
    }
    let out = std::process::Command::new("vcgencmd").arg("get_throttled").output().ok()?;
    out.status.success().then(|| parse_throttled(&String::from_utf8_lossy(&out.stdout)))?
}

/// Read the board's state from `root`, which is `/` outside a test.
pub fn read_from(root: &Path) -> SocSample {
    let throttle = read_trim(root, THROTTLED_SYSFS)
        .and_then(|t| parse_throttled(&t))
        .or_else(|| vcgencmd(root));

    let model = read_trim(root, "proc/device-tree/model")
        .or_else(|| read_trim(root, "sys/firmware/devicetree/base/model"));

    let mut cur_khz = None;
    let mut max_khz = None;
    let mut governor = None;
    if let Ok(dir) = fs::read_dir(root.join("sys/devices/system/cpu")) {
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_cpu = name.strip_prefix("cpu").is_some_and(|n| n.parse::<u32>().is_ok());
            if !is_cpu {
                continue;
            }
            let rel = format!("sys/devices/system/cpu/{name}/cpufreq");
            if let Some(c) = read_khz(root, &format!("{rel}/scaling_cur_freq")) {
                cur_khz = Some(cur_khz.map_or(c, |p: u64| p.max(c)));
            }
            // The *allowed* ceiling, so a cap shows up: `scaling_max_freq`
            // is what throttling lowers, `cpuinfo_max_freq` is the silicon.
            let ceiling = read_khz(root, &format!("{rel}/scaling_max_freq"))
                .or_else(|| read_khz(root, &format!("{rel}/cpuinfo_max_freq")));
            if let Some(m) = ceiling {
                max_khz = Some(max_khz.map_or(m, |p: u64| p.max(m)));
            }
            if governor.is_none() {
                governor = read_trim(root, &format!("{rel}/scaling_governor"));
            }
        }
    }
    SocSample { model, throttle, governor, cur_khz, max_khz }
}

pub fn read() -> SocSample {
    read_from(Path::new("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crabmon-soc-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    #[test]
    fn the_throttle_word_parses_in_every_spelling() {
        assert_eq!(parse_throttled("throttled=0x50005\n"), Some(ThrottleFlags(0x50005)));
        assert_eq!(parse_throttled("0x0"), Some(ThrottleFlags(0)));
        // Sysfs prints bare hex: `50005` is 0x50005, not fifty thousand.
        assert_eq!(parse_throttled("50005\n"), Some(ThrottleFlags(0x50005)));
        assert_eq!(parse_throttled("nonsense"), None);
        assert_eq!(parse_throttled(""), None);
    }

    #[test]
    fn now_and_since_boot_are_told_apart() {
        // Undervoltage and throttling right now, plus both having happened.
        let f = ThrottleFlags(0x50005);
        assert_eq!(f.now(), vec!["undervoltage", "throttled"]);
        assert!(f.undervoltage_now());
        assert_eq!(f.count_now(), 2);
        // Still in effect, so neither is "earlier".
        assert!(f.earlier().is_empty());
        assert_eq!(f.count_since_boot(), 2);

        // A brownout that is over: the "since boot" bit with no "now" bit.
        let past = ThrottleFlags(0x10000);
        assert!(past.now().is_empty());
        assert_eq!(past.earlier(), vec!["undervoltage"]);
        assert!(!past.undervoltage_now());
        assert_eq!(past.count_now(), 0);
        assert_eq!(past.count_since_boot(), 1);

        assert!(ThrottleFlags(0).now().is_empty());
    }

    // Unix only: the firmware's sysfs node is `soc:firmware`, and a `:` is not
    // allowed in a Windows path component.
    #[cfg(unix)]
    #[test]
    fn a_pi_is_read_from_sysfs_and_the_devicetree() {
        let root = tmpdir("pi");
        write(&root, THROTTLED_SYSFS, "50000\n");
        write(&root, "proc/device-tree/model", "Raspberry Pi 4 Model B Rev 1.5\0");
        write(&root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "600000\n");
        write(&root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "1500000\n");
        write(&root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "ondemand\n");
        write(&root, "sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "1500000\n");
        write(&root, "sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq", "1500000\n");
        // Not a core: must not be read as one.
        write(&root, "sys/devices/system/cpu/cpufreq/scaling_cur_freq", "9\n");

        let s = read_from(&root);
        assert_eq!(s.model.as_deref(), Some("Raspberry Pi 4 Model B Rev 1.5"));
        assert_eq!(s.throttle, Some(ThrottleFlags(0x50000)));
        assert_eq!(s.governor.as_deref(), Some("ondemand"));
        assert_eq!(s.cur_khz, Some(1_500_000), "the fastest core, not the first");
        assert_eq!(s.max_khz, Some(1_500_000));
        assert!(s.is_available());
        assert_eq!(s.clock_ratio(), Some(1.0));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_capped_clock_shows_as_a_ratio() {
        let s =
            SocSample { cur_khz: Some(600_000), max_khz: Some(1_800_000), ..Default::default() };
        let r = s.clock_ratio().unwrap();
        assert!((r - 1.0 / 3.0).abs() < 1e-9, "{r}");
        assert_eq!(SocSample::default().clock_ratio(), None);
        // A zero ceiling must not divide.
        assert_eq!(
            SocSample { cur_khz: Some(1), max_khz: Some(0), ..Default::default() }.clock_ratio(),
            None
        );
    }

    /// A PC has cpufreq and no devicetree: it is not a board, and must not
    /// get a panel line telling it so.
    #[test]
    fn an_ordinary_pc_is_not_a_board() {
        let root = tmpdir("pc");
        write(&root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "3000000\n");
        write(&root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "4500000\n");
        let s = read_from(&root);
        assert_eq!(s.cur_khz, Some(3_000_000));
        assert_eq!(s.throttle, None);
        assert!(!s.is_available());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_empty_root_reads_as_nothing() {
        let root = tmpdir("empty");
        let s = read_from(&root);
        assert_eq!(s, SocSample::default());
        assert!(!s.is_available());
        let _ = fs::remove_dir_all(&root);
    }
}
