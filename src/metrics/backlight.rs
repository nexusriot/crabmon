//! Screen backlight level, from `/sys/class/backlight`.
//!
//! On a laptop or a handheld the backlight is usually the biggest single
//! draw there is, and nothing else crabmon shows explains a battery that
//! drains faster at noon than at midnight. The kernel exposes each backlight
//! as a directory with `brightness`, `max_brightness` and, on some drivers,
//! `actual_brightness` — what the hardware is really doing, which can differ
//! from what was last asked for when an ambient-light sensor or the firmware
//! has stepped in.
//!
//! Reading needs nothing. *Writing* needs a udev rule or the `video` group on
//! most distributions, so a failed write is reported with that, not treated as
//! a fault.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Backlight {
    /// `intel_backlight`, `acpi_video0`, `rpi_backlight`, `backlight`...
    pub name: String,
    pub brightness: u64,
    pub max: u64,
    /// `raw`, `platform` or `firmware`: who controls the hardware. Firmware
    /// and platform controls are coarse (often 10-20 steps); raw ones are not.
    pub kind: String,
}

impl Backlight {
    /// 0..=100.
    pub fn percent(&self) -> f64 {
        if self.max == 0 {
            0.0
        } else {
            (self.brightness as f64 / self.max as f64 * 100.0).clamp(0.0, 100.0)
        }
    }
}

fn read_num(dir: &Path, file: &str) -> Option<u64> {
    fs::read_to_string(dir.join(file)).ok()?.trim().parse().ok()
}

fn read_text(dir: &Path, file: &str) -> Option<String> {
    let t = fs::read_to_string(dir.join(file)).ok()?.trim().to_string();
    (!t.is_empty()).then_some(t)
}

/// Read every backlight under `root` (`/sys/class/backlight`).
///
/// `actual_brightness` is preferred to `brightness` where it exists: the
/// latter is only the last value written, and on a machine whose firmware
/// dims the panel itself it can be a lie. A device with no maximum has no
/// scale to read a level against and is left out.
pub fn scan(root: &Path) -> Vec<Backlight> {
    let Ok(dir) = fs::read_dir(root) else { return Vec::new() };
    let mut out: Vec<Backlight> = dir
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let max = read_num(&path, "max_brightness").filter(|m| *m > 0)?;
            let brightness =
                read_num(&path, "actual_brightness").or_else(|| read_num(&path, "brightness"))?;
            Some(Backlight {
                name: e.file_name().to_string_lossy().to_string(),
                brightness: brightness.min(max),
                max,
                kind: read_text(&path, "type").unwrap_or_default(),
            })
        })
        .collect();
    // Raw controls first: they are the ones that actually move the panel,
    // where a firmware entry beside it is often a mirror of the same one.
    out.sort_by_key(|b| (b.kind != "raw", b.name.clone()));
    out
}

pub fn read() -> Vec<Backlight> {
    if cfg!(target_os = "linux") { scan(Path::new("/sys/class/backlight")) } else { Vec::new() }
}

/// How far one key press moves the level: a tenth of the range, but never
/// less than one step, so a 15-step firmware control still moves.
pub fn step_of(max: u64) -> u64 {
    (max / 10).max(1)
}

/// The level after one step up (`up`) or down, clamped.
///
/// The floor is one step, not zero: at zero most panels are black, and a
/// screen that cannot be seen cannot be brightened again with the same key.
pub fn stepped(current: u64, max: u64, up: bool) -> u64 {
    let step = step_of(max);
    let floor = step.min(max);
    if up { current.saturating_add(step).min(max) } else { current.saturating_sub(step).max(floor) }
}

/// Set a backlight's brightness under `root`. The error is the OS's own,
/// which for the common case — no permission — says so.
pub fn set(root: &Path, name: &str, value: u64) -> std::io::Result<()> {
    // A device name with a separator would write outside the directory.
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad backlight name"));
    }
    fs::write(root.join(name).join("brightness"), value.to_string())
}

/// Move the first backlight one step, returning what to tell the user.
pub fn adjust(root: &Path, up: bool) -> Result<String, String> {
    let lights = scan(root);
    let Some(b) = lights.first() else { return Err("no backlight to adjust".into()) };
    let to = stepped(b.brightness, b.max, up);
    if to == b.brightness {
        return Ok(format!("{} is already at {}", b.name, if up { "maximum" } else { "minimum" }));
    }
    match set(root, &b.name, to) {
        Ok(()) => Ok(format!("{} {:.0}%", b.name, to as f64 / b.max as f64 * 100.0)),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => Err(format!(
            "cannot set {}: permission denied (needs root, the video group or a udev rule)",
            b.name
        )),
        Err(e) => Err(format!("cannot set {}: {e}", b.name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crabmon-bl-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn light(root: &Path, name: &str, files: &[(&str, &str)]) {
        let d = root.join(name);
        fs::create_dir_all(&d).unwrap();
        for (f, v) in files {
            fs::write(d.join(f), v).unwrap();
        }
    }

    #[test]
    fn a_backlight_is_read_as_a_share_of_its_maximum() {
        let root = tmpdir("read");
        light(
            &root,
            "intel_backlight",
            &[("brightness", "48000\n"), ("max_brightness", "96000\n"), ("type", "raw\n")],
        );
        let b = scan(&root);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].name, "intel_backlight");
        assert_eq!(b[0].percent(), 50.0);
        assert_eq!(b[0].kind, "raw");
        let _ = fs::remove_dir_all(&root);
    }

    /// `brightness` is only what was last written; the hardware's own figure
    /// wins where it exists.
    #[test]
    fn the_hardwares_figure_beats_the_last_value_written() {
        let root = tmpdir("actual");
        light(
            &root,
            "bl",
            &[("brightness", "100\n"), ("actual_brightness", "25\n"), ("max_brightness", "100\n")],
        );
        assert_eq!(scan(&root)[0].brightness, 25);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_device_with_no_scale_is_left_out_and_a_level_above_max_is_clamped() {
        let root = tmpdir("odd");
        light(&root, "noscale", &[("brightness", "5\n"), ("max_brightness", "0\n")]);
        light(&root, "missing", &[("brightness", "5\n")]);
        light(&root, "over", &[("brightness", "900\n"), ("max_brightness", "100\n")]);
        let b = scan(&root);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].brightness, 100);
        assert_eq!(b[0].percent(), 100.0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn raw_controls_are_listed_before_firmware_mirrors() {
        let root = tmpdir("order");
        light(
            &root,
            "acpi_video0",
            &[("brightness", "5\n"), ("max_brightness", "10\n"), ("type", "firmware\n")],
        );
        light(
            &root,
            "intel_backlight",
            &[("brightness", "5\n"), ("max_brightness", "10\n"), ("type", "raw\n")],
        );
        assert_eq!(scan(&root)[0].name, "intel_backlight");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_machine_with_no_backlight_reads_as_none() {
        assert!(scan(&tmpdir("empty")).is_empty());
        assert!(scan(Path::new("/nonexistent/backlight")).is_empty());
    }

    #[test]
    fn a_step_is_a_tenth_of_the_range_and_never_less_than_one() {
        assert_eq!(step_of(96_000), 9_600);
        assert_eq!(step_of(15), 1);
        assert_eq!(step_of(0), 1);
    }

    #[test]
    fn stepping_clamps_at_both_ends_and_never_goes_black() {
        assert_eq!(stepped(50, 100, true), 60);
        assert_eq!(stepped(50, 100, false), 40);
        assert_eq!(stepped(95, 100, true), 100);
        // The floor is a step, not zero: a black screen cannot be undone by eye.
        assert_eq!(stepped(12, 100, false), 10);
        assert_eq!(stepped(10, 100, false), 10);
        assert_eq!(stepped(1, 15, false), 1);
        assert_eq!(stepped(0, 0, true), 0);
    }

    #[test]
    fn adjusting_writes_the_new_level_and_says_what_it_did() {
        let root = tmpdir("adjust");
        light(
            &root,
            "bl",
            &[("brightness", "50\n"), ("max_brightness", "100\n"), ("type", "raw\n")],
        );
        let msg = adjust(&root, true).unwrap();
        assert_eq!(msg, "bl 60%");
        assert_eq!(fs::read_to_string(root.join("bl/brightness")).unwrap(), "60");
        assert_eq!(adjust(&root, false).unwrap(), "bl 50%");

        // At the limit nothing is written and the answer says why.
        light(&root, "bl", &[("brightness", "100\n")]);
        assert_eq!(adjust(&root, true).unwrap(), "bl is already at maximum");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn adjusting_with_nothing_to_adjust_is_an_error_not_a_panic() {
        assert!(adjust(&tmpdir("none"), true).is_err());
        assert!(set(Path::new("/tmp"), "../etc", 1).is_err());
        assert!(set(Path::new("/tmp"), "", 1).is_err());
    }

    #[test]
    fn reading_this_machine_is_safe() {
        for b in read() {
            assert!(b.max > 0 && b.brightness <= b.max);
        }
    }
}
