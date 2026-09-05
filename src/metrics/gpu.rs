//! GPU stats without a vendor SDK: everything comes from `/sys/class/drm`.
//!
//! * amdgpu exposes `gpu_busy_percent` and VRAM counters.
//! * i915/xe expose current and max clocks but no busy percentage.
//! * NVIDIA exposes nothing useful in sysfs, so it is read via `nvidia-smi`
//!   only when the user opts in (`[gpu] nvidia_smi = true`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    /// 0..100 where the driver reports it.
    pub busy_percent: Option<f32>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub temp_c: Option<f32>,
    pub freq_mhz: Option<u64>,
    pub max_freq_mhz: Option<u64>,
}

impl GpuInfo {
    pub fn vram_ratio(&self) -> Option<f64> {
        match (self.vram_used, self.vram_total) {
            (Some(u), Some(t)) if t > 0 => Some((u as f64 / t as f64).clamp(0.0, 1.0)),
            _ => None,
        }
    }

    /// Clock as a fraction of maximum — the closest thing to a load figure on
    /// drivers with no busy counter.
    pub fn freq_ratio(&self) -> Option<f64> {
        match (self.freq_mhz, self.max_freq_mhz) {
            (Some(c), Some(m)) if m > 0 => Some((c as f64 / m as f64).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

pub fn vendor_name(id: &str) -> &'static str {
    match id.trim().trim_start_matches("0x").to_ascii_lowercase().as_str() {
        "8086" => "Intel",
        "1002" | "1022" => "AMD",
        "10de" => "NVIDIA",
        "15ad" => "VMware",
        "1af4" => "virtio",
        _ => "GPU",
    }
}

fn read_trim(p: PathBuf) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_num<T: std::str::FromStr>(p: PathBuf) -> Option<T> {
    read_trim(p)?.parse::<T>().ok()
}

/// Pull the `DRIVER=` line out of a sysfs `uevent`.
pub fn driver_from_uevent(uevent: &str) -> Option<String> {
    uevent.lines().find_map(|l| l.strip_prefix("DRIVER=").map(|d| d.trim().to_string()))
}

/// Read one `/sys/class/drm/cardN` directory. Split out from the scan so tests
/// can point it at a fabricated sysfs tree.
pub fn read_card(card: &Path) -> Option<GpuInfo> {
    let device = card.join("device");
    if !device.exists() {
        return None;
    }
    let vendor_id = read_trim(device.join("vendor")).unwrap_or_default();
    let vendor = vendor_name(&vendor_id).to_string();
    let driver = fs::read_to_string(device.join("uevent"))
        .ok()
        .and_then(|u| driver_from_uevent(&u))
        .unwrap_or_default();

    let card_label = card.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let name = if driver.is_empty() {
        format!("{vendor} {card_label}")
    } else {
        format!("{vendor} {driver}")
    };

    let temp_c = hwmon_temp(&device);

    let info = GpuInfo {
        name,
        vendor,
        busy_percent: read_num::<f32>(device.join("gpu_busy_percent")),
        vram_used: read_num::<u64>(device.join("mem_info_vram_used")),
        vram_total: read_num::<u64>(device.join("mem_info_vram_total")),
        temp_c,
        freq_mhz: read_num::<u64>(card.join("gt_cur_freq_mhz")),
        max_freq_mhz: read_num::<u64>(card.join("gt_max_freq_mhz")),
    };

    // A card that reports nothing at all is not worth a row.
    let empty = info.busy_percent.is_none()
        && info.vram_total.is_none()
        && info.temp_c.is_none()
        && info.freq_mhz.is_none();
    if empty {
        None
    } else {
        Some(info)
    }
}

/// hwmon temperatures are in millidegrees Celsius.
fn hwmon_temp(device: &Path) -> Option<f32> {
    let dir = fs::read_dir(device.join("hwmon")).ok()?;
    for entry in dir.flatten() {
        if let Some(milli) = read_num::<f32>(entry.path().join("temp1_input")) {
            return Some(milli / 1000.0);
        }
    }
    None
}

/// A DRM node is a real card (`card0`), not a connector (`card0-DP-1`).
pub fn is_card_dir(name: &str) -> bool {
    name.starts_with("card")
        && name["card".len()..].chars().all(|c| c.is_ascii_digit())
        && name.len() > "card".len()
}

pub fn scan_drm(root: &Path) -> Vec<GpuInfo> {
    let Ok(dir) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut cards: Vec<PathBuf> = dir
        .flatten()
        .filter(|e| is_card_dir(&e.file_name().to_string_lossy()))
        .map(|e| e.path())
        .collect();
    cards.sort();
    cards.iter().filter_map(|c| read_card(c)).collect()
}

/// `name, utilization.gpu, memory.used, memory.total, temperature.gpu` in CSV.
pub fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    out.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if f.len() < 5 {
                return None;
            }
            Some(GpuInfo {
                name: f[0].to_string(),
                vendor: "NVIDIA".into(),
                busy_percent: f[1].parse().ok(),
                // nvidia-smi reports MiB.
                vram_used: f[2].parse::<u64>().ok().map(|m| m * 1024 * 1024),
                vram_total: f[3].parse::<u64>().ok().map(|m| m * 1024 * 1024),
                temp_c: f[4].parse().ok(),
                freq_mhz: None,
                max_freq_mhz: None,
            })
        })
        .collect()
}

fn query_nvidia_smi() -> Vec<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match out {
        Ok(o) if o.status.success() => parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout)),
        _ => Vec::new(),
    }
}

pub fn read_all(use_nvidia_smi: bool) -> Vec<GpuInfo> {
    let mut gpus = scan_drm(Path::new("/sys/class/drm"));
    if use_nvidia_smi {
        gpus.extend(query_nvidia_smi());
    }
    gpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crabmon-gpu-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(path: PathBuf, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    #[test]
    fn connector_nodes_are_not_mistaken_for_cards() {
        assert!(is_card_dir("card0"));
        assert!(is_card_dir("card1"));
        assert!(!is_card_dir("card1-DP-1"), "connectors would double-count the GPU");
        assert!(!is_card_dir("renderD128"));
        assert!(!is_card_dir("card"));
        assert!(!is_card_dir("version"));
    }

    #[test]
    fn vendor_ids_map_to_names() {
        assert_eq!(vendor_name("0x8086"), "Intel");
        assert_eq!(vendor_name("0x1002"), "AMD");
        assert_eq!(vendor_name("0x10DE"), "NVIDIA");
        assert_eq!(vendor_name("0xbeef"), "GPU");
    }

    #[test]
    fn amd_card_reports_busy_vram_and_temperature() {
        let root = tmpdir("amd");
        let card = root.join("card0");
        write(card.join("device/vendor"), "0x1002\n");
        write(card.join("device/uevent"), "DRIVER=amdgpu\nPCI_ID=1002:73FF\n");
        write(card.join("device/gpu_busy_percent"), "42\n");
        write(card.join("device/mem_info_vram_used"), "1073741824\n");
        write(card.join("device/mem_info_vram_total"), "8589934592\n");
        write(card.join("device/hwmon/hwmon3/temp1_input"), "54000\n");

        let gpus = scan_drm(&root);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "AMD amdgpu");
        assert_eq!(gpus[0].busy_percent, Some(42.0));
        assert_eq!(gpus[0].temp_c, Some(54.0), "millidegrees must be scaled");
        assert_eq!(gpus[0].vram_ratio(), Some(0.125));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn intel_card_falls_back_to_clock_ratio() {
        // Matches the real layout on this machine: no busy counter, no VRAM.
        let root = tmpdir("intel");
        let card = root.join("card1");
        write(card.join("device/vendor"), "0x8086\n");
        write(card.join("device/uevent"), "DRIVER=i915\n");
        write(card.join("gt_cur_freq_mhz"), "600\n");
        write(card.join("gt_max_freq_mhz"), "2400\n");
        // A connector node that must be ignored.
        write(root.join("card1-DP-1/device/uevent"), "DRIVER=i915\n");

        let gpus = scan_drm(&root);
        assert_eq!(gpus.len(), 1, "the connector must not produce a second row");
        assert_eq!(gpus[0].name, "Intel i915");
        assert_eq!(gpus[0].busy_percent, None);
        assert_eq!(gpus[0].freq_ratio(), Some(0.25));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_card_with_no_readable_counters_is_dropped() {
        let root = tmpdir("bare");
        write(root.join("card0/device/vendor"), "0x1234\n");
        assert!(scan_drm(&root).is_empty());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn nvidia_smi_csv_parses_and_converts_mib() {
        let out = "NVIDIA GeForce RTX 4070, 37, 1024, 12282, 51\nbroken line\n";
        let gpus = parse_nvidia_smi(out);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].busy_percent, Some(37.0));
        assert_eq!(gpus[0].vram_total, Some(12_282 * 1024 * 1024));
        assert_eq!(gpus[0].temp_c, Some(51.0));
    }

    #[test]
    fn driver_is_read_from_uevent() {
        assert_eq!(driver_from_uevent("DRIVER=i915\nPCI_CLASS=30000\n").as_deref(), Some("i915"));
        assert_eq!(driver_from_uevent("PCI_CLASS=30000\n"), None);
    }
}
