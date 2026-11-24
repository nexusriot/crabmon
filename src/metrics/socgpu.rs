//! GPUs on ARM boards, which the DRM scan in `gpu.rs` cannot see.
//!
//! `gpu.rs` reads `/sys/class/drm/card*/device/` for a vendor id and a few
//! amdgpu or i915 files. A GPU on a SoC is a platform device: no PCI vendor,
//! none of those files, so the scan finds the card and drops it as empty. Two
//! families are worth reading, and neither speaks DRM sysfs:
//!
//! - **Mali** (Panfrost, Lima, the vendor `mali_kbase`) and most other SoC
//!   GPUs sit behind **devfreq**, which exposes the clock, its ceiling and,
//!   on some drivers, a load figure — and a thermal zone named for the GPU.
//! - **VideoCore** (the Raspberry Pi's `v3d` and `vc4`) exposes none of it
//!   in sysfs; the firmware answers through `vcgencmd`, which gives the
//!   clock and the memory reserved for the GPU.
//!
//! A load figure is not available for VideoCore without debugfs or walking
//! every process's `fdinfo`, so a Pi's GPU row has its clock and not a busy
//! percentage — and says so by showing the clock in the load column rather
//! than a zero.

use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::gpu::GpuInfo;

/// Substrings that mark a devfreq device as a GPU's.
const GPU_MARKERS: [&str; 6] = ["gpu", "mali", "g3d", "panfrost", "kbase", "v3d"];

/// Whether a devfreq entry is a GPU's, going by its name — `ff9a0000.gpu`,
/// `fb000000.gpu-mali`, `mali0`, `11800000.g3d`.
pub fn is_gpu_devfreq(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    GPU_MARKERS.iter().any(|m| n.contains(m))
}

fn read_trim(p: &Path) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn read_u64(p: &Path) -> Option<u64> {
    read_trim(p)?.parse().ok()
}

/// A devfreq `load` file: `12@600000000Hz`, or a bare `12`. The percentage is
/// the part before the `@`.
pub fn parse_load(text: &str) -> Option<f32> {
    let head = text.trim().split('@').next()?.trim();
    let v: f32 = head.parse().ok()?;
    (0.0..=100.0).contains(&v).then_some(v)
}

/// The fastest entry of `available_frequencies`, which is a space-separated
/// list in Hz and not necessarily sorted.
pub fn max_available(text: &str) -> Option<u64> {
    text.split_whitespace().filter_map(|f| f.parse::<u64>().ok()).max()
}

/// A thermal zone whose type names the GPU — `gpu-thermal`, `gpu_thermal`,
/// `mali-thermal`. `None` when the board has no zone of its own for it, which
/// is most of them: many SoCs have one sensor for the whole die.
pub fn gpu_thermal(thermal_root: &Path) -> Option<f32> {
    let dir = fs::read_dir(thermal_root).ok()?;
    let mut zones: Vec<_> = dir
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("thermal_zone"))
        .collect();
    zones.sort_by_key(|e| e.file_name());
    zones.iter().find_map(|e| {
        let kind = read_trim(&e.path().join("type"))?.to_ascii_lowercase();
        (kind.contains("gpu") || kind.contains("mali"))
            .then(|| read_u64(&e.path().join("temp")).map(|m| m as f32 / 1000.0))
            .flatten()
    })
}

/// What to call a devfreq GPU: the kernel driver it is bound to, which tells
/// Mali from the others better than a device-tree node name does.
fn gpu_name(entry: &Path, fallback: &str) -> String {
    let driver = fs::read_link(entry.join("device/driver"))
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .or_else(|| {
            fs::read_to_string(entry.join("device/uevent")).ok().and_then(|u| {
                u.lines().find_map(|l| l.strip_prefix("DRIVER=").map(|d| d.trim().to_string()))
            })
        });
    match driver.as_deref() {
        Some(
            d @ ("panfrost" | "lima" | "mali" | "mali_kbase" | "midgard_kbase" | "bifrost_kbase"),
        ) => {
            format!("Mali {d}")
        }
        Some(d) => format!("GPU {d}"),
        None => format!("GPU {fallback}"),
    }
}

/// GPUs behind devfreq, as the DRM scan would have reported them had they
/// been PCI devices. The clock is `cur_freq`, in Hz.
///
/// A device that reports no clock is dropped: a row of dashes is not a GPU.
pub fn scan_devfreq(devfreq_root: &Path, thermal_root: &Path) -> Vec<GpuInfo> {
    let Ok(dir) = fs::read_dir(devfreq_root) else { return Vec::new() };
    let mut entries: Vec<_> =
        dir.flatten().filter(|e| is_gpu_devfreq(&e.file_name().to_string_lossy())).collect();
    entries.sort_by_key(|e| e.file_name());

    let temp = gpu_thermal(thermal_root);
    entries
        .iter()
        .filter_map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let cur_hz = read_u64(&path.join("cur_freq"))?;
            let max_hz = read_u64(&path.join("max_freq"))
                .or_else(|| max_available(&read_trim(&path.join("available_frequencies"))?));
            // Two places a load can live: devfreq's own `load` file, and the
            // vendor Mali driver's `utilisation` beside the device.
            let busy = read_trim(&path.join("load")).and_then(|t| parse_load(&t)).or_else(|| {
                read_trim(&path.join("device/utilisation")).and_then(|t| parse_load(&t))
            });
            Some(GpuInfo {
                name: gpu_name(&path, &name),
                vendor: "ARM".into(),
                busy_percent: busy,
                vram_used: None,
                vram_total: None,
                temp_c: temp,
                freq_mhz: Some(cur_hz / 1_000_000),
                max_freq_mhz: max_hz.map(|h| h / 1_000_000),
            })
        })
        .collect()
}

// ---- VideoCore ----------------------------------------------------------

/// `frequency(46)=500000000` → MHz.
pub fn parse_clock(text: &str) -> Option<u64> {
    let hz: u64 = text.trim().rsplit('=').next()?.trim().parse().ok()?;
    (hz > 0).then_some(hz / 1_000_000)
}

/// `gpu=76M` → bytes. The memory the firmware reserves for the GPU, which is
/// a fixed split on a Pi 3 or earlier and mostly CMA-backed on a 4 and 5.
pub fn parse_mem(text: &str) -> Option<u64> {
    let v = text.trim().rsplit('=').next()?.trim();
    let mib: u64 = v.strip_suffix('M').unwrap_or(v).parse().ok()?;
    (mib > 0).then_some(mib * 1024 * 1024)
}

/// The VideoCore generation, from the board's devicetree model.
pub fn videocore_name(model: Option<&str>) -> String {
    let gen_ = match model {
        Some(m) if m.contains("Raspberry Pi 5") || m.contains("Raspberry Pi 500") => "VII",
        Some(m) if m.contains("Raspberry Pi 4") || m.contains("Raspberry Pi 400") => "VI",
        Some(m) if m.contains("Raspberry Pi 3") || m.contains("Raspberry Pi Zero 2") => "IV",
        _ => "",
    };
    if gen_.is_empty() { "VideoCore".into() } else { format!("VideoCore {gen_}") }
}

/// Whether a DRM card is a VideoCore's: its driver is `v3d` or `vc4`.
pub fn has_videocore(drm_root: &Path) -> bool {
    let Ok(dir) = fs::read_dir(drm_root) else { return false };
    dir.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().to_string();
        crate::metrics::gpu::is_card_dir(&n)
            && fs::read_to_string(e.path().join("device/uevent")).is_ok_and(|u| {
                u.lines().any(|l| {
                    l.strip_prefix("DRIVER=")
                        .is_some_and(|d| matches!(d.trim(), "v3d" | "vc4" | "vc4-drm"))
                })
            })
    })
}

fn vcgencmd(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("vcgencmd").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

/// The SoC's temperature, which on a Pi is also the GPU's: one sensor, one die.
fn soc_temp(thermal_root: &Path) -> Option<f32> {
    read_u64(&thermal_root.join("thermal_zone0/temp")).map(|m| m as f32 / 1000.0)
}

/// How long a `vcgencmd` answer is reused. Each call is a process spawn and a
/// mailbox round trip, and the clock does not change faster than a person can
/// read it.
const VC_TTL: Duration = Duration::from_secs(2);

static VC_CACHE: Mutex<Option<(Instant, Option<GpuInfo>)>> = Mutex::new(None);

/// The Pi's GPU, from the firmware. `None` where there is no `vcgencmd` or no
/// answer, so a Pi without the userland tools simply has no GPU row.
pub fn read_videocore(model: Option<&str>, thermal_root: &Path) -> Option<GpuInfo> {
    if let Ok(cache) = VC_CACHE.lock()
        && let Some((at, v)) = &*cache
        && at.elapsed() < VC_TTL
    {
        return v.clone();
    }
    let info = (|| {
        let freq_mhz = parse_clock(&vcgencmd(&["measure_clock", "v3d"])?)?;
        Some(GpuInfo {
            name: videocore_name(model),
            vendor: "Broadcom".into(),
            // Not available without debugfs: see the module docs.
            busy_percent: None,
            vram_used: None,
            vram_total: vcgencmd(&["get_mem", "gpu"]).and_then(|t| parse_mem(&t)),
            temp_c: soc_temp(thermal_root),
            freq_mhz: Some(freq_mhz),
            max_freq_mhz: None,
        })
    })();
    if let Ok(mut cache) = VC_CACHE.lock() {
        *cache = Some((Instant::now(), info.clone()));
    }
    info
}

/// Every SoC GPU under the given roots.
pub fn read(drm_root: &Path, devfreq_root: &Path, thermal_root: &Path) -> Vec<GpuInfo> {
    let mut out = scan_devfreq(devfreq_root, thermal_root);
    if has_videocore(drm_root) {
        let model = fs::read_to_string("/proc/device-tree/model")
            .ok()
            .map(|m| m.trim_matches(|c: char| c.is_whitespace() || c == '\0').to_string());
        out.extend(read_videocore(model.as_deref(), thermal_root));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crabmon-socgpu-{tag}-{}", std::process::id()));
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
    fn gpu_devfreq_entries_are_told_from_the_memory_and_bus_ones() {
        for gpu in ["ff9a0000.gpu", "fb000000.gpu-mali", "mali0", "11800000.g3d", "1c40000.gpu"] {
            assert!(is_gpu_devfreq(gpu), "{gpu}");
        }
        for other in ["dmc", "ff400000.dmc", "1c62000.ve", "soc:memory-controller"] {
            assert!(!is_gpu_devfreq(other), "{other}");
        }
    }

    #[test]
    fn a_load_file_yields_the_percentage_before_the_at_sign() {
        assert_eq!(parse_load("12@600000000Hz\n"), Some(12.0));
        assert_eq!(parse_load("87"), Some(87.0));
        assert_eq!(parse_load("0@0Hz"), Some(0.0));
        assert_eq!(parse_load("150"), None, "not a percentage");
        assert_eq!(parse_load("nonsense"), None);
        assert_eq!(parse_load(""), None);
    }

    #[test]
    fn the_ceiling_is_the_fastest_listed_frequency_in_any_order() {
        assert_eq!(max_available("200000000 800000000 400000000\n"), Some(800_000_000));
        assert_eq!(max_available(""), None);
        assert_eq!(max_available("junk"), None);
    }

    #[test]
    fn a_mali_behind_devfreq_reports_clock_ceiling_load_and_temperature() {
        let root = tmpdir("mali");
        let dev = root.join("devfreq");
        write(&dev, "ff9a0000.gpu/cur_freq", "600000000\n");
        write(&dev, "ff9a0000.gpu/max_freq", "800000000\n");
        write(&dev, "ff9a0000.gpu/load", "35@600000000Hz\n");
        write(&dev, "ff9a0000.gpu/device/uevent", "DRIVER=panfrost\nOF_NAME=gpu\n");
        // A memory controller beside it, which must not become a GPU.
        write(&dev, "dmc/cur_freq", "1056000000\n");
        let thermal = root.join("thermal");
        write(&thermal, "thermal_zone0/type", "soc-thermal\n");
        write(&thermal, "thermal_zone0/temp", "41000\n");
        write(&thermal, "thermal_zone1/type", "gpu-thermal\n");
        write(&thermal, "thermal_zone1/temp", "47500\n");

        let g = scan_devfreq(&dev, &thermal);
        assert_eq!(g.len(), 1, "{g:?}");
        assert_eq!(g[0].name, "Mali panfrost");
        assert_eq!(g[0].freq_mhz, Some(600));
        assert_eq!(g[0].max_freq_mhz, Some(800));
        assert_eq!(g[0].busy_percent, Some(35.0));
        assert_eq!(g[0].temp_c, Some(47.5), "the GPU's own zone, not the SoC's");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_vendor_mali_driver_reports_load_through_utilisation() {
        let root = tmpdir("kbase");
        let dev = root.join("devfreq");
        write(&dev, "fde60000.gpu/cur_freq", "400000000\n");
        write(&dev, "fde60000.gpu/available_frequencies", "200000000 400000000 700000000\n");
        write(&dev, "fde60000.gpu/device/utilisation", "62\n");
        let g = scan_devfreq(&dev, &root.join("none"));
        assert_eq!(g[0].busy_percent, Some(62.0));
        assert_eq!(
            g[0].max_freq_mhz,
            Some(700),
            "from the frequency table when there is no max_freq"
        );
        assert_eq!(g[0].temp_c, None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_gpu_with_no_clock_is_not_a_row() {
        let root = tmpdir("noclock");
        let dev = root.join("devfreq");
        write(&dev, "ff9a0000.gpu/governor", "simple_ondemand\n");
        assert!(scan_devfreq(&dev, &root).is_empty());
        assert!(scan_devfreq(Path::new("/nonexistent"), &root).is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn vcgencmd_answers_parse() {
        assert_eq!(parse_clock("frequency(46)=500000000\n"), Some(500));
        assert_eq!(parse_clock("frequency(46)=0"), None);
        assert_eq!(parse_clock("error=1 error_msg=\"Command not registered\""), None);
        assert_eq!(parse_mem("gpu=76M\n"), Some(76 * 1024 * 1024));
        assert_eq!(parse_mem("gpu=0M"), None);
        assert_eq!(parse_mem("garbage"), None);
    }

    #[test]
    fn the_videocore_generation_follows_the_board() {
        assert_eq!(videocore_name(Some("Raspberry Pi 5 Model B Rev 1.0")), "VideoCore VII");
        assert_eq!(videocore_name(Some("Raspberry Pi 4 Model B Rev 1.5")), "VideoCore VI");
        assert_eq!(videocore_name(Some("Raspberry Pi 3 Model B Plus")), "VideoCore IV");
        assert_eq!(videocore_name(Some("Raspberry Pi Compute Module 4")), "VideoCore");
        assert_eq!(videocore_name(None), "VideoCore");
    }

    #[test]
    fn a_videocore_is_recognised_by_its_drm_driver_and_a_pc_is_not() {
        let root = tmpdir("vc");
        write(&root, "card0/device/uevent", "DRIVER=v3d\n");
        write(&root, "card1/device/uevent", "DRIVER=vc4-drm\n");
        assert!(has_videocore(&root));

        let pc = tmpdir("pc");
        write(&pc, "card0/device/uevent", "DRIVER=i915\n");
        // A connector node named for the driver must not count.
        write(&pc, "card0-HDMI-A-1/device/uevent", "DRIVER=vc4\n");
        assert!(!has_videocore(&pc));
        assert!(!has_videocore(Path::new("/nonexistent")));
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&pc);
    }

    /// With no `vcgencmd` on the machine the Pi's GPU is simply absent.
    #[test]
    fn reading_this_machine_is_safe() {
        let _ = read(
            Path::new("/sys/class/drm"),
            Path::new("/sys/class/devfreq"),
            Path::new("/sys/class/thermal"),
        );
    }
}
