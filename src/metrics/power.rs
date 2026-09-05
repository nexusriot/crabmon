//! Batteries, AC adapters and package power draw.
//!
//! Everything comes from `/sys/class/power_supply` and `/sys/class/powercap`.
//! RAPL energy counters became root-only after CVE-2020-8694, so package watts
//! are reported when readable and simply omitted when they are not.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Battery {
    pub name: String,
    /// 0..100.
    pub percent: f64,
    /// `Charging`, `Discharging`, `Full`, `Not charging`, `Unknown`.
    pub status: String,
    /// Positive while charging, negative while discharging.
    pub power_w: Option<f64>,
    pub energy_wh: Option<f64>,
    pub energy_full_wh: Option<f64>,
    /// Time to empty while discharging, or to full while charging.
    pub time_remaining_secs: Option<u64>,
    /// Design capacity remaining, 0..100 — how worn the cell is.
    pub health_percent: Option<f64>,
}

impl Battery {
    pub fn is_charging(&self) -> bool {
        self.status.eq_ignore_ascii_case("charging")
    }
    pub fn is_discharging(&self) -> bool {
        self.status.eq_ignore_ascii_case("discharging")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PowerSample {
    pub batteries: Vec<Battery>,
    /// `None` when no mains supply is exposed at all.
    pub ac_online: Option<bool>,
    /// Package power draw from RAPL, when the counters are readable.
    pub rapl_watts: Option<f64>,
}

impl PowerSample {
    pub fn is_available(&self) -> bool {
        !self.batteries.is_empty() || self.ac_online.is_some() || self.rapl_watts.is_some()
    }
}

fn read_trim(p: PathBuf) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_num(p: PathBuf) -> Option<f64> {
    read_trim(p)?.parse().ok()
}

/// Batteries report either energy (µWh + µW) or charge (µAh + µA). Charge-based
/// devices need the voltage to become watt-hours.
fn energy_and_power(dir: &Path) -> (Option<f64>, Option<f64>, Option<f64>) {
    let micro = 1_000_000.0;
    if let Some(now) = read_num(dir.join("energy_now")) {
        let full = read_num(dir.join("energy_full"));
        let design = read_num(dir.join("energy_full_design"));
        let power = read_num(dir.join("power_now")).map(|p| p / micro);
        return (
            Some(now / micro),
            full.map(|f| f / micro),
            power.map(|p| (p, full, design)).map(|(p, _, _)| p),
        );
    }
    if let Some(now) = read_num(dir.join("charge_now")) {
        let volts = read_num(dir.join("voltage_now")).map(|v| v / micro).unwrap_or(0.0);
        let full = read_num(dir.join("charge_full"));
        let amps = read_num(dir.join("current_now")).map(|a| a / micro);
        return (
            Some(now / micro * volts),
            full.map(|f| f / micro * volts),
            amps.map(|a| a * volts),
        );
    }
    (None, None, None)
}

/// `energy / power`, guarding against a zero or missing rate.
pub fn time_remaining(energy_wh: f64, power_w: f64) -> Option<u64> {
    if power_w <= 0.0 || energy_wh < 0.0 {
        return None;
    }
    let hours = energy_wh / power_w;
    // A week of runtime means the rate is noise, not a real estimate.
    if !hours.is_finite() || hours > 168.0 {
        None
    } else {
        Some((hours * 3600.0) as u64)
    }
}

fn read_battery(dir: &Path) -> Option<Battery> {
    let name = dir.file_name()?.to_string_lossy().to_string();
    let status = read_trim(dir.join("status")).unwrap_or_else(|| "Unknown".into());
    let (energy_wh, energy_full_wh, power_w) = energy_and_power(dir);

    let percent = read_num(dir.join("capacity")).or_else(|| match (energy_wh, energy_full_wh) {
        (Some(now), Some(full)) if full > 0.0 => Some(now / full * 100.0),
        _ => None,
    })?;

    let design = read_num(dir.join("energy_full_design"))
        .or_else(|| read_num(dir.join("charge_full_design")));
    let full_raw = read_num(dir.join("energy_full")).or_else(|| read_num(dir.join("charge_full")));
    let health_percent = match (full_raw, design) {
        (Some(f), Some(d)) if d > 0.0 => Some((f / d * 100.0).clamp(0.0, 100.0)),
        _ => None,
    };

    let mut battery = Battery {
        name,
        percent: percent.clamp(0.0, 100.0),
        status,
        power_w,
        energy_wh,
        energy_full_wh,
        time_remaining_secs: None,
        health_percent,
    };

    battery.time_remaining_secs = match (battery.is_discharging(), energy_wh, energy_full_wh) {
        (true, Some(now), _) => power_w.and_then(|p| time_remaining(now, p)),
        (false, Some(now), Some(full)) if battery.is_charging() => {
            power_w.and_then(|p| time_remaining(full - now, p))
        }
        _ => None,
    };
    // Discharging batteries report a positive rate; the sign carries the meaning.
    if battery.is_discharging() {
        battery.power_w = battery.power_w.map(|p| -p.abs());
    }
    Some(battery)
}

pub fn scan_supplies(root: &Path) -> (Vec<Battery>, Option<bool>) {
    let Ok(entries) = fs::read_dir(root) else {
        return (Vec::new(), None);
    };
    let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    dirs.sort();

    let mut batteries = Vec::new();
    let mut ac_online = None;
    for dir in dirs {
        match read_trim(dir.join("type")).as_deref() {
            Some("Battery") => {
                if let Some(b) = read_battery(&dir) {
                    batteries.push(b);
                }
            }
            Some("Mains") => {
                if let Some(online) = read_trim(dir.join("online")) {
                    // Any adapter being online counts as on mains.
                    ac_online = Some(ac_online.unwrap_or(false) || online == "1");
                }
            }
            _ => {}
        }
    }
    (batteries, ac_online)
}

/// Cumulative RAPL energy in microjoules, summed over all top-level domains.
pub fn read_rapl_energy_uj(root: &Path) -> Option<u64> {
    let entries = fs::read_dir(root).ok()?;
    let mut total = 0u64;
    let mut found = false;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        // Top-level packages only (`intel-rapl:0`), not their subdomains
        // (`intel-rapl:0:1`), which would double-count.
        if !name.starts_with("intel-rapl:") || name.matches(':').count() != 1 {
            continue;
        }
        if let Some(uj) =
            read_trim(entry.path().join("energy_uj")).and_then(|s| s.parse::<u64>().ok())
        {
            total += uj;
            found = true;
        }
    }
    found.then_some(total)
}

/// Convert an energy delta to average watts over `secs`.
pub fn watts_from_energy(prev_uj: u64, cur_uj: u64, secs: f64) -> Option<f64> {
    if secs <= 0.0 || cur_uj < prev_uj {
        // The counter wrapped (they are 32- or 64-bit and do wrap); skip a
        // sample rather than reporting a spike of thousands of watts.
        return None;
    }
    let joules = (cur_uj - prev_uj) as f64 / 1_000_000.0;
    Some(joules / secs)
}

pub fn read(prev_rapl_uj: &mut Option<u64>, secs: f64) -> PowerSample {
    if !cfg!(target_os = "linux") {
        return PowerSample::default();
    }
    let (batteries, ac_online) = scan_supplies(Path::new("/sys/class/power_supply"));
    let cur = read_rapl_energy_uj(Path::new("/sys/class/powercap"));
    let rapl_watts = match (*prev_rapl_uj, cur) {
        (Some(prev), Some(cur)) => watts_from_energy(prev, cur, secs),
        _ => None,
    };
    *prev_rapl_uj = cur;
    PowerSample { batteries, ac_online, rapl_watts }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crabmon-power-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(path: PathBuf, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// The exact layout of BAT0 on the development machine.
    fn energy_battery(root: &Path, status: &str, power_uw: &str) {
        let bat = root.join("BAT0");
        write(bat.join("type"), "Battery\n");
        write(bat.join("status"), &format!("{status}\n"));
        write(bat.join("capacity"), "98\n");
        write(bat.join("energy_now"), "57400000\n");
        write(bat.join("energy_full"), "58750000\n");
        write(bat.join("energy_full_design"), "60000000\n");
        write(bat.join("power_now"), &format!("{power_uw}\n"));
    }

    #[test]
    fn an_energy_reporting_battery_is_read_in_watt_hours() {
        let root = tmpdir("energy");
        energy_battery(&root, "Discharging", "12000000");
        write(root.join("ADP1/type"), "Mains\n");
        write(root.join("ADP1/online"), "0\n");

        let (batteries, ac) = scan_supplies(&root);
        assert_eq!(batteries.len(), 1);
        let b = &batteries[0];
        assert_eq!(b.percent, 98.0);
        assert_eq!(b.energy_wh, Some(57.4));
        assert_eq!(b.energy_full_wh, Some(58.75));
        assert_eq!(ac, Some(false));
        // 57.4 Wh at 12 W is a little under five hours.
        let secs = b.time_remaining_secs.unwrap();
        assert!((17_000..17_300).contains(&secs), "{secs}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_discharging_battery_reports_negative_watts() {
        let root = tmpdir("sign");
        energy_battery(&root, "Discharging", "12000000");
        let (b, _) = scan_supplies(&root);
        assert_eq!(b[0].power_w, Some(-12.0), "the sign carries the direction");

        let root2 = tmpdir("sign2");
        energy_battery(&root2, "Charging", "20000000");
        let (b, _) = scan_supplies(&root2);
        assert_eq!(b[0].power_w, Some(20.0));
        // Charging estimates the time to *full*, not to empty.
        assert!(b[0].time_remaining_secs.unwrap() < 300, "1.35 Wh at 20 W is minutes");
        fs::remove_dir_all(&root).unwrap();
        fs::remove_dir_all(&root2).unwrap();
    }

    #[test]
    fn battery_health_comes_from_full_against_design_capacity() {
        let root = tmpdir("health");
        energy_battery(&root, "Full", "0");
        let (b, _) = scan_supplies(&root);
        // 58.75 of a designed 60 Wh.
        assert!((b[0].health_percent.unwrap() - 97.9).abs() < 0.1);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_charge_reporting_battery_is_converted_through_its_voltage() {
        // Many phones and handhelds report µAh instead of µWh.
        let root = tmpdir("charge");
        let bat = root.join("BAT0");
        write(bat.join("type"), "Battery\n");
        write(bat.join("status"), "Discharging\n");
        write(bat.join("capacity"), "50\n");
        write(bat.join("charge_now"), "2000000\n"); // 2 Ah
        write(bat.join("charge_full"), "4000000\n");
        write(bat.join("voltage_now"), "12000000\n"); // 12 V
        write(bat.join("current_now"), "1000000\n"); // 1 A

        let (b, _) = scan_supplies(&root);
        assert_eq!(b[0].energy_wh, Some(24.0), "2 Ah x 12 V");
        assert_eq!(b[0].power_w, Some(-12.0), "1 A x 12 V, discharging");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_machine_with_no_power_supplies_reports_nothing() {
        let root = tmpdir("empty");
        let (b, ac) = scan_supplies(&root);
        assert!(b.is_empty());
        assert_eq!(ac, None);
        assert!(!PowerSample::default().is_available());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn rapl_sums_packages_but_not_their_subdomains() {
        let root = tmpdir("rapl");
        write(root.join("intel-rapl:0/energy_uj"), "1000000\n");
        write(root.join("intel-rapl:0:0/energy_uj"), "400000\n"); // core, inside :0
        write(root.join("intel-rapl:0:1/energy_uj"), "100000\n"); // uncore, inside :0
        write(root.join("intel-rapl:1/energy_uj"), "500000\n"); // second package
        assert_eq!(read_rapl_energy_uj(&root), Some(1_500_000));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn unreadable_rapl_counters_yield_none_rather_than_zero_watts() {
        // The real case on this machine: energy_uj is mode 0400, root-owned.
        let root = tmpdir("rapl-denied");
        fs::create_dir_all(root.join("intel-rapl:0")).unwrap();
        assert_eq!(read_rapl_energy_uj(&root), None);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn energy_deltas_become_watts_and_wrapping_is_ignored() {
        assert_eq!(watts_from_energy(0, 10_000_000, 1.0), Some(10.0));
        assert_eq!(watts_from_energy(0, 10_000_000, 2.0), Some(5.0));
        assert_eq!(watts_from_energy(10, 5, 1.0), None, "counter wrapped");
        assert_eq!(watts_from_energy(0, 10, 0.0), None);
    }

    #[test]
    fn implausible_runtimes_are_suppressed() {
        assert_eq!(time_remaining(50.0, 0.0), None);
        assert_eq!(time_remaining(50.0, 0.001), None, "a week+ is noise, not an estimate");
        assert!(time_remaining(50.0, 10.0).is_some());
    }
}
