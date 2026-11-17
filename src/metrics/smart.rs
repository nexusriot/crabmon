//! Disk health from SMART, through `smartctl --json`. Opt-in (`[smart]`).
//!
//! The md, ZFS and btrfs checks in `storage` say a mirror has *lost* a disk.
//! SMART is the earlier warning: a disk remapping sectors, sectors it could
//! not read and is waiting to retry, an NVMe that has used up its rated
//! endurance or raised a critical warning. Each is a disk that still works
//! and is on its way out, which is the cheapest moment to replace it.
//!
//! Reading it needs a privileged ioctl — `CAP_SYS_RAWIO` for ATA
//! pass-through, `CAP_SYS_ADMIN` for NVMe admin commands — so it is off by
//! default and run through a configurable command: `smartctl` as root, or
//! `sudo -n smartctl` with a sudoers rule for just that binary. A disk crabmon
//! cannot open is not a healthy disk: the reason is carried in
//! `Snapshot::smart_unreadable` rather than a row that says "ok".
//!
//! A reading costs tens of milliseconds a disk and SMART changes on the scale
//! of hours, so it runs on a thread of its own every `interval_secs` (ten
//! minutes by default) with `-n standby`: a disk that has spun down is left
//! asleep and keeps its last reading, rather than being woken every ten
//! minutes by the thing watching it.
//!
//! Each disk becomes a `StorageArray` of kind `smart`, so the Disks panel,
//! the `storage` alert kind, `crabmon_storage_degraded`, `--remote`,
//! recordings and incident summaries all carry it without a second path.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::storage::{ArrayState, StorageArray};

/// What SMART says about one disk, beyond its overall state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SmartInfo {
    /// The disk's own self-assessment, `smart_status.passed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_c: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_on_hours: Option<u64>,
    /// NVMe "percentage used" of rated endurance. Can exceed 100.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wear_percent: Option<f64>,
    /// ATA attribute 5, or a SCSI disk's grown defect list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reallocated: Option<u64>,
    /// ATA attribute 197: sectors that failed a read and await a rewrite.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<u64>,
    /// ATA attribute 198 (or 187 where 198 is absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncorrectable: Option<u64>,
    /// NVMe media and data integrity errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_errors: Option<u64>,
    /// NVMe critical warning bits; 0 is none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub critical_warning: Option<u64>,
    /// NVMe spare capacity left, and the level below which it warns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spare_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spare_threshold: Option<f64>,
}

/// What one `smartctl` run said.
#[derive(Debug, Clone, PartialEq)]
pub enum Reading {
    Disk(Box<StorageArray>),
    /// Spun down; `-n standby` declined to wake it.
    Standby,
    /// Could not be read, and why: `Permission denied`, no such device.
    Unreadable(String),
}

/// A disk to read: its path and, from `--scan`, the `-d` type to read it as.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub path: String,
    pub kind: Option<String>,
}

impl Device {
    /// `sda`, `nvme0`, or `bus/0[megaraid,3]` where several disks share a path.
    pub fn name(&self) -> String {
        let base = self.path.strip_prefix("/dev/").unwrap_or(&self.path).to_string();
        match &self.kind {
            Some(k) if k.contains(',') => format!("{base}[{k}]"),
            _ => base,
        }
    }
}

/// The devices `smartctl --scan --json` lists.
pub fn parse_scan(json: &str) -> Vec<Device> {
    let Ok(v) = serde_json::from_str::<Value>(json) else { return Vec::new() };
    v["devices"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|d| {
                    Some(Device {
                        path: d["name"].as_str()?.to_string(),
                        kind: d["type"].as_str().map(str::to_string),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn error_message(v: &Value) -> Option<String> {
    let msgs = v["smartctl"]["messages"].as_array()?;
    msgs.iter()
        .find(|m| m["severity"].as_str() == Some("error"))
        .or_else(|| msgs.first())
        .and_then(|m| m["string"].as_str())
        .map(|s| s.trim().to_string())
}

fn u(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
}

/// One device's `smartctl --json -i -H -A` output.
pub fn parse_device(json: &str, name: &str) -> Reading {
    let Ok(v) = serde_json::from_str::<Value>(json) else {
        return Reading::Unreadable("smartctl gave no JSON (needs smartmontools 7.0+)".into());
    };
    let status = u(&v["smartctl"]["exit_status"]).unwrap_or(0);
    let message = error_message(&v);
    if message.as_deref().is_some_and(|m| m.to_ascii_lowercase().contains("standby")) {
        return Reading::Standby;
    }
    // Bit 0: the command line was refused. Bit 1: the device could not be
    // opened, or did not answer IDENTIFY. Either way there is no reading.
    if status & 0b11 != 0 {
        return Reading::Unreadable(message.unwrap_or_else(|| format!("exit status {status}")));
    }
    // A virtual disk or a USB bridge answers IDENTIFY and then has no SMART
    // to give. Read on as if it had, it drew a "healthy" disk at 0°C — which
    // is a verdict on a disk nothing measured. Found on a QEMU guest.
    match (v["smart_support"]["available"].as_bool(), v["smart_support"]["enabled"].as_bool()) {
        (Some(false), _) => {
            return Reading::Unreadable("no SMART support (a virtual or USB-bridged disk)".into());
        }
        (_, Some(false)) => {
            return Reading::Unreadable(
                "SMART is off on this disk; `smartctl -s on` enables it".into(),
            );
        }
        _ => {}
    }

    let mut info = SmartInfo {
        passed: v["smart_status"]["passed"].as_bool(),
        // 0 is what smartctl prints for "not reported", not a reading.
        temp_c: v["temperature"]["current"].as_f64().filter(|t| *t > 0.0),
        power_on_hours: u(&v["power_on_time"]["hours"]),
        ..Default::default()
    };
    // ATA: the attributes that mean "this disk has started to fail".
    let mut failing_attr = false;
    if let Some(table) = v["ata_smart_attributes"]["table"].as_array() {
        let mut reported_uncorrect = None;
        for attr in table {
            let raw = u(&attr["raw"]["value"]);
            match u(&attr["id"]) {
                Some(5) => info.reallocated = raw,
                Some(197) => info.pending = raw,
                Some(198) => info.uncorrectable = raw,
                Some(187) => reported_uncorrect = raw,
                _ => {}
            }
            // A pre-fail attribute below its threshold *now* is the drive
            // saying it expects to fail; "past" is history and stays out.
            if attr["when_failed"].as_str() == Some("now") {
                failing_attr = true;
            }
        }
        info.uncorrectable = info.uncorrectable.or(reported_uncorrect);
    }
    let nvme = &v["nvme_smart_health_information_log"];
    if nvme.is_object() {
        info.critical_warning = u(&nvme["critical_warning"]);
        info.wear_percent = nvme["percentage_used"].as_f64();
        info.media_errors = u(&nvme["media_errors"]);
        info.spare_percent = nvme["available_spare"].as_f64();
        info.spare_threshold = nvme["available_spare_threshold"].as_f64();
        info.temp_c = info.temp_c.or_else(|| nvme["temperature"].as_f64().filter(|t| *t > 0.0));
        info.power_on_hours = info.power_on_hours.or_else(|| u(&nvme["power_on_hours"]));
    }
    if let Some(grown) = u(&v["scsi_grown_defect_list"]) {
        info.reallocated = Some(grown);
    }

    let (state, reasons) = judge(&info, status, failing_attr);
    let mut detail = reasons;
    if let Some(t) = info.temp_c {
        detail.push(format!("{t:.0}°C"));
    }
    if let Some(w) = info.wear_percent {
        detail.push(format!("wear {w:.0}%"));
    }
    let model = v["model_name"]
        .as_str()
        .or_else(|| v["scsi_model_name"].as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    Reading::Disk(Box::new(StorageArray {
        kind: "smart".into(),
        name: name.to_string(),
        level: model,
        state,
        detail: detail.join(" "),
        smart: Some(info),
        ..Default::default()
    }))
}

/// Healthy, degraded (started to fail) or failed (the disk says so), and why.
///
/// Any remapped, pending or uncorrectable sector counts as degraded: a disk
/// with one has started to wear out in a way that tends to accelerate, and a
/// count that never moves is still worth knowing about once. The same goes
/// for NVMe media errors, a critical warning, spare capacity under its
/// threshold, and endurance used up.
fn judge(info: &SmartInfo, status: u64, failing_attr: bool) -> (ArrayState, Vec<String>) {
    let mut reasons = Vec::new();
    // Bit 3: "SMART status check returned DISK FAILING".
    let failing = info.passed == Some(false) || status & 0b1000 != 0;
    if failing {
        reasons.push("FAILING".to_string());
    } else if failing_attr {
        reasons.push("pre-fail attribute".to_string());
    }
    let mut count = |n: Option<u64>, what: &str| {
        if let Some(n) = n.filter(|n| *n > 0) {
            reasons.push(format!("{n} {what}"));
        }
    };
    count(info.reallocated, "reallocated");
    count(info.pending, "pending");
    count(info.uncorrectable, "uncorrectable");
    count(info.media_errors, "media errors");
    if let Some(w) = info.critical_warning.filter(|w| *w != 0) {
        reasons.push(format!("critical warning 0x{w:x}"));
    }
    if let (Some(spare), Some(min)) = (info.spare_percent, info.spare_threshold)
        && min > 0.0
        && spare < min
    {
        reasons.push(format!("spare {spare:.0}%<{min:.0}%"));
    }
    if info.wear_percent.is_some_and(|w| w >= 100.0) {
        reasons.push("worn out".to_string());
    }
    let state = match (failing || failing_attr, reasons.is_empty()) {
        (true, _) => ArrayState::Failed,
        (false, false) => ArrayState::Degraded,
        (false, true) => ArrayState::Healthy,
    };
    (state, reasons)
}

/// How the watcher is configured; see `[smart]`.
#[derive(Debug, Clone)]
pub struct SmartSettings {
    /// The program and any leading arguments: `["smartctl"]`, or
    /// `["sudo", "-n", "smartctl"]`.
    pub command: Vec<String>,
    /// Device paths to read; empty means whatever `--scan` finds.
    pub devices: Vec<String>,
    pub interval: Duration,
}

/// How long one `smartctl` may take before it is killed. A disk that does not
/// answer IDENTIFY in this long is a finding of its own.
const RUN_TIMEOUT: Duration = Duration::from_secs(30);

fn run(command: &[String], args: &[&str]) -> Result<String, String> {
    let Some((program, lead)) = command.split_first() else {
        return Err("[smart] command is empty".into());
    };
    let child = std::process::Command::new(program)
        .args(lead)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("{program}: {e}"))?;
    let out = crate::remote::wait_with_timeout(child, RUN_TIMEOUT)
        .map_err(|e| format!("{program}: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    // smartctl reports through its JSON and a bit-mapped exit status, so a
    // non-zero exit is not itself a failure. Something that is not smartctl
    // — `sudo` refusing — writes nothing to stdout.
    if stdout.trim().is_empty() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(match err.is_empty() {
            true => format!("{program} printed nothing (exit {})", out.status),
            false => err,
        });
    }
    Ok(stdout)
}

/// The devices to read: the configured list, or what `--scan` finds.
fn devices(settings: &SmartSettings) -> Result<Vec<Device>, String> {
    if !settings.devices.is_empty() {
        return Ok(settings
            .devices
            .iter()
            .map(|p| Device { path: p.clone(), kind: None })
            .collect());
    }
    let found = parse_scan(&run(&settings.command, &["--scan", "--json"])?);
    match found.is_empty() {
        true => Err("smartctl --scan found no disks".into()),
        false => Ok(found),
    }
}

fn read_device(command: &[String], dev: &Device) -> Reading {
    let mut args = vec!["--json", "-i", "-H", "-A", "-n", "standby"];
    if let Some(kind) = &dev.kind {
        args.extend(["-d", kind.as_str()]);
    }
    args.push(&dev.path);
    match run(command, &args) {
        Ok(out) => parse_device(&out, &dev.name()),
        Err(e) => Reading::Unreadable(e),
    }
}

/// One pass over every disk: rows worst first, and why nothing could be read
/// when nothing could. `last` keeps a sleeping disk's previous reading.
pub fn read_all(
    last: &mut HashMap<String, StorageArray>,
    read: &dyn Fn(&Device) -> Reading,
    list: &dyn Fn() -> Result<Vec<Device>, String>,
) -> (Vec<StorageArray>, Option<String>) {
    let devices = match list() {
        Ok(d) => d,
        Err(e) => return (Vec::new(), Some(e)),
    };
    let mut rows = Vec::new();
    let mut first_error = None;
    for dev in &devices {
        let name = dev.name();
        match read(dev) {
            Reading::Disk(row) => {
                let row = *row;
                last.insert(name, row.clone());
                rows.push(row);
            }
            Reading::Standby => {
                if let Some(row) = last.get(&name) {
                    rows.push(row.clone());
                }
            }
            Reading::Unreadable(e) => {
                last.remove(&name);
                first_error.get_or_insert(format!("{name}: {e}"));
            }
        }
    }
    rows.sort_by(|a, b| (b.state as u8).cmp(&(a.state as u8)).then_with(|| a.name.cmp(&b.name)));
    // Some disks read and some not is reported by the rows that are there;
    // none at all has to say why, or it reads as a machine with no disks.
    let unreadable = match rows.is_empty() {
        true => Some(first_error.unwrap_or_else(|| "every disk is in standby".into())),
        false => None,
    };
    (rows, unreadable)
}

type Latest = (Vec<StorageArray>, Option<String>);

#[derive(Default)]
struct Shared {
    latest: std::sync::Mutex<Option<Latest>>,
    ready: std::sync::Condvar,
}

/// Reads SMART every `interval` on a thread of its own.
pub struct SmartWatcher {
    shared: std::sync::Arc<Shared>,
    /// Dropped with the watcher, which is how the thread learns to stop.
    _stop: std::sync::mpsc::Sender<()>,
}

/// What a snapshot says before the first pass has finished.
pub const STILL_READING: &str = "still reading SMART";

impl SmartWatcher {
    /// How long the first sample waits for a first pass, so `--once` has one.
    const FIRST_WAIT: Duration = Duration::from_secs(3);

    pub fn spawn(settings: SmartSettings) -> SmartWatcher {
        let command = settings.command.clone();
        let list_settings = settings.clone();
        Self::spawn_with(
            settings,
            move |dev: &Device| read_device(&command, dev),
            move || devices(&list_settings),
        )
    }

    pub fn spawn_with(
        settings: SmartSettings,
        read: impl Fn(&Device) -> Reading + Send + 'static,
        list: impl Fn() -> Result<Vec<Device>, String> + Send + 'static,
    ) -> SmartWatcher {
        let shared = std::sync::Arc::new(Shared::default());
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let worker = shared.clone();
        let _ = std::thread::Builder::new().name("crabmon-smart".into()).spawn(move || {
            let mut last = HashMap::new();
            loop {
                let result = read_all(&mut last, &read, &list);
                if let Ok(mut latest) = worker.latest.lock() {
                    *latest = Some(result);
                }
                worker.ready.notify_all();
                match stopped.recv_timeout(settings.interval) {
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    _ => break,
                }
            }
        });
        SmartWatcher { shared, _stop: stop }
    }

    /// The last pass. Waits up to `FIRST_WAIT` for the first one.
    pub fn latest(&self) -> Latest {
        let Ok(mut latest) = self.shared.latest.lock() else {
            return (Vec::new(), Some("SMART watcher failed".into()));
        };
        if latest.is_none() {
            latest = match self.shared.ready.wait_timeout(latest, Self::FIRST_WAIT) {
                Ok((guard, _)) => guard,
                Err(_) => return (Vec::new(), Some("SMART watcher failed".into())),
            };
        }
        latest.clone().unwrap_or_else(|| (Vec::new(), Some(STILL_READING.into())))
    }
}

/// One pass now, for `--doctor`: how many disks read, or why none did.
pub fn probe(settings: &SmartSettings) -> Result<usize, String> {
    let command = settings.command.clone();
    let (rows, why) =
        read_all(&mut HashMap::new(), &|dev| read_device(&command, dev), &|| devices(settings));
    match why {
        Some(e) => Err(e),
        None => Ok(rows.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ata(attrs: &str, passed: bool, status: u64) -> String {
        format!(
            r#"{{"smartctl":{{"exit_status":{status}}},"model_name":"WDC WD40EFRX-68N32N0",
            "smart_status":{{"passed":{passed}}},"temperature":{{"current":38}},
            "power_on_time":{{"hours":31337}},
            "ata_smart_attributes":{{"table":[{attrs}]}}}}"#
        )
    }

    fn attr(id: u64, raw: u64, when_failed: &str) -> String {
        format!(r#"{{"id":{id},"raw":{{"value":{raw}}},"when_failed":"{when_failed}"}}"#)
    }

    fn disk(r: Reading) -> StorageArray {
        match r {
            Reading::Disk(d) => *d,
            other => panic!("expected a disk, got {other:?}"),
        }
    }

    #[test]
    fn a_clean_ata_disk_is_healthy_and_says_its_temperature() {
        let json = ata(&[attr(5, 0, ""), attr(197, 0, ""), attr(198, 0, "")].join(","), true, 0);
        let d = disk(parse_device(&json, "sda"));
        assert_eq!(d.kind, "smart");
        assert_eq!(d.state, ArrayState::Healthy);
        assert_eq!(d.level, "WDC WD40EFRX-68N32N0");
        assert_eq!(d.detail, "38°C");
        let s = d.smart.unwrap();
        assert_eq!(s.power_on_hours, Some(31337));
        assert_eq!(s.reallocated, Some(0));
    }

    #[test]
    fn remapped_and_pending_sectors_are_a_disk_that_has_started_to_fail() {
        let json = ata(&[attr(5, 8, ""), attr(197, 2, ""), attr(198, 0, "")].join(","), true, 0);
        let d = disk(parse_device(&json, "sdb"));
        assert_eq!(d.state, ArrayState::Degraded);
        assert_eq!(d.detail, "8 reallocated 2 pending 38°C");
    }

    #[test]
    fn a_failed_self_assessment_or_a_prefail_attribute_now_is_failed() {
        let d = disk(parse_device(&ata(&attr(5, 0, ""), false, 8), "sdc"));
        assert_eq!(d.state, ArrayState::Failed);
        assert!(d.detail.starts_with("FAILING"), "{}", d.detail);

        let d = disk(parse_device(&ata(&attr(1, 0, "now"), true, 16), "sdd"));
        assert_eq!(d.state, ArrayState::Failed);
        // A threshold crossed in the past is history, not a failure.
        let d = disk(parse_device(&ata(&attr(1, 0, "past"), true, 32), "sde"));
        assert_eq!(d.state, ArrayState::Healthy);
    }

    #[test]
    fn attribute_187_stands_in_when_198_is_absent() {
        let d = disk(parse_device(&ata(&attr(187, 3, ""), true, 0), "sdf"));
        assert_eq!(d.smart.unwrap().uncorrectable, Some(3));
        assert_eq!(d.state, ArrayState::Degraded);
    }

    const NVME: &str = r#"{"smartctl":{"exit_status":0},"model_name":"Samsung SSD 980 PRO 1TB",
        "smart_status":{"passed":true},
        "nvme_smart_health_information_log":{"critical_warning":0,"temperature":41,
          "available_spare":100,"available_spare_threshold":10,"percentage_used":3,
          "power_on_hours":5021,"media_errors":0}}"#;

    #[test]
    fn an_nvme_drive_reports_wear_temperature_and_hours() {
        let d = disk(parse_device(NVME, "nvme0"));
        assert_eq!(d.state, ArrayState::Healthy);
        assert_eq!(d.detail, "41°C wear 3%");
        let s = d.smart.unwrap();
        assert_eq!(s.power_on_hours, Some(5021));
        assert_eq!(s.wear_percent, Some(3.0));
    }

    #[test]
    fn nvme_media_errors_warnings_spare_and_wear_each_degrade_it() {
        for (field, value, expect) in [
            ("media_errors", "4", "4 media errors"),
            ("critical_warning", "4", "critical warning 0x4"),
            ("available_spare", "5", "spare 5%<10%"),
            ("percentage_used", "104", "worn out"),
        ] {
            let json = NVME.replacen(
                &format!("\"{field}\":"),
                &format!("\"{field}\":{value},\"x_{field}\":"),
                1,
            );
            let d = disk(parse_device(&json, "nvme0"));
            assert_eq!(d.state, ArrayState::Degraded, "{field}");
            assert!(d.detail.contains(expect), "{field}: {}", d.detail);
        }
    }

    /// What smartctl 7.5 says about a QEMU guest's disk, trimmed. It used to
    /// be drawn as a healthy disk at 0°C.
    const QEMU: &str = r#"{"smartctl":{"version":[7,5],"exit_status":0},
        "device":{"name":"/dev/sda","type":"scsi","protocol":"SCSI"},
        "model_name":"QEMU QEMU HARDDISK","scsi_model_name":"QEMU QEMU HARDDISK",
        "rotation_rate":0,"smart_support":{"available":false},
        "temperature":{"current":0,"drive_trip":0}}"#;

    #[test]
    fn a_disk_without_smart_is_unreadable_rather_than_healthy() {
        match parse_device(QEMU, "sda") {
            Reading::Unreadable(e) => assert!(e.contains("no SMART support"), "{e}"),
            other => panic!("a disk nothing measured got a verdict: {other:?}"),
        }
        let off = QEMU.replace(r#"{"available":false}"#, r#"{"available":true,"enabled":false}"#);
        match parse_device(&off, "sda") {
            Reading::Unreadable(e) => assert!(e.contains("smartctl -s on"), "{e}"),
            other => panic!("{other:?}"),
        }
        // A temperature of 0 is "not reported".
        let on = QEMU
            .replace(r#"{"available":false}"#, r#"{"available":true,"enabled":true}"#)
            .replace(r#""smartctl":{"#, r#""smart_status":{"passed":true},"smartctl":{"#);
        let d = disk(parse_device(&on, "sda"));
        assert_eq!(d.smart.unwrap().temp_c, None);
        assert_eq!(d.detail, "");
    }

    #[test]
    fn a_device_that_cannot_be_opened_is_unreadable_not_healthy() {
        let json = r#"{"smartctl":{"exit_status":2,"messages":[{"string":"Smartctl open device: /dev/nvme0 failed: Permission denied","severity":"error"}]}}"#;
        match parse_device(json, "nvme0") {
            Reading::Unreadable(e) => assert!(e.contains("Permission denied"), "{e}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(parse_device("not json", "x"), Reading::Unreadable(_)));
    }

    #[test]
    fn a_disk_in_standby_keeps_its_last_reading_and_is_not_woken() {
        let standby = r#"{"smartctl":{"exit_status":2,"messages":[{"string":"Device is in STANDBY mode, exit(2)","severity":"information"}]}}"#;
        assert_eq!(parse_device(standby, "sda"), Reading::Standby);

        let list = || Ok(vec![Device { path: "/dev/sda".into(), kind: Some("sat".into()) }]);
        let mut last = HashMap::new();
        let awake = |_: &Device| parse_device(&ata(&attr(5, 1, ""), true, 0), "sda");
        let (rows, why) = read_all(&mut last, &awake, &list);
        assert_eq!((rows.len(), why), (1, None));
        let asleep = |_: &Device| Reading::Standby;
        let (rows, why) = read_all(&mut last, &asleep, &list);
        assert_eq!(rows.len(), 1, "the last reading stands");
        assert_eq!(rows[0].state, ArrayState::Degraded);
        assert!(why.is_none());
    }

    #[test]
    fn nothing_readable_says_why_and_a_partial_read_does_not() {
        let list = || {
            Ok(vec![
                Device { path: "/dev/sda".into(), kind: None },
                Device { path: "/dev/sdb".into(), kind: None },
            ])
        };
        let denied = |_: &Device| Reading::Unreadable("Permission denied".into());
        let (rows, why) = read_all(&mut HashMap::new(), &denied, &list);
        assert!(rows.is_empty());
        assert_eq!(why.as_deref(), Some("sda: Permission denied"));

        let one = |d: &Device| match d.path.as_str() {
            "/dev/sda" => parse_device(NVME, "sda"),
            _ => Reading::Unreadable("gone".into()),
        };
        let (rows, why) = read_all(&mut HashMap::new(), &one, &list);
        assert_eq!((rows.len(), why), (1, None));

        let none = || Err::<Vec<Device>, _>("smartctl: No such file or directory".to_string());
        let (_, why) = read_all(&mut HashMap::new(), &denied, &none);
        assert!(why.unwrap().contains("No such file"));
    }

    #[test]
    fn rows_come_out_worst_first() {
        let list = || {
            Ok(["a", "b", "c"]
                .iter()
                .map(|n| Device { path: format!("/dev/sd{n}"), kind: None })
                .collect())
        };
        let read = |d: &Device| {
            let json = match d.path.as_str() {
                "/dev/sdb" => ata(&attr(197, 1, ""), true, 0),
                "/dev/sdc" => ata(&attr(5, 0, ""), false, 8),
                _ => ata(&attr(5, 0, ""), true, 0),
            };
            parse_device(&json, &d.name())
        };
        let (rows, _) = read_all(&mut HashMap::new(), &read, &list);
        let names: Vec<_> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["sdc", "sdb", "sda"]);
    }

    #[test]
    fn scan_output_lists_devices_with_their_types() {
        let json = r#"{"devices":[{"name":"/dev/sda","type":"sat"},{"name":"/dev/bus/0","type":"megaraid,3"},{"name":"/dev/nvme0","type":"nvme"}]}"#;
        let devs = parse_scan(json);
        let names: Vec<_> = devs.iter().map(Device::name).collect();
        assert_eq!(names, ["sda", "bus/0[megaraid,3]", "nvme0"]);
        assert!(parse_scan("garbage").is_empty());
    }

    #[test]
    fn the_watcher_publishes_its_first_pass_and_says_so_when_it_has_none() {
        let settings =
            SmartSettings { command: vec![], devices: vec![], interval: Duration::from_secs(60) };
        let w = SmartWatcher::spawn_with(
            settings,
            |_| parse_device(NVME, "nvme0"),
            || Ok(vec![Device { path: "/dev/nvme0".into(), kind: None }]),
        );
        let (rows, why) = w.latest();
        assert_eq!(rows.len(), 1);
        assert!(why.is_none());
    }
}
