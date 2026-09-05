//! Pressure Stall Information — `/proc/pressure/{cpu,memory,io}`.
//!
//! PSI answers "how much time did work actually lose to contention", which
//! neither a CPU percentage nor a load average can. `some` is the share of wall
//! time with at least one task stalled; `full` is the share with *every* task
//! stalled, i.e. the machine doing no useful work at all.

use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PressureLine {
    /// Percent of wall time stalled, averaged over the last 10/60/300 seconds.
    pub avg10: f64,
    pub avg60: f64,
    pub avg300: f64,
    /// Cumulative stall time in microseconds.
    pub total: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pressure {
    pub some: PressureLine,
    /// Absent for CPU on most kernels: a single runnable task is never "full".
    pub full: Option<PressureLine>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PsiSample {
    pub cpu: Option<Pressure>,
    pub memory: Option<Pressure>,
    pub io: Option<Pressure>,
}

impl PsiSample {
    pub fn is_available(&self) -> bool {
        self.cpu.is_some() || self.memory.is_some() || self.io.is_some()
    }

    /// The worst 10-second `some` figure across all three resources, which is
    /// what belongs in a one-line summary.
    pub fn worst_avg10(&self) -> f64 {
        [self.cpu, self.memory, self.io]
            .iter()
            .filter_map(|p| p.map(|p| p.some.avg10))
            .fold(0.0, f64::max)
    }
}

fn parse_line(line: &str) -> Option<(&str, PressureLine)> {
    let mut it = line.split_whitespace();
    let kind = it.next()?;
    if kind != "some" && kind != "full" {
        return None;
    }
    let mut out = PressureLine::default();
    for field in it {
        let (key, value) = field.split_once('=')?;
        match key {
            "avg10" => out.avg10 = value.parse().ok()?,
            "avg60" => out.avg60 = value.parse().ok()?,
            "avg300" => out.avg300 = value.parse().ok()?,
            "total" => out.total = value.parse().ok()?,
            _ => {}
        }
    }
    Some((kind, out))
}

pub fn parse_pressure(content: &str) -> Option<Pressure> {
    let mut some = None;
    let mut full = None;
    for line in content.lines() {
        match parse_line(line) {
            Some(("some", p)) => some = Some(p),
            Some(("full", p)) => full = Some(p),
            _ => {}
        }
    }
    some.map(|some| Pressure { some, full })
}

fn read_one(name: &str) -> Option<Pressure> {
    parse_pressure(&fs::read_to_string(format!("/proc/pressure/{name}")).ok()?)
}

/// Linux 4.20+ with `CONFIG_PSI=y`. Everything else reports nothing.
pub fn read() -> PsiSample {
    if !cfg!(target_os = "linux") {
        return PsiSample::default();
    }
    PsiSample { cpu: read_one("cpu"), memory: read_one("memory"), io: read_one("io") }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Copied verbatim from /proc/pressure/io on the development machine.
    const IO: &str = "some avg10=2.55 avg60=2.70 avg300=2.10 total=308957030
full avg10=2.55 avg60=2.65 avg300=2.06 total=304405703
";
    const CPU: &str = "some avg10=0.00 avg60=0.01 avg300=0.00 total=15945564
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
";

    #[test]
    fn both_pressure_lines_are_parsed() {
        let p = parse_pressure(IO).unwrap();
        assert_eq!(p.some.avg10, 2.55);
        assert_eq!(p.some.avg300, 2.10);
        assert_eq!(p.some.total, 308_957_030);
        let full = p.full.unwrap();
        assert_eq!(full.avg10, 2.55);
        assert_eq!(full.total, 304_405_703);
    }

    #[test]
    fn a_file_with_only_a_some_line_still_parses() {
        // Older kernels omit `full` for CPU entirely.
        let p = parse_pressure("some avg10=0.00 avg60=0.01 avg300=0.00 total=15945564\n").unwrap();
        assert_eq!(p.some.avg60, 0.01);
        assert!(p.full.is_none());
    }

    #[test]
    fn garbage_and_empty_files_yield_nothing_rather_than_zeros() {
        assert!(parse_pressure("").is_none());
        assert!(parse_pressure("not pressure data\n").is_none());
        // A malformed field must not silently produce a partial line.
        assert!(parse_pressure("some avg10=abc avg60=0 avg300=0 total=0\n").is_none());
    }

    #[test]
    fn the_worst_resource_wins_the_summary() {
        let s = PsiSample { cpu: parse_pressure(CPU), memory: None, io: parse_pressure(IO) };
        assert!(s.is_available());
        assert_eq!(s.worst_avg10(), 2.55, "IO is stalling, CPU is not");
        assert_eq!(PsiSample::default().worst_avg10(), 0.0);
        assert!(!PsiSample::default().is_available());
    }
}
