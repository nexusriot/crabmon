//! The parts of `/proc/meminfo` that a used/total gauge cannot express.
//!
//! "Used" on Linux is a derived number; what people actually want to know is
//! how much of it is reclaimable cache.

use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemDetail {
    pub buffers: u64,
    pub cached: u64,
    /// Reclaimable slab, which `free` folds into its cache column.
    pub sreclaimable: u64,
    pub shmem: u64,
    pub dirty: u64,
    pub writeback: u64,
    pub swap_cached: u64,
}

impl MemDetail {
    /// What `free -h` calls "buff/cache".
    pub fn buff_cache(&self) -> u64 {
        self.buffers + self.cached + self.sreclaimable
    }

    pub fn is_available(&self) -> bool {
        self.buff_cache() > 0
    }
}

/// `/proc/meminfo` values are in kibibytes; everything crabmon carries is bytes.
pub fn parse(content: &str) -> MemDetail {
    let mut out = MemDetail::default();
    for line in content.lines() {
        let Some((key, rest)) = line.split_once(':') else { continue };
        let Some(kib) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        let bytes = kib.saturating_mul(1024);
        match key {
            "Buffers" => out.buffers = bytes,
            "Cached" => out.cached = bytes,
            "SReclaimable" => out.sreclaimable = bytes,
            "Shmem" => out.shmem = bytes,
            "Dirty" => out.dirty = bytes,
            "Writeback" => out.writeback = bytes,
            "SwapCached" => out.swap_cached = bytes,
            _ => {}
        }
    }
    out
}

pub fn read() -> MemDetail {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            return parse(&content);
        }
    }
    let _ = fs::metadata("/proc/meminfo");
    MemDetail::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim from the development machine.
    const SAMPLE: &str = "MemTotal:       15659644 kB
MemAvailable:   10163940 kB
Buffers:         1014068 kB
Cached:          4748196 kB
SwapCached:            0 kB
Dirty:              1392 kB
Writeback:             0 kB
Shmem:            343952 kB
SReclaimable:     505528 kB
";

    #[test]
    fn kibibytes_are_converted_to_bytes() {
        let m = parse(SAMPLE);
        assert_eq!(m.buffers, 1_014_068 * 1024);
        assert_eq!(m.cached, 4_748_196 * 1024);
        assert_eq!(m.dirty, 1_392 * 1024);
        assert_eq!(m.writeback, 0);
    }

    #[test]
    fn buff_cache_matches_what_free_reports() {
        // free(1) sums Cached + Buffers + SReclaimable.
        let m = parse(SAMPLE);
        assert_eq!(m.buff_cache(), (1_014_068 + 4_748_196 + 505_528) * 1024);
        assert!(m.is_available());
    }

    #[test]
    fn missing_and_malformed_lines_are_skipped() {
        let m = parse("Buffers: not-a-number kB\nCached:  100 kB\nnonsense\n");
        assert_eq!(m.buffers, 0);
        assert_eq!(m.cached, 102_400);
        assert!(!parse("").is_available());
    }
}
