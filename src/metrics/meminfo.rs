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

/// FreeBSD's equivalent, from its counters: the buffer cache is `buffers`, and
/// pages the VM has parked as inactive or cached are reclaimable on demand,
/// which is what Linux's `Cached` is. Page counts are in pages.
pub fn from_bsd_counters(
    bufspace: u64,
    inactive_pages: u64,
    cache_pages: u64,
    page_size: u64,
) -> MemDetail {
    MemDetail {
        buffers: bufspace,
        cached: inactive_pages.saturating_add(cache_pages).saturating_mul(page_size),
        ..MemDetail::default()
    }
}

/// One integer `sysctl`, whose width the kernel picks: counters are `u_int`,
/// sizes `u_long`. The length it hands back says which came.
#[cfg(target_os = "freebsd")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut buf = [0u8; 8];
    let mut len: libc::size_t = buf.len();
    // SAFETY: `buf` is `len` bytes, the name is NUL-terminated, and no new
    // value is passed in.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (rc, len) {
        (0, 4) => Some(u32::from_ne_bytes(buf[..4].try_into().ok()?) as u64),
        (0, 8) => Some(u64::from_ne_bytes(buf)),
        _ => None,
    }
}

pub fn read() -> MemDetail {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            return parse(&content);
        }
    }
    #[cfg(target_os = "freebsd")]
    {
        if let (Some(buf), Some(page)) =
            (sysctl_u64("vfs.bufspace"), sysctl_u64("vm.stats.vm.v_page_size"))
        {
            let inactive = sysctl_u64("vm.stats.vm.v_inactive_count").unwrap_or(0);
            let cache = sysctl_u64("vm.stats.vm.v_cache_count").unwrap_or(0);
            return from_bsd_counters(buf, inactive, cache, page);
        }
    }
    let _ = fs::metadata("/proc/meminfo");
    MemDetail::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freebsd_counters_become_buffers_and_reclaimable_cache() {
        let d = from_bsd_counters(99_794_944, 1690, 10, 4096);
        assert_eq!(d.buffers, 99_794_944);
        assert_eq!(d.cached, 1700 * 4096);
        assert!(d.is_available());
        assert_eq!(from_bsd_counters(0, 0, 0, 4096), MemDetail::default());
        // Absurd counters saturate instead of wrapping.
        assert_eq!(from_bsd_counters(0, u64::MAX, 1, 4096).cached, u64::MAX);
    }

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
