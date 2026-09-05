//! Human-readable formatting helpers. All pure, all unit-tested.

const BYTE_UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
const RATE_UNITS: [&str; 7] = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s", "PiB/s", "EiB/s"];

fn scale(mut v: f64, table: &'static [&'static str; 7]) -> (f64, &'static str) {
    let mut u = 0;
    while v.abs() >= 1024.0 && u < table.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    (v, table[u])
}

pub fn human_bytes(x: u64) -> String {
    let (v, u) = scale(x as f64, &BYTE_UNITS);
    if u == "B" {
        format!("{v:.0} {u}")
    } else {
        format!("{v:.1} {u}")
    }
}

pub fn human_bps(bps: f64) -> String {
    let bps = if bps.is_finite() { bps.max(0.0) } else { 0.0 };
    let (v, u) = scale(bps, &RATE_UNITS);
    if u == "B/s" {
        format!("{v:.0} {u}")
    } else {
        format!("{v:.1} {u}")
    }
}

/// Compact byte count for narrow table columns: `1.2G`, `430M`, `12K`.
pub fn compact_bytes(x: u64) -> String {
    const SUF: [&str; 6] = ["B", "K", "M", "G", "T", "P"];
    let mut v = x as f64;
    let mut u = 0;
    while v >= 1024.0 && u < SUF.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 || v >= 100.0 {
        format!("{v:.0}{}", SUF[u])
    } else {
        format!("{v:.1}{}", SUF[u])
    }
}

/// `3d 04:12:33`, `04:12:33`, `12:33`.
pub fn human_duration(secs: u64) -> String {
    let d = secs / 86_400;
    let h = (secs % 86_400) / 3_600;
    let m = (secs % 3_600) / 60;
    let s = secs % 60;
    if d > 0 {
        format!("{d}d {h:02}:{m:02}:{s:02}")
    } else if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub fn human_hz_mhz(mhz: u64) -> String {
    if mhz >= 1000 {
        format!("{:.2} GHz", mhz as f64 / 1000.0)
    } else {
        format!("{mhz} MHz")
    }
}

/// Truncate to `max` *characters*, appending `…` when cut.
pub fn truncate_fit(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Parse `100`, `10K`, `1.5M`, `2G` into bytes. Used by the filter language.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (num, mult) = match s.chars().last().unwrap().to_ascii_uppercase() {
        'K' => (&s[..s.len() - 1], 1024u64),
        'M' => (&s[..s.len() - 1], 1024 * 1024),
        'G' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        'T' => (&s[..s.len() - 1], 1024u64.pow(4)),
        'B' => (&s[..s.len() - 1], 1),
        _ => (s, 1),
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_scale_and_label() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        // The regression that started this rewrite: 16 GB of RAM must read as GiB.
        assert_eq!(human_bytes(16_035_467_264), "14.9 GiB");
    }

    #[test]
    fn rates_never_go_negative_or_nan() {
        assert_eq!(human_bps(-5.0), "0 B/s");
        assert_eq!(human_bps(f64::NAN), "0 B/s");
        assert_eq!(human_bps(2048.0), "2.0 KiB/s");
    }

    #[test]
    fn compact_bytes_fits_narrow_columns() {
        assert_eq!(compact_bytes(512), "512B");
        assert_eq!(compact_bytes(475_815_936), "454M");
        assert_eq!(compact_bytes(1024 * 1024 * 1024 * 3 / 2), "1.5G");
        assert!(compact_bytes(u64::MAX).len() <= 6);
    }

    #[test]
    fn durations() {
        assert_eq!(human_duration(59), "00:59");
        assert_eq!(human_duration(3_661), "01:01:01");
        assert_eq!(human_duration(16_020), "04:27:00");
        assert_eq!(human_duration(90_061), "1d 01:01:01");
    }

    #[test]
    fn truncation_is_char_safe() {
        assert_eq!(truncate_fit("abc", 5), "abc");
        assert_eq!(truncate_fit("abcdef", 4), "abc…");
        assert_eq!(truncate_fit("日本語テスト", 3), "日本…");
        assert_eq!(truncate_fit("abc", 0), "");
    }

    #[test]
    fn sizes_parse_with_suffixes() {
        assert_eq!(parse_size("100"), Some(100));
        assert_eq!(parse_size("10K"), Some(10 * 1024));
        assert_eq!(parse_size("1.5M"), Some(1_572_864));
        assert_eq!(parse_size("2g"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_size("nope"), None);
        assert_eq!(parse_size(""), None);
    }

    #[test]
    fn mhz_formatting() {
        assert_eq!(human_hz_mhz(800), "800 MHz");
        assert_eq!(human_hz_mhz(4_200), "4.20 GHz");
    }
}
