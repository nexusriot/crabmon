//! IO scheduling priority: the fourth knob on a process.
//!
//! crabmon can already signal a process, renice it and pin it to a set of
//! cores. The one missing control is the one the question "this backup is
//! killing the box" usually wants: CPU priority does nothing for a process
//! that is saturating a disk, and `renice` on an IO-bound job moves nothing
//! except the nice column.
//!
//! Linux only, and deliberately thin: the encoding and the parsing are pure
//! functions tested on their own, and the syscall is three lines around them.

use std::fmt;

/// `IOPRIO_WHO_PROCESS`.
#[cfg(target_os = "linux")]
const WHO_PROCESS: libc::c_int = 1;

/// Scheduling classes, in the order `ioprio_set` numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoClass {
    /// The scheduler picks a class from the nice value. No level applies.
    None,
    /// Served before everything else. Needs `CAP_SYS_ADMIN`.
    Realtime,
    /// The default class for everything that has not asked.
    BestEffort,
    /// Served only when nothing else wants the disk.
    Idle,
}

impl IoClass {
    pub fn code(self) -> u16 {
        match self {
            IoClass::None => 0,
            IoClass::Realtime => 1,
            IoClass::BestEffort => 2,
            IoClass::Idle => 3,
        }
    }

    pub fn from_code(code: u16) -> Option<IoClass> {
        Some(match code {
            0 => IoClass::None,
            1 => IoClass::Realtime,
            2 => IoClass::BestEffort,
            3 => IoClass::Idle,
            _ => return None,
        })
    }

    /// The spelling `ionice(1)` uses, which is what anyone typing this into a
    /// prompt will reach for.
    pub fn name(self) -> &'static str {
        match self {
            IoClass::None => "none",
            IoClass::Realtime => "rt",
            IoClass::BestEffort => "be",
            IoClass::Idle => "idle",
        }
    }

    /// Whether a priority level means anything for this class. `none` and
    /// `idle` have no levels, so quoting one would be inventing detail.
    pub fn has_level(self) -> bool {
        matches!(self, IoClass::Realtime | IoClass::BestEffort)
    }
}

/// Lowest-priority level within a class; the levels run 0 (highest) to 7.
pub const MAX_LEVEL: u8 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoPriority {
    pub class: IoClass,
    pub level: u8,
}

impl fmt::Display for IoPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.class.has_level() {
            true => write!(f, "{} {}", self.class.name(), self.level),
            false => write!(f, "{}", self.class.name()),
        }
    }
}

impl IoPriority {
    pub fn new(class: IoClass, level: u8) -> IoPriority {
        IoPriority { class, level: level.min(MAX_LEVEL) }
    }

    /// `IOPRIO_PRIO_VALUE`: the class in the top three bits, the level under it.
    pub fn encode(self) -> u16 {
        (self.class.code() << 13) | (self.level.min(MAX_LEVEL) as u16 & 0x1fff)
    }

    pub fn decode(raw: u16) -> Option<IoPriority> {
        let class = IoClass::from_code(raw >> 13)?;
        Some(IoPriority { class, level: (raw & 0x1fff).min(MAX_LEVEL as u16) as u8 })
    }
}

/// Parse what someone types at the prompt: `be`, `be4`, `be 4`, `idle`, `rt 0`.
///
/// A level on a class that has none is refused rather than ignored, because
/// "idle 3" looks like it asked for something and silently did not.
pub fn parse(input: &str) -> Result<IoPriority, String> {
    let text = input.trim().to_ascii_lowercase();
    if text.is_empty() {
        return Err("needs a class: none, rt, be or idle".into());
    }
    let (name, rest) = match text.split_once(|c: char| c.is_whitespace() || c.is_ascii_digit()) {
        // Put the digit back: splitting on it consumed one character.
        Some((head, _)) if !head.is_empty() => {
            let at = head.len();
            (head.to_string(), text[at..].trim().to_string())
        }
        _ => (text.clone(), String::new()),
    };
    let class = match name.as_str() {
        "none" | "0" => IoClass::None,
        "rt" | "realtime" | "1" => IoClass::Realtime,
        "be" | "best-effort" | "besteffort" | "2" => IoClass::BestEffort,
        "idle" | "3" => IoClass::Idle,
        other => return Err(format!("unknown class: {other} (none, rt, be, idle)")),
    };
    if rest.is_empty() {
        // `be` on its own is the class's middle level, which is what the
        // kernel would have picked from a default nice value anyway.
        return Ok(IoPriority::new(class, if class.has_level() { 4 } else { 0 }));
    }
    if !class.has_level() {
        return Err(format!("{} takes no level", class.name()));
    }
    let level: u8 = rest.parse().map_err(|_| format!("bad level: {rest}"))?;
    if level > MAX_LEVEL {
        return Err(format!("level must be 0-{MAX_LEVEL}"));
    }
    Ok(IoPriority::new(class, level))
}

/// Read a process's IO priority.
#[cfg(target_os = "linux")]
pub fn get(pid: u32) -> Option<IoPriority> {
    // SAFETY: `ioprio_get` takes two integers and reports failure as -1.
    let raw = unsafe { libc::syscall(libc::SYS_ioprio_get, WHO_PROCESS, pid as libc::c_int) };
    if raw < 0 {
        return None;
    }
    IoPriority::decode(raw as u16)
}

#[cfg(not(target_os = "linux"))]
pub fn get(_pid: u32) -> Option<IoPriority> {
    None
}

/// Set a process's IO priority.
#[cfg(target_os = "linux")]
pub fn set(pid: u32, prio: IoPriority) -> std::io::Result<()> {
    // SAFETY: `ioprio_set` takes three integers and reports failure as -1.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_ioprio_set,
            WHO_PROCESS,
            pid as libc::c_int,
            prio.encode() as libc::c_int,
        )
    };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn set(_pid: u32, _prio: IoPriority) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "ionice is Linux-only"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encoding_matches_ioprio_prio_value() {
        // `ionice -c2 -n4` is 0x4004 in the kernel's encoding.
        assert_eq!(IoPriority::new(IoClass::BestEffort, 4).encode(), (2 << 13) | 4);
        assert_eq!(IoPriority::new(IoClass::Idle, 0).encode(), 3 << 13);
        assert_eq!(IoPriority::new(IoClass::Realtime, 7).encode(), (1 << 13) | 7);
        assert_eq!(IoPriority::new(IoClass::None, 0).encode(), 0);
    }

    #[test]
    fn every_priority_survives_a_round_trip() {
        for class in [IoClass::None, IoClass::Realtime, IoClass::BestEffort, IoClass::Idle] {
            for level in 0..=MAX_LEVEL {
                let p = IoPriority::new(class, level);
                assert_eq!(IoPriority::decode(p.encode()), Some(p), "{p}");
            }
        }
    }

    #[test]
    fn the_spellings_ionice_uses_are_accepted() {
        assert_eq!(parse("be 4"), Ok(IoPriority::new(IoClass::BestEffort, 4)));
        assert_eq!(parse("be4"), Ok(IoPriority::new(IoClass::BestEffort, 4)));
        assert_eq!(parse("BE 0"), Ok(IoPriority::new(IoClass::BestEffort, 0)));
        assert_eq!(parse("idle"), Ok(IoPriority::new(IoClass::Idle, 0)));
        assert_eq!(parse("rt 7"), Ok(IoPriority::new(IoClass::Realtime, 7)));
        assert_eq!(parse("none"), Ok(IoPriority::new(IoClass::None, 0)));
        // A bare class takes the middle of its range.
        assert_eq!(parse("be"), Ok(IoPriority::new(IoClass::BestEffort, 4)));
    }

    #[test]
    fn a_level_on_a_class_that_has_none_is_refused_rather_than_ignored() {
        assert!(parse("idle 3").is_err());
        assert!(parse("none 1").is_err());
    }

    #[test]
    fn nonsense_is_refused() {
        assert!(parse("").is_err());
        assert!(parse("be 9").is_err());
        assert!(parse("be -1").is_err());
        assert!(parse("quickly").is_err());
    }

    #[test]
    fn a_priority_reads_as_ionice_writes_it() {
        assert_eq!(IoPriority::new(IoClass::BestEffort, 4).to_string(), "be 4");
        assert_eq!(IoPriority::new(IoClass::Idle, 0).to_string(), "idle");
    }

    /// The class the kernel reports for an unset process decodes rather than
    /// reading as corrupt.
    #[test]
    fn a_raw_zero_decodes_to_none() {
        assert_eq!(IoPriority::decode(0), Some(IoPriority::new(IoClass::None, 0)));
    }
}
