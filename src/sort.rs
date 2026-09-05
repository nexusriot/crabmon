//! Process sort keys. Every column is sortable and every comparison falls back
//! to PID so equal values stop shuffling between frames.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::metrics::ProcRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortBy {
    Pid,
    Name,
    Cpu,
    Mem,
    Virt,
    Disk,
    Time,
    User,
    State,
    Threads,
    Nice,
}

pub const ALL_SORTS: [SortBy; 11] = [
    SortBy::Pid,
    SortBy::Name,
    SortBy::Cpu,
    SortBy::Mem,
    SortBy::Virt,
    SortBy::Disk,
    SortBy::Time,
    SortBy::User,
    SortBy::State,
    SortBy::Threads,
    SortBy::Nice,
];

impl SortBy {
    pub fn label(self) -> &'static str {
        match self {
            SortBy::Pid => "PID",
            SortBy::Name => "NAME",
            SortBy::Cpu => "CPU%",
            SortBy::Mem => "MEM",
            SortBy::Virt => "VIRT",
            SortBy::Disk => "DISK",
            SortBy::Time => "TIME",
            SortBy::User => "USER",
            SortBy::State => "S",
            SortBy::Threads => "THR",
            SortBy::Nice => "NI",
        }
    }

    pub fn parse(s: &str) -> Option<SortBy> {
        let s = s.trim().to_ascii_lowercase();
        ALL_SORTS.iter().copied().find(|k| {
            k.label().to_ascii_lowercase().trim_end_matches('%') == s || k.key_name() == s
        })
    }

    /// Stable machine-readable name, also what lands in the config file.
    pub fn key_name(self) -> &'static str {
        match self {
            SortBy::Pid => "pid",
            SortBy::Name => "name",
            SortBy::Cpu => "cpu",
            SortBy::Mem => "mem",
            SortBy::Virt => "virt",
            SortBy::Disk => "disk",
            SortBy::Time => "time",
            SortBy::User => "user",
            SortBy::State => "state",
            SortBy::Threads => "threads",
            SortBy::Nice => "nice",
        }
    }

    pub fn next(self) -> SortBy {
        let i = ALL_SORTS.iter().position(|k| *k == self).unwrap_or(0);
        ALL_SORTS[(i + 1) % ALL_SORTS.len()]
    }

    pub fn prev(self) -> SortBy {
        let i = ALL_SORTS.iter().position(|k| *k == self).unwrap_or(0);
        ALL_SORTS[(i + ALL_SORTS.len() - 1) % ALL_SORTS.len()]
    }
}

/// Compare two rows on `key`, descending when `desc`. PID is the tie-break in
/// both directions, which keeps rows from swapping places on every refresh.
pub fn compare(a: &ProcRow, b: &ProcRow, key: SortBy, desc: bool) -> Ordering {
    let primary = match key {
        SortBy::Pid => a.pid.cmp(&b.pid),
        SortBy::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        SortBy::Cpu => a.cpu.partial_cmp(&b.cpu).unwrap_or(Ordering::Equal),
        SortBy::Mem => a.mem.cmp(&b.mem),
        SortBy::Virt => a.virt.cmp(&b.virt),
        SortBy::Disk => (a.read_bps + a.write_bps)
            .partial_cmp(&(b.read_bps + b.write_bps))
            .unwrap_or(Ordering::Equal),
        SortBy::Time => a.run_time.cmp(&b.run_time),
        SortBy::User => a.user.cmp(&b.user),
        SortBy::State => a.state.cmp(&b.state),
        SortBy::Threads => a.threads.cmp(&b.threads),
        SortBy::Nice => a.nice.cmp(&b.nice),
    };
    let primary = if desc { primary.reverse() } else { primary };
    primary.then_with(|| a.pid.cmp(&b.pid))
}

pub fn sort_rows(rows: &mut [ProcRow], key: SortBy, desc: bool) {
    rows.sort_by(|a, b| compare(a, b, key, desc));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ProcRow;

    fn row(pid: u32, name: &str, cpu: f32, mem: u64) -> ProcRow {
        ProcRow { pid, name: name.into(), cpu, mem, ..ProcRow::default() }
    }

    #[test]
    fn every_sort_key_round_trips_through_its_name() {
        for k in ALL_SORTS {
            assert_eq!(SortBy::parse(k.key_name()), Some(k), "{}", k.key_name());
        }
        assert_eq!(SortBy::parse("CPU"), Some(SortBy::Cpu));
        assert_eq!(SortBy::parse("nonsense"), None);
    }

    #[test]
    fn cycling_forward_then_back_returns_to_the_start() {
        for k in ALL_SORTS {
            assert_eq!(k.next().prev(), k);
        }
    }

    #[test]
    fn ties_break_on_pid_in_both_directions() {
        let a = row(10, "same", 5.0, 100);
        let b = row(2, "same", 5.0, 100);
        assert_eq!(compare(&a, &b, SortBy::Cpu, true), Ordering::Greater);
        assert_eq!(compare(&a, &b, SortBy::Cpu, false), Ordering::Greater);
    }

    #[test]
    fn descending_cpu_puts_the_busiest_first() {
        let mut rows =
            vec![row(1, "idle", 0.0, 1), row(2, "busy", 90.0, 1), row(3, "mid", 20.0, 1)];
        sort_rows(&mut rows, SortBy::Cpu, true);
        assert_eq!(rows.iter().map(|r| r.pid).collect::<Vec<_>>(), vec![2, 3, 1]);
    }

    #[test]
    fn name_sorting_is_case_insensitive() {
        let mut rows = vec![row(1, "zsh", 0.0, 0), row(2, "Apache", 0.0, 0)];
        sort_rows(&mut rows, SortBy::Name, false);
        assert_eq!(rows[0].name, "Apache");
    }

    #[test]
    fn nan_cpu_does_not_panic_the_comparator() {
        let a = row(1, "a", f32::NAN, 0);
        let b = row(2, "b", 1.0, 0);
        let _ = compare(&a, &b, SortBy::Cpu, true);
    }
}
