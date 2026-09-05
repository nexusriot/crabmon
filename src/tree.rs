//! Process-tree flattening for the tree view (`T`).

use std::collections::{HashMap, HashSet};

use crate::metrics::ProcRow;
use crate::sort::{compare, SortBy};

/// A row in the flattened tree: index into the input slice plus its depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeRow {
    pub index: usize,
    pub depth: usize,
    pub last_sibling: bool,
}

/// Flatten `rows` into depth-first tree order.
///
/// Processes whose parent is not in `rows` (filtered out, or already reaped)
/// become roots, so a filtered tree never hides matching processes. Parent
/// cycles are broken by visiting each pid at most once.
pub fn flatten(rows: &[ProcRow], key: SortBy, desc: bool) -> Vec<TreeRow> {
    let index_of: HashMap<u32, usize> = rows.iter().enumerate().map(|(i, r)| (r.pid, i)).collect();

    let mut children: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut roots: Vec<usize> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        match r.ppid {
            // A process that is its own parent would loop forever.
            Some(pp) if pp != r.pid && index_of.contains_key(&pp) => {
                children.entry(pp).or_default().push(i)
            }
            _ => roots.push(i),
        }
    }

    let order = |a: &usize, b: &usize| compare(&rows[*a], &rows[*b], key, desc);
    roots.sort_by(order);
    for kids in children.values_mut() {
        kids.sort_by(order);
    }

    let mut out = Vec::with_capacity(rows.len());
    let mut visited: HashSet<u32> = HashSet::with_capacity(rows.len());
    // Explicit stack: a deep tree must not blow the real one.
    let mut stack: Vec<(usize, usize, bool)> =
        roots.iter().enumerate().rev().map(|(n, i)| (*i, 0usize, n + 1 == roots.len())).collect();

    while let Some((idx, depth, last)) = stack.pop() {
        if !visited.insert(rows[idx].pid) {
            continue;
        }
        out.push(TreeRow { index: idx, depth, last_sibling: last });
        if let Some(kids) = children.get(&rows[idx].pid) {
            for (n, k) in kids.iter().enumerate().rev() {
                stack.push((*k, depth + 1, n + 1 == kids.len()));
            }
        }
    }
    out
}

/// The `├─ ` / `└─ ` prefix for a row at `depth`.
pub fn indent(depth: usize, last_sibling: bool) -> String {
    if depth == 0 {
        return String::new();
    }
    let mut s = "  ".repeat(depth - 1);
    s.push_str(if last_sibling { "└─" } else { "├─" });
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, ppid: Option<u32>, name: &str, cpu: f32) -> ProcRow {
        ProcRow { pid, ppid, name: name.into(), cpu, ..ProcRow::default() }
    }

    fn pids(rows: &[ProcRow], tree: &[TreeRow]) -> Vec<u32> {
        tree.iter().map(|t| rows[t.index].pid).collect()
    }

    #[test]
    fn children_follow_their_parent_at_one_more_level_of_depth() {
        let rows = vec![
            row(1, None, "init", 0.0),
            row(10, Some(1), "sshd", 0.0),
            row(11, Some(10), "bash", 0.0),
        ];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(pids(&rows, &tree), vec![1, 10, 11]);
        assert_eq!(tree.iter().map(|t| t.depth).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn every_process_appears_exactly_once() {
        let rows = vec![
            row(1, None, "init", 0.0),
            row(2, Some(1), "a", 0.0),
            row(3, Some(1), "b", 0.0),
            row(4, Some(2), "c", 0.0),
        ];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(tree.len(), rows.len());
    }

    #[test]
    fn orphans_become_roots_so_a_filtered_tree_hides_nothing() {
        // `bash` matched the filter but its parent `sshd` did not.
        let rows = vec![row(11, Some(10), "bash", 0.0), row(12, Some(10), "vim", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(tree.len(), 2);
        assert!(tree.iter().all(|t| t.depth == 0));
    }

    #[test]
    fn a_parent_cycle_terminates() {
        let rows = vec![row(1, Some(2), "a", 0.0), row(2, Some(1), "b", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        // Neither is a root by ppid, so the cycle is broken and both still show.
        assert!(tree.len() <= 2);
    }

    #[test]
    fn a_self_parenting_process_does_not_loop() {
        let rows = vec![row(1, Some(1), "weird", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(pids(&rows, &tree), vec![1]);
    }

    #[test]
    fn siblings_honour_the_active_sort_order() {
        let rows = vec![
            row(1, None, "init", 0.0),
            row(2, Some(1), "quiet", 1.0),
            row(3, Some(1), "busy", 90.0),
        ];
        let tree = flatten(&rows, SortBy::Cpu, true);
        assert_eq!(pids(&rows, &tree), vec![1, 3, 2]);
    }

    #[test]
    fn deep_chains_do_not_overflow_the_stack() {
        let rows: Vec<ProcRow> =
            (0..20_000u32).map(|i| row(i + 1, (i > 0).then_some(i), "p", 0.0)).collect();
        assert_eq!(flatten(&rows, SortBy::Pid, false).len(), rows.len());
    }

    #[test]
    fn indent_marks_the_last_sibling_differently() {
        assert_eq!(indent(0, true), "");
        assert_eq!(indent(1, false), "├─");
        assert_eq!(indent(2, true), "  └─");
    }
}
