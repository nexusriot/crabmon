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
    /// Children this row has, so a collapsed row can say how many it is
    /// standing in for. 0 for a leaf, which is what makes the marker
    /// meaningful rather than decoration.
    pub children: usize,
    /// Its subtree is hidden.
    pub collapsed: bool,
}

/// Flatten `rows` into depth-first tree order.
///
/// Processes whose parent is not in `rows` (filtered out, or already reaped)
/// become roots, so a filtered tree never hides matching processes. Parent
/// cycles are broken by visiting each pid at most once.
pub fn flatten(rows: &[ProcRow], key: SortBy, desc: bool) -> Vec<TreeRow> {
    flatten_collapsed(rows, key, desc, &HashSet::new())
}

/// As `flatten`, hiding the subtrees of any pid in `collapsed`.
///
/// A browser or a build with two hundred children buries everything else in
/// the table, and the tree view's whole claim is that structure makes a
/// process list readable. Folding one away is what makes that true on a real
/// machine.
pub fn flatten_collapsed(
    rows: &[ProcRow],
    key: SortBy,
    desc: bool,
    collapsed: &HashSet<u32>,
) -> Vec<TreeRow> {
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
    walk(rows, &children, &roots, collapsed, &mut out, &mut visited);

    // Anything a cycle left unreachable. Every member of a parent cycle has
    // its parent inside `rows`, so none of them is a root and the walk above
    // never reaches any of them — which used to drop the whole component from
    // the table. A tree view that silently omits processes is worse than one
    // that draws a cycle as a chain, so the stranded rows are re-seeded as
    // roots in sort order.
    if out.len() < rows.len() {
        let mut stranded: Vec<usize> =
            (0..rows.len()).filter(|i| !visited.contains(&rows[*i].pid)).collect();
        stranded.sort_by(order);
        walk(rows, &children, &stranded, collapsed, &mut out, &mut visited);
    }
    out
}

/// Depth-first walk from `seeds`, skipping pids already in `visited`.
///
/// An explicit stack: a deep tree must not blow the real one.
fn walk(
    rows: &[ProcRow],
    children: &HashMap<u32, Vec<usize>>,
    seeds: &[usize],
    collapsed: &HashSet<u32>,
    out: &mut Vec<TreeRow>,
    visited: &mut HashSet<u32>,
) {
    let mut stack: Vec<(usize, usize, bool)> =
        seeds.iter().enumerate().rev().map(|(n, i)| (*i, 0usize, n + 1 == seeds.len())).collect();

    while let Some((idx, depth, last)) = stack.pop() {
        let pid = rows[idx].pid;
        if !visited.insert(pid) {
            continue;
        }
        let kids = children.get(&pid);
        // The *whole* subtree, not just the direct children: "collapsed, 40
        // hidden" has to mean forty rows that are not on screen.
        let hidden = collapsed.contains(&pid);
        out.push(TreeRow {
            index: idx,
            depth,
            last_sibling: last,
            children: match hidden {
                true => subtree_size(rows, children, pid),
                false => kids.map_or(0, Vec::len),
            },
            collapsed: hidden,
        });
        if hidden {
            // Its descendants are marked visited so a cycle cannot smuggle
            // them back in through another parent.
            mark_subtree(rows, children, pid, visited);
            continue;
        }
        if let Some(kids) = kids {
            for (n, k) in kids.iter().enumerate().rev() {
                stack.push((*k, depth + 1, n + 1 == kids.len()));
            }
        }
    }
}

/// Every descendant of `pid`, counted once.
fn subtree_size(rows: &[ProcRow], children: &HashMap<u32, Vec<usize>>, pid: u32) -> usize {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = vec![pid];
    let mut count = 0;
    while let Some(next) = stack.pop() {
        for idx in children.get(&next).into_iter().flatten() {
            let child = rows[*idx].pid;
            if seen.insert(child) {
                count += 1;
                stack.push(child);
            }
        }
    }
    count
}

/// Mark a hidden subtree as visited so nothing else draws it.
fn mark_subtree(
    rows: &[ProcRow],
    children: &HashMap<u32, Vec<usize>>,
    pid: u32,
    visited: &mut HashSet<u32>,
) {
    let mut stack = vec![pid];
    while let Some(next) = stack.pop() {
        for idx in children.get(&next).into_iter().flatten() {
            let child = rows[*idx].pid;
            if visited.insert(child) {
                stack.push(child);
            }
        }
    }
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

/// What a folded row says it is standing in for.
///
/// `+40` rather than a bare marker, because the useful question about a
/// folded subtree is how much of the table it is hiding.
pub fn fold_marker(row: &TreeRow) -> String {
    match (row.collapsed, row.children) {
        (true, n) => format!("+{n} "),
        (false, _) => String::new(),
    }
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
    fn a_parent_cycle_terminates_without_losing_the_processes_in_it() {
        let rows = vec![row(1, Some(2), "a", 0.0), row(2, Some(1), "b", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        // Neither is a root by ppid. The cycle has to be broken *and* both
        // rows still drawn; `<= 2` used to pass with an empty table.
        assert_eq!(pids(&rows, &tree), vec![1, 2]);
    }

    /// A cycle must not take unrelated processes with it, and must not hide
    /// the ones inside it: the table is the whole point.
    #[test]
    fn a_cycle_beside_a_normal_tree_hides_neither() {
        let rows = vec![
            row(1, Some(2), "a", 0.0),
            row(2, Some(1), "b", 0.0),
            row(3, None, "init", 0.0),
            row(4, Some(3), "child", 0.0),
        ];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(tree.len(), rows.len(), "{tree:?}");
        let mut seen = pids(&rows, &tree);
        seen.sort();
        assert_eq!(seen, vec![1, 2, 3, 4]);
    }

    /// Every row reachable or not, the indices `flatten` hands back are still
    /// unique — `rebuild_view` moves rows out by index and would blank one.
    #[test]
    fn indices_are_unique_even_with_a_cycle() {
        let rows =
            vec![row(1, Some(2), "a", 0.0), row(2, Some(3), "b", 0.0), row(3, Some(1), "c", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        let mut idx: Vec<usize> = tree.iter().map(|t| t.index).collect();
        idx.sort();
        idx.dedup();
        assert_eq!(idx.len(), tree.len());
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn a_self_parenting_process_does_not_loop() {
        let rows = vec![row(1, Some(1), "weird", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert_eq!(pids(&rows, &tree), vec![1]);
    }

    /// A browser with two hundred children buries everything else, and the
    /// tree view's whole claim is that structure makes a list readable.
    #[test]
    fn a_folded_subtree_is_hidden_and_says_how_much_it_is_hiding() {
        let rows = vec![
            row(1, None, "init", 0.0),
            row(10, Some(1), "browser", 0.0),
            row(11, Some(10), "tab-a", 0.0),
            row(12, Some(10), "tab-b", 0.0),
            row(13, Some(11), "renderer", 0.0),
            row(20, Some(1), "sshd", 0.0),
        ];
        let folded = HashSet::from([10]);
        let tree = flatten_collapsed(&rows, SortBy::Pid, false, &folded);

        assert_eq!(pids(&rows, &tree), vec![1, 10, 20], "the subtree is still drawn");
        let browser = tree.iter().find(|t| rows[t.index].pid == 10).unwrap();
        assert!(browser.collapsed);
        // Three descendants, not two children: the useful number is how many
        // rows are not on screen.
        assert_eq!(browser.children, 3);
        assert_eq!(fold_marker(browser), "+3 ");
    }

    #[test]
    fn an_unfolded_row_carries_no_marker() {
        let rows = vec![row(1, None, "init", 0.0), row(2, Some(1), "child", 0.0)];
        let tree = flatten(&rows, SortBy::Pid, false);
        assert!(tree.iter().all(|t| !t.collapsed));
        assert_eq!(fold_marker(&tree[0]), "");
        assert_eq!(tree[0].children, 1, "a parent knows its children either way");
        assert_eq!(tree[1].children, 0);
    }

    /// Folding must not lose processes that are not under the folded one.
    #[test]
    fn folding_one_branch_leaves_the_others_alone() {
        let rows = vec![
            row(1, None, "init", 0.0),
            row(10, Some(1), "a", 0.0),
            row(11, Some(10), "a-child", 0.0),
            row(20, Some(1), "b", 0.0),
            row(21, Some(20), "b-child", 0.0),
        ];
        let tree = flatten_collapsed(&rows, SortBy::Pid, false, &HashSet::from([10]));
        assert_eq!(pids(&rows, &tree), vec![1, 10, 20, 21]);
    }

    /// A fold on a pid that is not in the table changes nothing.
    #[test]
    fn folding_a_process_that_is_gone_is_harmless() {
        let rows = vec![row(1, None, "init", 0.0), row(2, Some(1), "child", 0.0)];
        let tree = flatten_collapsed(&rows, SortBy::Pid, false, &HashSet::from([999]));
        assert_eq!(tree.len(), 2);
    }

    /// Folding inside a cycle must still terminate.
    #[test]
    fn folding_a_cycle_terminates() {
        let rows = vec![row(1, Some(2), "a", 0.0), row(2, Some(1), "b", 0.0)];
        let tree = flatten_collapsed(&rows, SortBy::Pid, false, &HashSet::from([1, 2]));
        assert!(tree.len() <= 2);
        assert!(!tree.is_empty(), "a folded cycle still has to draw its roots");
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
