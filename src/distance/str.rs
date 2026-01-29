//! Euler string representation and string edit distance for tree distance lower bounds.
//!
//! The Euler string of a tree is obtained by a depth-first traversal, recording each
//! node label when entering and leaving. For a tree with n nodes, the Euler string
//! has length 2n.
//!
//! The key property is: EDS(s(T1), s(T2)) ≤ 2 · EDT(T1, T2)
//! which gives us: EDT(T1, T2) ≥ EDS(s(T1), s(T2)) / 2
//!
//! This provides a lower bound on tree edit distance that can be computed in O(n·m) time
//! using standard string edit distance, useful for pruning in tree distance computations.

use super::nodes::Label;
use super::tree::TreeNode;
use super::zs::EditCosts;

/// Compute the Euler string of a tree.
///
/// The Euler string records each node label twice: once when entering the subtree
/// and once when leaving. This gives a string of length 2n for a tree with n nodes.
pub fn euler_string<L: Label>(tree: &TreeNode<L>) -> Vec<L> {
    fn euler_string_rec<LL: Label>(node: &TreeNode<LL>, out: &mut Vec<LL>) {
        out.push(node.label().clone());
        for child in node.children() {
            euler_string_rec(child, out);
        }
        out.push(node.label().clone());
    }
    let mut result = Vec::with_capacity(tree.size() * 2);
    euler_string_rec(tree, &mut result);
    result
}

/// Compute string edit distance between two sequences using the given cost function.
pub fn string_edit_distance<L, C: EditCosts<L>>(s1: &[L], s2: &[L], costs: &C) -> usize {
    let n = s1.len();
    let m = s2.len();

    if n == 0 {
        return s2.iter().map(|l| costs.insert(l)).sum();
    }
    if m == 0 {
        return s1.iter().map(|l| costs.delete(l)).sum();
    }

    // Use two-row optimization for O(min(n,m)) space
    let mut prev = Vec::with_capacity(m + 1);
    prev.push(0);
    for l in s2 {
        prev.push(prev.last().unwrap() + costs.insert(l));
    }
    let mut curr = vec![0; m + 1];

    for c1 in s1 {
        curr[0] = prev[0] + costs.delete(c1);
        for (j, c2) in s2.iter().enumerate() {
            let relabel = prev[j] + costs.relabel(c1, c2);
            let delete = prev[j + 1] + costs.delete(c1);
            let insert = curr[j] + costs.insert(c2);
            curr[j + 1] = relabel.min(delete).min(insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Compute a lower bound on tree edit distance using Euler string edit distance.
///
/// Returns `ceil(EDS(euler(t1), euler(t2)) / 2)` which is a valid lower bound
/// on the tree edit distance between t1 and t2.
pub fn tree_distance_euler_bound<L: Label, C: EditCosts<L>>(
    t1: &TreeNode<L>,
    t2: &TreeNode<L>,
    costs: &C,
) -> usize {
    let s1 = euler_string(t1);
    let s2 = euler_string(t2);
    let eds = string_edit_distance(&s1, &s2, costs);
    // EDT ≥ EDS / 2, so we use ceiling division
    eds.div_ceil(2)
}

/// Precomputed Euler string for a reference tree.
///
/// Reuse this when comparing against multiple candidate trees.
pub struct EulerString<L> {
    string: Vec<L>,
}

impl<L: Label> EulerString<L> {
    /// Create an Euler string from a tree.
    pub fn new(tree: &TreeNode<L>) -> Self {
        Self {
            string: euler_string(tree),
        }
    }

    /// Compute a lower bound on tree edit distance to the given tree.
    pub fn lower_bound<C: EditCosts<L>>(&self, tree: &TreeNode<L>, costs: &C) -> usize {
        let other = euler_string(tree);
        let eds = string_edit_distance(&self.string, &other, costs);
        eds.div_ceil(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::zs::UnitCost;

    fn leaf(label: &str) -> TreeNode<String> {
        TreeNode::leaf(label.to_owned())
    }

    fn node(label: &str, children: Vec<TreeNode<String>>) -> TreeNode<String> {
        TreeNode::new(label.to_owned(), children)
    }

    #[test]
    fn euler_string_leaf() {
        let tree = leaf("a");
        let euler = euler_string(&tree);
        assert_eq!(euler, vec!["a", "a"]);
    }

    #[test]
    fn euler_string_simple() {
        //     a
        //    / \
        //   b   c
        let tree = node("a", vec![leaf("b"), leaf("c")]);
        let euler = euler_string(&tree);
        // Enter a, enter b, leave b, enter c, leave c, leave a
        assert_eq!(euler, vec!["a", "b", "b", "c", "c", "a"]);
    }

    #[test]
    fn euler_string_nested() {
        //     a
        //     |
        //     b
        //     |
        //     c
        let tree = node("a", vec![node("b", vec![leaf("c")])]);
        let euler = euler_string(&tree);
        assert_eq!(euler, vec!["a", "b", "c", "c", "b", "a"]);
    }

    #[test]
    fn string_edit_distance_identical() {
        let s = vec!["a", "b", "c"];
        assert_eq!(string_edit_distance(&s, &s, &UnitCost), 0);
    }

    #[test]
    fn string_edit_distance_empty() {
        let empty: Vec<&str> = vec![];
        let s = vec!["a", "b", "c"];
        assert_eq!(string_edit_distance(&empty, &s, &UnitCost), 3);
        assert_eq!(string_edit_distance(&s, &empty, &UnitCost), 3);
    }

    #[test]
    fn string_edit_distance_one_diff() {
        let s1 = vec!["a", "b", "c"];
        let s2 = vec!["a", "x", "c"];
        assert_eq!(string_edit_distance(&s1, &s2, &UnitCost), 1);
    }

    #[test]
    fn lower_bound_identical() {
        let tree = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance_euler_bound(&tree, &tree, &UnitCost), 0);
    }

    #[test]
    fn lower_bound_valid() {
        // The lower bound should always be <= actual tree edit distance
        let t1 = node("a", vec![leaf("b"), leaf("c")]);
        let t2 = node("a", vec![leaf("b")]);

        let lb = tree_distance_euler_bound(&t1, &t2, &UnitCost);
        // Actual distance is 1 (delete c), lower bound should be <= 1
        assert!(lb <= 1);
    }

    #[test]
    fn euler_string_precomputed() {
        let t1 = node("a", vec![leaf("b"), leaf("c")]);
        let t2 = node("a", vec![leaf("b")]);

        let euler_ref = EulerString::new(&t1);
        let lb = euler_ref.lower_bound(&t2, &UnitCost);
        assert_eq!(lb, tree_distance_euler_bound(&t1, &t2, &UnitCost));
    }
}
