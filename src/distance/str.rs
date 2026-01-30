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

/// Euler string element - distinguishes entering vs leaving a node.
///
/// The paper defines s(T)[i₁(e)] = L(e) and s(T)[i₂(e)] = L̄(e) where L̄(e) is a
/// distinct "barred" version of the label. This distinction encodes directionality
/// (going down vs up in the tree) and ensures the distance bound holds properly.
#[derive(PartialEq, Eq, Hash, Debug)]
pub enum EulerSymbol<'a, L> {
    Enter(&'a L),
    Leave(&'a L),
}

impl<L> EulerSymbol<'_, L> {
    fn label(&self) -> &L {
        match self {
            EulerSymbol::Enter(l) | EulerSymbol::Leave(l) => l,
        }
    }
}

/// Compute string edit distance between two Euler strings using the given cost function.
///
/// The cost function operates on the underlying labels. Enter and Leave variants
/// with the same label have the same insert/delete cost. Relabeling between
/// Enter(a) and Leave(a) (same label, different variant) costs 1.
fn euler_string_edit_distance<L: Label, C: EditCosts<L>>(
    s1: &[EulerSymbol<L>],
    s2: &[EulerSymbol<L>],
    costs: &C,
) -> usize {
    let n = s1.len();
    let m = s2.len();

    if n == 0 {
        return s2.iter().map(|sym| costs.insert(sym.label())).sum();
    }
    if m == 0 {
        return s1.iter().map(|sym| costs.delete(sym.label())).sum();
    }

    // Use two-row optimization for O(min(n,m)) space
    let mut prev = Vec::with_capacity(m + 1);
    prev.push(0);
    for sym in s2 {
        prev.push(prev.last().unwrap() + costs.insert(sym.label()));
    }
    let mut curr = vec![0; m + 1];

    for c1 in s1 {
        curr[0] = prev[0] + costs.delete(c1.label());
        for (j, c2) in s2.iter().enumerate() {
            let relabel = prev[j] + relabel_cost(c1, c2, costs);
            let delete = prev[j + 1] + costs.delete(c1.label());
            let insert = curr[j] + costs.insert(c2.label());
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
    EulerString::new(t1).lower_bound(t2, costs)
}

// Cost for relabeling Euler symbols
fn relabel_cost<L: Label, C: EditCosts<L>>(
    a: &EulerSymbol<L>,
    b: &EulerSymbol<L>,
    costs: &C,
) -> usize {
    if a == b {
        0
    } else if a.label() == b.label() {
        // Same label but different variant (Enter vs Leave)
        costs.euler_in_out(a.label())
    } else {
        costs.relabel(a.label(), b.label())
    }
}

/// Precomputed Euler string for a reference tree.
///
/// Reuse this when comparing against multiple candidate trees.
pub struct EulerString<'a, L> {
    string: Vec<EulerSymbol<'a, L>>,
}

impl<'a, L: Label> EulerString<'a, L> {
    /// Create an Euler string from a tree.
    ///
    /// The Euler string records each node with Enter when entering the subtree
    /// and Leave when leaving. This gives a string of length 2n for a tree with n nodes.
    pub fn new(tree: &'a TreeNode<L>) -> Self {
        fn build<'b, LL: Label>(node: &'b TreeNode<LL>, out: &mut Vec<EulerSymbol<'b, LL>>) {
            out.push(EulerSymbol::Enter(node.label()));
            for child in node.children() {
                build(child, out);
            }
            out.push(EulerSymbol::Leave(node.label()));
        }
        let mut string = Vec::with_capacity(tree.size() * 2);
        build(tree, &mut string);
        Self { string }
    }

    /// Compute a lower bound on tree edit distance to the given tree.
    pub fn lower_bound<C: EditCosts<L>>(&self, tree: &'a TreeNode<L>, costs: &C) -> usize {
        let other = Self::new(tree);
        let eds = euler_string_edit_distance(&self.string, &other.string, costs);
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
        let euler = EulerString::new(&tree);
        assert_eq!(
            euler.string,
            vec![
                EulerSymbol::Enter(tree.label()),
                EulerSymbol::Leave(tree.label()),
            ]
        );
    }

    #[test]
    fn euler_string_simple() {
        //     a
        //    / \
        //   b   c
        let tree = node("a", vec![leaf("b"), leaf("c")]);
        let euler = EulerString::new(&tree);
        let [b, c] = tree.children() else { panic!() };
        // Enter a, enter b, leave b, enter c, leave c, leave a
        assert_eq!(
            euler.string,
            vec![
                EulerSymbol::Enter(tree.label()),
                EulerSymbol::Enter(b.label()),
                EulerSymbol::Leave(b.label()),
                EulerSymbol::Enter(c.label()),
                EulerSymbol::Leave(c.label()),
                EulerSymbol::Leave(tree.label()),
            ]
        );
    }

    #[test]
    fn euler_string_nested() {
        //     a
        //     |
        //     b
        //     |
        //     c
        let tree = node("a", vec![node("b", vec![leaf("c")])]);
        let euler = EulerString::new(&tree);
        let [b] = tree.children() else { panic!() };
        let [c] = b.children() else { panic!() };
        assert_eq!(
            euler.string,
            vec![
                EulerSymbol::Enter(tree.label()),
                EulerSymbol::Enter(b.label()),
                EulerSymbol::Enter(c.label()),
                EulerSymbol::Leave(c.label()),
                EulerSymbol::Leave(b.label()),
                EulerSymbol::Leave(tree.label()),
            ]
        );
    }

    #[test]
    fn euler_edit_distance_identical() {
        let (a, b) = ("a".to_owned(), "b".to_owned());
        let s: Vec<EulerSymbol<String>> = vec![
            EulerSymbol::Enter(&a),
            EulerSymbol::Enter(&b),
            EulerSymbol::Leave(&b),
            EulerSymbol::Leave(&a),
        ];
        assert_eq!(euler_string_edit_distance(&s, &s, &UnitCost), 0);
    }

    #[test]
    fn euler_edit_distance_empty() {
        let (a, b) = ("a".to_owned(), "b".to_owned());
        let empty: Vec<EulerSymbol<String>> = vec![];
        let s: Vec<EulerSymbol<String>> = vec![
            EulerSymbol::Enter(&a),
            EulerSymbol::Enter(&b),
            EulerSymbol::Leave(&b),
        ];
        assert_eq!(euler_string_edit_distance(&empty, &s, &UnitCost), 3);
        assert_eq!(euler_string_edit_distance(&s, &empty, &UnitCost), 3);
    }

    #[test]
    fn euler_edit_distance_one_diff() {
        let (a, b, x) = ("a".to_owned(), "b".to_owned(), "x".to_owned());
        let s1: Vec<EulerSymbol<String>> = vec![
            EulerSymbol::Enter(&a),
            EulerSymbol::Enter(&b),
            EulerSymbol::Leave(&b),
        ];
        let s2: Vec<EulerSymbol<String>> = vec![
            EulerSymbol::Enter(&a),
            EulerSymbol::Enter(&x),
            EulerSymbol::Leave(&x),
        ];
        // Changing "b" to "x" requires 2 edits (Enter(b)->Enter(x) and Leave(b)->Leave(x))
        assert_eq!(euler_string_edit_distance(&s1, &s2, &UnitCost), 2);
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
