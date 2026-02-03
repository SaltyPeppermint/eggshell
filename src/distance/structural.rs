use super::{EditCosts, Label, TreeNode};

/// Very simple structural diff.
/// The idea is to ignore the labels and just look at how much of the goal tree,
/// starting from the root, is already present.
pub fn structural_diff<L: Label, C: EditCosts<L>>(
    reference: &TreeNode<L>,
    candidate: &TreeNode<L>,
    cost: &C,
) -> usize {
    if reference.children().len() != candidate.children().len() {
        return cost_rec(reference, cost);
    }
    reference
        .children()
        .iter()
        .zip(candidate.children())
        .map(|(r, c)| structural_diff(r, c, cost))
        .sum()
}

fn cost_rec<L: Label, C: EditCosts<L>>(tree: &TreeNode<L>, cost: &C) -> usize {
    cost.insert(tree.label())
        + tree
            .children()
            .iter()
            .map(|c| cost_rec(c, cost))
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::UnitCost;

    fn leaf<L: Label>(label: L) -> TreeNode<L> {
        TreeNode::leaf(label)
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::new(label, children)
    }

    #[test]
    fn identical_trees() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 0);
    }

    #[test]
    fn identical_leaves() {
        let tree1 = leaf("a".to_owned());
        let tree2 = leaf("b".to_owned());
        // Labels are ignored, both are leaves with no children
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 0);
    }

    #[test]
    fn different_child_count_at_root() {
        let tree1 = node("a".to_owned(), vec![leaf("b".to_owned())]);
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        // Different number of children -> cost of entire reference tree
        // tree1 has 2 nodes: "a" and "b"
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 2);
    }

    #[test]
    fn different_child_count_nested() {
        // Tree 1:    a
        //            |
        //            b
        //           / \
        //          c   d
        let tree1 = node(
            "a".to_owned(),
            vec![node(
                "b".to_owned(),
                vec![leaf("c".to_owned()), leaf("d".to_owned())],
            )],
        );
        // Tree 2:    a
        //            |
        //            b
        //            |
        //            c
        let tree2 = node(
            "a".to_owned(),
            vec![node("b".to_owned(), vec![leaf("c".to_owned())])],
        );
        // Root matches (1 child each), but b's children differ (2 vs 1)
        // Cost is the subtree rooted at b in reference: b, c, d = 3 nodes
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 3);
    }

    #[test]
    fn same_structure_different_labels() {
        // Labels are ignored, only structure matters
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = node(
            "x".to_owned(),
            vec![leaf("y".to_owned()), leaf("z".to_owned())],
        );
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 0);
    }

    #[test]
    fn deep_matching_structure() {
        // Both have the same deep structure: a - b - c - d
        let tree1 = node(
            "a".to_owned(),
            vec![node(
                "b".to_owned(),
                vec![node("c".to_owned(), vec![leaf("d".to_owned())])],
            )],
        );
        let tree2 = node(
            "w".to_owned(),
            vec![node(
                "x".to_owned(),
                vec![node("y".to_owned(), vec![leaf("z".to_owned())])],
            )],
        );
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 0);
    }

    #[test]
    fn mismatch_at_different_depths() {
        // Tree 1:       a
        //             / | \
        //            b  c  d
        //           /|
        //          e f
        let tree1 = node(
            "a".to_owned(),
            vec![
                node(
                    "b".to_owned(),
                    vec![leaf("e".to_owned()), leaf("f".to_owned())],
                ),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        );

        // Tree 2:       a
        //             / | \
        //            b  c  d
        //            |
        //            e
        let tree2 = node(
            "a".to_owned(),
            vec![
                node("b".to_owned(), vec![leaf("e".to_owned())]),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        );

        // Root has 3 children in both - match
        // b has 2 children in tree1 vs 1 in tree2 - mismatch at b
        // Cost is subtree at b: b, e, f = 3 nodes
        // c and d match (both leaves)
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 3);
    }

    #[test]
    fn leaf_vs_node_with_children() {
        let tree1 = leaf("a".to_owned());
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        // tree1 (reference) is a leaf (0 children), tree2 has 2 children
        // Mismatch at root -> cost of reference = 1 node
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 1);
    }

    #[test]
    fn node_vs_leaf() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = leaf("a".to_owned());
        // tree1 (reference) has 2 children, tree2 has 0
        // Mismatch at root -> cost of reference = 3 nodes
        assert_eq!(structural_diff(&tree1, &tree2, &UnitCost), 3);
    }
}
