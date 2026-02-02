use super::nodes::Label;
use super::tree::TreeNode;

/// Postorder traversal information for a tree node.
#[derive(Debug, Clone)]
struct PostorderNode<'a, L: Label> {
    label: &'a L,
    leftmost_leaf: usize,
}

/// Preprocessed tree for Zhang-Shasha algorithm.
///
/// Reuse this when computing distances against multiple candidate trees.
pub struct PreprocessedTree<'a, L: Label> {
    nodes: Vec<PostorderNode<'a, L>>,
    keyroots: Vec<usize>,
}

impl<'a, L: Label> PreprocessedTree<'a, L> {
    /// Create a preprocessed tree from a tree node.
    /// This performs a single postorder traversal to compute leftmost leaf descendants
    /// and keyroots.
    pub fn new(root: &'a TreeNode<L>) -> Self {
        let mut nodes = Vec::new();

        // Perform postorder traversal and compute leftmost leaf descendants
        Self::postorder_traverse(root, &mut nodes);

        // Compute keyroots: a node is a keyroot if it's the last node (in postorder)
        // with its particular leftmost leaf value. This is equivalent to: a node is
        // a keyroot if it has no parent, or it is not the leftmost child of its parent.
        let mut keyroots = Vec::new();
        let mut leftmost_to_keyroot = vec![0; nodes.len()];

        for (i, n) in nodes.iter().enumerate() {
            // Each time we see a leftmost leaf value, update to the latest node
            leftmost_to_keyroot[n.leftmost_leaf] = i;
        }

        // Collect unique keyroots
        let mut seen = vec![false; nodes.len()];
        for &kr in &leftmost_to_keyroot {
            if !seen[kr] {
                seen[kr] = true;
                keyroots.push(kr);
            }
        }

        keyroots.sort_unstable();

        PreprocessedTree { nodes, keyroots }
    }

    fn postorder_traverse(node: &'a TreeNode<L>, nodes: &mut Vec<PostorderNode<'a, L>>) -> usize {
        // First, traverse all children
        let child_indices = node
            .children()
            .iter()
            .map(|child| Self::postorder_traverse(child, nodes))
            .collect::<Vec<_>>();

        // Current node's postorder index
        let current_idx = nodes.len();

        // Compute leftmost leaf
        let leftmost_leaf = if node.is_leaf() {
            current_idx
        } else {
            // Leftmost leaf is the leftmost leaf of the leftmost child
            nodes[child_indices[0]].leftmost_leaf
        };

        nodes.push(PostorderNode {
            label: node.label(),
            leftmost_leaf,
        });

        current_idx
    }

    /// Returns the number of nodes in the tree
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the postorder index of the leftmost leaf descendant of node i
    pub fn leftmost_leaf(&self, i: usize) -> usize {
        self.nodes[i].leftmost_leaf
    }

    /// Returns the label of node i
    pub fn label(&self, i: usize) -> &L {
        self.nodes[i].label
    }

    /// Returns the keyroots of the tree (nodes that start new subproblems)
    pub fn keyroots(&self) -> &[usize] {
        &self.keyroots
    }
}

/// Cost functions for tree edit operations.
pub trait EditCosts<L>: Send + Sync {
    /// Cost of deleting a node.
    fn delete(&self, label: &L) -> usize;

    /// Cost of inserting a node.
    fn insert(&self, label: &L) -> usize;

    /// Cost of relabeling a node.
    fn relabel(&self, from: &L, to: &L) -> usize;

    /// Euler in-out-same-node
    fn euler_in_out(&self, label: &L) -> usize;
}

/// Unit cost model: all operations cost 1, relabeling identical labels costs 0.
pub struct UnitCost;

impl<L: Eq> EditCosts<L> for UnitCost {
    fn delete(&self, _label: &L) -> usize {
        1
    }

    fn insert(&self, _label: &L) -> usize {
        1
    }

    fn relabel(&self, from: &L, to: &L) -> usize {
        usize::from(from != to)
    }

    fn euler_in_out(&self, _label: &L) -> usize {
        1
    }
}

/// Compute the Zhang-Shasha tree edit distance between two trees.
pub fn tree_distance<L: Label, C: EditCosts<L>>(
    tree1: &TreeNode<L>,
    tree2: &TreeNode<L>,
    costs: &C,
) -> usize {
    let t1 = PreprocessedTree::new(tree1);
    let t2 = PreprocessedTree::new(tree2);
    tree_distance_preprocessed(&t1, &t2, costs)
}

/// Compute distance with a pre-preprocessed reference tree.
pub fn tree_distance_with_ref<L: Label, C: EditCosts<L>>(
    candidate: &TreeNode<L>,
    reference: &PreprocessedTree<L>,
    costs: &C,
) -> usize {
    let t1 = PreprocessedTree::new(candidate);
    tree_distance_preprocessed(&t1, reference, costs)
}

/// Compute distance between two preprocessed trees.
pub fn tree_distance_preprocessed<L: Label, C: EditCosts<L>>(
    t1: &PreprocessedTree<L>,
    t2: &PreprocessedTree<L>,
    costs: &C,
) -> usize {
    let n1 = t1.size();
    let n2 = t2.size();

    if n1 == 0 && n2 == 0 {
        return 0;
    }
    if n1 == 0 {
        return (0..n2).map(|j| costs.insert(t2.label(j))).sum();
    }
    if n2 == 0 {
        return (0..n1).map(|i| costs.delete(t1.label(i))).sum();
    }

    // Tree distance matrix (permanent)
    let mut td = vec![vec![0; n2]; n1];

    // Forest distance matrix (temporary, reused for each keyroot pair)
    // We need indices from -1, so we use size+1 and offset by 1
    let mut fd = vec![vec![0; n2 + 1]; n1 + 1];

    // Compute tree distance for each pair of keyroots
    for &i in t1.keyroots() {
        for &j in t2.keyroots() {
            compute_forest_distance(t1, t2, i, j, &mut td, &mut fd, costs);
        }
    }

    // The final answer is the distance between the full trees
    td[n1 - 1][n2 - 1]
}

fn compute_forest_distance<L: Label, C: EditCosts<L>>(
    t1: &PreprocessedTree<L>,
    t2: &PreprocessedTree<L>,
    i: usize,
    j: usize,
    td: &mut [Vec<usize>],
    fd: &mut [Vec<usize>],
    costs: &C,
) {
    let l1 = t1.leftmost_leaf(i);
    let l2 = t2.leftmost_leaf(j);

    // fd[x][y] represents the forest distance between:
    // - forest of t1 from l1 to x-1 (using 1-based indexing offset)
    // - forest of t2 from l2 to y-1
    // fd[0][0] = 0 (empty forests)

    // Initialize: deleting all nodes from t1's forest
    fd[0][0] = 0;
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        fd[x_idx][0] = fd[x_idx - 1][0] + costs.delete(t1.label(x));
    }

    // Initialize: inserting all nodes into empty forest from t2
    for y in l2..=j {
        let y_idx = y - l2 + 1;
        fd[0][y_idx] = fd[0][y_idx - 1] + costs.insert(t2.label(y));
    }

    // Fill in the forest distance matrix
    // Note: we intentionally use x and y as indices into td, as td stores
    // tree distances for all node pairs using their postorder indices
    #[expect(clippy::needless_range_loop)]
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        let lx = t1.leftmost_leaf(x);

        for y in l2..=j {
            let y_idx = y - l2 + 1;
            let ly = t2.leftmost_leaf(y);

            let delete_cost = fd[x_idx - 1][y_idx] + costs.delete(t1.label(x));
            let insert_cost = fd[x_idx][y_idx - 1] + costs.insert(t2.label(y));

            if lx == l1 && ly == l2 {
                // Both x and y have their leftmost leaves at the start of this subproblem,
                // meaning we're computing the full subtree distance for (x, y)
                let relabel_cost =
                    fd[x_idx - 1][y_idx - 1] + costs.relabel(t1.label(x), t2.label(y));
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(relabel_cost);
                // Store in permanent tree distance matrix
                td[x][y] = fd[x_idx][y_idx];
            } else {
                // At least one of x or y has its leftmost leaf before the start of this
                // subproblem, so we need to use the previously computed tree distance
                let match_cost = fd[lx - l1][ly - l2] + td[x][y];
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(match_cost);
            }
        }
    }
}

/// Compute tree edit distance with unit costs.
pub fn tree_distance_unit<L: Label>(tree1: &TreeNode<L>, tree2: &TreeNode<L>) -> usize {
    tree_distance(tree1, tree2, &UnitCost)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf<L: Label>(label: L) -> TreeNode<L> {
        TreeNode::leaf(label)
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::new(label, children)
    }

    #[test]
    fn basic_zhang_shasha() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        assert_eq!(tree_distance(&tree1, &tree2, &UnitCost), 0);

        let tree3 = node("a".to_owned(), vec![leaf("b".to_owned())]);
        assert_eq!(tree_distance(&tree1, &tree3, &UnitCost), 1);
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
        assert_eq!(tree_distance_unit(&tree1, &tree2), 0);
    }

    #[test]
    fn single_node_difference() {
        let tree1 = leaf("a".to_owned());
        let tree2 = leaf("b".to_owned());
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // relabel a -> b
    }

    #[test]
    fn insert_child() {
        let tree1 = node("a".to_owned(), vec![leaf("b".to_owned())]);
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // insert c
    }

    #[test]
    fn delete_child() {
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = node("a".to_owned(), vec![leaf("b".to_owned())]);
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // delete c
    }

    #[test]
    fn empty_to_tree() {
        // Empty tree represented as single node to non-empty
        let tree1 = leaf("a".to_owned());
        let tree2 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2); // insert b, insert c
    }

    #[test]
    fn different_structure() {
        // Tree 1:    a          Tree 2:    a
        //           /|                     |
        //          b c                     b
        //                                  |
        //                                  c
        let tree1 = node(
            "a".to_owned(),
            vec![leaf("b".to_owned()), leaf("c".to_owned())],
        );
        let tree2 = node(
            "a".to_owned(),
            vec![node("b".to_owned(), vec![leaf("c".to_owned())])],
        );
        // One way: delete c from tree1, insert c under b = 2 operations
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2);
    }

    #[test]
    fn completely_different() {
        let tree1 = node("a".to_owned(), vec![leaf("b".to_owned())]);
        let tree2 = node("x".to_owned(), vec![leaf("y".to_owned())]);
        // relabel a->x, relabel b->y = 2 operations
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2);
    }

    #[test]
    fn larger_trees() {
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
        //           /
        //          e
        let tree2 = node(
            "a".to_owned(),
            vec![
                node("b".to_owned(), vec![leaf("e".to_owned())]),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        );

        // Delete f from tree1
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1);
    }

    #[test]
    fn deep_vs_shexpect() {
        // Tree 1: a - b - c - d (linear chain)
        let tree1 = node(
            "a".to_owned(),
            vec![node(
                "b".to_owned(),
                vec![node("c".to_owned(), vec![leaf("d".to_owned())])],
            )],
        );

        // Tree 2:    a
        //          / | \
        //         b  c  d
        let tree2 = node(
            "a".to_owned(),
            vec![
                leaf("b".to_owned()),
                leaf("c".to_owned()),
                leaf("d".to_owned()),
            ],
        );

        // Need to restructure: this requires delete and insert operations
        // The exact cost depends on the optimal alignment
        let dist = tree_distance_unit(&tree1, &tree2);
        assert!(dist > 0);
        assert!(dist <= 4); // Upper bound: delete all and insert all (minus common)
    }
}
