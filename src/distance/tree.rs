//! Zhang-Shasha Tree Edit Distance Algorithm
//!
//! Zhang-Shasha computes the edit distance between two ordered labeled trees.
//! The algorithm runs in O(n1 * n2 * min(depth1, leaves1) * min(depth2, leaves2))
//! time and O(n1 * n2) space.

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp, SexpError};

use super::graph::EGraph;
use super::ids::{DataChildId, DataId, EClassId, FunId, NatId, TypeChildId};
use super::nodes::Label;

/// A node in a labeled, ordered tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct TreeNode<L: Label> {
    label: L,
    children: Vec<TreeNode<L>>,
}

impl<L: Label> TreeNode<L> {
    /// Create a leaf node with no children.
    pub fn leaf(label: L) -> Self {
        TreeNode {
            label,
            children: Vec::new(),
        }
    }

    /// Create a node with the given children.
    pub fn new(label: L, children: Vec<TreeNode<L>>) -> Self {
        TreeNode { label, children }
    }

    /// Returns true if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn children(&self) -> &[Self] {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Build a type tree from an e-class's type annotation.
    #[must_use]
    pub fn from_eclass(egraph: &EGraph<L>, id: EClassId) -> Self {
        let ty_id = egraph.class(id).ty();
        Self::from_type(egraph, ty_id)
    }

    fn from_type(egraph: &EGraph<L>, id: TypeChildId) -> Self {
        match id {
            TypeChildId::Nat(nat_id) => Self::from_nat(egraph, nat_id),
            TypeChildId::Type(fun_ty_id) => Self::from_fun(egraph, fun_ty_id),
            TypeChildId::Data(data_ty_id) => Self::from_data(egraph, data_ty_id),
        }
    }

    fn from_fun(egraph: &EGraph<L>, id: FunId) -> Self {
        let node = egraph.fun_ty(id).label().to_owned();
        let children = egraph
            .fun_ty(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_type(egraph, c_id))
            .collect();
        TreeNode::new(node, children)
    }

    #[must_use]
    pub fn from_data(egraph: &EGraph<L>, id: DataId) -> Self {
        let node = egraph.data_ty(id).label().to_owned();
        let children = egraph
            .data_ty(id)
            .children()
            .iter()
            .map(|&c_id| match c_id {
                DataChildId::Nat(nat_id) => Self::from_nat(egraph, nat_id),
                DataChildId::DataType(data_ty_id) => Self::from_data(egraph, data_ty_id),
            })
            .collect();
        TreeNode::new(node, children)
    }

    #[must_use]
    pub fn from_nat(egraph: &EGraph<L>, id: NatId) -> Self {
        let node = egraph.nat(id).label().to_owned();
        let children = egraph
            .nat(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_nat(egraph, c_id))
            .collect();
        TreeNode::new(node, children)
    }

    // /// Remove type annotation wrappers from this tree.
    // #[must_use]
    // pub fn strip_types(&self) -> Self {
    //     if self.label.is_type_of() {
    //         self.children()[0].strip_types()
    //     } else {
    //         self.clone()
    //     }
    // }

    /// Count total number of nodes in this tree.
    pub fn size(&self) -> usize {
        1 + self.children.iter().map(Self::size).sum::<usize>()
    }

    // /// Remove the typeof node by pushing the typeof in the rightmost child
    // /// trades height and node number vs child number
    // #[allow(clippy::missing_panics_doc)]
    // #[must_use]
    // pub fn squash_types(mut self) -> Self {
    //     if self.label().is_type_of() {
    //         let mut child_iter = self.children.into_iter();
    //         let mut expr_tree = child_iter.next().unwrap().squash_types();
    //         let type_tree = child_iter.next().unwrap();
    //         expr_tree.children.push(type_tree);
    //         expr_tree
    //     } else {
    //         self.children = self
    //             .children
    //             .into_iter()
    //             .map(|c| c.squash_types())
    //             .collect();
    //         self
    //     }
    // }

    /// Strip the types for quicker comparison
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn strip_types(&self) -> Self {
        if self.label().is_type_of() {
            self.children[0].strip_types()
        } else {
            TreeNode {
                label: self.label().clone(),
                children: self.children.iter().map(|c| c.strip_types()).collect(),
            }
        }
    }
}

impl FromStr for TreeNode<String> {
    type Err = SexpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        /// Parse a type tree (no typeOf wrappers).
        fn parse_expr(sexp: Sexp) -> Result<TreeNode<String>, SexpError> {
            match sexp {
                Sexp::String(s) => Ok(TreeNode::leaf(s)),
                Sexp::List(sexps) => {
                    let mut iter = sexps.into_iter();
                    let Some(Sexp::String(label)) = iter.next() else {
                        return Err(SexpError::Other("expected (label ...)".to_owned()));
                    };
                    Ok(TreeNode::new(
                        label,
                        iter.map(parse_expr).collect::<Result<_, _>>()?,
                    ))
                }
                Sexp::Empty => Err(SexpError::Other("empty sexp".to_owned())),
            }
        }

        symbolic_expressions::parser::parse_str(s).and_then(parse_expr)
    }
}

impl IntoSexp for TreeNode<String> {
    fn into_sexp(&self) -> Sexp {
        if self.is_leaf() {
            // Leaf with no type - just the label
            Sexp::String(self.label.clone())
        } else {
            Sexp::List(
                vec![Sexp::String(self.label.clone())]
                    .into_iter()
                    .chain(self.children().iter().map(Sexp::from))
                    .collect::<Vec<_>>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sexp_roundtrip_simple() {
        use symbolic_expressions::IntoSexp;

        // Parse a simple s-expression and serialize it back
        let input = "(f a b)";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "f");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "a");
        assert_eq!(tree.children()[1].label(), "b");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_nested() {
        use symbolic_expressions::IntoSexp;

        // Nested s-expressions
        let input = "(a (b c) (d e))";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "a");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "b");
        assert_eq!(tree.children()[0].children()[0].label(), "c");
        assert_eq!(tree.children()[1].label(), "d");
        assert_eq!(tree.children()[1].children()[0].label(), "e");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_complex_type() {
        use symbolic_expressions::IntoSexp;

        // Expression with a type-like structure
        let input = "(-> int (-> int int))";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "->");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "int");
        assert_eq!(tree.children()[1].label(), "->");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_large() {
        use symbolic_expressions::IntoSexp;

        let input = "(natLam (natLam (natLam (lam (lam (app (app map (lam (app (app map (lam (app (app (app reduce add) 0.0) (app (app map (lam (app (app mul (app fst $e0)) (app snd $e0)))) (app (app zip $e1) $e0))))) (app transpose $e1)))) $e1))))))";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "natLam");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_leaf() {
        use symbolic_expressions::IntoSexp;

        let input = "x";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "x");
        assert!(tree.is_leaf());

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }
}
