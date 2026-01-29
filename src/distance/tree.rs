//! Zhang-Shasha Tree Edit Distance Algorithm
//!
//! Zhang-Shasha computes the edit distance between two ordered labeled trees.
//! The algorithm runs in O(n1 * n2 * min(depth1, leaves1) * min(depth2, leaves2))
//! time and O(n1 * n2) space.

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp, SexpError};

use super::graph::EGraph;
use super::ids::{DataTyId, EClassId, FunTyId, NatId, NatOrDTId, TypeId};
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

    fn from_type(egraph: &EGraph<L>, id: TypeId) -> Self {
        match id {
            TypeId::Nat(nat_id) => Self::from_nat(egraph, nat_id),
            TypeId::Type(fun_ty_id) => Self::from_fun(egraph, fun_ty_id),
            TypeId::DataType(data_ty_id) => Self::from_data_ty(egraph, data_ty_id),
        }
    }

    fn from_fun(egraph: &EGraph<L>, id: FunTyId) -> Self {
        let node = egraph.fun_ty(id).label().to_owned();
        let children = egraph
            .fun_ty(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_type(egraph, c_id))
            .collect();
        TreeNode::new(node, children)
    }

    fn from_data_ty(egraph: &EGraph<L>, id: DataTyId) -> Self {
        let node = egraph.data_ty(id).label().to_owned();
        let children = egraph
            .data_ty(id)
            .children()
            .iter()
            .map(|&c_id| match c_id {
                NatOrDTId::Nat(nat_id) => Self::from_nat(egraph, nat_id),
                NatOrDTId::DataType(data_ty_id) => Self::from_data_ty(egraph, data_ty_id),
            })
            .collect();
        TreeNode::new(node, children)
    }

    fn from_nat(egraph: &EGraph<L>, id: NatId) -> Self {
        let node = egraph.nat(id).label().to_owned();
        let children = egraph
            .nat(id)
            .children()
            .iter()
            .map(|&c_id| Self::from_nat(egraph, c_id))
            .collect();
        TreeNode::new(node, children)
    }

    /// Remove type annotation wrappers from this tree.
    #[must_use]
    pub fn strip_types(&self) -> Self {
        if self.label.is_type_of() {
            self.children()[0].strip_types()
        } else {
            self.clone()
        }
    }

    /// Count total number of nodes in this tree.
    pub fn size(&self) -> usize {
        1 + self.children.iter().map(Self::size).sum::<usize>()
    }
}

impl FromStr for TreeNode<String> {
    type Err = SexpError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn head(sexps: Vec<Sexp>) -> Result<(String, std::vec::IntoIter<Sexp>), SexpError> {
            let mut iter = sexps.into_iter();
            if let Some(Sexp::String(label)) = iter.next() {
                return Ok((label, iter));
            }
            Err(SexpError::Other("expected (label ...)".to_owned()))
        }

        /// Parse a typeOf-wrapped expression, returns node with type as last child.
        fn parse_expr(sexp: Sexp) -> Result<TreeNode<String>, SexpError> {
            use crate::distance::nodes::Label;
            let Sexp::List(sexps) = sexp else {
                return Err({
                    SexpError::Other(format!("expected typeOf wrapper, got: {sexp:?}"))
                });
            };
            let (label, mut rest) = head(sexps)?;
            if !label.is_type_of() {
                return Err(SexpError::Other(format!("expected typeOf, got: {label}")));
            }
            let Some(expr) = rest.next() else {
                return Err(SexpError::Other("typeOf missing expr".to_owned()));
            };
            let Some(ty) = rest.next() else {
                return Err(SexpError::Other("typeOf missing type".to_owned()));
            };

            parse_expr_body(expr, parse_type(ty)?)
        }

        /// Parse expression body, attaching type as last child.
        fn parse_expr_body(
            sexp: Sexp,
            ty: TreeNode<String>,
        ) -> Result<TreeNode<String>, SexpError> {
            let (label, children) = match sexp {
                Sexp::String(label) => (label, vec![]),
                Sexp::List(sexps) => {
                    let (label, rest) = head(sexps)?;
                    (label, rest.map(parse_child).collect::<Result<_, _>>()?)
                }
                Sexp::Empty => return Err(SexpError::Other("empty sexp".to_owned())),
            };
            Ok(TreeNode::new(
                label,
                children.into_iter().chain([ty]).collect(),
            ))
        }

        /// Parse a child of an expression - either typeOf-wrapped or a bare value (e.g., nat literal).
        /// Bare values (atoms or non-typeOf lists) are parsed as plain trees without type annotation.
        fn parse_child(sexp: Sexp) -> Result<TreeNode<String>, SexpError> {
            use crate::distance::nodes::Label;
            if let Sexp::List(sexps) = &sexp
                && let Some(Sexp::String(h)) = sexps.first()
                && h.is_type_of()
            {
                return parse_expr(sexp);
            }
            parse_type(sexp)
        }

        /// Parse a type tree (no typeOf wrappers).
        fn parse_type(sexp: Sexp) -> Result<TreeNode<String>, SexpError> {
            match sexp {
                Sexp::String(s) => Ok(TreeNode::leaf(s)),
                Sexp::List(sexps) => {
                    let (label, rest) = head(sexps)?;
                    Ok(TreeNode::new(
                        label,
                        rest.map(parse_type).collect::<Result<_, _>>()?,
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
        /// Convert to s-expression without typeOf wrapper (for type trees).
        fn to_sexp_raw(expr: &TreeNode<String>) -> Sexp {
            if expr.is_leaf() {
                Sexp::String(expr.label.clone())
            } else {
                let l = vec![Sexp::String(expr.label.clone())]
                    .into_iter()
                    .chain(expr.children.iter().map(to_sexp_raw))
                    .collect::<Vec<_>>();
                Sexp::List(l)
            }
        }
        if self.is_leaf() {
            // Leaf with no type - just the label
            Sexp::String(self.label.clone())
        } else {
            // Expression nodes: last child is the type, wrap with typeOf
            let expr_children = &self.children[..self.children.len() - 1];
            let type_child = &self.children[self.children.len() - 1];

            let expr = if expr_children.is_empty() {
                Sexp::String(self.label.clone())
            } else {
                let l = vec![Sexp::String(self.label.clone())]
                    .into_iter()
                    .chain(expr_children.iter().map(Sexp::from))
                    .collect::<Vec<_>>();
                Sexp::List(l)
            };

            Sexp::List(vec![
                Sexp::String("typeOf".to_owned()),
                expr,
                to_sexp_raw(type_child), // Type tree - no wrapping
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sexp_roundtrip_simple() {
        use symbolic_expressions::IntoSexp;

        // Parse a typeOf expression, serialize it back, and verify it matches
        let input = "(typeOf a 0)";
        let tree: TreeNode<String> = input.parse().unwrap();

        // Internal representation: a(0)
        assert_eq!(tree.label(), "a");
        assert_eq!(tree.children().len(), 1);
        assert_eq!(tree.children()[0].label(), "0");

        // Serialize back to typeOf format
        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_nested() {
        use symbolic_expressions::IntoSexp;

        // Nested typeOf expressions
        let input = "(typeOf (a (typeOf b 0) (typeOf c 0)) 0)";
        let tree: TreeNode<String> = input.parse().unwrap();

        // Internal: a(b(0), c(0), 0)
        assert_eq!(tree.label(), "a");
        assert_eq!(tree.children().len(), 3);
        assert_eq!(tree.children()[0].label(), "b");
        assert_eq!(tree.children()[1].label(), "c");
        assert_eq!(tree.children()[2].label(), "0");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_complex_type() {
        use symbolic_expressions::IntoSexp;

        // Expression with a complex type tree (-> int int)
        let input = "(typeOf (f (typeOf x int)) (-> int int))";
        let tree: TreeNode<String> = input.parse().unwrap();

        // Internal: f(x(int), (-> int int))
        assert_eq!(tree.label(), "f");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "x");
        assert_eq!(tree.children()[1].label(), "->");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }

    #[test]
    fn sexp_roundtrip_large() {
        use symbolic_expressions::IntoSexp;

        let input = "(typeOf (natLam (typeOf (natLam (typeOf (natLam (typeOf (lam (typeOf (lam (typeOf (app (typeOf (app (typeOf map (fun (fun (arrT $n0 f32) (arrT $n1 f32)) (fun (arrT $n2 (arrT $n0 f32)) (arrT $n2 (arrT $n1 f32))))) (typeOf (lam (typeOf (app (typeOf (app (typeOf map (fun (fun (arrT $n0 f32) f32) (fun (arrT $n1 (arrT $n0 f32)) (arrT $n1 f32)))) (typeOf (lam (typeOf (app (typeOf (app (typeOf (app (typeOf reduce (fun (fun f32 (fun f32 f32)) (fun f32 (fun (arrT $n0 f32) f32)))) (typeOf add (fun f32 (fun f32 f32)))) (fun f32 (fun (arrT $n0 f32) f32))) (typeOf 0.0 f32)) (fun (arrT $n0 f32) f32)) (typeOf (app (typeOf (app (typeOf map (fun (fun (pairT f32 f32) f32) (fun (arrT $n0 (pairT f32 f32)) (arrT $n0 f32)))) (typeOf (lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf $e0 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf $e0 (pairT f32 f32))) f32)) f32)) (fun (pairT f32 f32) f32))) (fun (arrT $n0 (pairT f32 f32)) (arrT $n0 f32))) (typeOf (app (typeOf (app (typeOf zip (fun (arrT $n0 f32) (fun (arrT $n0 f32) (arrT $n0 (pairT f32 f32))))) (typeOf $e1 (arrT $n0 f32))) (fun (arrT $n0 f32) (arrT $n0 (pairT f32 f32)))) (typeOf $e0 (arrT $n0 f32))) (arrT $n0 (pairT f32 f32)))) (arrT $n0 f32))) f32)) (fun (arrT $n0 f32) f32))) (fun (arrT $n1 (arrT $n0 f32)) (arrT $n1 f32))) (typeOf (app (typeOf transpose (fun (arrT $n0 (arrT $n1 f32)) (arrT $n1 (arrT $n0 f32)))) (typeOf $e1 (arrT $n0 (arrT $n1 f32)))) (arrT $n1 (arrT $n0 f32)))) (arrT $n1 f32))) (fun (arrT $n0 f32) (arrT $n1 f32)))) (fun (arrT $n2 (arrT $n0 f32)) (arrT $n2 (arrT $n1 f32)))) (typeOf $e1 (arrT $n2 (arrT $n0 f32)))) (arrT $n2 (arrT $n1 f32)))) (fun (arrT $n0 (arrT $n1 f32)) (arrT $n2 (arrT $n1 f32))))) (fun (arrT $n2 (arrT $n0 f32)) (fun (arrT $n0 (arrT $n1 f32)) (arrT $n2 (arrT $n1 f32)))))) (natFun (fun (arrT $n2 (arrT $n0 f32)) (fun (arrT $n0 (arrT $n1 f32)) (arrT $n2 (arrT $n1 f32))))))) (natFun (natFun (fun (arrT $n2 (arrT $n0 f32)) (fun (arrT $n0 (arrT $n1 f32)) (arrT $n2 (arrT $n1 f32)))))))) (natFun (natFun (natFun (fun (arrT $n2 (arrT $n0 f32)) (fun (arrT $n0 (arrT $n1 f32)) (arrT $n2 (arrT $n1 f32))))))))";
        let tree: TreeNode<String> = input.parse().unwrap();

        assert_eq!(tree.label(), "natLam");

        let output = tree.into_sexp().to_string();
        assert_eq!(output, input);
    }
}
