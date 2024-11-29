use std::fmt::{Display, Formatter};

use egg::RecExpr;

use crate::features::{AsFeatures, SymbolList};
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawAst {
    name: String,
    children: Box<[RawAst]>,
    features: Vec<f64>,
}

impl RawAst {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn features(&self) -> &[f64] {
        &self.features
    }
}

impl Tree for RawAst {
    fn children(&self) -> &[Self] {
        &self.children
    }
}

impl Display for RawAst {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.children.is_empty() {
            write!(f, "{}", self.name)
        } else {
            let inner = self
                .children
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "({} {inner})", self.name)
        }
    }
}

impl<L: AsFeatures + Display> From<&RecExpr<L>> for RawAst {
    fn from(expr: &RecExpr<L>) -> Self {
        fn rec<L: AsFeatures + Display>(
            node: &L,
            expr: &RecExpr<L>,
            symbol_list: &SymbolList<L>,
        ) -> RawAst {
            RawAst {
                name: node.to_string(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| rec(&expr[*child_id], expr, symbol_list))
                    .collect(),
                features: node.features(symbol_list),
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = expr.as_ref().last().unwrap();
        let symbol_list = L::symbols();
        rec(root, expr, &symbol_list)
    }
}
