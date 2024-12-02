use std::fmt::{Display, Formatter};

use egg::RecExpr;

use crate::features::{AsFeatures, Feature, Featurizer};
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawAst {
    name: String,
    children: Box<[RawAst]>,
    features: Feature,
}

impl RawAst {
    pub fn new<L: AsFeatures + Display>(expr: &RecExpr<L>, variable_names: Vec<String>) -> Self {
        fn rec<L: AsFeatures + Display>(
            node: &L,
            expr: &RecExpr<L>,
            featurizer: &Featurizer<L>,
        ) -> RawAst {
            RawAst {
                name: node.to_string(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| rec(&expr[*child_id], expr, featurizer))
                    .collect(),
                features: featurizer.features(node),
            }
        }
        let symbol_list = L::symbol_list(variable_names);
        let root = expr.as_ref().last().unwrap();
        rec(root, expr, &symbol_list)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn arity(&self) -> usize {
        self.children.len()
    }

    pub fn features(&self) -> &Feature {
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
