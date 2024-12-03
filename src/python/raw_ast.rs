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
    pub fn new<L: AsFeatures + Display>(
        expr: &RecExpr<L>,
        variable_names: Vec<String>,
    ) -> (Self, usize) {
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
        let feature_vec_len = symbol_list.feature_vec_len();
        let root = expr.as_ref().last().unwrap();
        let raw_ast = rec(root, expr, &symbol_list);
        (raw_ast, feature_vec_len)
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

    pub fn flatten(&self) -> Vec<&RawAst> {
        fn rec<'a>(raw_ast: &'a RawAst, flat: &mut Vec<&'a RawAst>) {
            flat.push(raw_ast);
            for c in raw_ast.children() {
                rec(c, flat);
            }
        }
        let mut flat = Vec::new();
        rec(self, &mut flat);
        flat.reverse();
        flat
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

#[cfg(test)]
mod tests {
    use crate::trs::{Halide, Simple, TermRewriteSystem};

    use super::*;

    #[test]
    fn simple_expr() {
        let expr = "(* (+ a b) 1)"
            .parse::<RecExpr<<Simple as TermRewriteSystem>::Language>>()
            .unwrap();
        let raw_ast = RawAst::new(&expr, vec!["a".into(), "b".into()]).0;

        assert_eq!(&Feature::NonLeaf(1), raw_ast.features());
        assert_eq!(&Feature::NonLeaf(0), raw_ast.children()[0].features());
        assert_eq!(
            &Feature::Leaf(vec![1.0, 0.0, 0.0, 1.0]),
            raw_ast.children()[1].features()
        );
        assert_eq!(
            &Feature::Leaf(vec![0.0, 1.0, 0.0, 0.0]),
            raw_ast.children()[0].children()[0].features()
        );
        assert_eq!(
            &Feature::Leaf(vec![0.0, 0.0, 1.0, 0.0]),
            raw_ast.children()[0].children()[1].features()
        );
    }

    #[test]
    fn invert_simple() {
        let expr = "(* (+ a b) 1)"
            .parse::<RecExpr<<Simple as TermRewriteSystem>::Language>>()
            .unwrap();
        let (raw_ast, _) = RawAst::new(&expr, vec!["a".into(), "b".into()]);
        let inverted_flattened = raw_ast.flatten();

        assert_eq!(
            &RawAst {
                name: "1".to_owned(),
                children: Box::new([]),
                features: Feature::Leaf(vec![1.0, 0.0, 0.0, 1.0])
            },
            inverted_flattened[0]
        );
    }

    #[test]
    fn halide_expr() {
        let expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();
        let raw_ast = RawAst::new(&expr, vec!["v0".into(), "v1".into(), "v2".into()]).0;

        assert_eq!(&Feature::NonLeaf(10), raw_ast.features());
        assert_eq!(&Feature::NonLeaf(0), raw_ast.children()[0].features());
        assert_eq!(
            &Feature::Leaf(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            raw_ast.children()[1].children()[1].features()
        );
        assert_eq!(
            &Feature::Leaf(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            raw_ast.children()[0].children()[0].children()[0].features()
        );
        assert_eq!(
            &Feature::Leaf(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            raw_ast.children()[0].children()[0].children()[1].features()
        );
    }
}
