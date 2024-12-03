use std::fmt::{Display, Formatter};

use egg::RecExpr;

use crate::features::{AsFeatures, Feature, Featurizer, SymbolType};
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawAst<L: AsFeatures + Display> {
    node: L,
    children: Box<[RawAst<L>]>,
    features: Feature,
}

impl<L: AsFeatures + Display> RawAst<L> {
    pub fn new(expr: &RecExpr<L>, variable_names: Vec<String>) -> (Self, Featurizer<L>) {
        fn rec<L: AsFeatures + Display>(
            node: &L,
            expr: &RecExpr<L>,
            featurizer: &Featurizer<L>,
        ) -> RawAst<L> {
            RawAst {
                node: node.clone(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| rec(&expr[*child_id], expr, featurizer))
                    .collect(),
                features: featurizer.features(node),
            }
        }
        let featurizer = L::symbol_list(variable_names);
        let root = expr.as_ref().last().unwrap();
        let raw_ast = rec(root, expr, &featurizer);
        (raw_ast, featurizer)
    }

    pub fn node(&self) -> &L {
        &self.node
    }

    fn count_symbol(&self, symbol: &L) -> usize {
        usize::from(if let SymbolType::Constant(_) = self.node.symbol_type() {
            self.node.discriminant() == symbol.discriminant()
        } else {
            symbol.matches(&self.node)
        }) + self
            .children
            .iter()
            .map(|c| c.count_symbol(symbol))
            .sum::<usize>()
    }

    pub fn count_symbols(&self, variable_names: Vec<String>) -> Vec<usize> {
        L::symbol_list(variable_names)
            .symbols()
            .iter()
            .map(|symbol| self.count_symbol(symbol))
            .collect()
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

    pub fn flatten(&self) -> Vec<&RawAst<L>> {
        fn rec<'a, IL: AsFeatures + Display>(
            raw_ast: &'a RawAst<IL>,
            flat: &mut Vec<&'a RawAst<IL>>,
        ) {
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

impl<L: AsFeatures + Display> Tree for RawAst<L> {
    fn children(&self) -> &[Self] {
        &self.children
    }
}

impl<L: AsFeatures + Display> Display for RawAst<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.children.is_empty() {
            write!(f, "{}", self.node)
        } else {
            let inner = self
                .children
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "({} {inner})", self.node)
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
                node: "1"
                    .parse::<RecExpr<<Simple as TermRewriteSystem>::Language>>()
                    .unwrap()
                    .as_ref()[0]
                    .clone(),
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
