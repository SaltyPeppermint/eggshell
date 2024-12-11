use std::fmt::{Display, Formatter};

use egg::RecExpr;

use crate::features::{AsFeatures, Feature, FeatureError, Featurizer, SymbolType};
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawAst<L: AsFeatures> {
    node: L,
    children: Box<[RawAst<L>]>,
    features: Feature,
}

impl<L: AsFeatures> RawAst<L> {
    pub fn new(expr: &RecExpr<L>, featurizer: &Featurizer<L>) -> Result<Self, FeatureError> {
        fn rec<L: AsFeatures>(
            node: &L,
            expr: &RecExpr<L>,
            featurizer: &Featurizer<L>,
        ) -> Result<RawAst<L>, FeatureError> {
            Ok(RawAst {
                node: node.clone(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| rec(&expr[*child_id], expr, featurizer))
                    .collect::<Result<_, _>>()?,
                features: featurizer.features(node)?,
            })
        }
        let root = expr.as_ref().last().unwrap();
        let raw_ast = rec(root, expr, featurizer)?;
        Ok(raw_ast)
    }

    pub fn node(&self) -> &L {
        &self.node
    }

    fn count_symbol(&self, symbol: &L) -> usize {
        usize::from(match self.node.symbol_type() {
            SymbolType::Constant(_) => self.node.discriminant() == symbol.discriminant(),
            SymbolType::Variable(name) => {
                if let SymbolType::Variable(other_name) = symbol.symbol_type() {
                    name == other_name && self.node.discriminant() == symbol.discriminant()
                } else {
                    false
                }
            }
            SymbolType::Operator | SymbolType::MetaSymbol => symbol.matches(&self.node),
        }) + self
            .children
            .iter()
            .map(|c| c.count_symbol(symbol))
            .sum::<usize>()
    }

    pub fn count_symbols(&self, featurizer: &Featurizer<L>) -> Vec<usize> {
        featurizer
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

impl<L: AsFeatures> Tree for RawAst<L> {
    fn children(&self) -> &[Self] {
        &self.children
    }
}

impl<L: AsFeatures> Display for RawAst<L> {
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
        let featurizer =
            <Simple as TermRewriteSystem>::Language::featurizer(vec!["a".into(), "b".into()]);
        let raw_ast = RawAst::new(&expr, &featurizer).unwrap();

        assert_eq!(&Feature::NonLeaf(1), raw_ast.features());
        assert_eq!(&Feature::NonLeaf(0), raw_ast.children()[0].features());
        assert_eq!(
            raw_ast.children()[1].features(),
            &Feature::Leaf(vec![1.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            raw_ast.children()[0].children()[0].features(),
            &Feature::Leaf(vec![0.0, 1.0, 0.0, 0.0])
        );
        assert_eq!(
            raw_ast.children()[0].children()[1].features(),
            &Feature::Leaf(vec![0.0, 0.0, 1.0, 0.0])
        );
    }

    #[test]
    fn invert_simple() {
        let expr = "(* (+ a b) 1)"
            .parse::<RecExpr<<Simple as TermRewriteSystem>::Language>>()
            .unwrap();
        let featurizer =
            <Simple as TermRewriteSystem>::Language::featurizer(vec!["a".into(), "b".into()]);
        let raw_ast = RawAst::new(&expr, &featurizer).unwrap();
        let inverted_flattened = raw_ast.flatten();

        assert_eq!(
            inverted_flattened[0],
            &RawAst {
                node: "1"
                    .parse::<RecExpr<<Simple as TermRewriteSystem>::Language>>()
                    .unwrap()
                    .as_ref()[0]
                    .clone(),
                children: Box::new([]),
                features: Feature::Leaf(vec![1.0, 0.0, 0.0, 1.0])
            }
        );
    }

    #[test]
    fn halide_expr() {
        let expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();
        let featurizer = <Halide as TermRewriteSystem>::Language::featurizer(vec![
            "v0".into(),
            "v1".into(),
            "v2".into(),
        ]);
        let raw_ast = RawAst::new(&expr, &featurizer).unwrap();

        assert_eq!(&Feature::NonLeaf(10), raw_ast.features());
        assert_eq!(&Feature::NonLeaf(0), raw_ast.children()[0].features());
        assert_eq!(
            raw_ast.children()[1].children()[1].features(),
            &Feature::Leaf(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            raw_ast.children()[0].children()[0].children()[0].features(),
            &Feature::Leaf(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            raw_ast.children()[0].children()[0].children()[1].features(),
            &Feature::Leaf(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        );
    }
}
