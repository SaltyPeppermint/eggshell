mod rules;

use egg::{Analysis, DidMerge, Id, PatternAst, Subst, Symbol, define_language};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{LangExtras, SymbolInfo, SymbolType, TermRewriteSystem};

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

pub type Constant = NotNan<f64>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize, Deserialize, EnumDiscriminants, EnumCount)]
    #[strum_discriminants(derive(EnumIter))]
    pub enum Math {
        "d" = Diff([Id; 2]),
        "i" = Integral([Id; 2]),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
        "ln" = Ln(Id),
        "sqrt" = Sqrt(Id),

        "sin" = Sin(Id),
        "cos" = Cos(Id),

        Constant(Constant),
        Symbol(Symbol),
    }
}

impl LangExtras for Math {
    fn symbol_info(&self) -> SymbolInfo {
        let id = MathDiscriminants::iter()
            .position(|x| x == self.into())
            .unwrap();
        match self {
            Math::Symbol(name) => SymbolInfo::new(id, SymbolType::Variable(name.to_string())),
            Math::Constant(value) => SymbolInfo::new(id, SymbolType::Constant(value.to_string())),
            _ => SymbolInfo::new(id, SymbolType::Operator),
        }
    }

    fn operators() -> Vec<&'static str> {
        vec![
            "d", "i", "+", "-", "*", "/", "pow", "ln", "sqrt", "sin", "cos",
        ]
    }

    const MAX_ARITY: usize = 2;
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<Math> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Math>)>;

    fn make(egraph: &mut EGraph, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            Math::Constant(c) => (*c, format!("{c}").parse().unwrap()),
            Math::Add([a, b]) => (
                x(a)? + x(b)?,
                format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Sub([a, b]) => (
                x(a)? - x(b)?,
                format!("(- {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Mul([a, b]) => (
                x(a)? * x(b)?,
                format!("(* {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Div([a, b]) if x(b) != Some(NotNan::new(0.0).unwrap()) => (
                x(a)? / x(b)?,
                format!("(/ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{c}").parse().unwrap(),
                    &Subst::default(),
                    "constant_fold".to_owned(),
                );
            } else {
                let added = egraph.add(Math::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Arithmetic;

impl TermRewriteSystem for Arithmetic {
    type Language = Math;
    type Analysis = ConstantFold;

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        self::rules::rules()
    }
}
