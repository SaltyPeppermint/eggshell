mod rules;
mod substitute;

use egg::{define_language, Analysis, DidMerge, Id, Language, RecExpr, Symbol};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use strum::{EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{MetaInfo, SymbolType, TermRewriteSystem};
// use crate::typing::{Type, Typeable, TypingInfo};

// Big thanks to @Bastacyclop for implementing this all
// https://github.com/Bastacyclop/egg-rise/blob/main/src/main.rs

// Defining aliases to reduce code.
type EGraph = egg::EGraph<RiseLang, RiseAnalysis>;
type Rewrite = egg::Rewrite<RiseLang, RiseAnalysis>;

define_language! {
    #[derive(Serialize, Deserialize, EnumDiscriminants)]
    #[strum_discriminants(derive(EnumIter))]
    pub enum RiseLang {
        "var" = Var(Id),
        "app" = App([Id; 2]),
        "lam" = Lambda([Id; 2]),

        "let" = Let([Id; 3]),
        // "fix"

        ">>" = Then([Id; 2]),

        // Rise builtins
        "toMem"= ToMem,
        "iterateStream"= IterateStream,
        "map" = Map,
        "mapSeq" = MapSeq,

        "split" = Split,
        "join"= Join,

        "transpose" = Transpose,

        "rotateValues" = RotateValues,
        "slide" = Slide,

        "reduce" = Reduce,
        "reduceSeqUnroll" = ReduceSeqUnroll,

        "zip" = Zip,

        "fst" = Fst,
        "snd" = Snd,

        Number(i32),
        Symbol(Symbol),
    }
}

impl MetaInfo for RiseLang {
    fn symbol_type(&self) -> SymbolType {
        match self {
            RiseLang::Symbol(name) => SymbolType::Variable(name.as_str()),
            RiseLang::Number(value) => SymbolType::Constant(0, (*value).into()),
            _ => {
                let position = RiseLangDiscriminants::iter()
                    .position(|x| x == self.into())
                    .unwrap();
                SymbolType::Operator(position + Self::N_CONST_TYPES)
            }
        }
    }

    fn operator_names() -> Vec<&'static str> {
        vec![
            "var",
            "app",
            "lam",
            "let",
            ">>",
            "toMem",
            "iterateStream",
            "map",
            "mapSeq",
            "split",
            "join",
            "transpose",
            "rotateValues",
            "slide",
            "reduce",
            "reduceSeqUnroll",
            "zip",
            "fst",
            "snd",
        ]
    }

    const N_CONST_TYPES: usize = 1;

    // fn operators() -> Vec<&'static Self::EnumDiscriminant> {
    //     let mut o = RiseLangDiscriminants::VARIANTS.to_vec();
    //     o.truncate(o.len() - 2);
    //     o
    // }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct RiseAnalysis;

#[derive(Default, Debug, Clone, Serialize)]
pub struct Data {
    pub free: HashSet<Id>,
    pub beta_extract: RecExpr<RiseLang>,
}

impl Analysis<RiseLang> for RiseAnalysis {
    type Data = Data;

    fn merge(&mut self, to: &mut Data, from: Data) -> DidMerge {
        let before_len = to.free.len();
        to.free.extend(from.free);
        let mut did_change = before_len != to.free.len();
        if !from.beta_extract.as_ref().is_empty()
            && (to.beta_extract.as_ref().is_empty()
                || to.beta_extract.as_ref().len() > from.beta_extract.as_ref().len())
        {
            to.beta_extract = from.beta_extract;
            did_change = true;
        }
        DidMerge(did_change, true) // TODO: more precise second bool
    }

    fn make(egraph: &mut EGraph, enode: &RiseLang) -> Data {
        let extend = |free: &mut HashSet<Id>, i: &Id| {
            free.extend(&egraph[*i].data.free);
        };
        let mut free = HashSet::default();
        match enode {
            RiseLang::Var(v) => {
                free.insert(*v);
            }
            RiseLang::Lambda([v, a]) => {
                extend(&mut free, a);
                free.remove(v);
            }
            RiseLang::Let([v, a, b]) => {
                extend(&mut free, b);
                if free.remove(v) {
                    extend(&mut free, a);
                }
            }
            _ => {
                enode.for_each(|c| extend(&mut free, &c));
            }
        }
        let empty = enode.any(|id| egraph[id].data.beta_extract.as_ref().is_empty());
        let beta_extract = if empty {
            vec![].into()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };
        Data { free, beta_extract }
    }
}

fn unwrap_symbol(n: &RiseLang) -> Symbol {
    match n {
        &RiseLang::Symbol(s) => s,
        _ => panic!("expected symbol"),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Rise;

impl Rise {
    #[must_use]
    pub fn rules(names: &[&str], use_explicit_subs: bool) -> Vec<Rewrite> {
        self::rules::filtered_rules(names, use_explicit_subs)
    }
}

impl TermRewriteSystem for Rise {
    type Language = RiseLang;
    type Analysis = RiseAnalysis;

    fn full_rules() -> Vec<Rewrite> {
        self::rules::rules(true).into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_1() {
        let dot_str = "(lam a (lam b (app (app (app reduce add) 0) (app (app map (lam mt (app (app mul (app fst (var mt))) (app snd (var mt))))) (app (app zip (var a)) (var b))))))";
        let _ = dot_str.parse::<RecExpr<RiseLang>>().unwrap();
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("fst"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("snd"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("reduce"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("zip"))));
    }

    #[test]
    fn operators() {
        let known_operators = vec![
            "var",
            "app",
            "lam",
            "let",
            ">>",
            "toMem",
            "iterateStream",
            "map",
            "mapSeq",
            "split",
            "join",
            "transpose",
            "rotateValues",
            "slide",
            "reduce",
            "reduceSeqUnroll",
            "zip",
            "fst",
            "snd",
        ];
        assert_eq!(
            RiseLang::operator_names()
                .iter()
                .map(|x| (*x).to_owned())
                .collect::<Vec<_>>(),
            known_operators
        );
    }

    #[test]
    fn get_var_type() {
        let symbol = RiseLang::Var(Id::from(0));
        let symbol_type = symbol.symbol_type();
        assert_eq!(symbol_type, SymbolType::Operator(1));
    }

    #[test]
    fn get_var_type2() {
        let symbol = RiseLang::Map;
        let symbol_type = symbol.symbol_type();
        assert_eq!(symbol_type, SymbolType::Operator(8));
    }

    #[test]
    fn get_num_type() {
        let symbol = RiseLang::Number(1);
        let symbol_type = symbol.symbol_type();
        assert_eq!(symbol_type, SymbolType::Constant(0, 1.0));
    }

    #[test]
    fn get_symbol_type() {
        let symbol = RiseLang::Symbol("BLA".into());
        let symbol_type = symbol.symbol_type();
        assert_eq!(symbol_type, SymbolType::Variable("BLA"));
    }
}
