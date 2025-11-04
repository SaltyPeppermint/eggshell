mod rules;
mod substitute;

use egg::{Analysis, DidMerge, Id, Language, RecExpr, Symbol};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

// Big thanks to @Bastacyclop for implementing this all
// https://github.com/Bastacyclop/egg-rise/blob/main/src/main.rs

// Defining aliases to reduce code.
type EGraph = egg::EGraph<DummyRiseLang, DummyRiseAnalysis>;
type Rewrite = egg::Rewrite<DummyRiseLang, DummyRiseAnalysis>;

egg::define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum DummyRiseLang {
        "var" = Var(Id),
        "app" = App([Id; 2]),
        "lam" = Lambda([Id; 2]),
        "let" = Let([Id; 3]),
        ">>" = Then([Id; 2]),

        Number(i32),
        Symbol(Symbol),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct DummyRiseAnalysis;

#[derive(Default, Debug, Clone, Serialize)]
pub struct Data {
    pub free: HashSet<Id>,
    pub beta_extract: RecExpr<DummyRiseLang>,
}

impl Analysis<DummyRiseLang> for DummyRiseAnalysis {
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

    fn make(egraph: &mut EGraph, enode: &DummyRiseLang) -> Data {
        let extend = |free: &mut HashSet<Id>, i: &Id| {
            free.extend(&egraph[*i].data.free);
        };
        let mut free = HashSet::default();
        match enode {
            DummyRiseLang::Var(v) => {
                free.insert(*v);
            }
            DummyRiseLang::Lambda([v, a]) => {
                extend(&mut free, a);
                free.remove(v);
            }
            DummyRiseLang::Let([v, a, b]) => {
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
            Vec::new().into()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };
        Data { free, beta_extract }
    }
}

fn unwrap_symbol(n: &DummyRiseLang) -> Symbol {
    match n {
        &DummyRiseLang::Symbol(s) => s,
        _ => panic!("expected symbol"),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Rise;

#[must_use]
pub fn rules(names: &[&str], use_explicit_subs: bool) -> Vec<Rewrite> {
    self::rules::filtered_rules(names, use_explicit_subs)
}

#[must_use]
pub fn full_rules() -> Vec<Rewrite> {
    self::rules::rules(true).into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_1() {
        let dot_str = "(lam a (lam b (app (app (app reduce add) 0) (app (app map (lam mt (app (app mul (app fst (var mt))) (app snd (var mt))))) (app (app zip (var a)) (var b))))))";
        let _ = dot_str.parse::<RecExpr<DummyRiseLang>>().unwrap();
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("fst"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("snd"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("reduce"))));
        // assert!(!dot_rec_expr.contains(&RiseLang::Symbol(Symbol::new("zip"))));
    }
}
