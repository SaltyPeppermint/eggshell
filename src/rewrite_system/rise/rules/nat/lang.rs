use egg::{Analysis, DidMerge, EGraph, Id, RecExpr};
use serde::{Deserialize, Serialize};

use super::super::Index;

use super::Rise;

egg::define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum RiseMath {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),

        Var(Index),
        Constant(u32),
        // Symbol(Symbol),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<RiseMath> for ConstantFold {
    type Data = Option<u32>;

    fn make(egraph: &mut EGraph<RiseMath, ConstantFold>, enode: &RiseMath) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        Some(match enode {
            RiseMath::Constant(c) => *c,
            RiseMath::Add([a, b]) => x(a)? + x(b)?,
            RiseMath::Sub([a, b]) => x(a)? - (x(b)?),
            RiseMath::Mul([a, b]) => x(a)? * x(b)?,
            RiseMath::Pow([a, b]) => x(a)?.pow(x(b)?),
            RiseMath::Div([a, b]) if x(a)? % x(b)? == 0 => x(a)? / x(b)?,
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph<RiseMath, ConstantFold>, id: Id) {
        let data = egraph[id].data;
        if let Some(c) = data {
            let added = egraph.add(RiseMath::Constant(c));
            egraph.union(id, added);

            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

pub fn to_nat_expr(rise_expr: &RecExpr<Rise>) -> RecExpr<RiseMath> {
    let t = rise_expr
        .into_iter()
        .map(|n| match n {
            Rise::Var(index) => Ok(RiseMath::Var(*index)),

            Rise::NatAdd([a, b]) => Ok(RiseMath::Add([*a, *b])),
            Rise::NatSub([a, b]) => Ok(RiseMath::Sub([*a, *b])),
            Rise::NatMul([a, b]) => Ok(RiseMath::Mul([*a, *b])),
            Rise::NatDiv([a, b]) => Ok(RiseMath::Div([*a, *b])),
            Rise::NatPow([a, b]) => Ok(RiseMath::Pow([*a, *b])),

            Rise::Integer(i) => Ok(RiseMath::Constant((*i).try_into().unwrap())),
            _ => Err(()),
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    RecExpr::from(t)
}

pub fn to_rise_expr(nat_expr: &RecExpr<RiseMath>) -> RecExpr<Rise> {
    let t = nat_expr
        .into_iter()
        .map(|n| match n {
            RiseMath::Var(index) => Rise::Var(*index),

            RiseMath::Add([a, b]) => Rise::NatAdd([*a, *b]),
            RiseMath::Sub([a, b]) => Rise::NatSub([*a, *b]),
            RiseMath::Mul([a, b]) => Rise::NatMul([*a, *b]),
            RiseMath::Div([a, b]) => Rise::NatDiv([*a, *b]),
            RiseMath::Pow([a, b]) => Rise::NatPow([*a, *b]),

            RiseMath::Constant(c) => Rise::Integer((*c).try_into().unwrap()),
        })
        .collect::<Vec<_>>();
    RecExpr::from(t)
}
