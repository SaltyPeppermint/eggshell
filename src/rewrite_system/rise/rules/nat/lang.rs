use egg::{Analysis, DidMerge, EGraph, Id, RecExpr};
use serde::{Deserialize, Serialize};

use super::Rise;
use crate::rewrite_system::rise::Index;

egg::define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum Math {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),

        Var(Index),
        Constant(i32),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<Math> for ConstantFold {
    type Data = Option<i32>;

    fn make(egraph: &mut EGraph<Math, ConstantFold>, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        Some(match enode {
            Math::Constant(c) => *c,
            Math::Add([a, b]) => x(a)? + x(b)?,
            Math::Sub([a, b]) => x(a)? - x(b)?,
            Math::Mul([a, b]) => {
                // println!("trying {} * {}", x(a)?, x(b)?);
                x(a)? * x(b)?
            }
            Math::Pow([a, b]) => x(a)?.pow(x(b)?.try_into().ok()?),
            // if ((x(a)? % x(b)?) == Ratio::<i32>::ZERO) not needed we got proper ratios
            Math::Div([a, b]) => {
                // println!("trying {} / {}", x(a)?, x(b)?);
                x(a)? / x(b)?
            }
            Math::Var(_) => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph<Math, ConstantFold>, id: Id) {
        let data = egraph[id].data;
        if let Some(c) = data {
            let added = egraph.add(Math::Constant(c));
            egraph.union(id, added);

            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

pub fn to_nat_expr(rise_expr: &RecExpr<Rise>) -> RecExpr<Math> {
    let t = rise_expr
        .into_iter()
        .map(|n| match n {
            Rise::Var(index) => Ok(Math::Var(*index)),
            Rise::Integer(i) => Ok(Math::Constant(*i)),

            Rise::NatAdd([a, b]) => Ok(Math::Add([*a, *b])),
            Rise::NatSub([a, b]) => Ok(Math::Sub([*a, *b])),
            Rise::NatMul([a, b]) => Ok(Math::Mul([*a, *b])),
            Rise::NatDiv([a, b]) => Ok(Math::Div([*a, *b])),
            Rise::NatPow([a, b]) => Ok(Math::Pow([*a, *b])),

            _ => Err(()),
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    RecExpr::from(t)
}

pub fn to_rise_expr(nat_expr: &RecExpr<Math>) -> RecExpr<Rise> {
    let t = nat_expr
        .into_iter()
        .map(|n| match n {
            Math::Var(index) => Rise::Var(*index),

            Math::Add([a, b]) => Rise::NatAdd([*a, *b]),
            Math::Sub([a, b]) => Rise::NatSub([*a, *b]),
            Math::Mul([a, b]) => Rise::NatMul([*a, *b]),
            Math::Div([a, b]) => Rise::NatDiv([*a, *b]),
            Math::Pow([a, b]) => Rise::NatPow([*a, *b]),

            Math::Constant(c) => Rise::Integer(*c),
        })
        .collect::<Vec<_>>();
    RecExpr::from(t)
}
