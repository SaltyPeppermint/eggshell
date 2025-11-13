use egg::{Analysis, DidMerge, EGraph, Id, Symbol};
use serde::{Deserialize, Serialize};

egg::define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum RiseNat {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),

        Constant(u32),
        Symbol(Symbol),
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<RiseNat> for ConstantFold {
    type Data = Option<u32>;

    fn make(egraph: &mut EGraph<RiseNat, ConstantFold>, enode: &RiseNat) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        Some(match enode {
            RiseNat::Constant(c) => *c,
            RiseNat::Add([a, b]) => x(a)? + x(b)?,
            RiseNat::Sub([a, b]) => x(a)? - (x(b)?),
            RiseNat::Mul([a, b]) => x(a)? * x(b)?,
            RiseNat::Pow([a, b]) => x(a)?.pow(x(b)?),
            RiseNat::Div([a, b]) if x(a)? % x(b)? == 0 => x(a)? / x(b)?,
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph<RiseNat, ConstantFold>, id: Id) {
        let data = egraph[id].data;
        if let Some(c) = data {
            let added = egraph.add(RiseNat::Constant(c));
            egraph.union(id, added);

            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}
