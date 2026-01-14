use egg::{Applier, EGraph, Id, PatternAst, RecExpr, Subst, Symbol, Var};
use hashbrown::HashSet;

use crate::rise::db::{Cutoff, Shift};
use crate::rise::kind::Kindable;
use crate::rise::{Rise, RiseAnalysis};

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: Shift,
    cutoff: Cutoff,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(
        var_str: &str,
        shifted_var_str: &str,
        shift: Shift,
        cutoff: Cutoff,
        applier: A,
    ) -> Self {
        Shifted {
            var: var_str.parse().unwrap(),
            new_var: shifted_var_str.parse().unwrap(),
            shift,
            cutoff,
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for Shifted<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(mut extract) = egraph[subst[self.var]].data.small_repr(egraph) else {
            return Vec::new();
        };
        shift_mut(&mut extract, self.shift, self.cutoff);

        let mut new_subst = subst.clone();
        let added_expr_id = egraph.add_expr(&extract);
        new_subst.insert(self.new_var, added_expr_id);

        self.applier
            .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
    }
}

pub struct ShiftedCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: Shift,
    cutoff: Cutoff,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ShiftedCheck<A> {
    #[expect(unused)]
    pub fn new(
        var_str: &str,
        shifted_var_str: &str,
        shift: Shift,
        cutoff: Cutoff,
        applier: A,
    ) -> Self {
        ShiftedCheck {
            var: var_str.parse().unwrap(),
            new_var: shifted_var_str.parse().unwrap(),
            shift,
            cutoff,
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ShiftedCheck<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(expected) = egraph[subst[self.new_var]].data.small_repr(egraph) else {
            return Vec::new();
        };
        let Some(mut extract) = egraph[subst[self.var]].data.small_repr(egraph) else {
            return Vec::new();
        };
        shift_mut(&mut extract, self.shift, self.cutoff);

        if extract == expected {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

pub fn shift_mut(expr: &mut RecExpr<Rise>, shift: Shift, cutoff: Cutoff) {
    fn rec(
        expr: &mut RecExpr<Rise>,
        id: Id,
        shift: Shift,
        cutoff: Cutoff,
        already_shifted: &mut HashSet<Id>,
    ) {
        if !already_shifted.insert(id) {
            return;
        }

        match expr[id] {
            Rise::Var(index) => {
                if index.value() >= cutoff.of_kind(index.kind()) {
                    let shifted_index = index + shift;
                    expr[id] = Rise::Var(shifted_index);
                }
            }
            Rise::Lambda(l, e) => {
                let new_cutoff = cutoff.inc(l.kind());
                rec(expr, e, shift, new_cutoff, already_shifted);
            }
            Rise::App(_, [c_id_a, c_id_b])
            | Rise::TypeOf([c_id_a, c_id_b])
            | Rise::FunType([c_id_a, c_id_b])
            | Rise::ArrType([c_id_a, c_id_b])
            | Rise::VecType([c_id_a, c_id_b])
            | Rise::PairType([c_id_a, c_id_b])
            | Rise::NatAdd([c_id_a, c_id_b])
            | Rise::NatSub([c_id_a, c_id_b])
            | Rise::NatMul([c_id_a, c_id_b])
            | Rise::NatDiv([c_id_a, c_id_b])
            | Rise::NatPow([c_id_a, c_id_b]) => {
                // Mean bug lurking here, we must not shift down twice if the children point to the same
                // Eclass
                // this is prevented by the already_shifted check
                rec(expr, c_id_a, shift, cutoff, already_shifted);
                rec(expr, c_id_b, shift, cutoff, already_shifted);
            }
            Rise::NatFun(c_id)
            | Rise::DataFun(c_id)
            | Rise::AddrFun(c_id)
            | Rise::NatNatFun(c_id)
            | Rise::IndexType(c_id) => rec(expr, c_id, shift, cutoff, already_shifted),

            Rise::Let
            | Rise::NatType
            | Rise::F32
            | Rise::AsVector
            | Rise::AsScalar
            | Rise::VectorFromScalar
            | Rise::Snd
            | Rise::Fst
            | Rise::Add
            | Rise::Mul
            | Rise::ToMem
            | Rise::Split
            | Rise::Join
            | Rise::Generate
            | Rise::Transpose
            | Rise::Zip
            | Rise::Unzip
            | Rise::Map
            | Rise::MapPar
            | Rise::Reduce
            | Rise::ReduceSeq
            | Rise::ReduceSeqUnroll
            | Rise::Integer(_)
            | Rise::Float(_) => (),
        }
    }
    rec(expr, expr.root(), shift, cutoff, &mut HashSet::new());
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn shift_applier() {
        fn check(
            to_shift: &str,
            ground_truth: &str,
            cutoff: (u32, u32, u32, u32, u32),
            shift: (i32, i32, i32, i32, i32),
        ) {
            let mut a = to_shift.parse().unwrap();
            let b = &ground_truth.parse().unwrap();
            shift_mut(&mut a, shift.into(), cutoff.into());
            assert_eq!(&a, b);
        }
        check(
            "(app %e0 %e1)",
            "(app %e1 %e2)",
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e1 %e2) f32)",
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e0 %e2) f32)",
            (1, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e3) f32)) f32)",
            (1, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e1) f32)) f32)",
            (1, 0, 0, 0, 0),
            (-1, 0, 0, 0, 0),
        );
        check(
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) f32))",
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) f32))",
            (0, 0, 0, 0, 0),
            (2, 0, 0, 0, 0),
        );
    }
}
