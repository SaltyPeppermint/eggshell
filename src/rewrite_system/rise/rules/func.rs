use egg::{Applier, EGraph, Id, Pattern, PatternAst, RecExpr, Subst, Symbol, Var};
use hashbrown::HashSet;

use crate::rewrite_system::rise::Index;

use super::{Rise, RiseAnalysis};

pub fn pat(pat: &str) -> impl Applier<Rise, RiseAnalysis> {
    pat.parse::<Pattern<Rise>>().unwrap()
}

pub fn not_free_in<A>(var: &str, index: u32, applier: A) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    NotFreeIn {
        var: var.parse().unwrap(),
        index: Index(index),
        applier,
    }
}

struct NotFreeIn<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    index: Index,
    applier: A,
}

impl<A> Applier<Rise, RiseAnalysis> for NotFreeIn<A>
where
    A: Applier<Rise, RiseAnalysis>,
{
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let free_in = egraph[subst[self.var]].data.free.contains(&self.index);
        if free_in {
            Vec::new()
        } else {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        }
    }
}

pub fn vectorize_scalar_fun<A>(
    var: &str,
    size_var: &str,
    vectorized_var: &str,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    VectorizeScalaFun {
        var: var.parse().unwrap(),
        size_var: size_var.parse().unwrap(),
        vectorized_var: vectorized_var.parse().unwrap(),
        applier,
    }
}

struct VectorizeScalaFun<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    size_var: Var,
    vectorized_var: Var,
    applier: A,
}

impl<A> Applier<Rise, RiseAnalysis> for VectorizeScalaFun<A>
where
    A: Applier<Rise, RiseAnalysis>,
{
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let extracted = &egraph[subst[self.var]].data.beta_extract;
        let size_extracted = &egraph[subst[self.size_var]].data.beta_extract;
        let n = extracted_to_u32(size_extracted);
        if let Some(vectorized_expr) =
            vectorize_expr(extracted, n, egraph, HashSet::new(), extracted.root())
        {
            let mut substitution = subst.clone();
            substitution.insert(self.vectorized_var, egraph.add_expr(&vectorized_expr));
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn extracted_to_u32(expr: &RecExpr<Rise>) -> u32 {
    if let Rise::Integer(i) = expr[0.into()] {
        return u32::try_from(i).unwrap();
    }
    panic!("Unexpected thing in expr")
}

fn vectorize_expr(
    expr: &RecExpr<Rise>,
    n: u32,
    egraph: &EGraph<Rise, RiseAnalysis>,
    v_env: HashSet<u32>,
    id: Id,
) -> Option<RecExpr<Rise>> {
    match &expr[id] {
        Rise::Var(index) => todo!(),
        Rise::App(_) => todo!(),
        Rise::Lambda(id) => todo!(),
        Rise::Symbol(global_symbol) => todo!(),
        Rise::Integer(_) => todo!(),

        Rise::TypeOf(_)
        | Rise::ArrType
        | Rise::VecType
        | Rise::PairType
        | Rise::IndexType
        | Rise::NatType
        | Rise::F32
        | Rise::ToMem
        | Rise::Split
        | Rise::Join
        | Rise::NatAdd(_)
        | Rise::NatSub(_)
        | Rise::NatMul(_)
        | Rise::NatDiv(_)
        | Rise::NatPow(_)
        | Rise::AsVector
        | Rise::AsScalar
        | Rise::Snd
        | Rise::Fst
        | Rise::Generate
        | Rise::Transpose
        | Rise::Unzip
        | Rise::Zip
        | Rise::MapPar
        | Rise::Reduce
        | Rise::ReduceSeq
        | Rise::ReduceSeqUnroll => unimplemented!(),
    }
    // expr.node match {
    //   case Var(i) if vEnv(i) =>
    //     for { tv <- vecDT(expr.t, n, eg) }
    //       yield ExprWithHashCons(Var(i), tv)
    //   case Var(_) => None
    //   case App(f, e) =>
    //     for { fv <- vectorizeExpr(f, n, eg, vEnv); ev <- vectorizeExpr(e, n, eg, vEnv) }
    //       yield ExprWithHashCons(App(fv, ev), eg(fv.t).asInstanceOf[FunType[TypeId]].outT)
    //   case Lambda(e) =>
    //     for { ev <- vectorizeExpr(e, n, eg, vEnv.map(_ + 1) + 0);
    //           xtv <- vecDT(eg(expr.t).asInstanceOf[FunType[TypeId]].inT, n, eg) }
    //       yield ExprWithHashCons(Lambda(ev),  eg.add(FunType(xtv, ev.t)))
    //   case NatApp(_, _) => None
    //   case DataApp(_, _) => None
    //   case AddrApp(_, _) => None
    //   case AppNatToNat(_, _) => None
    //   case NatLambda(_) => None
    //   case DataLambda(_) => None
    //   case AddrLambda(_) => None
    //   case LambdaNatToNat(_) => None
    //   case Literal(_) | NatLiteral(_) | IndexLiteral(_, _) =>
    //     for { tv <- vecDT(expr.t, n, eg) }
    //       yield ExprWithHashCons(App(
    //         ExprWithHashCons(Primitive(rcp.vectorFromScalar.primitive), eg.add(FunType(expr.t, tv))),
    //         expr), tv)
    //   case Primitive(rcp.add() | rcp.mul() | rcp.fst() | rcp.snd()) =>
    //     for { tv <- vecT(expr.t, n, eg) }
    //       yield ExprWithHashCons(expr.node, tv)
    //   case Primitive(_) => None
    //   case Composition(f, g) =>
    //     for { fv <- vectorizeExpr(f, n, eg, vEnv); gv <- vectorizeExpr(g, n, eg, vEnv) }
    //       yield ExprWithHashCons(Composition(fv, gv), eg.add(FunType(
    //         eg(fv.t).asInstanceOf[FunType[TypeId]].inT,
    //         eg(gv.t).asInstanceOf[FunType[TypeId]].outT,
    //       )))
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pat_rule_1() {
        pat(
            "(typeOf (lam (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt0) ?dt3)) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3)))) (typeOf ?e0 (app (app fun ?dt0) ?dt3))) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3))) (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt4) ?dt0)) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0)))) (typeOf (lam (typeOf ?e2 ?dt5)) (app (app fun ?dt4) ?dt0))) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0))) (typeOf %0 (app (app arrT ?n1) ?dt4))) (app (app arrT ?n1) ?dt0))) (app (app arrT ?n1) ?dt3))) (app (app fun (app (app arrT ?n0) ?dt1)) (app (app arrT ?n0) ?dt2)))",
        );
    }
}
