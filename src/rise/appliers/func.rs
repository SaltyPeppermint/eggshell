use egg::{Applier, EGraph, Id, Language, PatternAst, RecExpr, Subst, Symbol, Var};
use hashbrown::HashSet;

use crate::rise::lang::Primitive;
use crate::utils;

use crate::rise::db::Index;
use crate::rise::kind::{Kind, Kindable};
use crate::rise::lang::{Application, Nat};
use crate::rise::{Rise, RiseAnalysis};

pub struct NotFreeIn<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    index: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> NotFreeIn<A> {
    pub fn new(var_str: &str, index: u32, applier: A) -> Self {
        let var: Var = var_str.parse().unwrap();
        let kind = var.kind();
        NotFreeIn {
            var,
            index: Index::new(kind, index),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for NotFreeIn<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let free_in = egraph[subst[self.var]].data.free.contains(&self.index);
        // dbg!(free_in);
        if free_in {
            Vec::new()
        } else {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        }
    }
}

pub struct VectorizeScalarFun<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    size_var: Var,
    vectorized_var: Var,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> VectorizeScalarFun<A> {
    pub fn new(var: &str, size_var: &str, vectorized_var: &str, applier: A) -> Self {
        VectorizeScalarFun {
            var: var.parse().unwrap(),
            size_var: size_var.parse().unwrap(),
            vectorized_var: vectorized_var.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for VectorizeScalarFun<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(extracted) = egraph[egraph[subst[self.var]].parents().next().unwrap()]
            .data
            .small_repr(egraph)
        else {
            return Vec::new();
        };
        let Some(size_extracted) = &egraph[subst[self.size_var]].data.small_repr(egraph) else {
            return Vec::new();
        };
        let n = extracted_nat(size_extracted);
        if let Some((vectorized_expr, _, expr_id)) =
            vec_expr(&extracted, n, HashSet::new(), extracted.root())
        {
            let new_expr = vectorized_expr[expr_id].build_recexpr(|i| vectorized_expr[i].clone());
            let mut new_subst = subst.clone();
            let added_expr_id = egraph.add_expr(&new_expr);
            new_subst.insert(self.vectorized_var, added_expr_id);

            self.applier
                .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn extracted_nat(expr: &RecExpr<Rise>) -> i64 {
    if let Rise::NatCst(n) = expr[0.into()] {
        return n.0;
    }
    panic!("Unexpected thing in expr")
}

// Expr, ty_id, expr_id
fn vec_expr(
    expr: &RecExpr<Rise>,
    n: i64,
    v_env: HashSet<Index>,
    id: Id,
) -> Option<(RecExpr<Rise>, Id, Id)> {
    match &expr[id] {
        Rise::TypedVar(index, ty) if v_env.contains(index) => {
            let mut new = RecExpr::default();
            let vec_ty_id = utils::add_expr(&mut new, vec_ty(expr, n, *ty)?);
            let var_id = new.add(Rise::TypedVar(*index, vec_ty_id));
            Some((new, vec_ty_id, var_id))
        }

        Rise::App(app, [f, e, _]) if app.is_expr() => {
            let (fv, fv_ty_id, _) = vec_expr(expr, n, v_env.clone(), *f)?;
            let (ev, _, _) = vec_expr(expr, n, v_env.clone(), *e)?;

            let Rise::FunType([_, output_ty_id]) = fv[fv_ty_id] else {
                panic!("No Fun type wrapped in here: {:?}", &fv[fv_ty_id]);
            };

            let mut new = RecExpr::default();
            let fv_id = utils::add_expr(&mut new, fv);
            // The index for fv is preserved since add_expr only appends the second arg
            let ev_id = utils::add_expr(&mut new, ev);

            let app_id = new.add(Rise::App(*app, [fv_id, ev_id, output_ty_id]));

            Some((new, output_ty_id, app_id))
        }
        Rise::Lambda(lam, [e, ty]) if lam.is_expr() => {
            let new_v_env = v_env
                .into_iter()
                .map(|i| i.inc())
                .chain([Index::zero(Kind::Expr)])
                .collect::<HashSet<_>>();

            // Vectorize e
            let (ev, ev_ty_id, _) = vec_expr(expr, n, new_v_env, *e)?;
            let mut new = RecExpr::default();
            let typed_ev_id = utils::add_expr(&mut new, ev);

            // Get input type of the original expr and vectorize it (xtv in scala)
            let Rise::FunType([input_ty_id, _]) = expr[*ty] else {
                panic!("No Fun type wrapped in here: {:?}", &expr[*ty])
            };
            let vec_input_ty = vec_ty(expr, n, input_ty_id)?;

            // Wrap in a lambda and append to the existing recexpr

            let vec_input_ty_id = utils::add_expr(&mut new, vec_input_ty);
            let fun_id = new.add(Rise::FunType([vec_input_ty_id, ev_ty_id]));
            let lam_id = new.add(Rise::Lambda(*lam, [typed_ev_id, fun_id]));

            // Add the fun type of the new lam and wrap it in a typeof
            // evt index is valid since add_expr only appends

            Some((new, fun_id, lam_id))
        }
        Rise::IntLit(value, ty) => {
            let mut new = RecExpr::default();

            let new_ty_id = utils::add_expr(&mut new, utils::build(expr, *ty));
            let new_expr_id = new.add(Rise::IntLit(*value, new_ty_id));

            let vec_ty = vec_ty(expr, n, *ty)?;
            let vec_ty_id = utils::add_expr(&mut new, vec_ty);
            let prim_ty = new.add(Rise::FunType([new_ty_id, vec_ty_id]));
            let prim_id = new.add(Rise::Prim(Primitive::VectorFromScalar, prim_ty)); // asVector ?!

            let app_id = new.add(Rise::App(
                Application::new(Kind::Expr),
                [prim_id, new_expr_id, vec_ty_id],
            ));

            Some((new, vec_ty_id, app_id))
        }
        Rise::Prim(prim, ty) => match prim {
            &Primitive::Fst | Primitive::Snd | Primitive::Add | Primitive::Mul => {
                let mut typed_prim = RecExpr::default();
                let vec_prim_ty_id = utils::add_expr(&mut typed_prim, vec_ty(expr, n, *ty)?);
                let prim_id = typed_prim.add(Rise::Prim(*prim, vec_prim_ty_id));
                Some((typed_prim, vec_prim_ty_id, prim_id))
            }
            _ => None,
        },
        Rise::FloatLit(value, ty) => {
            let mut new = RecExpr::default();

            let new_ty_id = utils::add_expr(&mut new, utils::build(expr, *ty));
            let new_expr_id = new.add(Rise::FloatLit(*value, new_ty_id));

            let vec_ty = vec_ty(expr, n, *ty)?;
            let vec_ty_id = utils::add_expr(&mut new, vec_ty);
            let prim_ty = new.add(Rise::FunType([new_ty_id, vec_ty_id]));
            let prim_id = new.add(Rise::Prim(Primitive::VectorFromScalar, prim_ty)); // asVector ?!

            let app_id = new.add(Rise::App(
                Application::new(Kind::Expr),
                [prim_id, new_expr_id, vec_ty_id],
            ));

            Some((new, vec_ty_id, app_id))
        }
        Rise::TypedVar(_, _)
        | Rise::Var(_)
        | Rise::App(_, _)
        | Rise::Lambda(_, _)
        | Rise::Let(_) => None,
        Rise::FunType(_)
        | Rise::NatFun(_)
        | Rise::DataFun(_)
        | Rise::AddrFun(_)
        | Rise::NatNatFun(_)
        | Rise::ArrType(_)
        | Rise::VecType(_)
        | Rise::PairType(_)
        | Rise::IndexType(_)
        | Rise::I64
        | Rise::F32
        | Rise::NatAdd(_)
        | Rise::NatSub(_)
        | Rise::NatMul(_)
        | Rise::NatDiv(_)
        | Rise::NatPow(_)
        | Rise::NatCst(_) => {
            panic!("Cannot vectorize this in this fn: {:?}", &expr[id])
        }
    }
    //   case Composition(f, g) =>
    //     for { fv <- vectorizeExpr(f, n, eg, vEnv); gv <- vectorizeExpr(g, n, eg, vEnv) }
    //       yield ExprWithHashCons(Composition(fv, gv), eg.add(FunType(
    //         eg(fv.t).asInstanceOf[FunType[TypeId]].inT,
    //         eg(gv.t).asInstanceOf[FunType[TypeId]].outT,
    //       )))
    // }
}

fn vec_ty(expr: &RecExpr<Rise>, n: i64, id: Id) -> Option<RecExpr<Rise>> {
    match expr[id] {
        Rise::F32 | Rise::I64 => {
            let mut vec_ty = RecExpr::default();
            let scalar_ty = vec_ty.add(expr[id].clone());
            let width = vec_ty.add(Rise::NatCst(Nat(n)));
            vec_ty.add(Rise::VecType([width, scalar_ty]));
            Some(vec_ty)
        }

        Rise::VecType(_) | Rise::IndexType(_) | Rise::ArrType(_) => None,
        Rise::PairType([a, b]) => {
            let vec_fst_ty = vec_ty(expr, n, a)?;
            let vec_snd_ty = vec_ty(expr, n, b)?;

            let mut vec_pair_ty = RecExpr::default();
            let vec_fst_ty_id = utils::add_expr(&mut vec_pair_ty, vec_fst_ty);
            let vec_snd_ty_id = utils::add_expr(&mut vec_pair_ty, vec_snd_ty);
            vec_pair_ty.add(Rise::PairType([vec_fst_ty_id, vec_snd_ty_id]));
            Some(vec_pair_ty)
        }
        Rise::FunType([input_ty, output_ty]) => {
            let vec_input_ty = vec_ty(expr, n, input_ty)?;
            let vec_output_ty = vec_ty(expr, n, output_ty)?;

            let mut vec_fun_ty = RecExpr::default();
            let vec_in_ty_id = utils::add_expr(&mut vec_fun_ty, vec_input_ty);
            let vec_out_ty_id = utils::add_expr(&mut vec_fun_ty, vec_output_ty);
            vec_fun_ty.add(Rise::FunType([vec_in_ty_id, vec_out_ty_id]));

            Some(vec_fun_ty)
        }
        _ => panic!("Cannot vectorize {:?}", expr[id]),
    }
}
// private def vecT(t: TypeId, n: NatId, eg: EGraph): Option[TypeId] = {
//     t match {
//       case dt @ DataTypeId(_) => vecDT(dt, n, eg)
//       case ndt @ NotDataTypeId(_) => eg(ndt) match {
//         case FunType(inT, outT) =>
//           for { inTV <- vecT(inT, n, eg); outTV <- vecT(outT, n, eg) }
//             yield eg.add(FunType(inTV, outTV))
//         case NatFunType(t) => ???
//         case DataFunType(t) => ???
//         case AddrFunType(t) => ???
//         case NatToNatFunType(t) => ???
//         case _: DataTypeNode[_, _] => throw new Exception("this should not happen")
//       }
//     }
//   }

//   private def vecDT(t: TypeId, n: NatId, eg: EGraph): Option[TypeId] = {
//     t match {
//       case dt @ DataTypeId(_) => vecDT(dt, n, eg)
//       case NotDataTypeId(_) => None
//     }
//   }

//   private def vecDT(t: DataTypeId, n: NatId, eg: EGraph): Option[DataTypeId] = {
//     eg(t) match {
//       case DataTypeVar(_) => None
//       case ScalarType(_) => Some(eg.add(VectorType(n, t)))
//       case NatType => None
//       case VectorType(_, _) => None
//       case IndexType(_) => None
//       case PairType(a, b) =>
//         for { a2 <- vecDT(a, n, eg); b2 <- vecDT(b, n, eg) }
//           yield eg.add(PairType(a2, b2))
//       case ArrayType(_, _) => None
//     }
//   }
