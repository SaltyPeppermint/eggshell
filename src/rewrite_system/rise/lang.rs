// use core::panic;
// use std::fmt::Display;
// use std::mem::{Discriminant, discriminant};

// use egg::{
//     Applier, Condition, ConditionalApplier, ENodeOrVar, Id, Language, Pattern, PatternAst, RecExpr,
//     Rewrite, Symbol, Var,
// };
// use hashbrown::HashMap;
use egg::Id;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
// use thiserror::Error;

// use crate::rewrite_system::rise::analysis::RiseAnalysis;

use super::Index;

egg::define_language! {
  #[derive(Serialize, Deserialize)]
  pub enum Rise {
    Var(Index),
    "app" = App([Id; 2]),
    "natApp" = NatApp([Id; 2]),
    "dataApp" = DataApp([Id; 2]),
    "addrApp" = AddrApp([Id; 2]),
    "natNatApp" = NatNatApp([Id; 2]),

    "lam" = Lambda(Id),
    "natLam" = NatLambda(Id),
    "dataLam" = DataLambda(Id),
    "addrLam" = AddrLambda(Id),
    "natNatLam" = NatNatLambda(Id),

    "let" = Let,

    "typeOf" = TypeOf([Id; 2]),

    "fun" = FunType([Id; 2]),
    "natFun" = NatFunType(Id),
    "dataFun" = DataFunType(Id),
    "natNatFun" = NatNatFunType(Id),

    "arrT" = ArrType([Id; 2]),
    "vecT" = VecType([Id; 2]),
    "pairT" = PairType([Id; 2]),
    "idxT" = IndexType(Id),
    "natT" = NatType,

    "f32" = F32,

    // Natural number operations
    "natAdd" = NatAdd([Id; 2]),
    "natSub" = NatSub([Id; 2]),
    "natMul" = NatMul([Id; 2]),
    "natDiv" = NatDiv([Id; 2]),
    "natPow" = NatPow([Id; 2]),

    "asVector" = AsVector,
    "asScalar" = AsScalar,
    "vectorFromScalar" = VectorFromScalar,

    "snd" = Snd,
    "fst" = Fst,
    "add" = Add,
    "mul" = Mul,

    "toMem" = ToMem,
    "split" = Split,
    "join" = Join,

    "generate" = Generate,
    "transpose" = Transpose,

    "zip" = Zip,
    "unzip" = Unzip,

    "map" = Map,
    "mapPar" = MapPar,

    "reduce" = Reduce,
    "reduceSeq" = ReduceSeq,
    "reduceSeqUnroll" = ReduceSeqUnroll,

    // to implement explicit substitution:
    // "sig" = Sigma([Id; 3]),
    // "phi" = Phi([Id; 3]),

    Integer(i32),
    Float(NotNan<f32>),
    // Double(f64),
    // Symbol(Symbol),
  }
}

// #[derive(PartialEq, Eq, PartialOrd, Ord, Serialize, std::hash::Hash, Debug, Clone)]
// pub struct TypedRise {
//     pub expr: RiseOps,
//     pub ty: TypeOrId,
// }

// impl Display for TypedRise {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}: {}", self.expr, self.ty)
//     }
// }

// impl Language for TypedRise {
//     type Discriminant = (Discriminant<RiseOps>, Discriminant<TypeOrId>);

//     fn discriminant(&self) -> Self::Discriminant {
//         (discriminant(&self.expr), discriminant(&self.ty))
//     }

//     fn matches(&self, other: &Self) -> bool {
//         self.discriminant() == other.discriminant()
//     }

//     fn children(&self) -> &[Id] {
//         self.expr.children()
//     }

//     fn children_mut(&mut self) -> &mut [Id] {
//         self.expr.children_mut()
//     }
// }

// #[expect(unused)]
// pub fn convert_expr(untyped: &RecExpr<Rise>) -> Result<RecExpr<TypedRise>, ConvErr> {
//     fn rec(untyped: &RecExpr<Rise>, id: Id) -> Result<TypedRise, ConvErr> {
//         let Rise::TypeOf([expr_id, type_id]) = untyped[id] else {
//             return Err(ConvErr::MissingTypeOf(untyped[id].clone()));
//         };

//         let ty_root_node: RiseType = (&untyped[type_id]).try_into()?;
//         let ty = ty_root_node.try_build_recexpr(|i| (&untyped[i]).try_into())?;

//         Ok(TypedRise {
//             expr: (&untyped[expr_id]).try_into()?,
//             ty: TypeOrId::Inline(ty),
//         })
//     }
//     let new_node = rec(untyped, untyped.root())?;
//     new_node.try_build_recexpr(|i| rec(untyped, i))
// }

// pub fn convert_rewrite<A: Applier<Rise, RiseAnalysis>>(
//     name: &str,
//     lhs_pattern: Pattern<Rise>,
//     rhs: A,
// ) -> Rewrite<TypedRise, RiseAnalysis> {
//     let (searcher, var_typing) = convert_pattern(&lhs_pattern).unwrap();
//     let name = Symbol::new(name);
//     let condition = |
//         egraph: &mut egg::EGraph<TypedRise, RiseAnalysis>,
//         eclass: Id,
//         subst: &egg::Subst| -> bool {
//             for (k, v) in &var_typing {
//                 if let Some(id) = subst.get(*k) {
//                     _egraph[id].data.
//                 }
//             }
//         };
//     let applier = ConditionalApplier {
//         condition,
//         applier: rhs,
//     }
//     Rewrite::new(name, searcher, applier)
// }

// pub fn convert_pattern(
//     untyped: &Pattern<Rise>,
// ) -> Result<(Pattern<TypedRise>, HashMap<Var, RecExpr<RiseType>>), PatConvErr> {
//     fn rec(
//         untyped: &PatternAst<Rise>,
//         id: Id,
//         var_typing: &mut HashMap<Var, RecExpr<RiseType>>,
//     ) -> Result<ENodeOrVar<TypedRise>, PatConvErr> {
//         let (expr_id, type_id) = match peel(untyped, id)? {
//             Rise::TypeOf([expr_id, type_id]) => (expr_id, type_id),
//             x => {
//                 return Err(ConvErr::MissingTypeOf(x.clone()).into());
//             }
//         };
//         let ty_root_node: RiseType = (peel(untyped, *type_id)?).try_into()?;
//         let ty = ty_root_node.try_build_recexpr(|i| match peel(untyped, i) {
//             Ok(n) => n.try_into().map_err(|e: ConvErr| e.into()),
//             Err(e) => Err(e),
//         })?;
//         match &untyped[*expr_id] {
//             ENodeOrVar::ENode(n) => Ok(ENodeOrVar::ENode(TypedRise {
//                 expr: n.try_into()?,
//                 ty: TypeOrId::Inline(ty),
//             })),
//             ENodeOrVar::Var(var) => {
//                 var_typing.insert(*var, ty);
//                 Ok(ENodeOrVar::Var(*var))
//             }
//         }
//     }
//     let mut var_typing = HashMap::new();
//     let new_node = rec(&untyped.ast, untyped.ast.root(), &mut var_typing)?;
//     let typed_pattern = new_node.try_build_recexpr(|i| rec(&untyped.ast, i, &mut var_typing))?;
//     Ok((Pattern::new(typed_pattern), var_typing))
// }

// fn peel<L: Language + Display>(untyped: &RecExpr<ENodeOrVar<L>>, i: Id) -> Result<&L, PatConvErr> {
//     let n = match &untyped[i] {
//         ENodeOrVar::ENode(n) => n,
//         ENodeOrVar::Var(var) => return Err(PatConvErr::UnexpectedVar(*var)),
//     };
//     Ok(n)
// }

// egg::define_language! {
//   #[derive(Serialize)]
//   pub enum RiseOps {
//     Var(Index),
//     "app" = App([Id; 2]),
//     "natApp" = NatApp([Id; 2]),
//     "dataApp" = DataApp([Id; 2]),
//     "addrApp" = AddrApp([Id; 2]),
//     "natNatApp" = NatNatApp([Id; 2]),

//     "lam" = Lambda(Id),
//     "natLam" = NatLambda(Id),
//     "dataLam" = DataLambda(Id),
//     "addrLam" = AddrLambda(Id),
//     "natNatLam" = NatNatLambda(Id),

//     "let" = Let,

//     Nat(NatOrId),

//     "asVector" = AsVector,
//     "asScalar" = AsScalar,
//     "vectorFromScalar" = VectorFromScalar,

//     "snd" = Snd,
//     "fst" = Fst,
//     "add" = Add,
//     "mul" = Mul,

//     "toMem" = ToMem,
//     "split" = Split,
//     "join" = Join,

//     "generate" = Generate,
//     "transpose" = Transpose,

//     "zip" = Zip,
//     "unzip" = Unzip,

//     "map" = Map,
//     "mapPar" = MapPar,

//     "reduce" = Reduce,
//     "reduceSeq" = ReduceSeq,
//     "reduceSeqUnroll" = ReduceSeqUnroll,

//     // Integer(i32),
//     Float(NotNan<f32>),
//   }
// }

// impl TryFrom<&Rise> for RiseOps {
//     type Error = ConvErr;

//     fn try_from(value: &Rise) -> Result<Self, Self::Error> {
//         Ok(match value {
//             Rise::Var(index) => RiseOps::Var(*index),
//             Rise::App(ids) => RiseOps::App(*ids),
//             Rise::NatApp(ids) => RiseOps::NatApp(*ids),
//             Rise::DataApp(ids) => RiseOps::DataApp(*ids),
//             Rise::AddrApp(ids) => RiseOps::AddrApp(*ids),
//             Rise::NatNatApp(ids) => RiseOps::NatNatApp(*ids),
//             Rise::Lambda(id) => RiseOps::Lambda(*id),
//             Rise::NatLambda(id) => RiseOps::NatLambda(*id),
//             Rise::DataLambda(id) => RiseOps::DataLambda(*id),
//             Rise::AddrLambda(id) => RiseOps::AddrLambda(*id),
//             Rise::NatNatLambda(id) => RiseOps::NatNatLambda(*id),
//             Rise::Let => RiseOps::Let,

//             Rise::AsVector => RiseOps::AsVector,
//             Rise::AsScalar => RiseOps::AsScalar,
//             Rise::VectorFromScalar => RiseOps::VectorFromScalar,
//             Rise::Snd => RiseOps::Snd,
//             Rise::Fst => RiseOps::Fst,
//             Rise::Add => RiseOps::Add,
//             Rise::Mul => RiseOps::Mul,
//             Rise::ToMem => RiseOps::ToMem,
//             Rise::Split => RiseOps::Split,
//             Rise::Join => RiseOps::Join,
//             Rise::Generate => RiseOps::Generate,
//             Rise::Transpose => RiseOps::Transpose,
//             Rise::Zip => RiseOps::Zip,
//             Rise::Unzip => RiseOps::Unzip,
//             Rise::Map => RiseOps::Map,
//             Rise::MapPar => RiseOps::MapPar,
//             Rise::Reduce => RiseOps::Reduce,
//             Rise::ReduceSeq => RiseOps::ReduceSeq,
//             Rise::ReduceSeqUnroll => RiseOps::ReduceSeqUnroll,
//             Rise::Float(not_nan) => RiseOps::Float(*not_nan),
//             _ => return Err(ConvErr::NoMatchingOps(value.to_owned())),
//         })
//     }
// }

// #[derive(Serialize, Clone, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
// pub enum TypeOrId {
//     Inline(RecExpr<RiseType>),
//     Id(usize),
// }

// impl Display for TypeOrId {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             TypeOrId::Inline(e) => write!(f, "{e}"),
//             TypeOrId::Id(id) => write!(f, "TYPE_ID_{id}"),
//         }
//     }
// }

// impl std::str::FromStr for TypeOrId {
//     type Err = egg::RecExprParseError<egg::FromOpError>;

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         let Some(no_prefix) = s.strip_prefix("type_") else {
//             return Err(egg::RecExprParseError::BadOp(egg::FromOpError::new(
//                 s,
//                 vec![],
//             )));
//         };
//         let expr = no_prefix.parse()?;
//         Ok(TypeOrId::Inline(expr))
//     }
// }

// egg::define_language! {
//   #[derive(Serialize)]
//   pub enum RiseType {
//     Var(Index),

//     "fun" = FunType([Id; 2]),
//     "natFun" = NatFunType(Id),
//     "dataFun" = DataFunType(Id),
//     "natNatFun" = NatNatFunType(Id),

//     "arrT" = ArrType([Id; 2]),
//     "vecT" = VecType([Id; 2]),
//     "pairT" = PairType([Id; 2]),
//     "idxT" = IndexType(Id),
//     "natT" = NatType,

//     "f32" = F32,

//     NatExpr(NatOrId),
//   }
// }

// impl TryFrom<&Rise> for RiseType {
//     type Error = ConvErr;

//     fn try_from(value: &Rise) -> Result<Self, Self::Error> {
//         Ok(match value {
//             Rise::Var(index) => RiseType::Var(*index),

//             Rise::FunType(ids) => RiseType::FunType(*ids),
//             Rise::NatFunType(id) => RiseType::NatFunType(*id),
//             Rise::DataFunType(id) => RiseType::DataFunType(*id),
//             Rise::NatNatFunType(id) => RiseType::NatNatFunType(*id),
//             Rise::ArrType(ids) => RiseType::ArrType(*ids),
//             Rise::VecType(ids) => RiseType::VecType(*ids),
//             Rise::PairType(ids) => RiseType::PairType(*ids),
//             Rise::IndexType(id) => RiseType::IndexType(*id),
//             Rise::NatType => RiseType::NatType,
//             Rise::F32 => RiseType::F32,
//             _ => return Err(ConvErr::NoMatchingType(value.to_owned())),
//         })
//     }
// }

// #[derive(Serialize, Clone, std::hash::Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
// pub enum NatOrId {
//     Inline(RecExpr<RiseNat>),
//     Id(usize),
// }

// impl Display for NatOrId {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             NatOrId::Inline(e) => write!(f, "{e}"),
//             NatOrId::Id(id) => write!(f, "TYPE_ID_{id}"),
//         }
//     }
// }

// impl std::str::FromStr for NatOrId {
//     type Err = egg::RecExprParseError<egg::FromOpError>;

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         let Some(no_prefix) = s.strip_prefix("nat_") else {
//             return Err(egg::RecExprParseError::BadOp(egg::FromOpError::new(
//                 s,
//                 vec![],
//             )));
//         };
//         let expr = no_prefix.parse()?;
//         Ok(NatOrId::Inline(expr))
//     }
// }

// egg::define_language! {
//   #[derive(Serialize, Deserialize)]
//   pub enum RiseNat {
//     Var(Index),

//     // "idxL" = IndexLiteral([Id; 2]),
//     "natL" = NatLiteral(Id),
//     // Natural number operations
//     "natAdd" = NatAdd([Id; 2]),
//     "natSub" = NatSub([Id; 2]),
//     "natMul" = NatMul([Id; 2]),
//     "natDiv" = NatDiv([Id; 2]),
//     "natPow" = NatPow([Id; 2]),
//     Integer(i32),
//   }
// }

// impl TryFrom<&Rise> for RiseNat {
//     type Error = ConvErr;

//     fn try_from(value: &Rise) -> Result<Self, Self::Error> {
//         Ok(match value {
//             Rise::Var(index) => RiseNat::Var(*index),

//             Rise::NatAdd(ids) => RiseNat::NatAdd(*ids),
//             Rise::NatSub(ids) => RiseNat::NatSub(*ids),
//             Rise::NatMul(ids) => RiseNat::NatMul(*ids),
//             Rise::NatDiv(ids) => RiseNat::NatDiv(*ids),
//             Rise::NatPow(ids) => RiseNat::NatPow(*ids),

//             Rise::Integer(i) => RiseNat::Integer(*i),
//             _ => return Err(ConvErr::NoMatchingNat(value.to_owned())),
//         })
//     }
// }

// #[expect(clippy::enum_variant_names)]
// #[derive(Debug, Error)]
// pub enum ConvErr {
//     #[error("No matching ops {0}")]
//     NoMatchingOps(Rise),
//     #[error("No matching ops {0}")]
//     NoMatchingType(Rise),
//     #[error("No matching ops {0}")]
//     NoMatchingNat(Rise),
//     #[error("TypeOf wrapper missing, found instead {0}")]
//     MissingTypeOf(Rise),
// }

// #[expect(clippy::enum_variant_names)]
// #[derive(Debug, Error)]
// pub enum PatConvErr {
//     #[error("Underlying Language Error")]
//     Inner(#[from] ConvErr),
//     #[error("Unexpected Var: {0}")]
//     UnexpectedVar(Var),
// }
