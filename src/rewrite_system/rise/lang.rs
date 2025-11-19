use core::{fmt, slice};
use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOpError, Id, Language};
use serde::{Deserialize, Serialize};

use super::indices::TypedIndex;

#[derive(PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize, Clone, Hash, Debug)]
pub enum Rise {
    Var(TypedIndex),
    TypeOf([Id; 2]),
    App(AppType, [Id; 2]),
    Lambda(LamType, Id),
    Nat(RiseNat),
    Primitive(RisePrimitives),
    Type(RiseTypes),
    Integer(i32),
    // Float(f32),
    // Double(f64),
    // Symbol(Symbol),
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize, Clone, Copy, Hash, Debug)]
pub enum AppType {
    App,
    NatApp,
    DataApp,
    NatNatApp,
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize, Clone, Copy, Hash, Debug)]
pub enum LamType {
    Lam,
    NatLam,
    DataLam,
    NatNatLam,
    AddrLam,
}

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum RiseDiscriminant {
    Rise(Discriminant<Rise>),
    Type(Discriminant<RiseTypes>),
    Primitive(Discriminant<RisePrimitives>),
    Nat(Discriminant<RiseNat>),
}

impl Language for Rise {
    type Discriminant = RiseDiscriminant;

    // Clippy says [inline(always)] on a discriminant is a bad idea
    fn discriminant(&self) -> Self::Discriminant {
        match self {
            Rise::Nat(nat) => RiseDiscriminant::Nat(discriminant(nat)),
            Rise::Primitive(p) => RiseDiscriminant::Primitive(discriminant(p)),
            Rise::Type(ty) => RiseDiscriminant::Type(discriminant(ty)),
            _ => RiseDiscriminant::Rise(discriminant(self)),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }

    fn children(&self) -> &[Id] {
        match self {
            Rise::Var(_) | Rise::Integer(_) => &[], // | Rise::Symbol(_)
            Rise::TypeOf(c) | Rise::App(_, c) => c,
            Rise::Lambda(_, id) => slice::from_ref(id),
            Rise::Nat(rise_nat) => rise_nat.children(),
            Rise::Primitive(rise_primitives) => rise_primitives.children(),
            Rise::Type(rise_types) => rise_types.children(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Rise::Var(_) | Rise::Integer(_) => &mut [], // | Rise::Symbol(_)
            Rise::TypeOf(c) | Rise::App(_, c) => c,
            Rise::Lambda(_, id) => slice::from_mut(id),
            Rise::Nat(rise_nat) => rise_nat.children_mut(),
            Rise::Primitive(rise_primitives) => rise_primitives.children_mut(),
            Rise::Type(rise_types) => rise_types.children_mut(),
        }
    }
}

impl Display for Rise {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Rise::Var(typed_index) => write!(f, "{typed_index}"),
            Rise::TypeOf(_) => write!(f, "typeOf"),
            Rise::App(at, _) => match at {
                AppType::App => write!(f, "app"),
                AppType::NatApp => write!(f, "natApp"),
                AppType::DataApp => write!(f, "dataApp"),
                AppType::NatNatApp => write!(f, "natNatApp"),
            },
            Rise::Lambda(lt, _) => match lt {
                LamType::Lam => write!(f, "lam"),
                LamType::NatLam => write!(f, "natLam"),
                LamType::DataLam => write!(f, "dataLam"),
                LamType::NatNatLam => write!(f, "natNatLam"),
                LamType::AddrLam => write!(f, "addrLam"),
            },
            Rise::Nat(rise_nat) => write!(f, "{rise_nat}"),
            Rise::Primitive(rise_primitives) => write!(f, "{rise_primitives}"),
            Rise::Type(rise_types) => write!(f, "{rise_types}"),
            Rise::Integer(i) => write!(f, "{i}"),
            // Rise::Symbol(global_symbol) => write!(f, "{global_symbol}"),
        }
    }
}

impl egg::FromOp for Rise {
    type Error = FromOpError;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        if let Ok(i) = op.parse::<i32>()
            && children.is_empty()
        {
            return Ok(Rise::Integer(i));
        }
        if let Ok(typed_index) = op.parse::<TypedIndex>()
            && children.is_empty()
        {
            return Ok(Rise::Var(typed_index));
        }

        match (op, children.len()) {
            ("typeOf", 2) => Ok(Self::TypeOf([children[0], children[1]])),

            ("app", 2) => Ok(Self::App(AppType::App, [children[0], children[1]])),
            ("appNat", 2) => Ok(Self::App(AppType::NatApp, [children[0], children[1]])),
            ("appData", 2) => Ok(Self::App(AppType::DataApp, [children[0], children[1]])),
            ("appNatNat", 2) => Ok(Self::App(AppType::NatNatApp, [children[0], children[1]])),

            ("lam", 1) => Ok(Self::Lambda(LamType::Lam, children[0])),
            ("natLam", 1) => Ok(Self::Lambda(LamType::NatLam, children[0])),
            ("dataLam", 1) => Ok(Self::Lambda(LamType::DataLam, children[0])),
            ("natNatLam", 1) => Ok(Self::Lambda(LamType::NatNatLam, children[0])),
            ("addrLam", 1) => Ok(Self::Lambda(LamType::AddrLam, children[0])),

            _ => RiseNat::from_op(op, children.clone())
                .map(Self::Nat)
                .or_else(|_| RisePrimitives::from_op(op, children.clone()).map(Self::Primitive))
                .or_else(|_| RiseTypes::from_op(op, children).map(Self::Type)),
        }
    }
}

egg::define_language! {
    #[derive(Serialize, Deserialize, Copy)]
    pub enum RisePrimitives {
        "asVector" = AsVector,
        "asScalar" = AsScalar,

        // Not sure about this one
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
    }
}

egg::define_language! {
  #[derive(Serialize, Deserialize, Copy)]
  pub enum RiseTypes {
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
  }
}

egg::define_language! {
  #[derive(Serialize, Deserialize, Copy)]
  pub enum RiseNat {
    // Natural number operations
    "natAdd" = NatAdd([Id; 2]),
    "natSub" = NatSub([Id; 2]),
    "natMul" = NatMul([Id; 2]),
    "natDiv" = NatDiv([Id; 2]),
    "natPow" = NatPow([Id; 2]),

  }
}

// egg::define_language! {
//   #[derive(Serialize, Deserialize)]
//   pub enum Rise {
//     TypedVar(TypedIndex, Id),
//     Var(TypedIndex),
//     "app" = App([Id; 2]),
//     "lam" = Lambda(Id),
//     "typeOf" = TypeOf([Id; 2]),

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

//     // Natural number operations
//     "natAdd" = NatAdd([Id; 2]),
//     "natSub" = NatSub([Id; 2]),
//     "natMul" = NatMul([Id; 2]),
//     "natDiv" = NatDiv([Id; 2]),
//     "natPow" = NatPow([Id; 2]),

//     "asVector" = AsVector,
//     "asScalar" = AsScalar,

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

//     // to implement explicit substitution:
//     // "sig" = Sigma([Id; 3]),
//     // "phi" = Phi([Id; 3]),

//     Integer(i32),
//     // Float(f32),
//     // Double(f64),
//     Symbol(Symbol),
//   }
// }
