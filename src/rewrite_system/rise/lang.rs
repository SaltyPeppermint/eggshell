use colored::{ColoredString, Colorize};
use egg::{Id, Language, RecExpr};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use super::{Index, Kind, Kindable};

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

    "natFun" = NatFun(Id),
    "dataFun" = DataFun(Id),
    "addrFun" = AddrFun(Id),
    "natNatFun" = NatNatFun(Id),

    "let" = Let,

    "typeOf" = TypeOf([Id; 2]),
    "fun" = FunType([Id; 2]),

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

impl Rise {
    pub fn kind(&self) -> Option<Kind> {
        Some(match self {
            Rise::Var(index) => index.kind()?,
            Rise::App(_)
            | Rise::Lambda(_)
            | Rise::Let
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
            | Rise::Float(_) => Kind::Expr,
            Rise::NatApp(_)
            | Rise::NatLambda(_)
            | Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_) => Kind::Nat,
            Rise::DataApp(_)
            | Rise::DataLambda(_)
            | Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            | Rise::IndexType(_)
            | Rise::NatType
            | Rise::F32 => Kind::Data,
            Rise::AddrApp(_) | Rise::AddrLambda(_) => Kind::Addr,
            Rise::TypeOf(_) | Rise::NatNatApp(_) | Rise::NatNatLambda(_) | Rise::Integer(_) => {
                return None;
            }
        })
    }
}
