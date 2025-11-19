use egg::{Id, Symbol};
use serde::{Deserialize, Serialize};

use super::indices::TypedIndex;

egg::define_language! {
  #[derive(Serialize, Deserialize)]
  pub enum Rise {
    TypedVar(TypedIndex, Id),
    Var(TypedIndex),
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
    // Float(f32),
    // Double(f64),
    Symbol(Symbol),
  }
}
