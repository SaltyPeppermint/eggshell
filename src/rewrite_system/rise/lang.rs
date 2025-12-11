use egg::Id;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{DBIndex, Kind, Kindable};

egg::define_language! {
  #[derive(Serialize, Deserialize)]
  pub enum Rise {
    Var(DBIndex),
    Lambda(Lambda, Id),
    App(Application, [Id; 2]),
    // "app" = App([Id; 2]),
    // "natApp" = NatApp([Id; 2]),
    // "dataApp" = DataApp([Id; 2]),
    // "addrApp" = AddrApp([Id; 2]),
    // "natNatApp" = NatNatApp([Id; 2]),


    // "lam" = Lambda(Id),
    // "natLam" = NatLambda(Id),
    // "dataLam" = DataLambda(Id),
    // "addrLam" = AddrLambda(Id),
    // "natNatLam" = NatNatLambda(Id),

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

#[derive(Error, Debug)]
pub enum ParseKindError {
    #[error("Not a lambda: {0}")]
    Lambda(String),
    #[error("Not a app: {0}")]
    App(String),
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy, Serialize, Deserialize,
)]
pub struct Lambda(Kind);

impl Lambda {
    pub fn new(kind: Kind) -> Self {
        Self(kind)
    }
}

impl std::fmt::Display for Lambda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Kind::Expr => write!(f, "lam"),
            Kind::Nat => write!(f, "natLam"),
            Kind::Data => write!(f, "dataLam"),
            Kind::Addr => write!(f, "addrLam"),
            Kind::Nat2Nat => write!(f, "natNatLam"),
        }
    }
}

impl std::str::FromStr for Lambda {
    type Err = ParseKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lam" => Ok(Lambda(Kind::Expr)),
            "natLam" => Ok(Lambda(Kind::Nat)),
            "dataLam" => Ok(Lambda(Kind::Data)),
            "addrLam" => Ok(Lambda(Kind::Addr)),
            "natNatLam" => Ok(Lambda(Kind::Nat2Nat)),
            _ => Err(ParseKindError::Lambda(s.to_owned())),
        }
    }
}

impl Kindable for Lambda {
    fn kind(&self) -> Kind {
        self.0
    }
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy, Serialize, Deserialize,
)]
pub struct Application(Kind);

impl Application {
    pub fn new(kind: Kind) -> Self {
        Self(kind)
    }
}

impl std::fmt::Display for Application {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Kind::Expr => write!(f, "app"),
            Kind::Nat => write!(f, "natApp"),
            Kind::Data => write!(f, "dataApp"),
            Kind::Addr => write!(f, "addrApp"),
            Kind::Nat2Nat => write!(f, "natNatApp"),
        }
    }
}

impl std::str::FromStr for Application {
    type Err = ParseKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "app" => Ok(Application(Kind::Expr)),
            "natApp" => Ok(Application(Kind::Nat)),
            "dataApp" => Ok(Application(Kind::Data)),
            "addrApp" => Ok(Application(Kind::Addr)),
            "natNatApp" => Ok(Application(Kind::Nat2Nat)),
            _ => Err(ParseKindError::App(s.to_owned())),
        }
    }
}

impl Kindable for Application {
    fn kind(&self) -> Kind {
        self.0
    }
}
