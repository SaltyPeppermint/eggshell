use std::fmt;
use std::num::ParseIntError;

use egg::Id;
use ordered_float::NotNan;
use thiserror::Error;

use super::db::Index;
use super::kind::{Kind, Kindable};

egg::define_language! {
  pub enum Rise {
    Var(Index),
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

    NatCst(Nat),

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

    IntLit(Int),
    FloatLit(NotNan<f64>),
    // Double(f64),
    // Symbol(Symbol),
  }
}

impl Kindable for Rise {
    fn kind(&self) -> Kind {
        match self {
            Rise::Var(index) => index.kind(),
            Rise::Lambda(lambda, _) => lambda.kind(),
            Rise::App(application, _) => application.kind(),
            Rise::Let
            | Rise::TypeOf(_)
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
            | Rise::FloatLit(_)
            | Rise::IntLit(_) => Kind::Expr,
            Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            | Rise::NatType
            | Rise::F32
            | Rise::IndexType(_) => Kind::Data,
            Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_)
            | Rise::NatCst(_) => Kind::Nat,
        }
    }
}

impl Rise {
    #[must_use]
    pub fn is_nat(&self) -> bool {
        if let Rise::Var(index) = self
            && index.kind() == Kind::Nat
        {
            return true;
        }
        matches!(
            self,
            Rise::NatAdd(_)
                | Rise::NatSub(_)
                | Rise::NatMul(_)
                | Rise::NatDiv(_)
                | Rise::NatPow(_)
                | Rise::IntLit(_)
        )
    }
}

#[derive(Error, Debug)]
pub enum KindParseError {
    #[error("Not a lambda: {0}")]
    Lambda(String),
    #[error("Not a app: {0}")]
    App(String),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub struct Lambda(Kind);

impl Lambda {
    pub fn new(kind: Kind) -> Self {
        Self(kind)
    }
}

impl fmt::Display for Lambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    type Err = KindParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lam" => Ok(Lambda(Kind::Expr)),
            "natLam" => Ok(Lambda(Kind::Nat)),
            "dataLam" => Ok(Lambda(Kind::Data)),
            "addrLam" => Ok(Lambda(Kind::Addr)),
            "natNatLam" => Ok(Lambda(Kind::Nat2Nat)),
            _ => Err(KindParseError::Lambda(s.to_owned())),
        }
    }
}

impl Kindable for Lambda {
    fn kind(&self) -> Kind {
        self.0
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub struct Application(Kind);

impl Application {
    pub fn new(kind: Kind) -> Self {
        Self(kind)
    }
}

impl fmt::Display for Application {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    type Err = KindParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "app" => Ok(Application(Kind::Expr)),
            "natApp" => Ok(Application(Kind::Nat)),
            "dataApp" => Ok(Application(Kind::Data)),
            "addrApp" => Ok(Application(Kind::Addr)),
            "natNatApp" => Ok(Application(Kind::Nat2Nat)),
            _ => Err(KindParseError::App(s.to_owned())),
        }
    }
}

impl Kindable for Application {
    fn kind(&self) -> Kind {
        self.0
    }
}

#[derive(Error, Debug)]
pub enum NumberParseError {
    #[error("Wrong Integer Prefix: {0}")]
    IntegerPrefix(String),
    #[error("Wrong Nat Prefix: {0}")]
    NatPrefix(String),
    #[error("Can't parse value: {0}")]
    IntegerValue(#[from] ParseIntError),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub struct Int(pub i64);

impl std::str::FromStr for Int {
    type Err = NumberParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let stripped_s = s
            .strip_suffix("i")
            .ok_or(NumberParseError::IntegerPrefix(s.to_owned()))?;

        let value = stripped_s.parse()?;
        Ok(Int(value))
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}i", self.0)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub struct Nat(pub i64);

impl std::str::FromStr for Nat {
    type Err = NumberParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let stripped_s = s
            .strip_suffix("n")
            .ok_or(NumberParseError::IntegerPrefix(s.to_owned()))?;

        let value = stripped_s.parse()?;
        Ok(Nat(value))
    }
}

impl fmt::Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}n", self.0)
    }
}
