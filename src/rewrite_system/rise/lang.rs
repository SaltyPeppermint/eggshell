use egg::Id;
use ordered_float::NotNan;
use thiserror::Error;

use super::{DBIndex, Kind, Kindable};

egg::define_language! {
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
                | Rise::Integer(_)
        )
    }
}

#[derive(Error, Debug)]
pub enum ParseKindError {
    #[error("Not a lambda: {0}")]
    Lambda(String),
    #[error("Not a app: {0}")]
    App(String),
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub enum Lambda {
    Expr,
    Nat,
    Data,
    Addr,
    Nat2Nat,
}

impl std::fmt::Display for Lambda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lambda::Expr => write!(f, "lam"),
            Lambda::Nat => write!(f, "natLam"),
            Lambda::Data => write!(f, "dataLam"),
            Lambda::Addr => write!(f, "addrLam"),
            Lambda::Nat2Nat => write!(f, "natNatLam"),
        }
    }
}

impl std::str::FromStr for Lambda {
    type Err = ParseKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lam" => Ok(Lambda::Expr),
            "natLam" => Ok(Lambda::Nat),
            "dataLam" => Ok(Lambda::Data),
            "addrLam" => Ok(Lambda::Addr),
            "natNatLam" => Ok(Lambda::Nat2Nat),
            _ => Err(ParseKindError::Lambda(s.to_owned())),
        }
    }
}

impl Kindable for Lambda {
    fn kind(&self) -> Kind {
        match self {
            Lambda::Expr => Kind::Expr,
            Lambda::Nat => Kind::Nat,
            Lambda::Data => Kind::Data,
            Lambda::Addr => Kind::Addr,
            Lambda::Nat2Nat => Kind::Nat2Nat,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub enum Application {
    Expr,
    Nat,
    Data,
    Addr,
    Nat2Nat,
}

impl std::fmt::Display for Application {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Application::Expr => write!(f, "app"),
            Application::Nat => write!(f, "natApp"),
            Application::Data => write!(f, "dataApp"),
            Application::Addr => write!(f, "addrApp"),
            Application::Nat2Nat => write!(f, "natNatApp"),
        }
    }
}

impl std::str::FromStr for Application {
    type Err = ParseKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "app" => Ok(Application::Expr),
            "natApp" => Ok(Application::Nat),
            "dataApp" => Ok(Application::Data),
            "addrApp" => Ok(Application::Addr),
            "natNatApp" => Ok(Application::Nat2Nat),
            _ => Err(ParseKindError::App(s.to_owned())),
        }
    }
}

impl Kindable for Application {
    fn kind(&self) -> Kind {
        match self {
            Application::Expr => Kind::Expr,
            Application::Nat => Kind::Nat,
            Application::Data => Kind::Data,
            Application::Addr => Kind::Addr,
            Application::Nat2Nat => Kind::Nat2Nat,
        }
    }
}
