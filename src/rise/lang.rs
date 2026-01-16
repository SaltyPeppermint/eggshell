use std::fmt;
use std::num::ParseIntError;

use egg::Id;
use ordered_float::NotNan;
use thiserror::Error;

use super::db::Index;
use super::kind::{Kind, Kindable};

// Convention is that last arg is type if applicable
egg::define_language! {
  pub enum Rise {
    Lambda(Lambda,[Id; 2]),
    App(Application,[Id; 3]),
    TypedVar(Index, Id),
    Var(Index),
    "let" = Let(Id),

    // Types, my types
    "fun" = FunType([Id; 2]),
    "natFun" = NatFun(Id),
    "dataFun" = DataFun(Id),
    "addrFun" = AddrFun(Id),
    "natNatFun" = NatNatFun(Id),

    "arrT" = ArrType([Id; 2]),
    "vecT" = VecType([Id; 2]),
    "pairT" = PairType([Id; 2]),
    "idxT" = IndexType(Id),
    // "natT" = NatType,

    "f32" = F32,
    "int" = I64,


    // Natural number operations
    "natAdd" = NatAdd([Id; 2]),
    "natSub" = NatSub([Id; 2]),
    "natMul" = NatMul([Id; 2]),
    "natDiv" = NatDiv([Id; 2]),
    "natPow" = NatPow([Id; 2]),
    NatCst(Nat),

    // Primitives
    Prim(Primitive, Id),

    // to implement explicit substitution:
    // "sig" = Sigma([Id; 3]),
    // "phi" = Phi([Id; 3]),

    IntLit(Int, Id),
    FloatLit(NotNan<f64>, Id),
    // Double(f64),
    // Symbol(Symbol),
  }
}

impl Kindable for Rise {
    fn kind(&self) -> Kind {
        match self {
            Rise::Lambda(lam,_) => lam.kind(),
            Rise::App(app,_) => app.kind(),
            Rise::TypedVar(index,_) | Rise::Var(index) => index.kind(),
            Rise::Let(_)
            | Rise::Prim(_,_ )
            | Rise::FloatLit(_, _)
            | Rise::IntLit(_, _)=> Kind::Expr,
            Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            // | Rise::NatType
            | Rise::F32
            | Rise::I64
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
    pub fn ty_id(&self) -> Option<Id> {
        match self {
            // Rise::NatType => None,
            Rise::Var(_)
            | Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            | Rise::IndexType(_)
            | Rise::F32
            | Rise::I64
            | Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_)
            | Rise::NatCst(_) => None,
            Rise::App(_, [_, _, ty_id])
            | Rise::Lambda(_, [_, ty_id])
            | Rise::Let(ty_id)
            | Rise::TypedVar(_, ty_id)
            | Rise::Prim(_, ty_id)
            | Rise::IntLit(_, ty_id)
            | Rise::FloatLit(_, ty_id) => Some(*ty_id),
        }
    }

    #[must_use]
    pub fn normal_children(&self) -> &[Id] {
        match self {
            Rise::App(_, ids) => &ids[..2],
            Rise::Lambda(_, ids) => &ids[..1],
            Rise::TypedVar(_, _)
            | Rise::Var(_)
            | Rise::Let(_)
            | Rise::F32
            | Rise::I64
            | Rise::NatCst(_)
            | Rise::Prim(_, _) => &[],
            Rise::FunType(ids)
            | Rise::ArrType(ids)
            | Rise::VecType(ids)
            | Rise::PairType(ids)
            | Rise::NatAdd(ids)
            | Rise::NatSub(ids)
            | Rise::NatMul(ids)
            | Rise::NatDiv(ids)
            | Rise::NatPow(ids) => ids,
            Rise::NatFun(id)
            | Rise::DataFun(id)
            | Rise::AddrFun(id)
            | Rise::NatNatFun(id)
            | Rise::IndexType(id)
            | Rise::IntLit(_, id)
            | Rise::FloatLit(_, id) => std::slice::from_ref(id),
            // Rise::NatType => &[],
        }
    }
}

macro_rules! define_kinded_type {
    ($name:ident, $base:literal, $base_cap:literal) => {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
        pub struct $name(Kind);

        impl $name {
            pub fn new(kind: Kind) -> Self {
                Self(kind)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Kind::Expr => write!(f, $base),
                    Kind::Nat => write!(f, concat!("nat", $base_cap)),
                    Kind::Data => write!(f, concat!("data", $base_cap)),
                    Kind::Addr => write!(f, concat!("addr", $base_cap)),
                    Kind::Nat2Nat => write!(f, concat!("natNat", $base_cap)),
                }
            }
        }

        impl std::str::FromStr for $name {
            type Err = RiseParseError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $base => Ok($name(Kind::Expr)),
                    concat!("nat", $base_cap) => Ok($name(Kind::Nat)),
                    concat!("data", $base_cap) => Ok($name(Kind::Data)),
                    concat!("addr", $base_cap) => Ok($name(Kind::Addr)),
                    concat!("natNat", $base_cap) => Ok($name(Kind::Nat2Nat)),
                    _ => Err(RiseParseError::$name(s.to_owned())),
                }
            }
        }

        impl Kindable for $name {
            fn kind(&self) -> Kind {
                self.0
            }
        }
    };
}

define_kinded_type!(Lambda, "lam", "Lam");
define_kinded_type!(Application, "app", "App");

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
pub enum Primitive {
    AsVector,
    AsScalar,
    VectorFromScalar,
    Snd,
    Fst,
    Add,
    Mul,
    ToMem,
    Split,
    Join,
    Generate,
    Transpose,
    Zip,
    Unzip,
    Map,
    MapPar,
    Reduce,
    ReduceSeq,
    ReduceSeqUnroll,
}
impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Primitive::AsVector => write!(f, "asVector"),
            Primitive::AsScalar => write!(f, "asScalar"),
            Primitive::VectorFromScalar => write!(f, "vectorFromScalar"),
            Primitive::Snd => write!(f, "snd"),
            Primitive::Fst => write!(f, "fst"),
            Primitive::Add => write!(f, "add"),
            Primitive::Mul => write!(f, "mul"),
            Primitive::ToMem => write!(f, "toMem"),
            Primitive::Split => write!(f, "split"),
            Primitive::Join => write!(f, "join"),
            Primitive::Generate => write!(f, "generate"),
            Primitive::Transpose => write!(f, "transpose"),
            Primitive::Zip => write!(f, "zip"),
            Primitive::Unzip => write!(f, "unzip"),
            Primitive::Map => write!(f, "map"),
            Primitive::MapPar => write!(f, "mapPar"),
            Primitive::Reduce => write!(f, "reduce"),
            Primitive::ReduceSeq => write!(f, "reduceSeq"),
            Primitive::ReduceSeqUnroll => write!(f, "reduceSeqUnroll"),
        }
    }
}
impl std::str::FromStr for Primitive {
    type Err = RiseParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "asVector" => Ok(Primitive::AsVector),
            "asScalar" => Ok(Primitive::AsScalar),
            "vectorFromScalar" => Ok(Primitive::VectorFromScalar),
            "snd" => Ok(Primitive::Snd),
            "fst" => Ok(Primitive::Fst),
            "add" => Ok(Primitive::Add),
            "mul" => Ok(Primitive::Mul),
            "toMem" => Ok(Primitive::ToMem),
            "split" => Ok(Primitive::Split),
            "join" => Ok(Primitive::Join),
            "generate" => Ok(Primitive::Generate),
            "transpose" => Ok(Primitive::Transpose),
            "zip" => Ok(Primitive::Zip),
            "unzip" => Ok(Primitive::Unzip),
            "map" => Ok(Primitive::Map),
            "mapPar" => Ok(Primitive::MapPar),
            "reduce" => Ok(Primitive::Reduce),
            "reduceSeq" => Ok(Primitive::ReduceSeq),
            "reduceSeqUnroll" => Ok(Primitive::ReduceSeqUnroll),
            _ => Err(RiseParseError::Primitive(s.to_owned())),
        }
    }
}

macro_rules! define_number_type {
    ($name:ident, $suffix:literal, $error_variant:ident) => {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash, Clone, Copy)]
        pub struct $name(pub i64);

        impl std::str::FromStr for $name {
            type Err = RiseParseError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let stripped_s = s
                    .strip_suffix($suffix)
                    .ok_or(RiseParseError::$error_variant(s.to_owned()))?;
                let value = stripped_s.parse()?;
                Ok($name(value))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!("{}", $suffix), self.0)
            }
        }
    };
}

define_number_type!(Int, "i", IntegerPrefix);
define_number_type!(Nat, "n", NatPrefix);

#[derive(Error, Debug)]
pub enum RiseParseError {
    #[error("Wrong Integer Prefix: {0}")]
    IntegerPrefix(String),
    #[error("Wrong Nat Prefix: {0}")]
    NatPrefix(String),
    #[error("Can't parse value: {0}")]
    IntegerValue(#[from] ParseIntError),
    #[error("Unknown primitive: {0}")]
    Primitive(String),
    #[error("Not a lambda: {0}")]
    Lambda(String),
    #[error("Not a app: {0}")]
    Application(String),
}
