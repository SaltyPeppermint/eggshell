//! Label type for Rise that implements the Label trait.

use std::fmt::{self, Display};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::primitive::Primitive;
use super::types::ScalarType;
use crate::distance::nodes::Label;

/// A compact label type for Rise that implements the Label trait.
///
/// This allows using Rise expressions directly with the e-graph infrastructure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiseLabel {
    // Expression labels
    Var(usize),
    App,
    Lambda,
    NatApp,
    NatLambda,
    DataApp,
    DataLambda,
    AddrApp,
    AddrLambda,
    NatNatLambda,
    IndexLiteral,
    TypeOf,
    // Literals
    BoolLit(bool),
    IntLit(i32),
    FloatLit(OrderedFloat<f32>),
    DoubleLit(OrderedFloat<f64>),
    NatLit(i64),
    // Primitives
    Primitive(Primitive),
    // Nat labels
    NatVar(usize),
    NatCst(i64),
    NatAdd,
    NatMul,
    NatPow,
    NatMod,
    NatFloorDiv,
    // Address labels
    AddrVar(usize),
    Global,
    Local,
    Private,
    Constant,
    // Type labels
    Fun,
    NatFun,
    DataFun,
    AddrFun,
    NatNatFun,
    // Data type labels
    DataVar(usize),
    Scalar(ScalarType),
    NatT,
    IdxT,
    PairT,
    ArrT,
    VecT,
}

impl Label for RiseLabel {
    fn type_of() -> Self {
        RiseLabel::TypeOf
    }
}

#[allow(clippy::match_same_arms)]
impl Display for RiseLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiseLabel::Var(i) => write!(f, "$e{i}"),
            RiseLabel::App => write!(f, "app"),
            RiseLabel::Lambda => write!(f, "lam"),
            RiseLabel::NatApp => write!(f, "natApp"),
            RiseLabel::NatLambda => write!(f, "natLam"),
            RiseLabel::DataApp => write!(f, "dataApp"),
            RiseLabel::DataLambda => write!(f, "dataLam"),
            RiseLabel::AddrApp => write!(f, "addrApp"),
            RiseLabel::AddrLambda => write!(f, "addrLam"),
            RiseLabel::NatNatLambda => write!(f, "natNatLam"),
            RiseLabel::IndexLiteral => write!(f, "idxL"),
            RiseLabel::TypeOf => write!(f, "typeOf"),
            RiseLabel::BoolLit(b) => write!(f, "{b}"),
            RiseLabel::IntLit(i) => write!(f, "{i}i"),
            RiseLabel::FloatLit(n) => write!(f, "{}f", n.0),
            RiseLabel::DoubleLit(n) => write!(f, "{}d", n.0),
            RiseLabel::NatLit(n) => write!(f, "{n}n"),
            RiseLabel::Primitive(p) => write!(f, "{p}"),
            RiseLabel::NatVar(i) => write!(f, "$n{i}"),
            RiseLabel::NatCst(n) => write!(f, "{n}n"),
            RiseLabel::NatAdd => write!(f, "natAdd"),
            RiseLabel::NatMul => write!(f, "natMul"),
            RiseLabel::NatPow => write!(f, "natPow"),
            RiseLabel::NatMod => write!(f, "natMod"),
            RiseLabel::NatFloorDiv => write!(f, "natFloorDiv"),
            RiseLabel::AddrVar(i) => write!(f, "$a{i}"),
            RiseLabel::Global => write!(f, "global"),
            RiseLabel::Local => write!(f, "local"),
            RiseLabel::Private => write!(f, "private"),
            RiseLabel::Constant => write!(f, "constant"),
            RiseLabel::Fun => write!(f, "fun"),
            RiseLabel::NatFun => write!(f, "natFun"),
            RiseLabel::DataFun => write!(f, "dataFun"),
            RiseLabel::AddrFun => write!(f, "addrFun"),
            RiseLabel::NatNatFun => write!(f, "natNatFun"),
            RiseLabel::DataVar(i) => write!(f, "$d{i}"),
            RiseLabel::Scalar(s) => write!(f, "{s}"),
            RiseLabel::NatT => write!(f, "natT"),
            RiseLabel::IdxT => write!(f, "idxT"),
            RiseLabel::PairT => write!(f, "pairT"),
            RiseLabel::ArrT => write!(f, "arrT"),
            RiseLabel::VecT => write!(f, "vecT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rise_label_implements_label() {
        assert_eq!(RiseLabel::type_of(), RiseLabel::TypeOf);
        assert!(RiseLabel::TypeOf.is_type_of());
        assert!(!RiseLabel::App.is_type_of());
    }
}
