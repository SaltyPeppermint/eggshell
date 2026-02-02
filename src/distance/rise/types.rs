//! Type expressions in Rise.

use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp};

use super::ParseError;
use super::label::RiseLabel;
use super::nat::{Nat, parse_nat};
use crate::distance::tree::TreeNode;

// ============================================================================
// Scalar types
// ============================================================================

/// Scalar types in Rise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Int,
    Bool,
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::F16 => write!(f, "f16"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::Int => write!(f, "int"),
            ScalarType::Bool => write!(f, "bool"),
        }
    }
}

impl IntoSexp for ScalarType {
    fn into_sexp(&self) -> Sexp {
        Sexp::String(self.to_string())
    }
}

/// Parse a scalar type from a string.
pub fn parse_scalar_type(s: &str) -> Option<ScalarType> {
    match s {
        "f16" => Some(ScalarType::F16),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "u8" => Some(ScalarType::U8),
        "u16" => Some(ScalarType::U16),
        "u32" => Some(ScalarType::U32),
        "u64" => Some(ScalarType::U64),
        "int" => Some(ScalarType::Int),
        "bool" => Some(ScalarType::Bool),
        _ => None,
    }
}

// ============================================================================
// Data types
// ============================================================================

/// Data types in Rise (types that represent actual data).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// Variable: $d<index>
    Var(usize),
    /// Scalar type
    Scalar(ScalarType),
    /// Natural number type
    NatT,
    /// Index type: (idxT n)
    Index(Box<Nat>),
    /// Pair type: (pairT dt1 dt2)
    Pair(Box<DataType>, Box<DataType>),
    /// Array type: (arrT n dt)
    Array(Box<Nat>, Box<DataType>),
    /// Vector type: (vecT n dt)
    Vector(Box<Nat>, Box<DataType>),
}

impl Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Var(i) => write!(f, "$d{i}"),
            DataType::Scalar(s) => write!(f, "{s}"),
            DataType::NatT => write!(f, "natT"),
            DataType::Index(n) => write!(f, "(idxT {n})"),
            DataType::Pair(a, b) => write!(f, "(pairT {a} {b})"),
            DataType::Array(n, dt) => write!(f, "(arrT {n} {dt})"),
            DataType::Vector(n, dt) => write!(f, "(vecT {n} {dt})"),
        }
    }
}

impl IntoSexp for DataType {
    fn into_sexp(&self) -> Sexp {
        match self {
            DataType::Var(i) => Sexp::String(format!("$d{i}")),
            DataType::Scalar(s) => s.into_sexp(),
            DataType::NatT => Sexp::String("natT".to_owned()),
            DataType::Index(n) => Sexp::List(vec![Sexp::String("idxT".to_owned()), n.into_sexp()]),
            DataType::Pair(a, b) => Sexp::List(vec![
                Sexp::String("pairT".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            DataType::Array(n, dt) => Sexp::List(vec![
                Sexp::String("arrT".to_owned()),
                n.into_sexp(),
                dt.into_sexp(),
            ]),
            DataType::Vector(n, dt) => Sexp::List(vec![
                Sexp::String("vecT".to_owned()),
                n.into_sexp(),
                dt.into_sexp(),
            ]),
        }
    }
}

impl DataType {
    /// Convert this data type to a `RiseLabel`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            DataType::Var(i) => RiseLabel::DataVar(*i),
            DataType::Scalar(s) => RiseLabel::Scalar(s.clone()),
            DataType::NatT => RiseLabel::NatT,
            DataType::Index(..) => RiseLabel::IdxT,
            DataType::Pair(..) => RiseLabel::PairT,
            DataType::Array(..) => RiseLabel::ArrT,
            DataType::Vector(..) => RiseLabel::VecT,
        }
    }

    /// Convert this data type to a `TreeNode<RiseLabel>`.
    #[must_use]
    pub fn to_tree(&self) -> TreeNode<RiseLabel> {
        match self {
            DataType::Var(_) | DataType::Scalar(_) | DataType::NatT => {
                TreeNode::leaf(self.to_label())
            }
            DataType::Index(n) => TreeNode::new(self.to_label(), vec![n.to_tree()]),
            DataType::Pair(a, b) => TreeNode::new(self.to_label(), vec![a.to_tree(), b.to_tree()]),
            DataType::Array(n, dt) | DataType::Vector(n, dt) => {
                TreeNode::new(self.to_label(), vec![n.to_tree(), dt.to_tree()])
            }
        }
    }
}

impl FromStr for DataType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let sexp = symbolic_expressions::parser::parse_str(s)?;
        parse_data_type(&sexp)
    }
}

/// Parse a data type from an S-expression.
pub fn parse_data_type(sexp: &Sexp) -> Result<DataType, ParseError> {
    match sexp {
        Sexp::String(s) => parse_data_type_atom(s),
        Sexp::List(items) => {
            let head = items
                .first()
                .and_then(|s| match s {
                    Sexp::String(s_inner) => Some(s_inner.as_str()),
                    _ => None,
                })
                .ok_or_else(|| ParseError::Type("expected data type expression".to_owned()))?;

            match head {
                "idxT" if items.len() == 2 => {
                    let n = parse_nat(&items[1])?;
                    Ok(DataType::Index(Box::new(n)))
                }
                "pairT" if items.len() == 3 => {
                    let a = parse_data_type(&items[1])?;
                    let b = parse_data_type(&items[2])?;
                    Ok(DataType::Pair(Box::new(a), Box::new(b)))
                }
                "arrT" if items.len() == 3 => {
                    let n = parse_nat(&items[1])?;
                    let dt = parse_data_type(&items[2])?;
                    Ok(DataType::Array(Box::new(n), Box::new(dt)))
                }
                "vecT" if items.len() == 3 => {
                    let n = parse_nat(&items[1])?;
                    let dt = parse_data_type(&items[2])?;
                    Ok(DataType::Vector(Box::new(n), Box::new(dt)))
                }
                _ => Err(ParseError::Type(format!("unknown data type form: {head}"))),
            }
        }
        Sexp::Empty => Err(ParseError::Type(
            "empty sexp in data type position".to_owned(),
        )),
    }
}

pub fn parse_data_type_atom(s: &str) -> Result<DataType, ParseError> {
    // Data type variable: $d<index>
    if let Some(rest) = s.strip_prefix("$d") {
        let idx = rest
            .parse::<usize>()
            .map_err(|reason| ParseError::VarIndex {
                input: s.to_owned(),
                reason,
            })?;
        return Ok(DataType::Var(idx));
    }

    // Scalar type
    if let Some(scalar) = parse_scalar_type(s) {
        return Ok(DataType::Scalar(scalar));
    }

    // NatT
    if s == "natT" {
        return Ok(DataType::NatT);
    }

    Err(ParseError::Type(format!(
        "cannot parse data type atom: {s}"
    )))
}

// ============================================================================
// Types
// ============================================================================

/// Types in Rise (including function types).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    /// Data type
    Data(DataType),
    /// Function type: (fun in out)
    Fun(Box<Type>, Box<Type>),
    /// Nat-polymorphic function: (natFun t)
    NatFun(Box<Type>),
    /// DataType-polymorphic function: (dataFun t)
    DataFun(Box<Type>),
    /// Address-polymorphic function: (addrFun t)
    AddrFun(Box<Type>),
    /// NatToNat-polymorphic function: (natNatFun t)
    NatNatFun(Box<Type>),
}

impl Type {
    /// Create a function type.
    #[must_use]
    pub fn fun(a: Type, b: Type) -> Self {
        Type::Fun(Box::new(a), Box::new(b))
    }

    /// Create a data type wrapper.
    #[must_use]
    pub fn data(dt: DataType) -> Self {
        Type::Data(dt)
    }

    /// Convert this type to a `RiseLabel`.
    /// Note: For `Type::Data`, this returns the label of the inner `DataType`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            Type::Data(dt) => dt.to_label(),
            Type::Fun(..) => RiseLabel::Fun,
            Type::NatFun(..) => RiseLabel::NatFun,
            Type::DataFun(..) => RiseLabel::DataFun,
            Type::AddrFun(..) => RiseLabel::AddrFun,
            Type::NatNatFun(..) => RiseLabel::NatNatFun,
        }
    }

    /// Convert this type to a `TreeNode<RiseLabel>`.
    #[must_use]
    pub fn to_tree(&self) -> TreeNode<RiseLabel> {
        match self {
            Type::Data(dt) => dt.to_tree(),
            Type::Fun(a, b) => TreeNode::new(self.to_label(), vec![a.to_tree(), b.to_tree()]),
            Type::NatFun(t) | Type::DataFun(t) | Type::AddrFun(t) | Type::NatNatFun(t) => {
                TreeNode::new(self.to_label(), vec![t.to_tree()])
            }
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Data(dt) => write!(f, "{dt}"),
            Type::Fun(a, b) => write!(f, "(fun {a} {b})"),
            Type::NatFun(t) => write!(f, "(natFun {t})"),
            Type::DataFun(t) => write!(f, "(dataFun {t})"),
            Type::AddrFun(t) => write!(f, "(addrFun {t})"),
            Type::NatNatFun(t) => write!(f, "(natNatFun {t})"),
        }
    }
}

impl IntoSexp for Type {
    fn into_sexp(&self) -> Sexp {
        match self {
            Type::Data(dt) => dt.into_sexp(),
            Type::Fun(a, b) => Sexp::List(vec![
                Sexp::String("fun".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            Type::NatFun(t) => Sexp::List(vec![Sexp::String("natFun".to_owned()), t.into_sexp()]),
            Type::DataFun(t) => Sexp::List(vec![Sexp::String("dataFun".to_owned()), t.into_sexp()]),
            Type::AddrFun(t) => Sexp::List(vec![Sexp::String("addrFun".to_owned()), t.into_sexp()]),
            Type::NatNatFun(t) => {
                Sexp::List(vec![Sexp::String("natNatFun".to_owned()), t.into_sexp()])
            }
        }
    }
}

impl FromStr for Type {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let sexp = symbolic_expressions::parser::parse_str(s)?;
        parse_type(&sexp)
    }
}

/// Parse a type from an S-expression.
pub fn parse_type(sexp: &Sexp) -> Result<Type, ParseError> {
    match sexp {
        Sexp::String(s) => {
            // Try parsing as data type first
            let dt = parse_data_type_atom(s)?;
            Ok(Type::Data(dt))
        }
        Sexp::List(items) => {
            let head = items
                .first()
                .and_then(|s| match s {
                    Sexp::String(s_inner) => Some(s_inner.as_str()),
                    _ => None,
                })
                .ok_or_else(|| ParseError::Type("expected type expression".to_owned()))?;

            match head {
                "fun" if items.len() == 3 => {
                    let a = parse_type(&items[1])?;
                    let b = parse_type(&items[2])?;
                    Ok(Type::Fun(Box::new(a), Box::new(b)))
                }
                "natFun" if items.len() == 2 => {
                    let t = parse_type(&items[1])?;
                    Ok(Type::NatFun(Box::new(t)))
                }
                "dataFun" if items.len() == 2 => {
                    let t = parse_type(&items[1])?;
                    Ok(Type::DataFun(Box::new(t)))
                }
                "addrFun" if items.len() == 2 => {
                    let t = parse_type(&items[1])?;
                    Ok(Type::AddrFun(Box::new(t)))
                }
                "natNatFun" if items.len() == 2 => {
                    let t = parse_type(&items[1])?;
                    Ok(Type::NatNatFun(Box::new(t)))
                }
                // Data type forms
                "idxT" | "pairT" | "arrT" | "vecT" => {
                    let dt = parse_data_type(sexp)?;
                    Ok(Type::Data(dt))
                }
                _ => Err(ParseError::Type(format!("unknown type form: {head}"))),
            }
        }
        Sexp::Empty => Err(ParseError::Type("empty sexp in type position".to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_type_scalar() {
        let ty: Type = "f32".parse().unwrap();
        assert_eq!(ty, Type::Data(DataType::Scalar(ScalarType::F32)));
    }

    #[test]
    fn parse_type_fun() {
        let ty: Type = "(fun f32 f32)".parse().unwrap();
        assert_eq!(
            ty,
            Type::Fun(
                Box::new(Type::Data(DataType::Scalar(ScalarType::F32))),
                Box::new(Type::Data(DataType::Scalar(ScalarType::F32))),
            )
        );
    }

    #[test]
    fn parse_type_array() {
        let ty: Type = "(arrT 10n f32)".parse().unwrap();
        assert_eq!(
            ty,
            Type::Data(DataType::Array(
                Box::new(Nat::Cst(10)),
                Box::new(DataType::Scalar(ScalarType::F32)),
            ))
        );
    }

    #[test]
    fn sexp_roundtrip_type() {
        let ty = Type::fun(
            Type::Data(DataType::Scalar(ScalarType::F32)),
            Type::Data(DataType::Array(
                Box::new(Nat::cst_node(10)),
                Box::new(DataType::Scalar(ScalarType::F32)),
            )),
        );
        let sexp = ty.into_sexp().to_string();
        let parsed: Type = sexp.parse().unwrap();
        assert_eq!(ty, parsed);
    }
}
