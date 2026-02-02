//! Natural number expressions in Rise.

use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp};

use super::ParseError;
use super::label::RiseLabel;
use crate::distance::tree::TreeNode;

/// Natural number expressions in Rise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Nat {
    /// Variable: $n<index>
    Var(usize),
    /// Constant: <value>n
    Cst(i64),
    /// Addition: (natAdd a b)
    Add(Box<Nat>, Box<Nat>),
    /// Multiplication: (natMul a b)
    Mul(Box<Nat>, Box<Nat>),
    /// Power: (natPow a b)
    Pow(Box<Nat>, Box<Nat>),
    /// Modulo: (natMod a b)
    Mod(Box<Nat>, Box<Nat>),
    /// Floor division: (natFloorDiv a b)
    FloorDiv(Box<Nat>, Box<Nat>),
}

impl Nat {
    /// Create a nat variable.
    #[must_use]
    pub fn var_node(index: usize) -> Self {
        Nat::Var(index)
    }

    /// Create a nat constant.
    #[must_use]
    pub fn cst_node(value: i64) -> Self {
        Nat::Cst(value)
    }

    /// Create a nat addition.
    #[must_use]
    pub fn add_node(a: Nat, b: Nat) -> Self {
        Nat::Add(Box::new(a), Box::new(b))
    }

    /// Create a nat multiplication.
    #[must_use]
    pub fn mul_node(a: Nat, b: Nat) -> Self {
        Nat::Mul(Box::new(a), Box::new(b))
    }

    /// Convert this nat to a `RiseLabel`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            Nat::Var(i) => RiseLabel::NatVar(*i),
            Nat::Cst(n) => RiseLabel::NatCst(*n),
            Nat::Add(..) => RiseLabel::NatAdd,
            Nat::Mul(..) => RiseLabel::NatMul,
            Nat::Pow(..) => RiseLabel::NatPow,
            Nat::Mod(..) => RiseLabel::NatMod,
            Nat::FloorDiv(..) => RiseLabel::NatFloorDiv,
        }
    }

    /// Convert this nat to a `TreeNode<RiseLabel>`.
    #[must_use]
    pub fn to_tree(&self) -> TreeNode<RiseLabel> {
        match self {
            Nat::Var(_) | Nat::Cst(_) => TreeNode::leaf(self.to_label()),
            Nat::Add(a, b)
            | Nat::Mul(a, b)
            | Nat::Pow(a, b)
            | Nat::Mod(a, b)
            | Nat::FloorDiv(a, b) => TreeNode::new(self.to_label(), vec![a.to_tree(), b.to_tree()]),
        }
    }
}

impl Display for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Nat::Var(i) => write!(f, "$n{i}"),
            Nat::Cst(n) => write!(f, "{n}n"),
            Nat::Add(a, b) => write!(f, "(natAdd {a} {b})"),
            Nat::Mul(a, b) => write!(f, "(natMul {a} {b})"),
            Nat::Pow(a, b) => write!(f, "(natPow {a} {b})"),
            Nat::Mod(a, b) => write!(f, "(natMod {a} {b})"),
            Nat::FloorDiv(a, b) => write!(f, "(natFloorDiv {a} {b})"),
        }
    }
}

impl IntoSexp for Nat {
    fn into_sexp(&self) -> Sexp {
        match self {
            Nat::Var(i) => Sexp::String(format!("$n{i}")),
            Nat::Cst(n) => Sexp::String(format!("{n}n")),
            Nat::Add(a, b) => Sexp::List(vec![
                Sexp::String("natAdd".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            Nat::Mul(a, b) => Sexp::List(vec![
                Sexp::String("natMul".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            Nat::Pow(a, b) => Sexp::List(vec![
                Sexp::String("natPow".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            Nat::Mod(a, b) => Sexp::List(vec![
                Sexp::String("natMod".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
            Nat::FloorDiv(a, b) => Sexp::List(vec![
                Sexp::String("natFloorDiv".to_owned()),
                a.into_sexp(),
                b.into_sexp(),
            ]),
        }
    }
}

impl FromStr for Nat {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let sexp = symbolic_expressions::parser::parse_str(s)?;
        parse_nat(&sexp)
    }
}

/// Parse a nat from an S-expression.
#[allow(clippy::missing_errors_doc)]
pub fn parse_nat(sexp: &Sexp) -> Result<Nat, ParseError> {
    match sexp {
        Sexp::String(s) => parse_nat_atom(s),
        Sexp::List(items) => {
            let head = items
                .first()
                .and_then(|s| match s {
                    Sexp::String(inner_s) => Some(inner_s.as_str()),
                    _ => None,
                })
                .ok_or_else(|| ParseError("expected nat expression".to_owned()))?;

            match head {
                "natAdd" if items.len() == 3 => {
                    let a = parse_nat(&items[1])?;
                    let b = parse_nat(&items[2])?;
                    Ok(Nat::Add(Box::new(a), Box::new(b)))
                }
                "natMul" if items.len() == 3 => {
                    let a = parse_nat(&items[1])?;
                    let b = parse_nat(&items[2])?;
                    Ok(Nat::Mul(Box::new(a), Box::new(b)))
                }
                "natPow" if items.len() == 3 => {
                    let a = parse_nat(&items[1])?;
                    let b = parse_nat(&items[2])?;
                    Ok(Nat::Pow(Box::new(a), Box::new(b)))
                }
                "natMod" if items.len() == 3 => {
                    let a = parse_nat(&items[1])?;
                    let b = parse_nat(&items[2])?;
                    Ok(Nat::Mod(Box::new(a), Box::new(b)))
                }
                "natFloorDiv" if items.len() == 3 => {
                    let a = parse_nat(&items[1])?;
                    let b = parse_nat(&items[2])?;
                    Ok(Nat::FloorDiv(Box::new(a), Box::new(b)))
                }
                _ => Err(ParseError(format!("unknown nat form: {head}"))),
            }
        }
        Sexp::Empty => Err(ParseError("empty sexp in nat position".to_owned())),
    }
}

fn parse_nat_atom(s: &str) -> Result<Nat, ParseError> {
    // Nat variable: $n<index>
    if let Some(rest) = s.strip_prefix("$n") {
        let idx = rest
            .parse::<usize>()
            .map_err(|e| ParseError(format!("invalid nat variable index: {s} ({e})")))?;
        return Ok(Nat::Var(idx));
    }

    // Nat constant: <value>n
    if let Some(num) = s.strip_suffix('n') {
        let value = num
            .parse::<i64>()
            .map_err(|e| ParseError(format!("invalid nat constant: {s} ({e})")))?;
        return Ok(Nat::Cst(value));
    }

    // Plain number (also a nat constant)
    if let Ok(value) = s.parse::<i64>() {
        return Ok(Nat::Cst(value));
    }

    Err(ParseError(format!("cannot parse nat atom: {s}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nat_var() {
        let nat: Nat = "$n0".parse().unwrap();
        assert_eq!(nat, Nat::Var(0));
    }

    #[test]
    fn parse_nat_cst() {
        let nat: Nat = "42n".parse().unwrap();
        assert_eq!(nat, Nat::Cst(42));
    }

    #[test]
    fn parse_nat_add() {
        let nat: Nat = "(natAdd $n0 5n)".parse().unwrap();
        assert_eq!(nat, Nat::Add(Box::new(Nat::Var(0)), Box::new(Nat::Cst(5))));
    }

    #[test]
    fn sexp_roundtrip_nat() {
        let nat = Nat::add_node(Nat::var_node(0), Nat::cst_node(5));
        let sexp = nat.into_sexp().to_string();
        let parsed: Nat = sexp.parse().unwrap();
        assert_eq!(nat, parsed);
    }
}
