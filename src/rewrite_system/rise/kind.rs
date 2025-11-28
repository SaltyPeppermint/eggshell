use std::fmt::Display;

use egg::Var;
use serde::{Deserialize, Serialize};

use super::Index;

pub trait Kindable {
    fn kind(&self) -> Option<Kind>;
}

impl<T: Kindable> Kindable for &T {
    fn kind(&self) -> Option<Kind> {
        (*self).kind()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum Kind {
    Expr,
    Nat,
    Data,
    Addr,
    Synthetic,
}

impl Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::Expr => write!(f, "EXPR"),
            Kind::Nat => write!(f, "NAT"),
            Kind::Data => write!(f, "DATA"),
            Kind::Addr => write!(f, "ADDR"),
            Kind::Synthetic => write!(f, "SYNTHETIC"),
        }
    }
}

impl Kindable for Var {
    fn kind(&self) -> Option<Kind> {
        let var_str = self.to_string();
        var_str.chars().nth(1).map(|c| match c {
            'd' | 't' => Kind::Data,
            'a' => Kind::Addr,
            'n' => Kind::Nat,
            _ => Kind::Expr,
        })
    }
}

impl Kindable for Index {
    fn kind(&self) -> Option<Kind> {
        Some(match self {
            Index::Expr(_) => Kind::Expr,
            Index::Nat(_) => Kind::Nat,
            Index::Data(_) => Kind::Data,
            Index::Addr(_) => Kind::Addr,
            Index::Synthetic(_) => Kind::Synthetic,
        })
    }
}
