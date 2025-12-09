use std::fmt::Display;

use egg::Var;
use serde::{Deserialize, Serialize};

pub trait Kindable {
    fn kind(&self) -> Kind;
}

impl<T: Kindable> Kindable for &T {
    fn kind(&self) -> Kind {
        (*self).kind()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum Kind {
    Expr,
    Nat,
    Data,
    Addr,
}

impl Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::Expr => write!(f, "EXPR"),
            Kind::Nat => write!(f, "NAT"),
            Kind::Data => write!(f, "DATA"),
            Kind::Addr => write!(f, "ADDR"),
        }
    }
}

impl Kindable for Var {
    fn kind(&self) -> Kind {
        let var_str = self.to_string();
        var_str
            .chars()
            .nth(1)
            .map(|c| match c {
                'd' | 't' => Kind::Data,
                'a' => Kind::Addr,
                'n' => Kind::Nat,
                x if x.is_numeric() => Kind::Expr,
                x => panic!("Wrong format {x}"),
            })
            .expect("Wrong format {x}")
    }
}
