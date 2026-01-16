use std::fmt;

use egg::Var;
use thiserror::Error;

pub trait Kindable {
    fn kind(&self) -> Kind;

    fn is_expr(&self) -> bool {
        self.kind() == Kind::Expr
    }
    // fn is_data(&self) -> bool {
    //     self.kind() == Kind::Data
    // }
    fn is_nat(&self) -> bool {
        self.kind() == Kind::Nat
    }
    // fn is_addr(&self) -> bool {
    //     self.kind() == Kind::Addr
    // }
    // fn is_nat2nat(&self) -> bool {
    //     self.kind() == Kind::Nat2Nat
    // }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum Kind {
    Expr,
    Nat,
    Data,
    Addr,
    Nat2Nat,
}

impl Kind {
    pub fn prefix(self) -> char {
        match self {
            Kind::Expr => 'e',
            Kind::Nat => 'n',
            Kind::Data => 'd',
            Kind::Addr => 'a',
            Kind::Nat2Nat => 'x',
        }
    }
}

impl TryFrom<char> for Kind {
    type Error = KindError;

    fn try_from(c: char) -> Result<Self, KindError> {
        Ok(match c {
            'd' | 't' => Kind::Data,
            'a' => Kind::Addr,
            'n' => Kind::Nat,
            'x' => Kind::Nat2Nat,
            'e' => Kind::Expr,
            x if x.is_numeric() => Kind::Expr,
            _ => return Err(KindError::ImproperTag(c)),
        })
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Expr => write!(f, "EXPR"),
            Kind::Nat => write!(f, "NAT"),
            Kind::Data => write!(f, "DATA"),
            Kind::Addr => write!(f, "ADDR"),
            Kind::Nat2Nat => write!(f, "NAT2NAT"),
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
                'x' => Kind::Nat2Nat,
                'e' => Kind::Expr,
                x if x.is_numeric() => Kind::Expr,
                x => panic!("Wrong format {x}"),
            })
            .expect("Wrong format {x}")
    }
}

#[derive(Error, Debug)]
pub enum KindError {
    #[error("Improper Prefix Character {0}")]
    ImproperTag(char),
}
