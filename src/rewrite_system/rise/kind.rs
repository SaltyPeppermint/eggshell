use std::fmt;

use egg::Var;

pub trait Kindable {
    fn kind(&self) -> Kind;
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum Kind {
    Expr,
    Nat,
    Data,
    Addr,
    Nat2Nat,
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
                x if x.is_numeric() => Kind::Expr,
                x => panic!("Wrong format {x}"),
            })
            .expect("Wrong format {x}")
    }
}
