use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{Kind, Kindable};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct DBIndex {
    kind: Kind,
    value: u32,
}

impl DBIndex {
    pub fn new(kind: Kind, value: u32) -> Self {
        Self { kind, value }
    }

    pub fn zero(kind: Kind) -> Self {
        Self { kind, value: 0 }
    }

    pub fn inc(self, kind: Kind) -> Self {
        self + DBShift::up(kind)
    }

    pub fn dec(self, kind: Kind) -> Self {
        self + DBShift::down(kind)
    }

    pub fn value(self) -> u32 {
        self.value
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl std::ops::Add<DBShift> for DBIndex {
    type Output = Self;

    fn add(self, rhs: DBShift) -> Self::Output {
        let shift = rhs.get(self.kind);
        Self {
            kind: self.kind,
            value: self.value.strict_add_signed(shift),
        }
    }
}

impl std::str::FromStr for DBIndex {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            if let Some((tag, i)) = stripped_s.split_at_checked(1) {
                let kind = match tag {
                    "e" => Kind::Expr,
                    "n" => Kind::Nat,
                    "d" => Kind::Data,
                    "a" => Kind::Addr,
                    "x" => Kind::Nat2Nat,
                    _ => return Err(IndexError::ImproperTag(stripped_s.to_owned())),
                };
                Ok(Self {
                    kind,
                    value: i.parse()?,
                })
            } else {
                Err(IndexError::MissingTag(stripped_s.to_owned()))
            }
        } else {
            Err(IndexError::MissingPercentPrefix(s.to_owned()))
        }
    }
}

impl std::fmt::Display for DBIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            Kind::Expr => write!(f, "%e{}", self.value),
            Kind::Nat => write!(f, "%n{}", self.value),
            Kind::Data => write!(f, "%d{}", self.value),
            Kind::Addr => write!(f, "%a{}", self.value),
            Kind::Nat2Nat => write!(f, "%x{}", self.value),
        }
    }
}

impl Kindable for DBIndex {
    fn kind(&self) -> Kind {
        self.kind
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct DBShift {
    expr: i32,
    nat: i32,
    data: i32,
    addr: i32,
    nat2nat: i32,
}

impl DBShift {
    pub fn up(kind: Kind) -> Self {
        Self::new_with(kind, 1)
    }

    pub fn down(kind: Kind) -> Self {
        Self::new_with(kind, -1)
    }

    pub fn get(&self, kind: Kind) -> i32 {
        match kind {
            Kind::Expr => self.expr,
            Kind::Nat => self.nat,
            Kind::Data => self.data,
            Kind::Addr => self.addr,
            Kind::Nat2Nat => self.nat2nat,
        }
    }

    fn new_with(kind: Kind, value: i32) -> Self {
        match kind {
            Kind::Expr => Self {
                expr: value,
                nat: 0,
                data: 0,
                addr: 0,
                nat2nat: 0,
            },
            Kind::Nat => Self {
                expr: 0,
                nat: value,
                data: 0,
                addr: 0,
                nat2nat: 0,
            },
            Kind::Data => Self {
                expr: 0,
                nat: 0,
                data: value,
                addr: 0,
                nat2nat: 0,
            },
            Kind::Addr => Self {
                expr: 0,
                nat: 0,
                data: 0,
                addr: value,
                nat2nat: 0,
            },
            Kind::Nat2Nat => Self {
                expr: 0,
                nat: 0,
                data: 0,
                addr: 0,
                nat2nat: value,
            },
        }
    }
}

impl TryFrom<(i32, i32, i32, i32, i32)> for DBShift {
    type Error = IndexError;

    fn try_from(value: (i32, i32, i32, i32, i32)) -> Result<Self, Self::Error> {
        if value.0 == 0 && value.1 == 0 && value.2 == 0 && value.3 == 0 && value.4 == 0 {
            return Err(IndexError::ZeroShift);
        }
        Ok(Self {
            expr: value.0,
            nat: value.1,
            data: value.2,
            addr: value.3,
            nat2nat: value.4,
        })
    }
}

impl std::fmt::Display for DBShift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {})",
            self.expr, self.nat, self.data, self.addr, self.nat2nat
        )
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct DBCutoff {
    expr: u32,
    nat: u32,
    data: u32,
    addr: u32,
    nat2nat: u32,
}

impl DBCutoff {
    pub fn get(&self, kind: Kind) -> u32 {
        match kind {
            Kind::Expr => self.expr,
            Kind::Nat => self.nat,
            Kind::Data => self.data,
            Kind::Addr => self.addr,
            Kind::Nat2Nat => self.nat2nat,
        }
    }

    pub fn inc(self, kind: Kind) -> Self {
        self + DBShift::up(kind)
    }

    // pub fn dec(self, kind: Kind) -> Self {
    //     self + DBShift::down(kind)
    // }

    pub fn zero() -> Self {
        Self {
            expr: 0,
            nat: 0,
            data: 0,
            addr: 0,
            nat2nat: 0,
        }
    }
}

impl std::ops::Add<DBShift> for DBCutoff {
    type Output = Self;

    fn add(self, rhs: DBShift) -> Self::Output {
        Self {
            expr: self.expr.strict_add_signed(rhs.expr),
            nat: self.expr.strict_add_signed(rhs.nat),
            data: self.expr.strict_add_signed(rhs.data),
            addr: self.expr.strict_add_signed(rhs.addr),
            nat2nat: self.expr.strict_add_signed(rhs.nat2nat),
        }
    }
}

impl From<(u32, u32, u32, u32, u32)> for DBCutoff {
    fn from(value: (u32, u32, u32, u32, u32)) -> Self {
        Self {
            expr: value.0,
            nat: value.1,
            data: value.2,
            addr: value.3,
            nat2nat: value.4,
        }
    }
}

impl std::fmt::Display for DBCutoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {})",
            self.expr, self.nat, self.data, self.addr, self.nat2nat
        )
    }
}

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Missing % Prefix: {0}")]
    MissingPercentPrefix(String),
    #[error("Improper Tag {0}")]
    ImproperTag(String),
    #[error("Missing Tag {0}")]
    MissingTag(String),
    #[error("Invalide zero shift")]
    ZeroShift,
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
