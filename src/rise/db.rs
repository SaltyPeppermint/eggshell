use std::fmt;

use thiserror::Error;

use super::kind::{Kind, Kindable};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum Index {
    Expr(u32),
    Nat(u32),
    Data(u32),
    Addr(u32),
    Nat2Nat(u32),
}

impl Index {
    pub fn new(kind: Kind, value: u32) -> Self {
        match kind {
            Kind::Expr => Self::Expr(value),
            Kind::Nat => Self::Nat(value),
            Kind::Data => Self::Data(value),
            Kind::Addr => Self::Addr(value),
            Kind::Nat2Nat => Self::Nat2Nat(value),
        }
    }

    pub fn zero(kind: Kind) -> Self {
        Self::new(kind, 0)
    }

    pub fn inc(self, kind: Kind) -> Self {
        self + Shift::up(kind)
    }

    pub fn dec(self, kind: Kind) -> Self {
        self + Shift::down(kind)
    }

    pub fn value(self) -> u32 {
        match self {
            Index::Expr(value)
            | Index::Nat(value)
            | Index::Data(value)
            | Index::Addr(value)
            | Index::Nat2Nat(value) => value,
        }
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl std::ops::Add<Shift> for Index {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        let shift = rhs.of_index(self);
        let new_value = self.value().strict_add_signed(shift);
        Self::new(self.kind(), new_value)
    }
}

impl Kindable for Index {
    fn kind(&self) -> Kind {
        match self {
            Index::Expr(_) => Kind::Expr,
            Index::Nat(_) => Kind::Nat,
            Index::Data(_) => Kind::Data,
            Index::Addr(_) => Kind::Addr,
            Index::Nat2Nat(_) => Kind::Nat2Nat,
        }
    }
}

impl std::str::FromStr for Index {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            if let Some((tag, i)) = stripped_s.split_at_checked(1) {
                let value = i.parse()?;
                Ok(match tag {
                    "e" => Self::Expr(value),
                    "n" => Self::Nat(value),
                    "d" => Self::Data(value),
                    "a" => Self::Addr(value),
                    "x" => Self::Nat2Nat(value),
                    _ => return Err(IndexError::ImproperTag(stripped_s.to_owned())),
                })
            } else {
                Err(IndexError::MissingTag(stripped_s.to_owned()))
            }
        } else {
            Err(IndexError::MissingPercentPrefix(s.to_owned()))
        }
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(value) => write!(f, "%e{value}"),
            Self::Nat(value) => write!(f, "%n{value}"),
            Self::Data(value) => write!(f, "%d{value}"),
            Self::Addr(value) => write!(f, "%a{value}"),
            Self::Nat2Nat(value) => write!(f, "%x{value}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum Shift {
    /// expr, nat, data, addr, nat2nat
    Expr((i32, i32, i32, i32, i32)),
    /// nat, nat2nat
    Nat((i32, i32)),
    /// nat, data, nat2nat
    Data((i32, i32, i32)),
    /// addr
    Addr(i32),
}

impl Shift {
    pub fn up(kind: Kind) -> Self {
        Self::new_with(kind, 1)
    }

    pub fn down(kind: Kind) -> Self {
        Self::new_with(kind, -1)
    }

    fn of_index(&self, index: Index) -> i32 {
        match index {
            Index::Expr(_) => self.expr_shift(),
            Index::Nat(_) => self.nat_shift(),
            Index::Data(_) => self.data_shift(),
            Index::Addr(_) => self.addr_shift(),
            Index::Nat2Nat(_) => self.nat2nat_shift(),
        }
    }

    pub fn expr_shift(&self) -> i32 {
        match self {
            Shift::Expr((expr_shift, _, _, _, _)) => *expr_shift,
            Shift::Nat(_) | Shift::Data(_) | Shift::Addr(_) => 0,
        }
    }

    pub fn nat_shift(&self) -> i32 {
        match self {
            Shift::Expr((_, nat_shift, _, _, _))
            | Shift::Nat((nat_shift, _))
            | Shift::Data((nat_shift, _, _)) => *nat_shift,
            Shift::Addr(_) => 0,
        }
    }

    pub fn data_shift(&self) -> i32 {
        match self {
            Shift::Expr((_, _, data_shift, _, _)) | Shift::Data((_, data_shift, _)) => *data_shift,
            Shift::Nat(_) | Shift::Addr(_) => 0,
        }
    }

    pub fn addr_shift(&self) -> i32 {
        match self {
            Shift::Expr((_, _, _, addr_shift, _)) | Shift::Addr(addr_shift) => *addr_shift,
            Shift::Nat(_) | Shift::Data(_) => 0,
        }
    }

    pub fn nat2nat_shift(&self) -> i32 {
        match self {
            Shift::Expr((_, _, _, _, nat2nat_shift))
            | Shift::Nat((_, nat2nat_shift))
            | Shift::Data((_, _, nat2nat_shift)) => *nat2nat_shift,
            Shift::Addr(_) => 0,
        }
    }

    fn new_with(kind: Kind, arg: i32) -> Shift {
        match kind {
            Kind::Expr => Shift::Expr((arg, 0, 0, 0, 0)),
            Kind::Nat => Shift::Nat((arg, 0)),
            Kind::Data => Shift::Data((arg, 0, 0)),
            Kind::Addr => Shift::Addr(arg),
            Kind::Nat2Nat => panic!("No Nat2Nat shift is possible"),
        }
    }
}

impl From<(i32, i32, i32, i32, i32)> for Shift {
    fn from(value: (i32, i32, i32, i32, i32)) -> Self {
        Self::Expr(value)
    }
}

impl From<(i32, i32, i32)> for Shift {
    fn from(value: (i32, i32, i32)) -> Self {
        Self::Data(value)
    }
}

impl From<(i32, i32)> for Shift {
    fn from(value: (i32, i32)) -> Self {
        Self::Nat(value)
    }
}

impl From<i32> for Shift {
    fn from(value: i32) -> Self {
        Self::Addr(value)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Default)]
pub struct Cutoff {
    expr: u32,
    nat: u32,
    data: u32,
    addr: u32,
    nat2nat: u32,
}

impl Cutoff {
    pub fn inc(self, kind: Kind) -> Self {
        self + Shift::up(kind)
    }

    // pub fn dec(self, kind: Kind) -> Self {
    //     self + DBShift::down(kind)
    // }

    pub fn of_index(self, index: Index) -> u32 {
        match index {
            Index::Expr(_) => self.expr,
            Index::Nat(_) => self.nat,
            Index::Data(_) => self.data,
            Index::Addr(_) => self.addr,
            Index::Nat2Nat(_) => self.nat2nat,
        }
    }

    pub fn zero() -> Self {
        Cutoff::default()
    }
}

impl std::ops::Add<Shift> for Cutoff {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        Self {
            expr: self.expr.strict_add_signed(rhs.expr_shift()),
            nat: self.nat.strict_add_signed(rhs.nat_shift()),
            data: self.data.strict_add_signed(rhs.data_shift()),
            addr: self.expr.strict_add_signed(rhs.addr_shift()),
            nat2nat: self.expr.strict_add_signed(rhs.nat2nat_shift()),
        }
    }
}

impl From<(u32, u32, u32, u32, u32)> for Cutoff {
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

impl From<(u32, u32)> for Cutoff {
    fn from(value: (u32, u32)) -> Self {
        Self {
            nat: value.0,
            nat2nat: value.1,
            ..Default::default()
        }
    }
}

impl From<(u32, u32, u32)> for Cutoff {
    fn from(value: (u32, u32, u32)) -> Self {
        Self {
            nat: value.0,
            data: value.1,
            nat2nat: value.2,
            ..Default::default()
        }
    }
}

impl From<u32> for Cutoff {
    fn from(value: u32) -> Self {
        Self {
            addr: value,
            ..Default::default()
        }
    }
}

impl fmt::Display for Cutoff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
