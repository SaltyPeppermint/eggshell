use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{Kind, Kindable};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum DBIndex {
    Expr(u32),
    Nat(u32),
    Data(u32),
    Addr(u32),
    Nat2Nat(u32),
}

impl DBIndex {
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
        self + DBShift::up(kind)
    }

    pub fn dec(self, kind: Kind) -> Self {
        self + DBShift::down(kind)
    }

    pub fn value(self) -> u32 {
        match self {
            DBIndex::Expr(value)
            | DBIndex::Nat(value)
            | DBIndex::Data(value)
            | DBIndex::Addr(value)
            | DBIndex::Nat2Nat(value) => value,
        }
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl std::ops::Add<DBShift> for DBIndex {
    type Output = Self;

    fn add(self, rhs: DBShift) -> Self::Output {
        let shift = rhs.of_index(self);
        let new_value = self.value().strict_add_signed(shift);
        Self::new(self.kind(), new_value)
    }
}

impl Kindable for DBIndex {
    fn kind(&self) -> Kind {
        match self {
            DBIndex::Expr(_) => Kind::Expr,
            DBIndex::Nat(_) => Kind::Nat,
            DBIndex::Data(_) => Kind::Data,
            DBIndex::Addr(_) => Kind::Addr,
            DBIndex::Nat2Nat(_) => Kind::Nat2Nat,
        }
    }
}

impl std::str::FromStr for DBIndex {
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

impl std::fmt::Display for DBIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expr(value) => write!(f, "%e{value}"),
            Self::Nat(value) => write!(f, "%n{value}"),
            Self::Data(value) => write!(f, "%d{value}"),
            Self::Addr(value) => write!(f, "%a{value}"),
            Self::Nat2Nat(value) => write!(f, "%x{value}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum DBShift {
    /// expr, nat, data, addr, nat2nat
    Expr((i32, i32, i32, i32, i32)),
    /// nat, nat2nat
    Nat((i32, i32)),
    /// nat, data, nat2nat
    Data((i32, i32, i32)),
    /// addr
    Addr(i32),
}

impl DBShift {
    pub fn up(kind: Kind) -> Self {
        Self::new_with(kind, 1)
    }

    pub fn down(kind: Kind) -> Self {
        Self::new_with(kind, -1)
    }

    fn of_index(&self, index: DBIndex) -> i32 {
        match index {
            DBIndex::Expr(_) => self.expr_shift(),
            DBIndex::Nat(_) => self.nat_shift(),
            DBIndex::Data(_) => self.data_shift(),
            DBIndex::Addr(_) => self.addr_shift(),
            DBIndex::Nat2Nat(_) => self.nat2nat_shift(),
        }
    }

    pub fn expr_shift(&self) -> i32 {
        match self {
            DBShift::Expr((expr_shift, _, _, _, _)) => *expr_shift,
            DBShift::Nat(_) | DBShift::Data(_) | DBShift::Addr(_) => 0,
        }
    }

    pub fn nat_shift(&self) -> i32 {
        match self {
            DBShift::Expr((_, nat_shift, _, _, _))
            | DBShift::Nat((nat_shift, _))
            | DBShift::Data((nat_shift, _, _)) => *nat_shift,
            DBShift::Addr(_) => 0,
        }
    }

    pub fn data_shift(&self) -> i32 {
        match self {
            DBShift::Expr((_, _, data_shift, _, _)) | DBShift::Data((_, data_shift, _)) => {
                *data_shift
            }
            DBShift::Nat(_) | DBShift::Addr(_) => 0,
        }
    }

    pub fn addr_shift(&self) -> i32 {
        match self {
            DBShift::Expr((_, _, _, addr_shift, _)) | DBShift::Addr(addr_shift) => *addr_shift,
            DBShift::Nat(_) | DBShift::Data(_) => 0,
        }
    }

    pub fn nat2nat_shift(&self) -> i32 {
        match self {
            DBShift::Expr((_, _, _, _, nat2nat_shift))
            | DBShift::Nat((_, nat2nat_shift))
            | DBShift::Data((_, _, nat2nat_shift)) => *nat2nat_shift,
            DBShift::Addr(_) => 0,
        }
    }

    fn new_with(kind: Kind, arg: i32) -> DBShift {
        match kind {
            Kind::Expr => DBShift::Expr((arg, 0, 0, 0, 0)),
            Kind::Nat => DBShift::Nat((arg, 0)),
            Kind::Data => DBShift::Data((arg, 0, 0)),
            Kind::Addr => DBShift::Addr(arg),
            Kind::Nat2Nat => panic!("No Nat2Nat shift is possible"),
        }
    }
}

impl From<(i32, i32, i32, i32, i32)> for DBShift {
    fn from(value: (i32, i32, i32, i32, i32)) -> Self {
        Self::Expr(value)
    }
}

impl From<(i32, i32, i32)> for DBShift {
    fn from(value: (i32, i32, i32)) -> Self {
        Self::Data(value)
    }
}

impl From<(i32, i32)> for DBShift {
    fn from(value: (i32, i32)) -> Self {
        Self::Nat(value)
    }
}

impl From<i32> for DBShift {
    fn from(value: i32) -> Self {
        Self::Addr(value)
    }
}

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Default, Serialize, Deserialize,
)]
pub struct DBCutoff {
    expr: u32,
    nat: u32,
    data: u32,
    addr: u32,
    nat2nat: u32,
}

impl DBCutoff {
    pub fn inc(self, kind: Kind) -> Self {
        self + DBShift::up(kind)
    }

    // pub fn dec(self, kind: Kind) -> Self {
    //     self + DBShift::down(kind)
    // }

    pub fn of_index(self, index: DBIndex) -> u32 {
        match index {
            DBIndex::Expr(_) => self.expr,
            DBIndex::Nat(_) => self.nat,
            DBIndex::Data(_) => self.data,
            DBIndex::Addr(_) => self.addr,
            DBIndex::Nat2Nat(_) => self.nat2nat,
        }
    }

    pub fn zero() -> Self {
        DBCutoff::default()
    }
}

impl std::ops::Add<DBShift> for DBCutoff {
    type Output = Self;

    fn add(self, rhs: DBShift) -> Self::Output {
        Self {
            expr: self.expr.strict_add_signed(rhs.expr_shift()),
            nat: self.nat.strict_add_signed(rhs.nat_shift()),
            data: self.data.strict_add_signed(rhs.data_shift()),
            addr: self.expr.strict_add_signed(rhs.addr_shift()),
            nat2nat: self.expr.strict_add_signed(rhs.nat2nat_shift()),
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

impl From<(u32, u32)> for DBCutoff {
    fn from(value: (u32, u32)) -> Self {
        Self {
            nat: value.0,
            nat2nat: value.1,
            ..Default::default()
        }
    }
}

impl From<(u32, u32, u32)> for DBCutoff {
    fn from(value: (u32, u32, u32)) -> Self {
        Self {
            nat: value.0,
            data: value.1,
            nat2nat: value.2,
            ..Default::default()
        }
    }
}

impl From<u32> for DBCutoff {
    fn from(value: u32) -> Self {
        Self {
            addr: value,
            ..Default::default()
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
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
