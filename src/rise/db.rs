use std::fmt;

use thiserror::Error;

use super::kind::{Kind, Kindable};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub struct Index {
    kind: Kind,
    value: u32,
}

impl Index {
    pub fn new(kind: Kind, value: u32) -> Self {
        Index { kind, value }
    }

    pub fn zero(kind: Kind) -> Self {
        Self::new(kind, 0)
    }

    pub fn inc(mut self) -> Self {
        self.value = self.value.checked_add_signed(1).unwrap();
        self
    }

    pub fn dec(mut self) -> Self {
        self.value = self.value.checked_add_signed(-1).unwrap();
        self
    }

    pub fn value(self) -> u32 {
        self.value
    }

    pub fn is_zero(self) -> bool {
        self.value == 0
    }
}

impl std::ops::Add<Shift> for Index {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        let shift = rhs.of_kind(self.kind());
        let new_value = self.value.strict_add_signed(shift);
        Self::new(self.kind(), new_value)
    }
}

impl Kindable for Index {
    fn kind(&self) -> Kind {
        self.kind
    }
}

impl std::str::FromStr for Index {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let stripped_s = s
            .strip_prefix("%")
            .ok_or(IndexError::MissingPercentPrefix(s.to_owned()))?;
        let mut chars = stripped_s.chars();
        let kind = chars
            .next()
            .ok_or(IndexError::MissingTag(s.to_owned()))?
            .try_into()?;
        let value = chars.as_str().parse()?;
        Ok(Index { kind, value })
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}{}", self.kind.prefix(), self.value)
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub struct Shift {
    expr: i32,
    nat: i32,
    data: i32,
    addr: i32,
    nat2nat: i32,
}

impl Shift {
    pub fn up(kind: Kind) -> Self {
        Self::new_with(kind, 1)
    }

    pub fn down(kind: Kind) -> Self {
        Self::new_with(kind, -1)
    }

    fn of_kind(&self, kind: Kind) -> i32 {
        match kind {
            Kind::Expr => self.expr(),
            Kind::Nat => self.nat(),
            Kind::Data => self.data(),
            Kind::Addr => self.addr(),
            Kind::Nat2Nat => self.nat2nat(),
        }
    }

    pub fn expr(&self) -> i32 {
        self.expr
    }

    pub fn nat(&self) -> i32 {
        self.nat
    }

    pub fn data(&self) -> i32 {
        self.data
    }

    pub fn addr(&self) -> i32 {
        self.addr
    }

    pub fn nat2nat(&self) -> i32 {
        self.nat2nat
    }

    fn new_with(kind: Kind, arg: i32) -> Shift {
        match kind {
            Kind::Expr => Shift {
                expr: arg,
                ..Default::default()
            },
            Kind::Nat => Shift {
                nat: arg,
                ..Default::default()
            },
            Kind::Data => Shift {
                data: arg,
                ..Default::default()
            },
            Kind::Addr => Shift {
                addr: arg,
                ..Default::default()
            },
            Kind::Nat2Nat => Shift {
                nat2nat: arg,
                ..Default::default()
            },
        }
    }
}

/// expr, nat, data, addr, nat2nat
impl From<(i32, i32, i32, i32, i32)> for Shift {
    fn from(value: (i32, i32, i32, i32, i32)) -> Self {
        Shift {
            expr: value.0,
            nat: value.1,
            data: value.2,
            addr: value.3,
            nat2nat: value.4,
        }
    }
}

/// nat, data, nat2nat
impl From<(i32, i32, i32)> for Shift {
    fn from(value: (i32, i32, i32)) -> Self {
        Shift {
            nat: value.0,
            data: value.1,
            nat2nat: value.2,
            ..Default::default()
        }
    }
}

/// nat, nat2nat
impl From<(i32, i32)> for Shift {
    fn from(value: (i32, i32)) -> Self {
        Shift {
            nat: value.0,
            nat2nat: value.1,
            ..Default::default()
        }
    }
}

/// addr
impl From<i32> for Shift {
    fn from(value: i32) -> Self {
        Shift {
            addr: value,
            ..Default::default()
        }
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

    pub fn of_kind(self, kind: Kind) -> u32 {
        match kind {
            Kind::Expr => self.expr,
            Kind::Nat => self.nat,
            Kind::Data => self.data,
            Kind::Addr => self.addr,
            Kind::Nat2Nat => self.nat2nat,
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
            expr: self.expr.strict_add_signed(rhs.expr()),
            nat: self.nat.strict_add_signed(rhs.nat()),
            data: self.data.strict_add_signed(rhs.data()),
            addr: self.expr.strict_add_signed(rhs.addr()),
            nat2nat: self.expr.strict_add_signed(rhs.nat2nat()),
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
    ImproperTag(#[from] super::kind::KindError),
    #[error("Missing Tag {0}")]
    MissingTag(String),
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
