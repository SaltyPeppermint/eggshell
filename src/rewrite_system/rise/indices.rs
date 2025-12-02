use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::Kind;

// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
// pub struct Index(u32);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum Index {
    Expr(u32),
    Nat(u32),
    Data(u32),
    Addr(u32),
}

impl Index {
    pub fn inc(self) -> Self {
        self + Shift::up()
    }

    pub fn dec(self) -> Self {
        self + Shift::down()
    }

    pub fn new(value: u32, kind: Kind) -> Self {
        match kind {
            Kind::Expr => Index::Expr(value),
            Kind::Nat => Index::Nat(value),
            Kind::Data => Index::Data(value),
            Kind::Addr => Index::Addr(value),
        }
    }

    pub fn zero(kind: Kind) -> Self {
        match kind {
            Kind::Expr => Index::Expr(0),
            Kind::Nat => Index::Nat(0),
            Kind::Data => Index::Data(0),
            Kind::Addr => Index::Addr(0),
        }
    }

    pub fn zero_like(other: Self) -> Self {
        match other {
            Index::Expr(_) => Index::Expr(0),
            Index::Nat(_) => Index::Nat(0),
            Index::Data(_) => Index::Data(0),
            Index::Addr(_) => Index::Addr(0),
        }
    }

    pub fn value(self) -> u32 {
        match self {
            Index::Expr(i) | Index::Nat(i) | Index::Data(i) | Index::Addr(i) => i,
        }
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl std::ops::Add<Shift> for Index {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        match self {
            Index::Expr(i) => Index::Expr(i.strict_add_signed(rhs.0)),
            Index::Nat(i) => Index::Nat(i.strict_add_signed(rhs.0)),
            Index::Data(i) => Index::Data(i.strict_add_signed(rhs.0)),
            Index::Addr(i) => Index::Addr(i.strict_add_signed(rhs.0)),
        }
    }
}

impl std::str::FromStr for Index {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            if let Some((tag, i)) = stripped_s.split_at_checked(1) {
                match tag {
                    "e" => Ok(Index::Expr(i.parse()?)),
                    "n" => Ok(Index::Nat(i.parse()?)),
                    "d" => Ok(Index::Data(i.parse()?)),
                    "a" => Ok(Index::Addr(i.parse()?)),
                    _ => Err(IndexError::ImproperTag(stripped_s.to_owned())),
                }
            } else {
                Err(IndexError::MissingTag(stripped_s.to_owned()))
            }
        } else {
            Err(IndexError::MissingPercentPrefix(s.to_owned()))
        }
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Index::Expr(i) => write!(f, "%e{i}"),
            Index::Nat(i) => write!(f, "%n{i}"),
            Index::Data(i) => write!(f, "%d{i}"),
            Index::Addr(i) => write!(f, "%a{i}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct Shift(i32);

impl Shift {
    pub fn up() -> Self {
        Self(1)
    }

    pub fn down() -> Self {
        Self(-1)
    }
}

impl TryFrom<i32> for Shift {
    type Error = IndexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(IndexError::ZeroShift);
        }
        Ok(Shift(value))
    }
}

impl std::fmt::Display for Shift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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
