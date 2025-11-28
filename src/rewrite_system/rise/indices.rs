use std::fmt::Display;

use egg::Var;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
// pub struct Index(u32);

#[derive(Debug, PartialEq, Eq, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum Index {
    Expr(u32),
    Nat(u32),
    Data(u32),
    Addr(u32),
    Synthetic(u32),
}

impl Index {
    pub fn new(i: u32) -> Self {
        Self::Synthetic(i)
    }

    pub fn zero() -> Self {
        Index::Synthetic(0)
    }

    pub fn upshifted(self) -> Self {
        self + Shift::up()
    }

    pub fn downshifted(self) -> Self {
        self + Shift::down()
    }

    fn value(self) -> u32 {
        match self {
            Index::Expr(i)
            | Index::Nat(i)
            | Index::Data(i)
            | Index::Addr(i)
            | Index::Synthetic(i) => i,
        }
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl PartialOrd for Index {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Index {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value().cmp(&other.value())
    }
}

impl std::ops::Add<Shift> for Index {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        let v = |i: u32| i.checked_add_signed(rhs.0).unwrap();

        match self {
            Index::Expr(i) => Index::Expr(v(i)),
            Index::Nat(i) => Index::Nat(v(i)),
            Index::Data(i) => Index::Data(v(i)),
            Index::Addr(i) => Index::Addr(v(i)),
            Index::Synthetic(i) => Index::Synthetic(v(i)),
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
            Index::Synthetic(i) => write!(f, "%synthetic{i}"),
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
