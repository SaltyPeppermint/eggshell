use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{Kind, Kindable};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub enum DBIndex {
    Expr(u32),
    Nat(u32),
    Data(u32),
    Addr(u32),
}

impl DBIndex {
    pub fn inc(self) -> Self {
        self + DBShift::up()
    }

    pub fn dec(self) -> Self {
        self + DBShift::down()
    }

    pub fn new(value: u32, kind: Kind) -> Self {
        match kind {
            Kind::Expr => DBIndex::Expr(value),
            Kind::Nat => DBIndex::Nat(value),
            Kind::Data => DBIndex::Data(value),
            Kind::Addr => DBIndex::Addr(value),
        }
    }

    pub fn zero(kind: Kind) -> Self {
        match kind {
            Kind::Expr => DBIndex::Expr(0),
            Kind::Nat => DBIndex::Nat(0),
            Kind::Data => DBIndex::Data(0),
            Kind::Addr => DBIndex::Addr(0),
        }
    }

    pub fn value(self) -> u32 {
        match self {
            DBIndex::Expr(i) | DBIndex::Nat(i) | DBIndex::Data(i) | DBIndex::Addr(i) => i,
        }
    }

    pub fn is_zero(self) -> bool {
        self.value() == 0
    }
}

impl std::ops::Add<DBShift> for DBIndex {
    type Output = Self;

    fn add(self, rhs: DBShift) -> Self::Output {
        match self {
            DBIndex::Expr(i) => DBIndex::Expr(i.strict_add_signed(rhs.0)),
            DBIndex::Nat(i) => DBIndex::Nat(i.strict_add_signed(rhs.0)),
            DBIndex::Data(i) => DBIndex::Data(i.strict_add_signed(rhs.0)),
            DBIndex::Addr(i) => DBIndex::Addr(i.strict_add_signed(rhs.0)),
        }
    }
}

impl std::str::FromStr for DBIndex {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            if let Some((tag, i)) = stripped_s.split_at_checked(1) {
                match tag {
                    "e" => Ok(DBIndex::Expr(i.parse()?)),
                    "n" => Ok(DBIndex::Nat(i.parse()?)),
                    "d" => Ok(DBIndex::Data(i.parse()?)),
                    "a" => Ok(DBIndex::Addr(i.parse()?)),
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

impl std::fmt::Display for DBIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DBIndex::Expr(i) => write!(f, "%e{i}"),
            DBIndex::Nat(i) => write!(f, "%n{i}"),
            DBIndex::Data(i) => write!(f, "%d{i}"),
            DBIndex::Addr(i) => write!(f, "%a{i}"),
        }
    }
}

impl Kindable for DBIndex {
    fn kind(&self) -> Kind {
        match self {
            DBIndex::Expr(_) => Kind::Expr,
            DBIndex::Nat(_) => Kind::Nat,
            DBIndex::Data(_) => Kind::Data,
            DBIndex::Addr(_) => Kind::Addr,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct DBShift(i32);

impl DBShift {
    pub fn up() -> Self {
        Self(1)
    }

    pub fn down() -> Self {
        Self(-1)
    }
}

impl TryFrom<i32> for DBShift {
    type Error = IndexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(IndexError::ZeroShift);
        }
        Ok(DBShift(value))
    }
}

impl std::fmt::Display for DBShift {
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
