use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct Index(u32);

impl Index {
    pub fn new(i: u32) -> Self {
        Self(i)
    }

    pub fn zero() -> Self {
        Self(0)
    }

    pub fn upshifted(self) -> Self {
        self + Shift::up()
    }

    pub fn downshifted(self) -> Self {
        self + Shift::down()
    }
}

impl std::ops::Add<Shift> for Index {
    type Output = Self;

    fn add(self, rhs: Shift) -> Self::Output {
        Self(self.0.checked_add_signed(rhs.0).unwrap())
    }
}

impl std::str::FromStr for Index {
    type Err = IndexError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            let i = stripped_s.parse()?;
            Ok(Index(i))
        } else {
            Err(IndexError::MissingPercentPrefix)
        }
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
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

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Missing % Prefix")]
    MissingPercentPrefix,
    #[error("Invalide zero shift")]
    ZeroShift,
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
