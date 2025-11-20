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
}

impl std::str::FromStr for Index {
    type Err = IndexParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            let i = stripped_s.parse()?;
            Ok(Index(i))
        } else {
            Err(IndexParseError::MissingPercentPrefix)
        }
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl std::ops::Add<u32> for Index {
    type Output = Self;

    fn add(self, other: u32) -> Self {
        Index(self.0 + other)
    }
}

impl std::ops::Add<i32> for Index {
    type Output = Self;

    fn add(self, other: i32) -> Self {
        Index(self.0.checked_add_signed(other).unwrap())
    }
}

impl std::ops::Sub<u32> for Index {
    type Output = Self;

    fn sub(self, other: u32) -> Self {
        Index(self.0 - other)
    }
}

#[derive(Error, Debug)]
pub enum IndexParseError {
    #[error("Missing % Prefix")]
    MissingPercentPrefix,
    #[error("Missing Type Prefix")]
    MissingTypePrefix,
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}
