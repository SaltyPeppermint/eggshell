use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct Index(pub u32);

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
    #[error("Invalide Type Prefix: {0}")]
    InvalideTypePrefix(#[from] strum::ParseError),
    #[error("Invalide Index: {0}")]
    InvalidIndex(#[from] std::num::ParseIntError),
}

// macro_rules! mk_index_type {
//     ($type_name: tt, $prefix: tt) => {
//         #[derive(
//             Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize,
//         )]
//         pub struct $type_name(pub u32);

//         impl std::str::FromStr for $type_name {
//             type Err = Option<std::num::ParseIntError>;

//             fn from_str(s: &str) -> Result<Self, Self::Err> {
//                 if let Some(stripped_s) = s.strip_prefix(concat!("%", $prefix)) {
//                     stripped_s.parse().map($type_name).map_err(Some)
//                 } else {
//                     Err(None)
//                 }
//             }
//         }

//         impl std::fmt::Display for $type_name {
//             fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//                 write!(f, concat!("%", $prefix, "{}"), self.0)
//             }
//         }
//     };
// }

// mk_index_type!(NatIndex, "n");
// mk_index_type!(DataIndex, "d");
// mk_index_type!(AddrIndex, "a");
// mk_index_type!(VarIndex, "v");
