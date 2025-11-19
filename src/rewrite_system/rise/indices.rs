use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy, Serialize, Deserialize)]
pub struct TypedIndex {
    index: u32,
    ty: IndexType,
}

impl TypedIndex {
    pub fn new(value: u32, ty: IndexType) -> Self {
        TypedIndex { index: value, ty }
    }

    pub fn value(self) -> u32 {
        self.index
    }

    pub fn ty(self) -> IndexType {
        self.ty
    }
}

impl std::str::FromStr for TypedIndex {
    type Err = IndexParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(stripped_s) = s.strip_prefix("%") {
            if let Some((ty_char, rest)) = stripped_s.split_at_checked(0) {
                let ty = ty_char.parse()?;
                let i = rest.parse()?;
                Ok(TypedIndex { index: i, ty })
            } else {
                Err(IndexParseError::MissingTypePrefix)
            }
        } else {
            Err(IndexParseError::MissingPercentPrefix)
        }
    }
}

impl std::fmt::Display for TypedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}{}", self.ty, self.index)
    }
}

#[derive(
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Hash,
    Copy,
    Serialize,
    Deserialize,
    EnumString,
    Display,
)]
pub enum IndexType {
    #[strum(serialize = "nat", serialize = "n")]
    Nat,
    #[strum(serialize = "data", serialize = "d")]
    Data,
    #[strum(serialize = "addr", serialize = "a")]
    Addr,
    #[strum(serialize = "var", serialize = "v")]
    Var,
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
