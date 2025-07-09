use serde::Serialize;

use crate::rewrite_system::RewriteSystemError;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum HalideData {
    Int(i64),
    Bool(bool),
}

impl TryFrom<HalideData> for i64 {
    type Error = RewriteSystemError;

    fn try_from(value: HalideData) -> Result<Self, Self::Error> {
        match value {
            HalideData::Int(x) => Ok(x),
            HalideData::Bool(_) => Err(RewriteSystemError::BadAnalysis(format!(
                "Tried to use {value:?} as integer"
            ))),
        }
    }
}

impl TryFrom<HalideData> for bool {
    type Error = RewriteSystemError;

    fn try_from(value: HalideData) -> Result<Self, Self::Error> {
        match value {
            HalideData::Int(_) => Err(RewriteSystemError::BadAnalysis(format!(
                "Tried to use {value:?} as bool"
            ))),
            HalideData::Bool(x) => Ok(x),
        }
    }
}

impl From<i64> for HalideData {
    fn from(value: i64) -> Self {
        HalideData::Int(value)
    }
}

impl From<bool> for HalideData {
    fn from(value: bool) -> Self {
        HalideData::Bool(value)
    }
}
