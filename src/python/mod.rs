pub mod halide;

use pyo3::prelude::*;
use serde::Serialize;

use crate::errors::EggShellError;

#[pyclass]
#[derive(PartialEq, Debug, Clone, Serialize)]
pub enum PyEqsatResult {
    Solved { result: String },
    Undecidable {},
    LimitReached { egraph_serialized: String },
}

#[pymethods]
impl PyEqsatResult {
    /// Returns `true` if the py prove result is [`Solved`].
    ///
    /// [`Solved`]: PyProveResult::Solved
    #[must_use]
    pub fn is_solved(&self) -> bool {
        matches!(self, Self::Solved { .. })
    }

    /// Returns `true` if the py prove result is [`Undecidable`].
    ///
    /// [`Undecidable`]: PyProveResult::Undecidable
    #[must_use]
    pub fn is_undecidable(&self) -> bool {
        matches!(self, Self::Undecidable { .. })
    }

    /// Returns `true` if the py prove result is [`LimitReached`].
    ///
    /// [`LimitReached`]: PyProveResult::LimitReached
    #[must_use]
    pub fn is_limit_reached(&self) -> bool {
        matches!(self, Self::LimitReached { .. })
    }

    /// Returns the type as a string
    #[must_use]
    pub fn type_str(&self) -> String {
        match self {
            PyEqsatResult::Solved { result: _ } => "Solved".into(),
            PyEqsatResult::Undecidable {} => "Undecidable".into(),
            PyEqsatResult::LimitReached {
                egraph_serialized: _,
            } => "LimitReached".into(),
        }
    }

    /// Returns the content of limit reched
    #[allow(clippy::type_complexity)]
    pub fn unpack_limit_reached(&self) -> Result<String, EggShellError> {
        if let Self::LimitReached { egraph_serialized } = self {
            return Ok(egraph_serialized.to_owned());
        }
        Err(EggShellError::TupleUnpacking("LimitReached".into()))
    }

    /// Returns the content of limit reched
    pub fn unpack_solved(&self) -> Result<String, EggShellError> {
        if let Self::Solved { result } = self {
            return Ok(result.clone());
        }
        Err(EggShellError::TupleUnpacking("Solved".into()))
    }
}
