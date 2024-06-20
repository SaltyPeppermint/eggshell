pub mod halide;

use hashbrown::HashMap as HashBrownMap;
use pyo3::prelude::*;
use serde::Serialize;

use crate::errors::EggShellError;
use crate::flattened::Vertex;

#[pyclass]
#[derive(PartialEq, Debug, Clone, Serialize)]
pub enum EqsatResult {
    Solved {
        result: String,
    },
    Undecidable {},
    LimitReached {
        vertices: Vec<Vertex>,
        edges: HashBrownMap<usize, Vec<usize>>,
    },
}

#[pymethods]
impl EqsatResult {
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
            EqsatResult::Solved { result: _ } => "Solved".into(),
            EqsatResult::Undecidable {} => "Undecidable".into(),
            EqsatResult::LimitReached {
                vertices: _,
                edges: _,
            } => "LimitReached".into(),
        }
    }

    /// Returns the content of limit reched
    #[allow(clippy::type_complexity)]
    pub fn unpack_limit_reached(
        &self,
    ) -> Result<(Vec<Vertex>, HashBrownMap<usize, Vec<usize>>), EggShellError> {
        if let Self::LimitReached { vertices, edges } = self {
            return Ok((vertices.clone(), edges.clone()));
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
