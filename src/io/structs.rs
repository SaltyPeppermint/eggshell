// use pyo3::prelude::*;
use serde::Serialize;

/// Used to represent an expression
#[derive(Serialize, Debug, Clone, PartialEq)]
// #[pyclass(frozen)]
pub struct Entry {
    /// Index of the expression
    pub index: usize,
    /// the string of the expression
    pub expr: String,
    /// the truth value of the expression
    pub truth_value: Option<String>,
}

// #[pymethods]
impl Entry {
    #[expect(clippy::cast_possible_wrap)]
    #[must_use]
    // #[getter]
    pub fn index(&self) -> i64 {
        self.index as i64
    }

    #[must_use]
    // #[getter]
    pub fn expr(&self) -> String {
        self.expr.clone()
    }
}

/// Holds (optional) additional data of other solver results
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct OtherSolverData {
    /// Other Solvers's result for proving the expression
    pub result: String,
    /// The time it took the other solver to prove the expression
    pub time: f64,
}
