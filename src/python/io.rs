use pyo3::prelude::*;
use serde::Serialize;

use crate::errors::EggShellError;
use crate::io::structs::Expression;

#[pyfunction]
pub fn read_expressions(file_path: &str) -> Result<Vec<PyExpression>, EggShellError> {
    let exprs = crate::io::reader::read_expressions(file_path)?;
    Ok(exprs
        .into_iter()
        .map(|x| PyExpression { expr: x })
        .collect())
}

#[pyclass]
#[derive(PartialEq, Debug, Clone, Serialize)]
pub struct PyExpression {
    expr: Expression,
}

#[pymethods]
impl PyExpression {
    #[allow(clippy::cast_possible_wrap)]
    #[getter]
    pub fn index(&self) -> i64 {
        self.expr.index as i64
    }

    #[getter]
    pub fn term(&self) -> String {
        self.expr.term.to_string()
    }
}
