use std::str::FromStr;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

use super::raw_sketch::{RawSketchError, RawSketchParseError};
use super::{FlatAst, RawSketch};

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PySketch(pub(crate) RawSketch);

#[pymethods]
impl PySketch {
    #[new]
    #[pyo3(signature = (node_type, children=vec![]))]
    fn new(node_type: &str, children: Vec<PySketch>) -> PyResult<Self> {
        let raw_children = children.into_iter().map(|c| c.0).collect();
        let raw_sketch = RawSketch::new(node_type, raw_children)?;
        Ok(PySketch(raw_sketch))
    }

    /// You will probably want to use this
    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let raw_sketch = s_expr_str.parse().map_err(RawSketchError::BadSexp)?;
        Ok(PySketch(raw_sketch))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    pub fn replace_child(&mut self, new_child: Self) {
        self.0.replace_child(new_child.0);
    }

    pub fn flat(&self) -> FlatAst {
        (&self.0).into()
    }
}

impl From<RawSketch> for PySketch {
    fn from(value: RawSketch) -> Self {
        PySketch(value)
    }
}

impl FromStr for PySketch {
    type Err = RawSketchParseError;

    fn from_str(s: &str) -> Result<Self, RawSketchParseError> {
        let raw_sketch = s.parse()?;
        Ok(PySketch(raw_sketch))
    }
}

create_exception!(
    eggshell,
    PySketchException,
    PyException,
    "Error dealing with a PySketch."
);

impl From<RawSketchError> for PyErr {
    fn from(err: RawSketchError) -> PyErr {
        PySketchException::new_err(err.to_string())
    }
}
