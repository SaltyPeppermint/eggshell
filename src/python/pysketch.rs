use std::str::FromStr;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

use super::raw_sketch::{RawSketch, RawSketchError, RawSketchParseError};
use super::FlatAst;
use crate::utils::Tree;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PySketch(pub(crate) RawSketch);

#[pymethods]
impl PySketch {
    /// This always generates a new node that has [open] as its children
    #[new]
    fn new(node: &str, arity: usize) -> PyResult<Self> {
        let new_children = vec![RawSketch::Open; arity];
        let raw_sketch = RawSketch::new(node, new_children)?;
        Ok(PySketch(raw_sketch))
    }

    /// Generate a new root with an [active] node
    #[staticmethod]
    pub fn new_root() -> Self {
        PySketch(RawSketch::Active)
    }

    /// Parse from string
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

    /// Appends at the current [active] node and turns an open [open]
    /// into a new [active]
    /// Returns if the sketch is finished
    pub fn append(&mut self, new_child: Self) -> bool {
        self.0.append(new_child.0)
    }

    /// Returns a flat representation of itself
    pub fn flat(&self) -> FlatAst {
        (&self.0).into()
    }

    /// Returns the number of nodes in the sketch
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns the maximum AST depth in the sketch
    pub fn depth(&self) -> usize {
        self.0.depth()
    }

    /// Checks if sketch has open [active]
    pub fn finished(&self) -> bool {
        self.0.finished()
    }

    /// Checks if sketch has open [active]
    pub fn sketch_symbols(&self) -> usize {
        self.0.sketch_symbols()
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
