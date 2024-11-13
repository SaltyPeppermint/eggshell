use std::str::FromStr;

use egg::RecExpr;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

use super::raw_ast::{RawAstError, RawAstParseError};
use super::{FlatAst, RawAst};
use crate::sketch::{PartialSketch, Sketch};
use crate::trs::TrsLang;
use crate::utils::Tree;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PyAst(pub(crate) RawAst);

#[pymethods]
impl PyAst {
    /// This always generates a new node that has [open] as its children
    #[new]
    fn new(node: &str, arity: usize) -> PyResult<Self> {
        let new_children = vec![RawAst::Open; arity];
        let raw_sketch = RawAst::new(node, new_children)?;
        Ok(PyAst(raw_sketch))
    }

    /// Generate a new root with an [active] node
    #[staticmethod]
    pub fn new_root() -> Self {
        PyAst(RawAst::Active)
    }

    /// Parse from string
    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let raw_sketch = s_expr_str.parse().map_err(RawAstError::BadSexp)?;
        Ok(PyAst(raw_sketch))
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

    /// Checks if it is a sketch
    pub fn is_sketch(&self) -> bool {
        self.0.is_sketch()
    }

    /// Checks if it is a sketch
    pub fn is_partial_sketch(&self) -> bool {
        self.0.is_partial_sketch()
    }

    pub fn features(&self) -> Option<Vec<f64>> {
        self.0.features()
    }
}

impl From<RawAst> for PyAst {
    fn from(value: RawAst) -> Self {
        PyAst(value)
    }
}

impl<L: TrsLang> From<&RecExpr<L>> for PyAst {
    fn from(expr: &RecExpr<L>) -> Self {
        let raw_ast = expr.into();
        PyAst(raw_ast)
    }
}

impl<L: TrsLang> From<&PartialSketch<L>> for PyAst {
    fn from(sketch: &PartialSketch<L>) -> Self {
        let raw_partial_sketch = sketch.into();
        PyAst(raw_partial_sketch)
    }
}

impl<L: TrsLang> From<&Sketch<L>> for PyAst {
    fn from(sketch: &Sketch<L>) -> Self {
        let raw_sketch = sketch.into();
        PyAst(raw_sketch)
    }
}

impl FromStr for PyAst {
    type Err = RawAstParseError;

    fn from_str(s: &str) -> Result<Self, RawAstParseError> {
        let raw_sketch = s.parse()?;
        Ok(PyAst(raw_sketch))
    }
}

impl From<RawAstError> for PyErr {
    fn from(err: RawAstError) -> PyErr {
        PySketchException::new_err(err.to_string())
    }
}

create_exception!(
    eggshell,
    PySketchException,
    PyException,
    "Error dealing with a PySketch."
);
