use std::fmt::Display;
use std::str::FromStr;

use egg::{Language, RecExpr};
use pyo3::exceptions::PyException;
use pyo3::{create_exception, prelude::*};

use super::raw_lang::{RawLang, RawLangParseError};
use super::FlatAst;
use crate::utils::Tree;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PyLang(pub(crate) RawLang);

#[pymethods]
impl PyLang {
    #[new]
    pub fn new(node: String, children: Vec<PyLang>) -> Self {
        let raw_children = children.into_iter().map(|c| c.0).collect();
        let raw_lang = RawLang::new(node, raw_children);
        PyLang(raw_lang)
    }

    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let raw_lang = s_expr_str.parse()?;
        Ok(PyLang(raw_lang))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    pub fn flat(&self) -> FlatAst {
        (&self.0).into()
    }

    pub fn size(&self) -> usize {
        self.0.size()
    }
}

impl From<RawLang> for PyLang {
    fn from(value: RawLang) -> Self {
        PyLang(value)
    }
}

impl FromStr for PyLang {
    type Err = RawLangParseError;

    fn from_str(s: &str) -> Result<Self, RawLangParseError> {
        let raw_lang = s.parse()?;
        Ok(PyLang(raw_lang))
    }
}

impl<L: Language + Display> From<&RecExpr<L>> for PyLang {
    fn from(value: &RecExpr<L>) -> Self {
        let raw_lang = value.into();
        PyLang(raw_lang)
    }
}

create_exception!(
    eggshell,
    PyLangException,
    PyException,
    "Error parsing a PyLang."
);

impl From<RawLangParseError> for PyErr {
    fn from(err: RawLangParseError) -> PyErr {
        PyLangException::new_err(err.to_string())
    }
}
