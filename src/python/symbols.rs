use indexmap::IndexMap;
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, PartialEq)]
pub struct SymbolTable {
    lut: IndexMap<String, SymbolMetaData>,
}

impl SymbolTable {
    /// Generates new Symbol table for TRS
    #[must_use]
    pub fn new(lut: IndexMap<String, SymbolMetaData>) -> Self {
        Self { lut }
    }
}

#[pymethods]
impl SymbolTable {
    pub fn add_sketch_symbols(&mut self) {
        self.lut
            .insert("?".to_owned(), SymbolMetaData::Sketch { arity: 0 });
        self.lut
            .insert("or".to_owned(), SymbolMetaData::Sketch { arity: 2 });
        self.lut
            .insert("contains".to_owned(), SymbolMetaData::Sketch { arity: 1 });
    }

    pub fn add_partial_symbols(&mut self) {
        self.lut
            .insert("[open]".to_owned(), SymbolMetaData::Partial);
        self.lut
            .insert("[active]".to_owned(), SymbolMetaData::Partial);
    }

    /// Get a Symbol from the table by name
    ///
    /// # Errors
    /// If Symbol not in the table
    pub fn get_symbol(&self, name: &str) -> PyResult<Symbol> {
        if let Ok(int_value) = name.parse() {
            Ok(Symbol {
                name: name.to_owned(),
                arity: 0,
                ion: IoN::Number(int_value),
            })
        } else {
            let (index, _, metadata) =
                self.lut
                    .get_full(name)
                    .ok_or(PyValueError::new_err(format!(
                        "No Symbol with the name {name}"
                    )))?;
            Ok(Symbol {
                name: name.to_owned(),
                arity: metadata.arity(),
                ion: IoN::Index(index),
            })
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.lut.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lut.is_empty()
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(frozen)]
#[derive(Debug, PartialEq)]
pub struct Symbol {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    arity: usize,
    ion: IoN,
}

#[derive(Debug, PartialEq)]
enum IoN {
    Index(usize),
    Number(f32),
}

impl Symbol {
    #[must_use]
    pub fn to_feature_vec(
        &self,
        f_vec_len: usize,
        n_symbols: usize,
        int_bounds: (f32, f32),
    ) -> Vec<f32> {
        let mut v = vec![0.0; f_vec_len];
        match self.ion {
            IoN::Index(idx) => {
                v[idx] = 1.0;
            }
            IoN::Number(int_value) => {
                let normalized = (int_value - int_bounds.0) / (int_bounds.1 - int_bounds.0);
                v[n_symbols] = normalized;
            }
        }
        v
    }
}

#[pymethods]
impl Symbol {
    #[must_use]
    pub fn is_num(&self) -> bool {
        matches!(self.ion, IoN::Number(_))
    }

    #[must_use]
    pub fn to_feature_np<'py>(
        &self,
        py: Python<'py>,
        f_vec_len: usize,
        n_symbols: usize,
        int_bounds: (f32, f32),
    ) -> Bound<'py, PyArray<f32, Ix1>> {
        self.to_feature_vec(f_vec_len, n_symbols, int_bounds)
            .into_pyarray_bound(py)
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[derive(Hash, Debug, PartialEq, Eq)]
pub enum SymbolMetaData {
    Lang { arity: usize },
    Sketch { arity: usize },
    Partial,
    Numerical,
}

impl SymbolMetaData {
    fn arity(&self) -> usize {
        match self {
            SymbolMetaData::Lang { arity } | SymbolMetaData::Sketch { arity } => *arity,
            SymbolMetaData::Partial | SymbolMetaData::Numerical => 0,
        }
    }
}
