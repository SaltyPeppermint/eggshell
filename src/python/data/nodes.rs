use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use serde::Serialize;

use crate::trs::SymbolInfo;

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct Node {
    #[pyo3(get)]
    raw_name: String,
    #[pyo3(get)]
    arity: usize,
    #[pyo3(get)]
    nth_child: usize,
    #[pyo3(get)]
    dfs_order: usize,
    #[pyo3(get)]
    depth: usize,
    symbol_info: SymbolInfo,
}

impl Node {
    #[must_use]
    pub fn new(
        raw_name: String,
        arity: usize,
        nth_child: usize,
        dfs_order: usize,
        depth: usize,
        symbol_info: SymbolInfo,
    ) -> Self {
        Self {
            raw_name,
            arity,
            nth_child,
            dfs_order,
            depth,
            symbol_info,
        }
    }

    #[must_use]
    pub fn symbol_info(&self) -> &SymbolInfo {
        &self.symbol_info
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Node {
    #[must_use]
    #[getter]
    pub fn id(&self) -> usize {
        self.symbol_info.id()
    }

    #[must_use]
    #[getter]
    pub fn value(&self) -> Option<String> {
        self.symbol_info.value()
    }

    #[must_use]
    #[getter]
    pub fn name(&self) -> String {
        match self.symbol_info.symbol_type() {
            crate::trs::SymbolType::Constant(_) => "[constant]".to_owned(),
            crate::trs::SymbolType::Variable(_) => "[variable]".to_owned(),
            crate::trs::SymbolType::MetaSymbol | crate::trs::SymbolType::Operator => {
                self.raw_name.clone()
            }
        }
    }

    #[must_use]
    pub fn arity(&self) -> usize {
        self.arity
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }
}
