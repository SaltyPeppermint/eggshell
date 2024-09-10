use pyo3::prelude::*;

use super::macros::monomorphize;

#[pyfunction]
pub fn sketch_symbols() -> Vec<(String, usize)> {
    vec![("?".into(), 0), ("contains".into(), 1), ("or".into(), 2)]
}

#[pyfunction]
pub fn todo_symbol() -> (String, usize) {
    ("[todo]".into(), 0)
}

#[pyfunction]
pub fn active_symbol() -> (String, usize) {
    ("[active]".into(), 0)
}

pub mod arithmatic {
    super::monomorphize!(crate::trs::Arithmetic);
}

pub mod halide {
    super::monomorphize!(crate::trs::Halide);
}

pub mod simple {
    super::monomorphize!(crate::trs::Simple);
}
