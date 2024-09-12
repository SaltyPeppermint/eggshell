use pyo3::prelude::*;

use super::macros::monomorphize;

#[pyfunction]
pub fn sketch_symbols() -> Vec<(String, usize)> {
    vec![
        ("?".to_owned(), 0),
        ("contains".to_owned(), 1),
        ("or".to_owned(), 2),
    ]
}

#[pyfunction]
pub fn open_symbol() -> (String, usize) {
    ("[open]".to_owned(), 0)
}

#[pyfunction]
pub fn active_symbol() -> (String, usize) {
    ("[active]".to_owned(), 0)
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
