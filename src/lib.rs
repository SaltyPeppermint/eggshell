#![warn(clippy::all, clippy::pedantic)]

pub mod argparse;
pub mod baseline;
pub mod cost_fn;
pub mod eqsat;
pub mod errors;
pub mod extraction;
pub mod flattened;
pub mod io;
mod python;
pub mod trs;
mod utils;

use pyo3::prelude::*;

use crate::eqsat::results::EqsatStats;
use crate::errors::EggShellException;
use crate::flattened::Vertex;

/// A Python module implemented in Rust.
#[allow(clippy::missing_errors_doc)]
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "EggShellException",
        m.py().get_type_bound::<EggShellException>(),
    )?;

    // IO
    let io_m = PyModule::new_bound(m.py(), "io")?;
    io_m.add_function(wrap_pyfunction!(python::io::read_expressions, m)?)?;
    io_m.add_class::<python::io::PyExpression>()?;
    m.add_submodule(&io_m)?;

    // EQSAT
    let eqsat_m = PyModule::new_bound(m.py(), "eqsat")?;
    eqsat_m.add_class::<python::eqsat::PyEqsatHalide>()?;
    eqsat_m.add_class::<python::eqsat::PyProveResult>()?;
    eqsat_m.add_class::<Vertex>()?;
    eqsat_m.add_class::<EqsatStats>()?;
    m.add_submodule(&eqsat_m)?;

    // EXTRACTION
    let extract_m = PyModule::new_bound(m.py(), "extraction")?;
    extract_m.add_function(wrap_pyfunction!(
        python::extraction::extract_with_costs_bottom_up,
        m
    )?)?;
    extract_m.add_function(wrap_pyfunction!(
        python::extraction::extract_ast_size_bottom_up,
        m
    )?)?;
    m.add_submodule(&extract_m)?;

    Ok(())
}
