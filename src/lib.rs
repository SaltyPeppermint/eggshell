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

use io::structs::Expression;
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
    m.add_class::<Vertex>()?;
    m.add_class::<EqsatStats>()?;

    // IO
    let io_m = PyModule::new_bound(m.py(), "io")?;
    io_m.add_function(wrap_pyfunction!(io::reader::read_expressions, m)?)?;
    io_m.add_class::<Expression>()?;
    m.add_submodule(&io_m)?;

    // Halide
    let halide_m = PyModule::new_bound(m.py(), "eqsat")?;
    halide_m.add_class::<python::halide::Eqsat>()?;
    halide_m.add_class::<python::halide::EqsatResult>()?;
    halide_m.add_function(wrap_pyfunction!(
        python::halide::extract_with_costs_bottom_up,
        m
    )?)?;
    halide_m.add_function(wrap_pyfunction!(
        python::halide::extract_ast_size_bottom_up,
        m
    )?)?;

    m.add_submodule(&halide_m)?;

    // EXTRACTION
    let extract_m = PyModule::new_bound(m.py(), "extraction")?;

    m.add_submodule(&extract_m)?;

    Ok(())
}
