#![warn(clippy::all, clippy::pedantic)]

pub mod argparse;
pub mod baseline;
pub mod eqsat;
pub mod errors;
pub mod io;
mod python;
pub mod trs;

use pyo3::prelude::*;

use crate::eqsat::results::EqsatStats;
use crate::errors::EggShellException;
use crate::io::structs::Expression;

/// A Python module implemented in Rust.
#[allow(clippy::missing_errors_doc)]
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "EggShellException",
        m.py().get_type_bound::<EggShellException>(),
    )?;
    m.add_class::<EqsatStats>()?;

    // IO
    let io_m = PyModule::new_bound(m.py(), "io")?;
    io_m.add_function(wrap_pyfunction!(io::reader::read_expressions, m)?)?;
    io_m.add_class::<Expression>()?;
    m.add_submodule(&io_m)?;

    // General Eqsat
    let eqsat_m = PyModule::new_bound(m.py(), "eqsat")?;
    eqsat_m.add_class::<python::PyEqsatResult>()?;
    m.add_submodule(&eqsat_m)?;

    // Halide
    let halide_m = PyModule::new_bound(eqsat_m.py(), "halide")?;
    halide_m.add_class::<python::halide::Eqsat>()?;
    eqsat_m.add_submodule(&halide_m)?;

    Ok(())
}
