#![warn(clippy::all, clippy::pedantic)]

mod eqsat;
mod errors;
mod io;
mod python;
mod sketches;
mod trs;
mod utils;

use pyo3::prelude::*;

// use crate::eqsat::results::EqsatStats;
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

    // IO
    let io_m = PyModule::new_bound(m.py(), "io")?;
    io_m.add_function(wrap_pyfunction!(io::reader::read_exprs, m)?)?;
    io_m.add_class::<Expression>()?;
    m.add_submodule(&io_m)?;

    // Eqsat
    let eqsat_m = PyModule::new_bound(m.py(), "eqsat")?;

    // Arithmatic specific generated eqsats
    let arithmatic_m = PyModule::new_bound(eqsat_m.py(), "arithmatic")?;
    arithmatic_m.add_class::<python::arithmatic::NewEqsat>()?;
    arithmatic_m.add_class::<python::arithmatic::FinishedEqsat>()?;
    eqsat_m.add_submodule(&arithmatic_m)?;

    // Halide specific generated eqsats
    let halide_m = PyModule::new_bound(eqsat_m.py(), "halide")?;
    halide_m.add_class::<python::halide::NewEqsat>()?;
    halide_m.add_class::<python::halide::FinishedEqsat>()?;
    eqsat_m.add_submodule(&halide_m)?;

    // SimpleLang specific generated eqsats
    let simple_m = PyModule::new_bound(eqsat_m.py(), "simple")?;
    simple_m.add_class::<python::simple::NewEqsat>()?;
    simple_m.add_class::<python::simple::FinishedEqsat>()?;
    eqsat_m.add_submodule(&simple_m)?;

    // Pylang works for all langs that implement Display
    eqsat_m.add_class::<python::PyLang>()?;
    m.add_submodule(&eqsat_m)?;

    Ok(())
}
