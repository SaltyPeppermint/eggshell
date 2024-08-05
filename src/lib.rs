#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::redundant_closure_for_method_calls,
    clippy::module_name_repetitions
)]

mod eqsat;
mod errors;
mod io;
mod python;
mod sketch;
mod trs;
mod utils;

use pyo3::prelude::*;

use crate::errors::EggShellException;
use crate::io::structs::Expression;

type HashMap<K, V> = hashbrown::HashMap<K, V>;
type HashSet<T> = hashbrown::HashSet<T>;
// type HashMap<K, V> = rustc_hash::crate::HashMap<K, V>;
// type HashSet<K, V> = rustc_hash::crate::HashSet<K, V>;

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
    arithmatic_m.add_class::<python::arithmatic::Eqsat>()?;
    arithmatic_m.add_class::<python::arithmatic::EqsatResult>()?;
    eqsat_m.add_submodule(&arithmatic_m)?;

    // Halide specific generated eqsats
    let halide_m = PyModule::new_bound(eqsat_m.py(), "halide")?;
    halide_m.add_class::<python::halide::Eqsat>()?;
    halide_m.add_class::<python::halide::EqsatResult>()?;
    eqsat_m.add_submodule(&halide_m)?;

    // SimpleLang specific generated eqsats
    let simple_m = PyModule::new_bound(eqsat_m.py(), "simple")?;
    simple_m.add_class::<python::simple::Eqsat>()?;
    simple_m.add_class::<python::simple::EqsatResult>()?;
    eqsat_m.add_submodule(&simple_m)?;

    // Pylang works for all langs that implement Display
    eqsat_m.add_class::<python::PyLang>()?;
    m.add_submodule(&eqsat_m)?;

    Ok(())
}
