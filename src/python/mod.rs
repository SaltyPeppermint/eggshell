// pub mod halide;
mod pyast;

use pyo3::prelude::*;

pub use pyast::*;

/// A Python module implemented in Rust.
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    // m.add_class::<python:>()?;
    pyast::simple::add_mod(m, "simple")?;
    pyast::arithmetic::add_mod(m, "arithmetic")?;
    pyast::halide::add_mod(m, "halide")?;
    pyast::rise::add_mod(m, "rise")?;

    m.add(
        "EggshellException",
        m.py().get_type::<crate::error::EggshellException>(),
    )?;

    Ok(())
}
