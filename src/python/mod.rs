mod err;
mod monomorphize;

use pyo3::prelude::*;

pub mod simple {
    super::monomorphize::monomorphize!(crate::rewrite_system::Simple, "eggshell.simple");
}

pub mod arithmetic {
    super::monomorphize::monomorphize!(crate::rewrite_system::Arithmetic, "eggshell.arithmetic");
}

pub mod halide {
    super::monomorphize::monomorphize!(crate::rewrite_system::Halide, "eggshell.halide");
}

pub mod rise {
    super::monomorphize::monomorphize!(crate::rewrite_system::Rise, "eggshell.rise");
}

/// A Python module implemented in Rust.
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    // m.add_class::<python:>()?;
    simple::add_mod(m, "simple")?;
    arithmetic::add_mod(m, "arithmetic")?;
    halide::add_mod(m, "halide")?;
    rise::add_mod(m, "rise")?;

    m.add(
        "EggshellException",
        m.py().get_type::<err::EggshellException>(),
    )?;
    // m.add_class::<tree_data::Node>()?;
    // m.add_class::<tree_data::TreeData>()?;
    // m.add_class::<probabilistic::FirstErrorDistance>()?;

    Ok(())
}
