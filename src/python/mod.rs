mod monomorphize;

use std::fmt::Display;

use egg::FromOp;
use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EggshellError<L>
where
    L::Error: Display,
    L: Display + FromOp,
{
    #[error(transparent)]
    BadRecExprParse(#[from] egg::RecExprParseError<egg::FromOpError>),
    #[error(transparent)]
    BadFromOp(#[from] egg::FromOpError),
    #[error(transparent)]
    BadSketchParse(#[from] egg::RecExprParseError<crate::sketch::SketchError<L>>),
}

create_exception!(
    eggshell,
    EggshellException,
    PyException,
    "Eggshell internal error."
);

impl<L> From<EggshellError<L>> for PyErr
where
    L::Error: Display,
    L: Display + FromOp,
{
    fn from(err: EggshellError<L>) -> PyErr {
        EggshellException::new_err(err.to_string())
    }
}

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
    simple::add_mod(m, "simple")?;
    arithmetic::add_mod(m, "arithmetic")?;
    halide::add_mod(m, "halide")?;
    rise::add_mod(m, "rise")?;

    m.add("EggshellException", m.py().get_type::<EggshellException>())?;
    // m.add_class::<probabilistic::FirstErrorDistance>()?;

    Ok(())
}
