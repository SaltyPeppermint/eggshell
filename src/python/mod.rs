mod monomorphize;

use std::fmt::Display;

use egg::FromOp;
use pyo3::PyErr;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use thiserror::Error;

use crate::sketch::SketchError;

#[derive(Debug, Error)]
pub enum EggshellError<L>
where
    L::Error: Display,
    L: Display + FromOp,
{
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<egg::FromOpError>),
    #[error(transparent)]
    FromOp(#[from] egg::FromOpError),
    #[error(transparent)]
    SketchParse(#[from] egg::RecExprParseError<SketchError<L>>),
}

pyo3::create_exception!(
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
    use crate::rewrite_system::simple;
    super::monomorphize::monomorphize!(simple::SimpleLang, (), simple::rules(), "eggshell.simple");
}

pub mod arithmetic {
    use crate::rewrite_system::arithmetic;
    super::monomorphize::monomorphize!(
        arithmetic::Math,
        arithmetic::ConstantFold,
        arithmetic::rules(),
        "eggshell.arithmetic"
    );
}

pub mod halide {
    use crate::rewrite_system::halide;
    super::monomorphize::monomorphize!(
        halide::HalideLang,
        halide::ConstantFold,
        halide::rules(halide::HalideRuleset::Full),
        "eggshell.halide"
    );
}

pub mod rise {
    use crate::rewrite_system::rise;
    super::monomorphize::monomorphize!(
        rise::RiseLang,
        rise::RiseAnalysis,
        rise::full_rules(),
        "eggshell.rise"
    );
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
