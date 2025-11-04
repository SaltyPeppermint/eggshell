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

pub mod dummy_rise {
    use crate::rewrite_system::dummy_rise;
    super::monomorphize::monomorphize!(
        dummy_rise::DummyRiseLang,
        dummy_rise::DummyRiseAnalysis,
        dummy_rise::full_rules(),
        "eggshell.dummy_rise"
    );
}

pub mod herbie {
    use crate::rewrite_system::herbie;
    super::monomorphize::monomorphize!(
        herbie::Math,
        herbie::ConstantFold,
        herbie::rules(herbie::HerbieRules::Ruleset242),
        "eggshell.herbie"
    );
}

/// A Python module implemented in Rust.
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    simple::add_mod(m, "simple")?;
    arithmetic::add_mod(m, "arithmetic")?;
    halide::add_mod(m, "halide")?;
    dummy_rise::add_mod(m, "dummy_rise")?;
    herbie::add_mod(m, "herbie")?;

    m.add("EggshellException", m.py().get_type::<EggshellException>())?;

    Ok(())
}
