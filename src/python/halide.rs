use std::fmt::Debug;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PyEqsatResult;
use crate::eqsat;
use crate::eqsat::results;
use crate::errors::EggShellError;
use crate::trs::halide::Halide;
use crate::trs::Trs;

/// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
#[pyclass]
#[derive(Debug, Clone)]
pub struct Eqsat(eqsat::Eqsat<Halide>);

#[pymethods]
impl Eqsat {
    /// Set up a new equality staturation with the term.
    ///
    /// # Errors
    ///
    /// Will error if the start_term could not be parsed.
    /// For more, see [`Eqsat`]
    #[new]
    #[pyo3(signature = (index, **py_kwargs))]
    fn new(index: usize, py_kwargs: Option<&Bound<'_, PyDict>>) -> Result<Self, PyErr> {
        let mut eqsat = eqsat::Eqsat::new(index);
        if let Some(bound) = py_kwargs {
            if let Some(time_limit) = bound.get_item("time_limit")? {
                let t = time_limit.extract()?;
                eqsat = eqsat.with_time_limit(Some(Duration::from_secs_f64(t)));
            }
            if let Some(phase_limit) = bound.get_item("phase_limit")? {
                let p = phase_limit.extract()?;
                eqsat = eqsat.with_phase_limit(Some(p));
            }
            if let Some(runner_args) = bound.get_item("phase_limit")? {
                let r = runner_args.extract()?;
                eqsat = eqsat.with_runner_args(r);
            }
        }
        Ok(Self(eqsat))
    }

    // /// See [`Eqsat`]
    // fn set_iteration_check(&mut self, iteration_check: bool) {
    //     self.0.set_iteration_check(iteration_check);
    // }
    #[allow(clippy::missing_errors_doc)]
    fn prove_once(&mut self, start_term: &str) -> Result<PyEqsatResult, PyErr> {
        let start_expr = start_term
            .parse()
            .map_err(|_| EggShellError::TermParse(start_term.into()))?;
        let goals = Halide::prove_goals();
        let r = match self.0.run_goal_once(&start_expr, &goals) {
            results::EqsatResult::Solved(result) => PyEqsatResult::Solved { result },
            results::EqsatResult::Undecidable => PyEqsatResult::Undecidable {},
            results::EqsatResult::LimitReached(egraph) => PyEqsatResult::LimitReached {
                egraph_serialized: format!("{:#?}", egraph.dump()),
            },
        };
        Ok(r)
    }

    #[allow(clippy::missing_errors_doc)]
    fn simplify_once(&mut self, start_term: &str) -> Result<PyEqsatResult, PyErr> {
        let start_expr = start_term
            .parse()
            .map_err(|_| EggShellError::TermParse(start_term.into()))?;
        let remaining = self.0.run_simplify_once(&start_expr);
        Ok(PyEqsatResult::LimitReached {
            egraph_serialized: format!("{:#?}", remaining.dump()),
        })
    }

    /// See [`Eqsat`]
    #[must_use]
    fn limit_reached(&self) -> bool {
        self.0.limit_reached()
    }

    #[must_use]
    fn stats_history(&self) -> Vec<results::EqsatStats> {
        self.0.stats_history().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_stmt = "( == 0 0 )";
        let mut eqsat = Eqsat::new(0, None).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyEqsatResult::Solved { result: "1".into() }, result);
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_stmt = "( == 1 0 )";
        let mut eqsat = Eqsat::new(0, None).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyEqsatResult::Solved { result: "0".into() }, result);
    }

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt = "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )";
        let mut eqsat = Eqsat::new(0, None).unwrap();
        let result = eqsat.prove_once(true_stmt).unwrap();
        assert_eq!(PyEqsatResult::Solved { result: "1".into() }, result);
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )";
        let mut eqsat = Eqsat::new(0, None).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyEqsatResult::Solved { result: "0".into() }, result);
    }
}
