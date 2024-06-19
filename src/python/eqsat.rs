use std::time::Duration;

use hashbrown::HashMap as HashBrownMap;
use pyo3::prelude::*;
use serde::Serialize;

use crate::eqsat::results::{EqsatStats, EqsatResult};
use crate::eqsat::utils::RunnerArgs;
use crate::eqsat::Eqsat;
use crate::errors::EggShellError;
use crate::flattened::Vertex;
use crate::trs::halide::Halide;
use crate::trs::Trs;

/// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
/// for the Halide Trs
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyEqsatHalide {
    pub(crate) eqsat: Eqsat<Halide>,
}

#[pymethods]
impl PyEqsatHalide {
    #[new]
    /// Set up a new equality staturation with the term.
    ///
    /// # Errors
    ///
    /// Will error if the start_term could not be parsed.
    /// For more, see [`Eqsat`]
    fn new(index: usize) -> Result<Self, EggShellError> {
        let eqsat = Eqsat::new(index)?;
        Ok(Self { eqsat })
    }

    /// See [`Eqsat`]
    fn set_phase_limit(&mut self, phase_limit: usize) {
        self.eqsat.set_phase_limit(Some(phase_limit));
    }

    /// See [`Eqsat`]
    fn unset_phase_limit(&mut self) {
        self.eqsat.set_phase_limit(None);
    }

    /// See [`Eqsat`]
    fn set_time_limit(&mut self, time_limit: f64) {
        self.eqsat
            .set_time_limit(Some(Duration::from_secs_f64(time_limit)));
    }

    /// See [`Eqsat`]
    fn unset_time_limit(&mut self) {
        self.eqsat.set_time_limit(None);
    }

    /// See [`Eqsat`]
    fn set_runner_arg_values(
        &mut self,
        iter: Option<usize>,
        nodes: Option<usize>,
        time: Option<f64>,
    ) {
        self.eqsat.set_runner_arg_values(iter, nodes, time);
    }

    /// See [`Eqsat`]
    fn set_runner_args(&mut self, runner_args: RunnerArgs) {
        self.eqsat.set_runner_args(runner_args);
    }

    /// See [`Eqsat`]
    fn set_iteration_check(&mut self, iteration_check: bool) {
        self.eqsat.set_iteration_check(iteration_check);
    }

    #[allow(clippy::missing_errors_doc)]
    fn prove_once(&mut self, start_term: &str) -> Result<PyProveResult, EggShellError> {
        let start_expr = start_term
            .parse()
            .map_err(|_| EggShellError::TermParse(start_term.into()))?;
        let goals = Halide::prove_goals();
        let r = match self.eqsat.run_goal_once(&start_expr, &goals) {
            EqsatResult::Solved(result) => PyProveResult::Solved { result },
            EqsatResult::Undecidable => PyProveResult::Undecidable {},
            EqsatResult::LimitReached(remaining) => PyProveResult::LimitReached {
                edges: remaining.edges,
                vertices: remaining.vertices,
            },
        };
        Ok(r)
    }

    #[allow(clippy::missing_errors_doc)]
    fn simplify_once(&mut self, start_term: &str) -> Result<PyProveResult, EggShellError> {
        let start_expr = start_term
            .parse()
            .map_err(|_| EggShellError::TermParse(start_term.into()))?;
        let remaining = self.eqsat.run_simplify_once(&start_expr);
        Ok(PyProveResult::LimitReached {
            edges: remaining.edges,
            vertices: remaining.vertices,
        })
    }

    /// See [`Eqsat`]
    #[must_use]
    fn limit_reached(&self) -> bool {
        self.eqsat.limit_reached()
    }

    #[must_use]
    fn stats_history(&self) -> Vec<EqsatStats> {
        self.eqsat.stats_history().to_vec()
    }
}

#[pyclass]
#[derive(PartialEq, Debug, Clone, Serialize)]
pub enum PyProveResult {
    Solved {
        result: String,
    },
    Undecidable {},
    LimitReached {
        vertices: Vec<Vertex>,
        edges: HashBrownMap<usize, Vec<usize>>,
    },
}

#[pymethods]
impl PyProveResult {
    /// Returns `true` if the py prove result is [`Solved`].
    ///
    /// [`Solved`]: PyProveResult::Solved
    #[must_use]
    pub fn is_solved(&self) -> bool {
        matches!(self, Self::Solved { .. })
    }

    /// Returns `true` if the py prove result is [`Undecidable`].
    ///
    /// [`Undecidable`]: PyProveResult::Undecidable
    #[must_use]
    pub fn is_undecidable(&self) -> bool {
        matches!(self, Self::Undecidable { .. })
    }

    /// Returns `true` if the py prove result is [`LimitReached`].
    ///
    /// [`LimitReached`]: PyProveResult::LimitReached
    #[must_use]
    pub fn is_limit_reached(&self) -> bool {
        matches!(self, Self::LimitReached { .. })
    }

    /// Returns the type as a string
    #[must_use]
    pub fn type_str(&self) -> String {
        match self {
            PyProveResult::Solved { result: _ } => "Solved".into(),
            PyProveResult::Undecidable {} => "Undecidable".into(),
            PyProveResult::LimitReached {
                vertices: _,
                edges: _,
            } => "LimitReached".into(),
        }
    }

    /// Returns the content of limit reched
    #[allow(clippy::type_complexity)]
    pub fn unpack_limit_reached(
        &self,
    ) -> Result<(Vec<Vertex>, HashBrownMap<usize, Vec<usize>>), EggShellError> {
        if let Self::LimitReached { vertices, edges } = self {
            return Ok((vertices.clone(), edges.clone()));
        }
        Err(EggShellError::TupleUnpacking("LimitReached".into()))
    }

    /// Returns the content of limit reched
    pub fn unpack_solved(&self) -> Result<String, EggShellError> {
        if let Self::Solved { result } = self {
            return Ok(result.clone());
        }
        Err(EggShellError::TupleUnpacking("Solved".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_stmt = "( == 0 0 )";
        let mut eqsat = PyEqsatHalide::new(0).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyProveResult::Solved { result: "1".into() }, result);
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_stmt = "( == 1 0 )";
        let mut eqsat = PyEqsatHalide::new(0).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyProveResult::Solved { result: "0".into() }, result);
    }

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt = "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )";
        let mut eqsat = PyEqsatHalide::new(0).unwrap();
        let result = eqsat.prove_once(true_stmt).unwrap();
        assert_eq!(PyProveResult::Solved { result: "1".into() }, result);
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )";
        let mut eqsat = PyEqsatHalide::new(0).unwrap();
        let result = eqsat.prove_once(false_stmt).unwrap();
        assert_eq!(PyProveResult::Solved { result: "0".into() }, result);
    }
}
