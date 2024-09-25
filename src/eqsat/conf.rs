use std::time::Duration;

use egg::{Analysis, Language, RecExpr, Runner};
use hashbrown::HashSet;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the arguments with which the [`egg::Runner`] is set up
#[expect(clippy::unsafe_derive_deserialize)]
#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq)]
pub struct EqsatConf {
    pub iter_limit: Option<usize>,
    pub node_limit: Option<usize>,
    pub time_limit: Option<Duration>,
    pub explanation: bool,
}

#[pymethods]
impl EqsatConf {
    #[must_use]
    #[new]
    #[pyo3(signature = (explanation=false, iter_limit=Some(10_000_000), node_limit=Some(100_000), time_limit=Some(10.0)))]
    pub fn new(
        explanation: bool,
        iter_limit: Option<usize>,
        node_limit: Option<usize>,
        time_limit: Option<f64>,
    ) -> Self {
        Self {
            iter_limit,
            node_limit,
            time_limit: time_limit.map(Duration::from_secs_f64),
            explanation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EqsatConfBuilder {
    pub iter_limit: Option<usize>,
    pub node_limit: Option<usize>,
    pub time_limit: Option<Duration>,
    pub explanation: bool,
}

impl EqsatConfBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            iter_limit: Some(10_000_000),
            node_limit: Some(100_000),
            // Do we actually want a time limit?
            time_limit: Some(Duration::new(10, 0)),
            explanation: false,
        }
    }

    #[must_use]
    pub fn iter_limit(mut self, iter: usize) -> Self {
        self.iter_limit = Some(iter);
        self
    }

    #[must_use]
    pub fn without_iter_limit(mut self) -> Self {
        self.iter_limit = None;
        self
    }

    #[must_use]
    pub fn node_limit(mut self, nodes: usize) -> Self {
        self.node_limit = Some(nodes);
        self
    }

    #[must_use]
    pub fn without_node_limit(mut self) -> Self {
        self.node_limit = None;
        self
    }

    #[must_use]
    pub fn time_limit(mut self, time: Duration) -> Self {
        self.time_limit = Some(time);
        self
    }

    #[must_use]
    pub fn without_time_limit(mut self) -> Self {
        self.time_limit = None;
        self
    }

    #[must_use]
    pub fn explanation(mut self, enabled: bool) -> Self {
        self.explanation = enabled;
        self
    }

    #[must_use]
    pub fn build(self) -> EqsatConf {
        EqsatConf {
            iter_limit: self.iter_limit,
            node_limit: self.node_limit,
            time_limit: self.time_limit,
            explanation: self.explanation,
        }
    }
}

impl Default for EqsatConfBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
pub(crate) fn build_runner<L, N>(
    conf: &EqsatConf,
    root_check: bool,
    exprs: &[RecExpr<L>],
) -> Runner<L, N>
where
    L: Language,
    N: Analysis<L> + Default,
{
    // Initialize a simple runner and run it.
    let mut runner = Runner::default();
    if let Some(iter_limit) = conf.iter_limit {
        runner = runner.with_iter_limit(iter_limit);
    };
    if let Some(node_limit) = conf.node_limit {
        runner = runner.with_node_limit(node_limit);
    };
    if let Some(time_limit) = conf.time_limit {
        runner = runner.with_time_limit(time_limit);
    };
    if conf.explanation {
        runner = runner.with_explanations_enabled();
    };

    if root_check {
        let hook = move |r: &mut Runner<L, N>| {
            let mut uniq = HashSet::new();
            if r.roots.iter().all(|x| uniq.insert(*x)) {
                Ok(())
            } else {
                Err("Duplicate in roots".into())
            }
        };
        runner = runner.with_hook(hook);
    }

    for expr in exprs {
        runner = runner.with_expr(expr);
    }
    runner
}
