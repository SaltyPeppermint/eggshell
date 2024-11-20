use std::time::Duration;

use bon::Builder;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the arguments with which the [`egg::Runner`] is set up
#[expect(clippy::unsafe_derive_deserialize)]
#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Builder)]
pub struct EqsatConf {
    #[builder(default = 1000)]
    pub iter_limit: usize,
    #[builder(default = 1_000_000)]
    pub node_limit: usize,
    #[builder(default = 1_000_000_000)]
    pub memory_limit: usize,
    #[builder(default = Duration::from_secs_f64(60.0))]
    pub time_limit: Duration,
    #[builder(default = false)]
    pub explanation: bool,
    #[builder(default = false)]
    pub root_check: bool,
    #[builder(default = false)]
    pub memory_log: bool,
}

#[pymethods]
impl EqsatConf {
    #[must_use]
    #[new]
    #[pyo3(signature = (explanation=false,root_check=false, memory_log=false, iter_limit=1000, node_limit=1_000_000_000, memory_limit=32_000_000_000, time_limit=60.0))]
    pub fn new(
        explanation: bool,
        root_check: bool,
        memory_log: bool,
        iter_limit: usize,
        node_limit: usize,
        memory_limit: usize,
        time_limit: f64,
    ) -> Self {
        Self {
            iter_limit,
            node_limit,
            memory_limit,
            time_limit: Duration::from_secs_f64(time_limit),
            explanation,
            root_check,
            memory_log,
        }
    }
}

impl Default for EqsatConf {
    fn default() -> Self {
        Self::builder().build()
    }
}
