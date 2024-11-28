use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::cli::Cli;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Builder)]
pub struct SampleConf {
    #[builder(default = 16)]
    pub samples_per_egraph: usize,
    #[builder(default = 16)]
    pub samples_per_eclass: usize,
    #[builder(default = 8)]
    pub loop_limit: usize,
    #[builder(default = 1024)]
    pub rng_seed: u64,
}

impl From<&Cli> for SampleConf {
    fn from(cli: &Cli) -> Self {
        SampleConf::builder()
            .rng_seed(cli.rng_seed())
            .samples_per_eclass(cli.eclass_samples())
            .build()
    }
}

impl Default for SampleConf {
    fn default() -> Self {
        Self::builder().build()
    }
}
