use bon::Builder;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Builder)]
pub struct SampleConf {
    #[builder(default = 16)]
    pub samples_per_egraph: usize,
    #[builder(default = 16)]
    pub samples_per_eclass: usize,
    #[builder(default = 8)]
    pub loop_limit: usize,
    #[builder(default = 2024)]
    pub rng_seed: u64,
}
