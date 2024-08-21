use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct SampleConf {
    pub samples_per_egraph: usize,
    pub samples_per_eclass: usize,
    pub loop_limit: usize,
    pub rng_seed: u64,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct SampleConfBuilder {
    samples_per_egraph: usize,
    samples_per_eclass: usize,
    loop_limit: usize,
    rng_seed: u64,
}

impl SampleConfBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples_per_egraph: 16,
            samples_per_eclass: 16,
            loop_limit: 8,
            rng_seed: 2024,
        }
    }

    #[must_use]
    pub fn samples_per_egraph(mut self, samples_per_eclass: usize) -> Self {
        self.samples_per_eclass = samples_per_eclass;
        self
    }
    #[must_use]
    pub fn samples_per_eclass(mut self, samples_per_eclass: usize) -> Self {
        self.samples_per_eclass = samples_per_eclass;
        self
    }
    #[must_use]
    pub fn loop_limit(mut self, loop_limit: usize) -> Self {
        self.loop_limit = loop_limit;
        self
    }

    #[must_use]
    pub fn rng_seed(mut self, rng_seed: u64) -> Self {
        self.rng_seed = rng_seed;
        self
    }

    #[must_use]
    pub fn build(self) -> SampleConf {
        SampleConf {
            samples_per_egraph: self.samples_per_egraph,
            samples_per_eclass: self.samples_per_eclass,
            loop_limit: self.loop_limit,
            rng_seed: self.rng_seed,
        }
    }
}

impl Default for SampleConfBuilder {
    fn default() -> Self {
        Self::new()
    }
}
