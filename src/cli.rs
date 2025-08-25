use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use clap::error::ErrorKind;
use clap::{Error, Parser};
use serde::{Deserialize, Serialize};

#[derive(Parser, Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(long)]
    file: PathBuf,

    /// Id of expr from which to seed egraphs
    #[arg(long)]
    expr_id: usize,

    /// Id of expr from which to seed egraphs
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// RNG Seed
    #[arg(long, default_value_t = 2024)]
    rng_seed: u64,

    /// Number of samples to take per `EClass`
    #[arg(long, default_value_t = 8)]
    eclass_samples: usize,

    /// Sample batch size
    #[arg(long)]
    sample_batch_size: Option<usize>,

    /// Sampling strategy
    #[arg(long, default_value_t = SampleStrategy::CountUniformly)]
    strategy: SampleStrategy,

    /// Memory limit for eqsat in bytes
    #[arg(long)]
    memory_limit: Option<usize>,

    /// Memory limit for eqsat in bytes
    #[arg(long)]
    iter_limit: Option<usize>,

    /// RewriteSystem of the input
    #[arg(long)]
    rewrite_system: RewriteSystemName,
}

impl Cli {
    #[must_use]
    pub fn expr_id(&self) -> usize {
        self.expr_id
    }

    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[must_use]
    pub fn strategy(&self) -> SampleStrategy {
        self.strategy
    }

    #[must_use]
    pub fn eclass_samples(&self) -> usize {
        self.eclass_samples
    }

    #[must_use]
    pub fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    #[must_use]
    pub fn rewrite_system(&self) -> RewriteSystemName {
        self.rewrite_system
    }

    #[must_use]
    pub fn file(&self) -> &PathBuf {
        &self.file
    }

    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }

    #[must_use]
    pub fn iter_limit(&self) -> Option<usize> {
        self.iter_limit
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    CountUniformly,
    CountSizeRange,
    Greedy,
    CostWeighted,
}

impl Display for SampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleStrategy::CountUniformly => write!(f, "CountUniformly"),
            SampleStrategy::CountSizeRange => write!(f, "CountSizeRange"),
            SampleStrategy::Greedy => write!(f, "Greedy"),
            SampleStrategy::CostWeighted => write!(f, "CostWeighted"),
        }
    }
}

impl FromStr for SampleStrategy {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('_', "").as_str() {
            "countuniformly" => Ok(Self::CountUniformly),
            "countsizerange" => Ok(Self::CountSizeRange),
            "greedy" => Ok(Self::Greedy),
            "costweighted" => Ok(Self::CostWeighted),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum RewriteSystemName {
    Halide,
    Rise,
}

impl Display for RewriteSystemName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Halide => write!(f, "halide"),
            Self::Rise => write!(f, "rise"),
        }
    }
}

impl FromStr for RewriteSystemName {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('_', "").as_str() {
            "halide" => Ok(Self::Halide),
            "rise" => Ok(Self::Rise),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}
