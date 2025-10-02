use std::fmt::{self, Display, Formatter};
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

    /// `RewriteSystem` of the input
    #[arg(long)]
    rewrite_system: RewriteSystemName,

    /// Id of expr from which to seed egraphs
    #[arg(long)]
    expr_id: usize,

    /// Id of expr from which to seed egraphs
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// RNG Seed
    #[arg(long, default_value_t = 2024)]
    rng_seed: u64,

    /// Number of chains
    #[arg(long)]
    n_chains: u64,

    /// Memory limit for eqsat in bytes
    #[arg(long)]
    iter_distance: usize,

    /// Memory limit for eqsat in bytes
    #[arg(long, default_value_t = 128)]
    max_retries: usize,

    /// Memory limit for eqsat in bytes
    #[arg(long, default_value_t = 1024)]
    chain_length: usize,
}

impl Cli {
    #[must_use]
    pub fn file(&self) -> &PathBuf {
        &self.file
    }

    #[must_use]
    pub fn rewrite_system(&self) -> RewriteSystemName {
        self.rewrite_system
    }

    #[must_use]
    pub fn expr_id(&self) -> usize {
        self.expr_id
    }

    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[must_use]
    pub fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    #[must_use]
    pub fn n_chains(&self) -> u64 {
        self.n_chains
    }

    #[must_use]
    pub fn iter_distance(&self) -> usize {
        self.iter_distance
    }

    #[must_use]
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    #[must_use]
    pub fn chain_length(&self) -> usize {
        self.chain_length
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
