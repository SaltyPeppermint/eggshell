use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use clap::error::ErrorKind;
use clap::{Args, Error, Parser};
use serde::{Deserialize, Serialize};

#[derive(Parser, Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(long)]
    file: PathBuf,

    /// Id of expr from which to seed egraphs
    #[arg(long)]
    expr_id: usize,

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
    #[arg(long, default_value_t = SampleStrategy::CountWeightedUniformly)]
    strategy: SampleStrategy,

    /// Calculate and save explanations
    #[arg(long, default_value_t = false)]
    with_explanations: bool,

    // /// Calculate and save explanations
    // #[arg(long, default_value_t = false)]
    // with_baselines: bool,
    /// Node limit for egraph in seconds
    #[arg(long)]
    node_limit: Option<usize>,

    /// Memory limit for eqsat in bytes
    #[arg(long)]
    memory_limit: Option<usize>,

    /// Time limit for eqsat in seconds
    #[arg(long)]
    time_limit: Option<usize>,

    /// UUID to identify run
    #[arg(long)]
    uuid: String,

    /// Trs of the input
    #[arg(long)]
    trs: TrsName,

    #[command(flatten)]
    baseline_args: Option<BaselineArgs>,
}

#[derive(Debug, Args, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct BaselineArgs {
    /// Number of random goals to pick
    #[arg(long, default_value_t = 1, group = "baselines")]
    random_goals: usize,

    /// Number of random guides to pick
    #[arg(long, group = "baselines")]
    random_guides: usize,
}

impl BaselineArgs {
    #[must_use]
    pub fn random_goals(&self) -> usize {
        self.random_goals
    }

    #[must_use]
    pub fn random_guides(&self) -> usize {
        self.random_guides
    }
}

impl Cli {
    #[must_use]
    pub fn expr_id(&self) -> usize {
        self.expr_id
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
    pub fn uuid(&self) -> &str {
        &self.uuid
    }

    #[must_use]
    pub fn with_explanations(&self) -> bool {
        self.with_explanations
    }

    #[must_use]
    pub(crate) fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    #[must_use]
    pub fn trs(&self) -> TrsName {
        self.trs
    }

    #[must_use]
    pub fn file(&self) -> &PathBuf {
        &self.file
    }

    #[must_use]
    pub(crate) fn node_limit(&self) -> Option<usize> {
        self.node_limit
    }

    #[must_use]
    pub(crate) fn time_limit(&self) -> Option<usize> {
        self.time_limit
    }

    #[must_use]
    pub(crate) fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }

    #[must_use]
    pub(crate) fn sample_batch_size(&self) -> Option<usize> {
        self.sample_batch_size
    }

    #[must_use]
    pub fn baseline_args(&self) -> Option<&BaselineArgs> {
        self.baseline_args.as_ref()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    CountWeightedUniformly,
    CountWeighted,
    CostWeighted,
}

impl Display for SampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleStrategy::CountWeightedUniformly => write!(f, "CountWeightedUniformly"),
            SampleStrategy::CountWeighted => write!(f, "CountWeighted"),
            SampleStrategy::CostWeighted => write!(f, "CostWeighted"),
        }
    }
}

impl FromStr for SampleStrategy {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('_', "").as_str() {
            "countweighteduniformly" => Ok(Self::CountWeightedUniformly),
            "countweighted" => Ok(Self::CountWeighted),
            "costweighted" => Ok(Self::CostWeighted),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TrsName {
    Halide,
    Rise,
}

impl Display for TrsName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Halide => write!(f, "halide"),
            Self::Rise => write!(f, "rise"),
        }
    }
}

impl FromStr for TrsName {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('_', "").as_str() {
            "halide" => Ok(Self::Halide),
            "rise" => Ok(Self::Rise),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}
