use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::Local;
use clap::error::ErrorKind;
use clap::{Error, Parser, Subcommand};
use egg::{AstSize, EGraph, Id, Language, RecExpr, Rewrite, StopReason};
use hashbrown::HashMap;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::{prelude::*, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult};
use eggshell::io::reader;
use eggshell::io::structs::Expression;
use eggshell::sampling::strategy::{CostWeighted, Strategy, TermCountWeighted};
use eggshell::sampling::{SampleConf, SampleConfBuilder};
use eggshell::trs::{Halide, Trs};

// const CHUNKSIZE: usize = 3;
// const N_SEEDS: usize = 20;
// const RNG_SEED: u64 = 2024;
// const SEED_FILE: &str = "data/prefix/5k_dataset.csv";

// const N_SEEDS: usize = 20;
// const RNG_SEED: u64 = 2024;
// const SEED_FILE: &str = "data/prefix/5k_dataset.csv";

fn main() {
    let cli = Cli::parse();

    let exprs = reader::read_exprs_json(&cli.file, &[]);
    let sample_conf_builder = SampleConfBuilder::new()
        .rng_seed(cli.rng_seed)
        .samples_per_eclass(cli.eclass_samples);
    let sample_conf = match cli.sample_mode {
        SampleMode::Full { egraph_samples } => sample_conf_builder
            .samples_per_egraph(egraph_samples)
            .build(),
        SampleMode::JustRoot => sample_conf_builder.build(),
    };
    let eqsat_conf = EqsatConf {
        iter_limit: None,
        node_limit: cli.node_limit,
        time_limit: None,
        explanation: cli.explanations,
    };
    if let Some(p) = cli.processes {
        ThreadPoolBuilder::new()
            .num_threads(p)
            .build_global()
            .unwrap();
    }

    let now = Local::now();
    let uuid = Uuid::new_v4();

    let folder = format!(
        "data/generated_samples/5k_dataset_{}-{}",
        now.format("%Y-%m-%d_%H:%M:%S"),
        uuid
    );
    fs::create_dir_all(&folder).unwrap();

    write_metadata(
        uuid,
        &folder,
        now.timestamp(),
        cli.strategy,
        &sample_conf,
        &eqsat_conf,
        &cli,
    );
    let rules = Halide::full_rules();
    gen_data::<Halide>(
        exprs,
        &eqsat_conf,
        &sample_conf,
        &cli.strategy,
        &folder,
        rules.as_slice(),
        &cli,
    );
}

#[derive(Parser, Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    file: PathBuf,

    /// Number of terms from which to seed egraphs
    #[arg(long, default_value_t = 100)]
    n_terms: usize,

    /// RNG Seed
    #[arg(long, default_value_t = 2024)]
    rng_seed: u64,

    /// Number of samples to take per EClass
    #[arg(long, default_value_t = 8)]
    eclass_samples: usize,

    /// Sampling strategy
    #[arg(long, default_value_t = SampleStrategy::TermSizeCount)]
    strategy: SampleStrategy,

    /// Calculate and save explanations
    #[arg(long, default_value_t = false)]
    explanations: bool,

    /// Number of processes to run in parallel
    #[arg(long)]
    processes: Option<usize>,

    /// Node limit for egraph
    #[arg(long)]
    node_limit: Option<usize>,

    /// Calculate and save n baselines
    #[arg(long)]
    baseline: Option<usize>,

    #[command(subcommand)]
    sample_mode: SampleMode,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
enum SampleStrategy {
    TermSizeCount,
    CostWeighted,
}

impl Display for SampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleStrategy::TermSizeCount => write!(f, "TermSizeCount"),
            SampleStrategy::CostWeighted => write!(f, "CostWeighted"),
        }
    }
}

impl FromStr for SampleStrategy {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace("_", "").as_str() {
            "termsizecount" => Ok(Self::TermSizeCount),
            "costweighted" => Ok(Self::CostWeighted),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}

#[derive(Subcommand, Serialize, Deserialize, Debug, Eq, PartialEq, Clone, Copy)]
enum SampleMode {
    Full {
        /// Number of samples to take for each EGraph
        #[arg(short = 'g', long, default_value_t = 16)]
        egraph_samples: usize,
    },
    JustRoot,
}

fn write_metadata(
    uuid: Uuid,
    folder: &str,
    timestamp: i64,
    strategy: SampleStrategy,
    sample_conf: &SampleConf,
    eqsat_conf: &EqsatConf,
    cli: &Cli,
) {
    let mut metadata_f = BufWriter::new(File::create(format!("{folder}/metadata.json")).unwrap());
    let metadata = MetaData {
        id: uuid,
        folder: folder.to_owned(),
        cli: cli.to_owned(),
        timestamp,
        strategy,
        sample_conf: sample_conf.clone(),
        eqsat_conf: eqsat_conf.clone(),
    };
    serde_json::to_writer(&mut metadata_f, &metadata).unwrap();
    metadata_f.flush().unwrap();
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
struct MetaData {
    id: Uuid,
    folder: String,
    cli: Cli,
    timestamp: i64,
    strategy: SampleStrategy,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

fn gen_data<R: Trs>(
    exprs: Vec<Expression>,
    eqsat_conf: &EqsatConf,
    sample_conf: &SampleConf,
    strategy: &SampleStrategy,
    folder: &str,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: &Cli,
) {
    let file_id_ctr = AtomicUsize::new(0);
    let bar = ProgressBar::new(exprs.len() as u64);

    exprs
        .into_par_iter()
        .take(cli.n_terms)
        .enumerate()
        .for_each_init(
            || (Vec::new(), StdRng::seed_from_u64(sample_conf.rng_seed)),
            |(sample_list, rng), (seed_id, expr)| {
                let seed_expr = expr.term.parse::<RecExpr<R::Language>>().unwrap();

                let eqsat: EqsatResult<R> = Eqsat::new(vec![seed_expr.clone()])
                    .with_conf(eqsat_conf.clone())
                    .run(rules);
                let egraph = eqsat.egraph();
                assert!(egraph.clean);

                let eclass_data = match strategy {
                    SampleStrategy::TermSizeCount => {
                        let min_size = seed_expr.as_ref().len();
                        let strategy = TermCountWeighted::new(egraph, rng, min_size + 3);
                        let samples = gen_samples(cli.sample_mode, &eqsat, sample_conf, strategy);
                        gen_associated_data(samples, cli, eqsat, rules, rng)
                    }
                    SampleStrategy::CostWeighted => {
                        let strategy =
                            CostWeighted::new(egraph, AstSize, rng, sample_conf.loop_limit);
                        let samples = gen_samples(cli.sample_mode, &eqsat, sample_conf, strategy);
                        gen_associated_data(samples, cli, eqsat, rules, rng)
                    }
                };
                let truth_value = match expr.truth_value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => panic!("Wrong truth_value"),
                };

                sample_list.push(DataEntry {
                    seed_id,
                    seed_expr,
                    eclass_data,
                    truth_value,
                });
                bar.inc(1);

                if sample_list.len() == 50 {
                    let file_id = file_id_ctr.fetch_add(1, Ordering::SeqCst);
                    let mut f =
                        BufWriter::new(File::create(format!("{folder}/{file_id}.json")).unwrap());
                    serde_json::to_writer(&mut f, &sample_list).unwrap();
                    f.flush().unwrap();
                    sample_list.clear();
                }
            },
        );
}

fn gen_samples<'a, R, S>(
    sample_mode: SampleMode,
    eqsat: &EqsatResult<R>,
    sample_conf: &SampleConf,
    mut strategy: S,
) -> HashMap<Id, Vec<RecExpr<<R as Trs>::Language>>>
where
    R: Trs,
    R::Language: 'a,
    R::Analysis: 'a,
    S: Strategy<'a, R::Language, R::Analysis>,
{
    let generated = match sample_mode {
        SampleMode::Full { egraph_samples: _ } => {
            let root_id = *eqsat.roots().first().unwrap();
            let root_samples = strategy.sample_eclass(sample_conf, root_id).unwrap();
            let mut random_samples = strategy.sample(sample_conf).unwrap();
            random_samples.insert(root_id, root_samples);
            random_samples
        }
        SampleMode::JustRoot => {
            let root_id = *eqsat.roots().first().unwrap();
            let root_samples = strategy.sample_eclass(sample_conf, root_id).unwrap();
            HashMap::from([(root_id, root_samples)])
        }
    }
    .into_iter()
    .map(|(k, v)| (k, Vec::from_iter(v)))
    .collect::<HashMap<_, Vec<_>>>();
    generated
}

fn gen_associated_data<R: Trs>(
    generated: HashMap<Id, Vec<RecExpr<<R as Trs>::Language>>>,
    cli: &Cli,
    mut eqsat: EqsatResult<R>,
    rules: &[Rewrite<<R as Trs>::Language, <R as Trs>::Analysis>],
    rng: &mut StdRng,
) -> Vec<EClassData<<R as Trs>::Language>> {
    let eclass_data = generated
        .into_iter()
        .map(|(id, generated)| {
            let explanations = if cli.explanations {
                Some(gen_explanations::<R>(&generated, eqsat.egraph_mut()))
            } else {
                None
            };
            let baselines = cli
                .baseline
                .map(|n_samples| gen_baseline::<R>(&generated, rules, n_samples, rng));
            EClassData {
                id,
                generated,
                baselines,
                explanations,
            }
        })
        .collect();
    eclass_data
}

fn gen_explanations<R: Trs>(
    generated: &[RecExpr<R::Language>],
    egraph: &mut EGraph<R::Language, R::Analysis>,
) -> Vec<ExplanationData> {
    generated
        .iter()
        .enumerate()
        .flat_map(|(lhs_idx, lhs)| {
            generated
                .iter()
                .enumerate()
                .flat_map(move |(rhs_idx, rhs)| {
                    if lhs_idx == rhs_idx {
                        return None;
                    }
                    Some((lhs_idx, lhs, rhs_idx, rhs))
                })
        })
        .map(|(lhs_idx, lhs, rhs_idx, rhs)| ExplanationData {
            from: lhs_idx,
            to: rhs_idx,
            explanation: egraph.explain_equivalence(lhs, rhs).get_flat_string(),
        })
        .collect()
}

fn gen_baseline<R: Trs>(
    generated: &[RecExpr<R::Language>],
    rules: &[Rewrite<R::Language, R::Analysis>],
    n_samples: usize,
    rng: &mut StdRng,
) -> Vec<BaselineData> {
    generated
        .iter()
        .enumerate()
        .flat_map(|(lhs_idx, lhs)| {
            generated
                .iter()
                .enumerate()
                .flat_map(move |(rhs_idx, rhs)| {
                    if lhs_idx == rhs_idx {
                        return None;
                    }
                    Some((lhs_idx, lhs, rhs_idx, rhs))
                })
        })
        .choose_multiple(rng, n_samples)
        .into_iter()
        .map(|(lhs_idx, lhs, rhs_idx, rhs)| {
            let result = Eqsat::<R>::new(vec![lhs.clone(), rhs.clone()]).run(rules);
            BaselineData {
                from: lhs_idx,
                to: rhs_idx,
                stop_reason: result.report().stop_reason.to_owned(),
                total_time: result.report().total_time,
                total_nodes: result.report().egraph_nodes,
                total_iters: result.report().iterations,
            }
        })
        .collect()
}

#[derive(Serialize, Clone, Debug)]
struct DataEntry<L: Language + Display> {
    seed_id: usize,
    seed_expr: RecExpr<L>,
    truth_value: bool,
    eclass_data: Vec<EClassData<L>>,
}

#[derive(Serialize, Clone, Debug)]
struct EClassData<L: Language + Display> {
    id: Id,
    generated: Vec<RecExpr<L>>,
    baselines: Option<Vec<BaselineData>>,
    explanations: Option<Vec<ExplanationData>>,
}

#[derive(Serialize, Clone, Debug)]
struct BaselineData {
    from: usize,
    to: usize,
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}

#[derive(Serialize, Clone, Eq, PartialEq, Debug)]
struct ExplanationData {
    from: usize,
    to: usize,
    explanation: String,
}
