use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use chrono::Local;
use clap::error::ErrorKind;
use clap::{Error, Parser};
use egg::{AstSize, Language, RecExpr, Rewrite, StopReason};
use hashbrown::HashSet;
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult};
use eggshell::io::reader;
use eggshell::io::structs::Expression;
use eggshell::sampling::strategy::{CostWeighted, Strategy, TermCountWeighted};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Rise, TermRewriteSystem};

#[derive(Parser, Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    file: PathBuf,

    /// Number of terms from which to seed egraphs
    #[arg(long)]
    seed_term_id: usize,

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
    with_explanations: bool,

    /// Calculate and save explanations
    #[arg(long, default_value_t = false)]
    with_baselines: bool,

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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
enum TrsName {
    Halide,
    Rise,
}

impl Display for TrsName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Halide => write!(f, "Halide"),
            Self::Rise => write!(f, "Rise"),
        }
    }
}

impl FromStr for TrsName {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace("_", "").as_str() {
            "halide" => Ok(Self::Halide),
            "rise" => Ok(Self::Rise),
            _ => Err(Error::new(ErrorKind::InvalidValue)),
        }
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let exprs = reader::read_exprs_json(&cli.file, &[]);
    let sample_conf_builder = SampleConf::builder()
        .rng_seed(cli.rng_seed)
        .samples_per_eclass(cli.eclass_samples);
    let sample_conf = sample_conf_builder.build();

    let eqsat_conf = EqsatConf::builder()
        .maybe_node_limit(cli.node_limit)
        .maybe_time_limit(cli.time_limit.map(|x| Duration::from_secs_f64(x as f64)))
        .maybe_memory_limit(cli.memory_limit)
        .explanation(cli.with_explanations)
        .root_check(false)
        .memory_log(false)
        .build();

    let now = Local::now();

    let folder = format!(
        "data/generated_samples/{}/{}-{}-{}",
        cli.trs,
        cli.file.file_stem().unwrap().to_str().unwrap(),
        now.format("%Y-%m-%d"),
        cli.uuid
    );
    fs::create_dir_all(&folder).unwrap();

    match cli.trs {
        TrsName::Halide => {
            run_eqsat::<Halide>(
                exprs,
                eqsat_conf,
                sample_conf,
                folder,
                Halide::full_rules().as_slice(),
                cli,
            );
        }
        TrsName::Rise => {
            run_eqsat::<Rise>(
                exprs,
                eqsat_conf,
                sample_conf,
                folder,
                Rise::full_rules().as_slice(),
                cli,
            );
        }
    }
}

fn run_eqsat<R: TermRewriteSystem>(
    exprs: Vec<Expression>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: Cli,
) {
    let rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    let expr = exprs
        .into_iter()
        .nth(cli.seed_term_id)
        .expect("Must be in the file!");

    info!("Starting work on expr {}: {}", cli.seed_term_id, expr.term);

    info!("Running eqsat {}", cli.seed_term_id);

    let seed_expr = expr.term.parse::<RecExpr<R::Language>>().unwrap();
    let eqsat: EqsatResult<R> = Eqsat::new(vec![seed_expr.clone()])
        .with_conf(eqsat_conf.clone())
        .run(rules);
    assert!(eqsat.egraph().clean);

    let mem = memory_stats::memory_stats().unwrap().physical_mem;
    info!("eqsat took {mem} bytes of memory");
    info!("Finished Eqsat {}!", cli.seed_term_id);

    let samples: Vec<_> = sample(&cli, &seed_expr, &eqsat, rng, &sample_conf)
        .into_iter()
        .collect();

    info!("Finished sampling {}!", cli.seed_term_id);
    info!("Took {} samples!", samples.len());

    info!("Generating associated data for {}...", cli.seed_term_id);
    let associated_data = mk_sample_data(&seed_expr, samples, &cli, eqsat, rules);
    info!(
        "Finished generating associated data for {}!",
        cli.seed_term_id
    );

    let data = DataEntry {
        seed_expr,
        associated_data,
        metadata: MetaData {
            uuid: cli.uuid.clone(),
            folder: folder.to_owned(),
            cli: cli.to_owned(),
            timestamp: Local::now().timestamp(),
            sample_conf,
            eqsat_conf,
        },
    };
    info!("Finished work on expr {}", cli.seed_term_id);

    let mut f =
        BufWriter::new(File::create(format!("{folder}/{}.json", cli.seed_term_id)).unwrap());
    serde_json::to_writer(&mut f, &data).unwrap();
    // });
}

fn sample<R: TermRewriteSystem>(
    cli: &Cli,
    seed_expr: &RecExpr<<R as TermRewriteSystem>::Language>,
    eqsat: &EqsatResult<R>,
    mut rng: StdRng,
    sample_conf: &SampleConf,
) -> HashSet<RecExpr<<R as TermRewriteSystem>::Language>> {
    let root_id = eqsat.roots()[0];

    match &cli.strategy {
        SampleStrategy::TermSizeCount => {
            let min_size = seed_expr.as_ref().len();
            info!("Using min_size {min_size}");
            TermCountWeighted::new(eqsat.egraph(), &mut rng, min_size + 8)
                .sample_eclass(sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CostWeighted => {
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit)
                .sample_eclass(sample_conf, root_id)
                .unwrap()
        }
    }
}

fn mk_sample_data<R: TermRewriteSystem>(
    seed_expr: &RecExpr<R::Language>,
    samples: Vec<RecExpr<R::Language>>,
    cli: &Cli,
    mut eqsat: EqsatResult<R>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Vec<SampleData<R::Language>> {
    samples
        .into_iter()
        .map(|sample| {
            let baseline = mk_baseline::<R>(cli, seed_expr, &sample, rules);
            let explanation = mk_explanation(&sample, cli, &mut eqsat, seed_expr);
            SampleData {
                sample,
                baseline,
                explanation,
            }
        })
        .collect()
}

fn mk_baseline<R: TermRewriteSystem>(
    cli: &Cli,
    seed_expr: &RecExpr<R::Language>,
    sample: &RecExpr<R::Language>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Option<BaselineData> {
    if cli.with_baselines {
        info!("Running baseline for \"{sample}\"...");
        let result = Eqsat::<R>::new(vec![seed_expr.clone(), sample.clone()]).run(rules);
        let b = BaselineData {
            stop_reason: result.report().stop_reason.clone(),
            total_time: result.report().total_time,
            total_nodes: result.report().egraph_nodes,
            total_iters: result.report().iterations,
        };
        info!("Baseline run!");
        Some(b)
    } else {
        None
    }
}

fn mk_explanation<R: TermRewriteSystem>(
    sample: &RecExpr<R::Language>,
    cli: &Cli,
    eqsat: &mut EqsatResult<R>,
    seed_expr: &RecExpr<R::Language>,
) -> Option<String> {
    if cli.with_explanations {
        info!("Constructing explanation of \"{seed_expr} == {sample}\"...");
        let expl = eqsat
            .egraph_mut()
            .explain_equivalence(seed_expr, sample)
            .get_flat_string();
        info!("Explanation constructed!");
        Some(expl)
    } else {
        None
    }
}

#[derive(Serialize, Clone, Debug)]
struct DataEntry<L: Language + Display> {
    seed_expr: RecExpr<L>,
    associated_data: Vec<SampleData<L>>,
    metadata: MetaData,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
struct MetaData {
    uuid: String,
    folder: String,
    cli: Cli,
    timestamp: i64,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

#[derive(Serialize, Clone, Debug)]
struct SampleData<L: Language + Display> {
    sample: RecExpr<L>,
    baseline: Option<BaselineData>,
    explanation: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
struct BaselineData {
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}
