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
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult};
use eggshell::io::reader;
use eggshell::io::structs::Expression;
use eggshell::sampling::strategy::{CostWeighted, Strategy, TermCountWeighted};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Trs};

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
        "data/generated_samples/5k_dataset_{}-{}",
        now.format("%Y-%m-%d"),
        cli.uuid
    );
    fs::create_dir_all(&folder).unwrap();

    let rules = Halide::full_rules();
    gen_data::<Halide>(
        exprs,
        eqsat_conf,
        sample_conf,
        folder,
        rules.as_slice(),
        cli,
    );
}

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

fn gen_data<R: Trs>(
    exprs: Vec<Expression>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: Cli,
) {
    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);

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
    let samples = match &cli.strategy {
        SampleStrategy::TermSizeCount => {
            let min_size = seed_expr.as_ref().len();
            let strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, min_size + 3);
            gen_samples(&eqsat, &sample_conf, strategy)
        }
        SampleStrategy::CostWeighted => {
            let strategy =
                CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
            gen_samples(&eqsat, &sample_conf, strategy)
        }
    };

    info!("Finished sampling {}!", cli.seed_term_id);

    info!("Generating associated data for {}...", cli.seed_term_id);
    let associated_data = gen_associated_data(&seed_expr, samples, &cli, eqsat, rules);
    info!(
        "Finished generating associated data for {}!",
        cli.seed_term_id
    );

    let truth_value = match expr.truth_value.as_str() {
        "true" => true,
        "false" => false,
        _ => panic!("Wrong truth_value"),
    };

    let metadata = MetaData {
        uuid: cli.uuid.clone(),
        folder: folder.to_owned(),
        cli: cli.to_owned(),
        timestamp: Local::now().timestamp(),
        strategy: cli.strategy,
        sample_conf,
        eqsat_conf,
    };

    let data = DataEntry {
        seed_id: cli.seed_term_id,
        seed_expr,
        associated_data,
        truth_value,
        metadata,
    };
    info!("Finished work on expr {}", cli.seed_term_id);

    let mut f =
        BufWriter::new(File::create(format!("{folder}/{}.json", cli.seed_term_id)).unwrap());
    serde_json::to_writer(&mut f, &data).unwrap();
    // });
}

fn gen_samples<'a, R, S>(
    // sample_mode: SampleMode,
    eqsat: &EqsatResult<R>,
    sample_conf: &SampleConf,
    mut strategy: S,
) -> Vec<RecExpr<R::Language>>
where
    R: Trs,
    R::Language: 'a,
    R::Analysis: 'a,
    S: Strategy<'a, R::Language, R::Analysis>,
{
    let root_id = eqsat.roots()[0];

    strategy
        .sample_eclass(sample_conf, root_id)
        .unwrap()
        .into_iter()
        .collect()
}

fn gen_associated_data<R: Trs>(
    seed_expr: &RecExpr<R::Language>,
    samples: Vec<RecExpr<R::Language>>,
    cli: &Cli,
    eqsat: EqsatResult<R>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Vec<SampleData<R::Language>> {
    let expls = gen_explanations(&samples, cli, eqsat, seed_expr);

    samples
        .into_iter()
        .zip(expls)
        .map(|(sample, explanation)| {
            let baseline = gen_baseline::<R>(cli, seed_expr, &sample, rules);
            SampleData {
                sample,
                baseline,
                explanation,
            }
        })
        .collect()
}

fn gen_baseline<R: Trs>(
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

fn gen_explanations<R: Trs>(
    samples: &[RecExpr<R::Language>],
    cli: &Cli,
    mut eqsat: EqsatResult<R>,
    seed_expr: &RecExpr<R::Language>,
) -> Vec<Option<String>> {
    samples
        .iter()
        .map(|sample| {
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
        })
        .collect::<Vec<_>>()
}

#[derive(Serialize, Clone, Debug)]
struct DataEntry<L: Language + Display> {
    seed_id: usize,
    seed_expr: RecExpr<L>,
    truth_value: bool,
    associated_data: Vec<SampleData<L>>,
    metadata: MetaData,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
struct MetaData {
    uuid: String,
    folder: String,
    cli: Cli,
    timestamp: i64,
    strategy: SampleStrategy,
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
