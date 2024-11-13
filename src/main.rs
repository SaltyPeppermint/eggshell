use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use chrono::Local;
use clap::error::ErrorKind;
use clap::{Error, Parser};
use egg::{AstSize, Language, RecExpr, Rewrite, StopReason};
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    seed_terms: usize,

    /// Number of terms from which to seed egraphs
    #[arg(long, default_value_t = 50)]
    saving_batchsize: usize,

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
    // #[command(subcommand)]
    // sample_mode: SampleMode,
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
    let bar = ProgressBar::with_draw_target(Some(exprs.len() as u64), ProgressDrawTarget::stdout());
    let mut file_id = 0;
    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    for (seed_id, expr) in exprs.into_iter().take(cli.seed_terms).enumerate() {
        let mut data_buf = Vec::new();

        info!("Starting work on expr {seed_id}: {}", expr.term);

        info!("Running eqsat {seed_id}");

        let seed_expr = expr.term.parse::<RecExpr<R::Language>>().unwrap();
        let eqsat: EqsatResult<R> = Eqsat::new(vec![seed_expr.clone()])
            .with_conf(eqsat_conf.clone())
            .run(rules);
        assert!(eqsat.egraph().clean);

        let mem = memory_stats::memory_stats().unwrap().physical_mem;
        info!("eqsat took {mem} bytes of memory");
        info!("Finished Eqsat {seed_id}!");
        let samples = match strategy {
            SampleStrategy::TermSizeCount => {
                let min_size = seed_expr.as_ref().len();
                let strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, min_size + 3);
                gen_samples(&eqsat, sample_conf, strategy)
            }
            SampleStrategy::CostWeighted => {
                let strategy =
                    CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
                gen_samples(&eqsat, sample_conf, strategy)
            }
        };

        info!("Finished sampling {seed_id}!");

        info!("Generating associated data for {seed_id}...");
        let associated_data = gen_associated_data(&seed_expr, samples, cli, eqsat, rules);
        info!("Finished generating associated data for {seed_id}!");

        let truth_value = match expr.truth_value.as_str() {
            "true" => true,
            "false" => false,
            _ => panic!("Wrong truth_value"),
        };

        data_buf.push(DataEntry {
            seed_id,
            seed_expr,
            associated_data,
            truth_value,
        });
        bar.inc(1);
        info!("Finished work on expr {}", seed_id);

        if seed_id % cli.saving_batchsize == 0 {
            let mut f = BufWriter::new(File::create(format!("{folder}/{file_id}.json")).unwrap());
            serde_json::to_writer(&mut f, &data_buf).unwrap();
            f.flush().unwrap();
            file_id += 1;
            data_buf.clear();
        }
    }
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
) -> HashMap<RecExpr<R::Language>, SampleData> {
    let expls = gen_explanations(&samples, cli, eqsat, seed_expr);

    samples
        .into_iter()
        .zip(expls)
        .map(|(sample, explanation)| {
            let baseline = gen_baseline::<R>(cli, seed_expr, &sample, rules);
            (
                sample,
                SampleData {
                    baseline,
                    explanation,
                },
            )
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
            from: seed_expr.to_string(),
            to: sample.to_string(),
            stop_reason: result.report().stop_reason.to_owned(),
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
    associated_data: HashMap<RecExpr<L>, SampleData>,
}

#[derive(Serialize, Clone, Debug)]
struct SampleData {
    baseline: Option<BaselineData>,
    explanation: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
struct BaselineData {
    from: String,
    to: String,
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}
