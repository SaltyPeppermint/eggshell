use core::panic;
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
use egg::{Analysis, AstSize, CostFunction, EGraph, Language, RecExpr, Rewrite, StopReason};
use hashbrown::HashSet;
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use eggshell::eqsat::{EqsatConf, StartMaterial};
use eggshell::io::reader;
use eggshell::io::structs::Entry;
use eggshell::sampling::strategy::{CostWeighted, SizeCountWeighted, Strategy};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Rise, TermRewriteSystem, TrsEqsat, TrsEqsatResult};

#[derive(Parser, Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    file: PathBuf,

    /// Id of expr from which to seed egraphs
    #[arg(long)]
    expr_id: usize,

    /// RNG Seed
    #[arg(long, default_value_t = 2024)]
    rng_seed: u64,

    /// Number of samples to take per EClass
    #[arg(long, default_value_t = 8)]
    eclass_samples: usize,

    /// Sampling strategy
    #[arg(long, default_value_t = SampleStrategy::SizeCount)]
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
    SizeCount,
    CostWeighted,
}

impl Display for SampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleStrategy::SizeCount => write!(f, "SizeCount"),
            SampleStrategy::CostWeighted => write!(f, "CostWeighted"),
        }
    }
}

impl FromStr for SampleStrategy {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace("_", "").as_str() {
            "sizecount" => Ok(Self::SizeCount),
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

// static mut COUNTER: u32 = 0;
// pub fn fresh_id() -> u32 {
//     unsafe {
//         let c = COUNTER;
//         COUNTER += 1;
//         c
//     }
// }

// fn lambda(f: impl FnOnce(&str) -> String) -> String {
//     let n = fresh_id();
//     let x = format!("x{}", n);
//     format!("(lam {} {})", x, f(x.as_str()))
// }

// trait Dsl {
//     // f1 >> f2
//     fn then<S: Into<String>>(self, other: S) -> String;
//     // v |> f
//     fn pipe<S: Into<String>>(self, other: S) -> String;
// }

// impl Dsl for String {
//     fn then<S: Into<String>>(self, other: S) -> String {
//         let c = fresh_id();
//         format!(
//             "(lam x{} (app {} (app {} (var x{}))))",
//             c,
//             other.into(),
//             self,
//             c
//         )
//     }

//     fn pipe<S: Into<String>>(self, other: S) -> String {
//         format!("(app {} {})", other.into(), self)
//     }
// }

// impl Dsl for &str {
//     fn then<S: Into<String>>(self, other: S) -> String {
//         String::from(self).then(other)
//     }

//     fn pipe<S: Into<String>>(self, other: S) -> String {
//         String::from(self).pipe(other)
//     }
// }

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let sample_conf_builder = SampleConf::builder()
        .rng_seed(cli.rng_seed)
        .samples_per_eclass(cli.eclass_samples);
    let sample_conf = sample_conf_builder.build();

    let eqsat_conf = EqsatConf::builder()
        .maybe_node_limit(cli.node_limit)
        .maybe_time_limit(cli.time_limit.map(|x| Duration::from_secs_f64(x as f64)))
        .maybe_memory_limit(cli.memory_limit)
        .iter_limit(1) // Iter limit of one since we manually run the eqsat
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

    let exprs = match cli.file.extension().unwrap().to_str().unwrap() {
        "csv" => reader::read_exprs_csv(&cli.file),
        "json" => reader::read_exprs_json(&cli.file),
        extension => panic!("Unknown file extension {}", extension),
    };

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
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: Cli,
) {
    let rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    let entry = exprs
        .into_iter()
        .nth(cli.expr_id)
        .expect("Must be in the file!");

    info!(
        "Starting work on expression {}: {}...",
        cli.expr_id, entry.expr
    );

    info!("Starting eqsat on expression {}...", cli.expr_id);

    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let mut eqsat = TrsEqsat::<R>::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
        .with_conf(eqsat_conf.clone());

    let mut intermediate_egraphs = Vec::new();
    let mut iter_count = 0;

    let last_result = loop {
        let result = eqsat.run(rules);
        iter_count += 1;
        info!("Iteration {iter_count} finished.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                intermediate_egraphs.push(result.egraph().clone());
                eqsat = TrsEqsat::<R>::new(result.into()).with_conf(eqsat_conf.clone());
            }
            _ => {
                info!("Limits reached after {iter_count} iterations!");
                info!(
                    "Max Memory Consumption: {:?}",
                    result
                        .iterations()
                        .last()
                        .expect("Should be at least one")
                        .mem_usage
                );
                break result;
            }
        }
    };

    info!("Finished Eqsat {}!", cli.expr_id);
    info!("Starting sampling...");

    let samples: Vec<_> = sample::<R>(&cli, &start_expr, &last_result, rng, &sample_conf)
        .into_iter()
        .collect();

    info!("Finished sampling {}!", cli.expr_id);
    info!("Took {} samples.", samples.len());

    info!("Generating associated data for {}...", cli.expr_id);
    let associated_data = mk_sample_data::<R>(
        &start_expr,
        samples,
        intermediate_egraphs.as_slice(),
        &cli,
        last_result,
        rules,
    );
    info!("Finished generating associated data for {}!", cli.expr_id);

    let data = DataEntry {
        start_expr,
        sample_data: associated_data,
        metadata: MetaData {
            uuid: cli.uuid.clone(),
            folder: folder.to_owned(),
            cli: cli.to_owned(),
            timestamp: Local::now().timestamp(),
            sample_conf,
            eqsat_conf,
        },
    };
    info!("Finished work on expr {}!", cli.expr_id);

    let mut f = BufWriter::new(File::create(format!("{folder}/{}.json", cli.expr_id)).unwrap());
    serde_json::to_writer(&mut f, &data).unwrap();
    // });
}

fn sample<R: TermRewriteSystem>(
    cli: &Cli,
    start_expr: &RecExpr<R::Language>,
    eqsat: &TrsEqsatResult<R>,
    mut rng: StdRng,
    sample_conf: &SampleConf,
) -> HashSet<RecExpr<<R as TermRewriteSystem>::Language>> {
    let root_id = eqsat.roots()[0];

    match &cli.strategy {
        SampleStrategy::SizeCount => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 1.5) as usize;
            SizeCountWeighted::new_with_limit(eqsat.egraph(), &mut rng, start_expr, limit)
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
    start_expr: &RecExpr<R::Language>,
    samples: Vec<RecExpr<R::Language>>,
    intermediate_egraphs: &[EGraph<R::Language, R::Analysis>],
    cli: &Cli,
    mut eqsat: TrsEqsatResult<R>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Vec<SampleData<R::Language>> {
    samples
        .into_iter()
        .map(|sample| {
            let baseline = mk_baseline::<R>(cli, start_expr, &sample, rules);
            let explanation = mk_explanation::<R>(&sample, cli, &mut eqsat, start_expr);
            let generation = find_generation(&sample, intermediate_egraphs);
            SampleData {
                sample,
                generation,
                baseline,
                explanation,
            }
        })
        .collect()
}

fn mk_baseline<R: TermRewriteSystem>(
    cli: &Cli,
    start_expr: &RecExpr<R::Language>,
    sample: &RecExpr<R::Language>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Option<BaselineData> {
    if cli.with_baselines {
        info!("Running baseline for \"{sample}\"...");
        let starting_exprs = StartMaterial::RecExprs(vec![start_expr.clone(), sample.clone()]);
        let result = TrsEqsat::<R>::new(starting_exprs).run(rules);
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
    eqsat: &mut TrsEqsatResult<R>,
    start_expr: &RecExpr<R::Language>,
) -> Option<String> {
    if cli.with_explanations {
        info!("Constructing explanation of \"{start_expr} == {sample}\"...");
        let expl = eqsat
            .egraph_mut()
            .explain_equivalence(start_expr, sample)
            .get_flat_string();
        info!("Explanation constructed!");
        Some(expl)
    } else {
        None
    }
}

fn find_generation<L: Language, N: Analysis<L>>(
    sample: &RecExpr<L>,
    intermediate_egraphs: &[EGraph<L, N>],
) -> usize {
    match intermediate_egraphs
        .iter()
        .position(|egraph| egraph.lookup_expr(sample).is_some())
    {
        Some(generation) => generation,
        None => intermediate_egraphs.len(),
    }
}

#[derive(Serialize, Clone, Debug)]
struct DataEntry<L: Language + Display> {
    start_expr: RecExpr<L>,
    sample_data: Vec<SampleData<L>>,
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
    generation: usize,
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
