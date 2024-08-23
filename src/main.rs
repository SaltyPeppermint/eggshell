use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

use chrono::Local;
use clap::Parser;
use egg::{Id, Language, RecExpr, Rewrite};
use eggshell::io::structs::Expression;
use hashbrown::{HashMap, HashSet};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use eggshell::eqsat::utils::{EqsatConf, EqsatConfBuilder};
use eggshell::eqsat::{Eqsat, EqsatResult};
use eggshell::io::reader;
use eggshell::sampling::SampleConfBuilder;
use eggshell::sampling::{self, SampleConf};
use eggshell::trs::{halide, Halide, Trs};

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
    let sample_conf = SampleConfBuilder::new().rng_seed(cli.rng_seed).build();
    let eqsat_conf = EqsatConfBuilder::new()
        .time_limit(Duration::from_secs_f64(0.5))
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
        &sample_conf,
        &eqsat_conf,
        &cli,
    );
    let rules = Halide::rules(&halide::Ruleset::ArithMinMax);

    sample::<Halide>(exprs, eqsat_conf, sample_conf, &folder, rules, &cli);
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    file: PathBuf,

    /// Sets a custom config file
    #[arg(short, long, default_value_t = 50)]
    chunksize: usize,

    #[arg(short, long, default_value_t = 100)]
    n_seeds: usize,

    #[arg(short, long, default_value_t = 2024)]
    rng_seed: u64,
}

fn write_metadata(
    uuid: Uuid,
    folder: &str,
    timestamp: i64,
    sample_conf: &SampleConf,
    eqsat_conf: &EqsatConf,
    cli: &Cli,
) {
    let mut metadata_f = BufWriter::new(File::create(format!("{folder}/metadata.json")).unwrap());
    let metadata = MetaData {
        id: uuid,
        folder: folder.to_owned(),
        seed_file: cli.file.to_owned(),
        n_seeds: cli.n_seeds,
        timestamp,
        sample_conf: sample_conf.clone(),
        eqsat_conf: eqsat_conf.clone(),
    };
    serde_json::to_writer(&mut metadata_f, &metadata).unwrap();
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
struct MetaData {
    id: Uuid,
    folder: String,
    seed_file: PathBuf,
    n_seeds: usize,
    timestamp: i64,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

fn sample<R: Trs>(
    exprs: Vec<Expression>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: &str,
    rules: Vec<Rewrite<R::Language, R::Analysis>>,
    cli: &Cli,
) {
    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    let mut sample_list = Vec::new();

    for (index, expr) in exprs.into_iter().take(cli.n_seeds).enumerate() {
        let seed_expr = expr.term.parse::<RecExpr<R::Language>>().unwrap();
        println!("Working on expr: {}", seed_expr);

        let eqsat: EqsatResult<R> = Eqsat::new(vec![seed_expr.clone()])
            .with_conf(eqsat_conf.clone())
            .run(&rules);

        let generated = sampling::sample(eqsat.egraph(), &sample_conf, &mut rng);

        sample_list.push(DataEntry {
            seed_id: index,
            seed_expr,
            generated,
        });

        if index % cli.chunksize == 0 {
            let mut f = BufWriter::new(
                File::create(format!("{folder}/{}.json", index / cli.chunksize)).unwrap(),
            );
            serde_json::to_writer(&mut f, &sample_list).unwrap();
            f.flush().unwrap();
            sample_list.clear();
        }
    }
}

#[derive(Serialize, Clone, Eq, PartialEq)]
struct DataEntry<L: Language + Display> {
    seed_id: usize,
    seed_expr: RecExpr<L>,
    generated: HashMap<Id, HashSet<RecExpr<L>>>,
}
