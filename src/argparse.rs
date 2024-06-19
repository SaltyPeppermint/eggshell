use clap::Args;
use clap::Parser;
use clap::Subcommand;

/// Parser for the cli options
// Example args:
// cargo run --release dataset `results/expressions_egg.csv` 1000000 10000000 5 5 3 0 4
// cargo run --release `prove_exprs_passes` `data/prefix/expressions_egg.csv` 10000000 10000000 3 $i
// cargo run --release `prove_exprs_fast` `data/prefix/expressions_egg.csv` 10000000 10000000 3
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CliArgs {
    /// Caviar or other mode
    #[command(subcommand)]
    pub mode: Mode,

    #[command(flatten)]
    pub args: EqsatArgs,
}

/// Subcommands for the specific modes
#[derive(Subcommand, Debug)]
pub enum Mode {
    /// Prove mode with a goal in mind.
    Prove {
        #[command(subcommand)]
        strategy: Strategy,

        #[arg(short, long)]
        iteration_check: bool,
    },
    /// Just simplify the expression
    Simplify {
        #[command(subcommand)]
        strategy: Strategy,
    },
}

/// Parameters shared by all subcommands
#[derive(Args, Debug)]
pub struct EqsatArgs {
    /// Location of the file of expressions to churn through with Eqsat
    #[arg(short, long)]
    pub expressions_file: String,
    /// Maximum number of iterations per phase
    #[arg(short, long, default_value_t = 30)]
    pub iter: usize,
    /// Maximum number of nodes per phase
    #[arg(short, long, default_value_t = 10000)]
    pub nodes: usize,
    /// Maximum time spent in a single phase
    #[arg(short, long, default_value_t = 3.0)]
    pub time: f64,
}

/// Subcommands for the specific strategies
#[derive(Subcommand, Debug)]
pub enum Strategy {
    /// Use Pulsing
    Pulse {
        /// Maximum number of total time spent in phases!
        // TODO Change to maximum number of phases, the division stuff is weird.
        #[arg(short, long)]
        time_limit: f64,

        /// Maximum number of total time spent in phases!
        // TODO Change to maximum number of phases, the division stuff is weird.
        #[arg(short, long)]
        phase_limit: usize,
    },
    /// Just one simple run
    Simple,
}
