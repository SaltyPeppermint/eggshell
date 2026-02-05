use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser};

use serde::de::DeserializeOwned;

use eggshell::distance::{EGraph, Expr, Label, TreeNode, UnitCost, find_min_sampling_zs};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
#[command(after_help = "\
Examples:
  # Reference tree from file
  extract graph.json -f trees.txt -n blocking_goal

  # Reference tree from command line
  extract graph.json -e '(+ 1 2)'

  # With revisits and quiet mode
  extract graph.json -e '(foo bar)' -r 2 -q
")]
struct Args {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    #[command(flatten)]
    reference: RefSource,

    /// Target weight
    #[arg(short, long, default_value_t = 0)]
    target_weight: usize,

    /// Number of samples
    #[arg(short = 'n', long, default_value_t = 10000)]
    samples: usize,

    /// Include the types in the comparison
    #[arg(short, long)]
    with_types: bool,

    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,
    // /// Use structural distance instead of Zhang-Shasha tree edit distance
    // #[arg(short, long)]
    // structural: bool,

    // /// Ignore the labels when using the structural option
    // #[arg(short, long, requires_all = ["structural"])]
    // ignore_labels: bool,
}

#[derive(ClapArgs)]
struct RefSource {
    /// Reference tree as an s-expression
    #[arg(short = 'e', long = "expr", conflicts_with_all = ["file", "name"])]
    expr: Option<String>,

    /// Path to file containing named trees
    #[arg(short = 'f', long, requires = "name")]
    file: Option<String>,

    /// Name of the reference tree (requires --file)
    #[arg(short = 'n', long, requires = "file")]
    name: Option<String>,
}

fn main() {
    let args = Args::parse();

    if args.raw_strings {
        run(&args, |sexpr| {
            TreeNode::<String>::from_str(sexpr).expect("Failed to parse s-expression")
        });
    } else {
        run(&args, |sexpr| {
            sexpr
                .parse::<Expr>()
                .expect("Failed to parse Rise expression")
                .to_tree(args.with_types)
        });
    }
}

fn run<L, F>(args: &Args, parse_tree: F)
where
    L: Label + std::fmt::Display + DeserializeOwned,
    F: Fn(&str) -> TreeNode<L>,
{
    println!("Loading e-graph from: {}", args.egraph);

    let graph: EGraph<L> = EGraph::parse_from_file(Path::new(&args.egraph));

    println!("  Root e-class: {:?}", graph.root());

    let ref_tree = parse_ref(args, parse_tree);

    run_extraction(&graph, &ref_tree, args);
}

fn parse_ref<L, F>(args: &Args, parse_tree: F) -> TreeNode<L>
where
    L: Label + std::fmt::Display + DeserializeOwned,
    F: Fn(&str) -> TreeNode<L>,
{
    let ref_tree: TreeNode<L> = if let Some(expr) = &args.reference.expr {
        println!("Parsing reference tree from command line...");
        parse_tree(expr)
    } else {
        let file = args.reference.file.as_ref().unwrap();
        let name = args.reference.name.as_ref().unwrap();
        println!("Parsing reference tree '{name}' from file...");
        let content =
            fs::read_to_string(file).unwrap_or_else(|e| panic!("Failed to read '{file}': {e}"));
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .find_map(|line| {
                let (n, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
                if n.trim() == name {
                    Some(parse_tree(sexpr.trim()))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("No tree with name {name} found"))
    };
    ref_tree
}

#[expect(clippy::cast_precision_loss)]
fn run_extraction<L: Label>(graph: &EGraph<L>, ref_tree: &TreeNode<L>, args: &Args) {
    let ref_node_count = ref_tree.size();
    let ref_stripped_count = ref_tree.strip_types().size();
    println!("  Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    println!("\n--- Zhang-Shasha extraction (with lower-bound pruning) ---");
    if let (Some(result), stats) = find_min_sampling_zs(
        graph,
        ref_tree,
        &UnitCost,
        args.with_types,
        args.samples,
        args.target_weight,
        100,
    ) {
        println!("  Best distance: {}", result.1);
        println!("  Time: {:.2?}", start.elapsed());
        println!("\n  Statistics:");
        println!("    Trees enumerated:   {}", stats.trees_enumerated);
        println!(
            "    Trees size pruned:  {} ({:.1}%)",
            stats.size_pruned,
            100.0 * stats.size_pruned as f64 / stats.trees_enumerated as f64
        );
        println!(
            "    Trees euler pruned: {} ({:.1}%)",
            stats.euler_pruned,
            100.0 * stats.euler_pruned as f64 / stats.trees_enumerated as f64
        );
        println!(
            "    Full comparisons:   {} ({:.1}%)",
            stats.full_comparisons,
            100.0 * stats.full_comparisons as f64 / stats.trees_enumerated as f64
        );
        println!("\n  Best tree:");
        println!("{}", result.0);
    } else {
        println!("  No result found!");
    }
}
