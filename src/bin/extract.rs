use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser};

use serde::de::DeserializeOwned;

use eggshell::distance::{EGraph, Expr, Label, TreeNode, UnitCost, find_min_struct, find_min_zs};

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
#[expect(clippy::struct_excessive_bools)]
struct Args {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    #[command(flatten)]
    reference: RefSource,

    /// Maximum number of times a node may be revisited (for cycles)
    #[arg(short = 'r', long, default_value_t = 0)]
    max_revisits: usize,

    /// Include the types in the comparison
    #[arg(short, long)]
    with_types: bool,

    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,

    /// Use structural distance instead of Zhang-Shasha tree edit distance
    #[arg(short, long)]
    structural: bool,

    /// Ignore the labels when using the structural option
    #[arg(short, long, requires_all = ["structural"])]
    ignore_labels: bool,
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

    run_extraction(&graph, &ref_tree, args);
}

fn run_extraction<L: Label + std::fmt::Display>(
    graph: &EGraph<L>,
    ref_tree: &TreeNode<L>,
    args: &Args,
) {
    let ref_node_count = ref_tree.size();
    println!("  Reference tree has {ref_node_count} nodes");

    // Count trees in the e-graph
    println!(
        "\nCounting trees in e-graph (max_revisits={})...",
        args.max_revisits
    );
    let count_start = Instant::now();
    let tree_count = graph.count_trees(args.max_revisits);
    let count_time = count_start.elapsed();
    println!("  Found {tree_count} trees in {count_time:.2?}");

    if tree_count == 0 {
        println!("No trees found in e-graph!");
        return;
    }

    if args.structural {
        run_structural(graph, ref_tree, args);
    } else {
        run_zs(graph, ref_tree, args);
    }
}

#[expect(clippy::cast_precision_loss)]
fn run_zs<L: Label>(graph: &EGraph<L>, ref_tree: &TreeNode<L>, args: &Args) {
    let start = Instant::now();
    println!("\n--- Zhang-Shasha extraction (with lower-bound pruning) ---");
    if let (Some(result), stats) = find_min_zs(
        graph,
        ref_tree,
        &UnitCost,
        args.max_revisits,
        args.with_types,
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

fn run_structural<L: Label>(graph: &EGraph<L>, ref_tree: &TreeNode<L>, args: &Args) {
    let start = Instant::now();
    println!("\n--- Structural distance extraction ---");
    if let Some((tree, distance)) = find_min_struct(
        graph,
        ref_tree,
        &UnitCost,
        args.max_revisits,
        args.with_types,
        args.ignore_labels,
    ) {
        println!("  Best distance: {distance}");
        println!("  Time: {:.2?}", start.elapsed());
        println!("\n  Best tree:");
        println!("{tree}");
    } else {
        println!("  No result found!");
    }
}
