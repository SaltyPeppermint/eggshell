use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser};

use eggshell::distance::{EGraph, TreeNode, UnitCost, find_min};
use symbolic_expressions::Sexp;

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

    /// Maximum number of times a node may be revisited (for cycles)
    #[arg(short = 'r', long, default_value_t = 0)]
    max_revisits: usize,

    /// Include the types in the comparison
    #[arg(short, long)]
    with_types: bool,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    cmd: bool,
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

#[allow(clippy::cast_precision_loss)]
fn main() {
    let args = Args::parse();
    // Load and parse the e-graph
    eprintln!("Loading e-graph from: {}", args.egraph);

    let graph = EGraph::<String>::parse_from_file(Path::new(&args.egraph));

    eprintln!("  Root e-class: {:?}", graph.root());

    // Parse the reference tree
    let ref_tree = if let Some(expr) = args.reference.expr {
        eprintln!("Parsing reference tree from command line...");
        TreeNode::from_str(&expr).expect("Failed to parse s-expression")
    } else {
        let file = args.reference.file.unwrap();
        let name = args.reference.name.unwrap();
        eprintln!("Parsing reference tree '{name}' from file...");
        let content =
            fs::read_to_string(&file).unwrap_or_else(|e| panic!("Failed to read '{file}': {e}"));
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .find_map(|line| {
                let (n, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
                if n.trim() == name {
                    Some(TreeNode::from_str(sexpr.trim()).expect("Failed to parse s-expression"))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("No tree with name {name} found"))
    };

    let ref_node_count = ref_tree.size();
    eprintln!("  Reference tree has {ref_node_count} nodes");

    // Count trees in the e-graph
    eprintln!(
        "\nCounting trees in e-graph (max_revisits={})...",
        args.max_revisits
    );
    let count_start = Instant::now();
    let tree_count = graph.count_trees(args.max_revisits);
    let count_time = count_start.elapsed();
    eprintln!("  Found {tree_count} trees in {count_time:.2?}");

    if tree_count == 0 {
        println!("No trees found in e-graph!");
        return;
    }

    // Run filtered extraction

    eprintln!("\n--- Filtered extraction (with lower-bound pruning) ---");
    let start = Instant::now();

    if let (Some(result), stats) = find_min(
        &graph,
        &ref_tree,
        &UnitCost,
        args.max_revisits,
        args.with_types,
    ) {
        let best_tree = Sexp::from(&result.0).to_string();

        eprintln!("  Best distance: {}", result.1);
        eprintln!("  Time: {:.2?}", start.elapsed());
        eprintln!("\n  Statistics:");
        eprintln!("    Trees enumerated:   {}", stats.trees_enumerated);
        eprintln!(
            "    Trees size pruned:  {} ({:.1}%)",
            stats.size_pruned,
            100.0 * stats.size_pruned as f64 / stats.trees_enumerated as f64
        );
        eprintln!(
            "    Trees euler pruned: {} ({:.1}%)",
            stats.euler_pruned,
            100.0 * stats.euler_pruned as f64 / stats.trees_enumerated as f64
        );
        eprintln!(
            "    Full comparisons:   {} ({:.1}%)",
            stats.full_comparisons,
            100.0 * stats.full_comparisons as f64 / stats.trees_enumerated as f64
        );
        eprintln!("\n  Best tree:");
        eprintln!("{best_tree}");
        if args.cmd {
            println!("{best_tree}");
        }
    } else if !args.cmd {
        eprintln!("  No result found!");
    }
}
