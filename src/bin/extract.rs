use std::str::FromStr;
use std::time::Instant;
use std::{fs, path::Path};

use clap::Parser;
use eggshell::distance::{
    EGraph, TreeNode, min_distance_extract_filtered, min_distance_extract_unit,
};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
struct Args {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    /// Path to the serialized reference tree JSON file
    trees: String,

    /// Name of the reference tree
    ref_name: String,

    /// Maximum number of times a node may be revisited (for cycles)
    #[arg(short, long, default_value_t = 0)]
    max_revisits: usize,

    /// Skip the filtered extraction (only run unfiltered for comparison)
    #[arg(long)]
    no_filter: bool,

    /// Skip the baseline comparison
    #[arg(long, default_value_t = false)]
    baseline: bool,
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let args = Args::parse();

    // Load and parse the e-graph
    println!("Loading e-graph from: {}", args.egraph);

    let graph = EGraph::<String>::parse_from_file(Path::new(&args.egraph));

    println!("  Root e-class: {:?}", graph.root());

    // Parse the reference tree
    println!("Parsing reference tree...");
    let content = fs::read_to_string(&args.trees)
        .unwrap_or_else(|e| panic!("Failed to read '{}': {e}", args.trees));
    let ref_tree = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| {
            let (name, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
            (name.trim() == args.ref_name)
                .then(|| TreeNode::from_str(sexpr.trim()).expect("Failed to parse s-expression"))
        })
        .unwrap_or_else(|| panic!("No tree with name {} found", args.ref_name));

    let ref_node_count = ref_tree.node_count();
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

    // Run filtered extraction
    if !args.no_filter {
        println!("\n--- Filtered extraction (with lower-bound pruning) ---");
        let start = Instant::now();

        if let (Some(result), stats) =
            min_distance_extract_filtered(&graph, &ref_tree, args.max_revisits)
        {
            println!("  Best distance: {}", result.distance);
            println!("  Time: {:.2?}", start.elapsed());
            println!("\n  Statistics:");
            println!("    Trees enumerated: {}", stats.trees_enumerated);
            println!(
                "    Trees pruned:     {} ({:.1}%)",
                stats.trees_pruned,
                100.0 * stats.trees_pruned as f64 / stats.trees_enumerated as f64
            );
            println!(
                "    Full comparisons: {} ({:.1}%)",
                stats.full_comparisons,
                100.0 * stats.full_comparisons as f64 / stats.trees_enumerated as f64
            );

            // Print the best tree
            println!("\n  Best tree: {:?}", result.tree);
        } else {
            println!("  No result found!");
        }
    }

    // Run unfiltered extraction for comparison
    if args.baseline {
        println!("\n--- Unfiltered extraction (baseline) ---");
        let start = Instant::now();
        if let Some(result) = min_distance_extract_unit(&graph, &ref_tree, args.max_revisits, false)
        {
            println!("  Best distance: {}", result.distance);
            println!("  Time: {:.2?}", start.elapsed());
        } else {
            println!("  No result found!");
        }
    }
}
