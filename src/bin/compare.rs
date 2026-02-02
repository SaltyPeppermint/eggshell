use std::fs;

use clap::Parser;

use eggshell::distance::{Expr, Label, TreeNode, tree_distance_unit};

#[derive(Parser)]
#[command(about = "Compare trees using Zhang-Shasha edit distance")]
struct Args {
    /// File containing trees in "Name: sexpr" format
    file: String,

    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,

    /// Strip type annotations from trees before comparison
    #[arg(long)]
    strip_types: bool,
}

fn main() {
    let args = Args::parse();
    let content = fs::read_to_string(&args.file)
        .unwrap_or_else(|e| panic!("Failed to read '{}': {e}", args.file));

    if args.raw_strings {
        run(&content, |sexpr| {
            let raw_tree: TreeNode<String> = sexpr.parse().expect("Failed to parse s-expression");
            if args.strip_types {
                raw_tree.strip_types()
            } else {
                raw_tree
            }
        });
    } else {
        run(&content, |sexpr| {
            let expr: Expr = sexpr.parse().expect("Failed to parse Rise expression");
            if args.strip_types {
                expr.to_untyped_tree()
            } else {
                expr.to_typed_tree()
            }
        });
    }
}

fn run<L, F>(content: &str, parse_tree: F)
where
    L: Label,
    F: Fn(&str) -> TreeNode<L>,
{
    let trees: Vec<(String, TreeNode<L>)> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (name, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
            (name.trim().to_owned(), parse_tree(sexpr.trim()))
        })
        .collect();

    print_distance_matrix(&trees);
}

fn print_distance_matrix<L>(trees: &[(String, TreeNode<L>)])
where
    L: Label,
{
    if trees.is_empty() {
        println!("No trees found in file.");
        return;
    }

    println!("Loaded {} trees:", trees.len());
    for (name, _) in trees {
        println!("  - {name}");
    }
    println!();

    // Find max name length for alignment
    let max_name_len = trees.iter().map(|(n, _)| n.len()).max().unwrap_or(0);

    // Print header
    print!("{:width$}", "", width = max_name_len + 2);
    for (name, _) in trees {
        print!("{:>width$}", name, width = max_name_len + 2);
    }
    println!();

    // Print separator
    let total_width = (max_name_len + 2) * (trees.len() + 1);
    println!("{}", "-".repeat(total_width));

    // Compute distance matrix (only upper triangle since symmetric)
    let n = trees.len();
    let mut distances = vec![vec![0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = tree_distance_unit(&trees[i].1, &trees[j].1);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Print distance matrix
    for (i, (name_i, _)) in trees.iter().enumerate() {
        print!("{:width$}", name_i, width = max_name_len + 2);
        for dist in &distances[i] {
            print!("{dist:>width$}", width = max_name_len + 2);
        }
        println!();
    }
}
