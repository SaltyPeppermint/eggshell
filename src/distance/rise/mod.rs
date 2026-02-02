//! Typed representation of the Rise language for tree edit distance computation.
//!
//! This module provides a proper typed AST representation of Rise expressions,
//! with S-expression parsing and serialization support.

mod address;
mod expr;
mod label;
mod nat;
mod primitive;
mod types;

use std::num::ParseIntError;

pub use address::Address;
pub use expr::{Expr, ExprNode, LiteralData};
pub use label::RiseLabel;
pub use nat::Nat;
pub use primitive::Primitive;
pub use types::{DataType, ScalarType, Type};

use symbolic_expressions::SexpError;
use thiserror::Error;

/// Error type for parsing Rise expressions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseError {
    #[error("S-expression parse error: {0}")]
    Sexp(String),
    #[error("invalid expression: {0}")]
    Expr(String),
    #[error("invalid primitive: {0}")]
    Prim(String),
    #[error("invalid type: {0}")]
    Type(String),
    #[error("invalid nat: {0}")]
    Nat(String),
    #[error("invalid address: {0}")]
    Address(String),
    #[error("invalid label: {0}")]
    Label(String),
    #[error("invalid variable index '{input}': {reason}")]
    VarIndex {
        input: String,
        reason: ParseIntError,
    },
    #[error("invalid literal '{input}': {reason}")]
    Literal {
        input: String,
        reason: ParseIntError,
    },
}

impl From<SexpError> for ParseError {
    fn from(e: SexpError) -> Self {
        ParseError::Sexp(format!("{e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use crate::distance::tree::TreeNode;
    use crate::distance::zs::tree_distance_unit;

    #[test]
    fn rise_label_works_with_zs() {
        // Create two simple trees with RiseLabel
        let tree1 = TreeNode::new(
            RiseLabel::App,
            vec![
                TreeNode::leaf(RiseLabel::Primitive(Primitive::Map)),
                TreeNode::new(RiseLabel::Lambda, vec![TreeNode::leaf(RiseLabel::Var(0))]),
            ],
        );

        let tree2 = TreeNode::new(
            RiseLabel::App,
            vec![
                TreeNode::leaf(RiseLabel::Primitive(Primitive::Map)),
                TreeNode::new(RiseLabel::Lambda, vec![TreeNode::leaf(RiseLabel::Var(1))]),
            ],
        );

        // Same structure, different variable index - should be distance 1
        let distance = tree_distance_unit(&tree1, &tree2);
        assert_eq!(distance, 1);

        // Identical trees should have distance 0
        let distance_same = tree_distance_unit(&tree1, &tree1);
        assert_eq!(distance_same, 0);
    }

    #[test]
    fn rise_label_with_floats_in_zs() {
        use ordered_float::OrderedFloat;

        let tree1 = TreeNode::leaf(RiseLabel::FloatLit(OrderedFloat(3.11)));
        let tree2 = TreeNode::leaf(RiseLabel::FloatLit(OrderedFloat(3.11)));
        let tree3 = TreeNode::leaf(RiseLabel::FloatLit(OrderedFloat(2.71)));

        // Same float value - distance 0
        assert_eq!(tree_distance_unit(&tree1, &tree2), 0);

        // Different float value - distance 1 (relabel)
        assert_eq!(tree_distance_unit(&tree1, &tree3), 1);
    }

    #[test]
    fn rise_expr_to_tree_with_zs() {
        let expr1: Expr = "(app map (lam $e0))".parse().unwrap();
        let expr2: Expr = "(app map (lam $e1))".parse().unwrap();

        let tree1 = expr1.to_untyped_tree();
        let tree2 = expr2.to_untyped_tree();

        // Different variable index - distance 1
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1);
    }

    #[test]
    fn rise_label_egraph_deserialize() {
        use crate::distance::EGraph;
        use crate::distance::ids::NumericId;
        use std::path::Path;

        let path = Path::new(
            "data/rise/egraph_jsons/ser_egraph_vectorization_SRL_step_2_iteration_0_root_150.json",
        );
        if !path.exists() {
            return;
        }

        let graph: EGraph<RiseLabel> = EGraph::parse_from_file(path);

        // Verify root is correct
        assert_eq!(graph.root().to_index(), 150);

        // Verify we can access the root class
        let root_class = graph.class(graph.root());
        assert!(!root_class.nodes().is_empty());

        // Verify the first node has the expected label
        let first_node = &root_class.nodes()[0];
        assert_eq!(*first_node.label(), RiseLabel::NatLambda);
    }

    #[test]
    fn parse_rise_file_sample() {
        let path = Path::new("data/tree_comps/rise.txt");
        if !path.exists() {
            // Skip test if file doesn't exist (e.g., in CI)
            return;
        }

        let content = fs::read_to_string(path).unwrap();

        let mut success = 0;
        let mut failures = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let (name, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");

            match sexpr.trim().parse::<Expr>() {
                Ok(_) => success += 1,
                Err(e) => failures.push((name.trim().to_owned(), e.to_string())),
            }
        }

        // Print results
        eprintln!("\nParsed {success} expressions successfully");
        if !failures.is_empty() {
            eprintln!("{} failures:", failures.len());
            for (name, err) in &failures[..failures.len().min(5)] {
                eprintln!("  {name}: {err}");
            }
        }

        // All expressions should parse successfully
        assert!(
            failures.is_empty(),
            "Failed to parse {} expressions",
            failures.len()
        );
    }
}
