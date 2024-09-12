use pyo3::prelude::*;

use crate::utils::Tree;

use super::{RawLang, RawSketch};

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAst {
    #[pyo3(get)]
    nodes: Vec<FlatNode>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
}

impl From<&RawSketch> for FlatAst {
    fn from(value: &RawSketch) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            ast: &RawSketch,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
        ) {
            nodes.push(FlatNode {
                name: ast.to_string(),
                root_distance,
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            match ast {
                RawSketch::Active | RawSketch::Open | RawSketch::Any => (),
                RawSketch::Node {
                    lang_node: _,
                    children,
                } => {
                    for c in children {
                        rec(current_idx, root_distance + 1, c, nodes, edges);
                    }
                }
                RawSketch::Or(children) => {
                    rec(current_idx, root_distance + 1, &children[0], nodes, edges);
                    rec(current_idx, root_distance + 1, &children[1], nodes, edges);
                }
                RawSketch::Contains(node) => {
                    rec(current_idx, root_distance + 1, node, nodes, edges);
                }
            }
        }

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        nodes.push(FlatNode {
            name: value.to_string(),
            root_distance: 0,
        });

        rec(0, 1, value, &mut nodes, &mut edges);
        FlatAst { nodes, edges }
    }
}

impl From<&RawLang> for FlatAst {
    fn from(value: &RawLang) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            ast: &RawLang,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
        ) {
            nodes.push(FlatNode {
                name: ast.to_string(),
                root_distance,
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            for c in ast.children() {
                rec(current_idx, root_distance + 1, c, nodes, edges);
            }
        }

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        nodes.push(FlatNode {
            name: value.to_string(),
            root_distance: 0,
        });

        rec(0, 1, value, &mut nodes, &mut edges);
        FlatAst { nodes, edges }
    }
}

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct FlatNode {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    root_distance: usize,
}
