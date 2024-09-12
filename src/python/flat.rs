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

#[pymethods]
impl FlatAst {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl From<&RawSketch> for FlatAst {
    fn from(root: &RawSketch) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            sketch_node: &RawSketch,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
        ) {
            nodes.push(FlatNode {
                name: sketch_node.name().to_owned(),
                root_distance,
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            match sketch_node {
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

        rec(0, 0, root, &mut nodes, &mut edges);
        FlatAst { nodes, edges }
    }
}

impl From<&RawLang> for FlatAst {
    fn from(root: &RawLang) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            lang_node: &RawLang,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
        ) {
            nodes.push(FlatNode {
                name: lang_node.name().to_owned(),
                root_distance,
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            for c in lang_node.children() {
                rec(current_idx, root_distance + 1, c, nodes, edges);
            }
        }

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        rec(0, 0, root, &mut nodes, &mut edges);
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

#[pymethods]
impl FlatNode {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}
