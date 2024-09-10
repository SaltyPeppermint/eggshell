use pyo3::prelude::*;

use super::{PyLang, PySketch};

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct FlatAst {
    #[pyo3(get)]
    nodes: Vec<FlatNode>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
}

impl From<&PySketch> for FlatAst {
    fn from(value: &PySketch) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            ast: &PySketch,
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
                PySketch::Active {} | PySketch::Todo {} | PySketch::Any {} => (),
                PySketch::Node {
                    lang_node: _,
                    children,
                }
                | PySketch::Or { children } => {
                    for c in children {
                        rec(current_idx, root_distance + 1, c, nodes, edges);
                    }
                }
                PySketch::Contains { node } => {
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

impl From<&PyLang> for FlatAst {
    fn from(value: &PyLang) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            ast: &PyLang,
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
