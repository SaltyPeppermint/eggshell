use std::fmt::{Debug, Display};

use egg::{Analysis, EGraph, Id, Language};
use pyo3::prelude::*;

use crate::{utils::Tree, HashMap};

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

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct FlatEGraph {
    #[pyo3(get)]
    nodes: Vec<FlatVertex>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
    #[pyo3(get)]
    roots: Vec<usize>,
}

#[pymethods]
impl FlatEGraph {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub enum FlatVertex {
    EClass { id: usize, analysis: String },
    ENode { symbol: String, position: usize },
}

#[pymethods]
impl FlatVertex {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl<L, N> From<(&EGraph<L, N>, &[Id])> for FlatEGraph
where
    L: Language + Display,
    N: Analysis<L>,
    N::Data: Debug,
{
    fn from(value: (&EGraph<L, N>, &[Id])) -> Self {
        fn rec<L, N>(
            parent_idx: usize,
            id: Id,
            egraph: &EGraph<L, N>,
            nodes: &mut Vec<FlatVertex>,
            edges: &mut Vec<(usize, usize)>,
            visited_classes: &mut HashMap<Id, usize>,
        ) where
            L: Language + Display,
            N: Analysis<L>,
            N::Data: Debug,
        {
            let eclass_idx = insert_eclass(nodes, id, egraph, edges, parent_idx, visited_classes);

            for (position, node) in egraph[id].nodes.iter().enumerate() {
                let node_idx = insert_enode(nodes, node, position, edges, eclass_idx);

                for child_id in node.children() {
                    if let Some(child_eclass_idx) = visited_classes.get(child_id) {
                        edges.push((node_idx, *child_eclass_idx));
                    } else {
                        rec(parent_idx, id, egraph, nodes, edges, visited_classes);
                    }
                }
            }
        }

        fn insert_eclass<L, N>(
            nodes: &mut Vec<FlatVertex>,
            id: Id,
            egraph: &EGraph<L, N>,
            edges: &mut Vec<(usize, usize)>,
            parent_idx: usize,
            visited_classes: &mut HashMap<Id, usize>,
        ) -> usize
        where
            L: Language,
            N: Analysis<L>,
            N::Data: Debug,
        {
            nodes.push(FlatVertex::EClass {
                id: id.into(),
                analysis: format!("{:?}", egraph[id].data),
            });
            let eclass_idx = nodes.len() - 1;
            edges.push((parent_idx, eclass_idx));
            visited_classes.insert(id, eclass_idx);
            eclass_idx
        }

        fn insert_enode<L: Language + Display>(
            nodes: &mut Vec<FlatVertex>,
            node: &L,
            position: usize,
            edges: &mut Vec<(usize, usize)>,
            eclass_idx: usize,
        ) -> usize {
            nodes.push(FlatVertex::ENode {
                symbol: node.to_string(),
                position,
            });
            let node_idx = nodes.len() - 1;
            edges.push((eclass_idx, node_idx));
            node_idx
        }

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut roots = Vec::new();

        let mut visited_classes = HashMap::new();
        for root_id in value.1 {
            let root = *root_id;
            nodes.push(FlatVertex::EClass {
                id: root.into(),
                analysis: format!("{:?}", value.0[root].data),
            });
            let root_idx = nodes.len() - 1;
            roots.push(root_idx);
            rec(
                root_idx,
                root,
                value.0,
                &mut nodes,
                &mut edges,
                &mut visited_classes,
            );
        }

        FlatEGraph {
            nodes,
            edges,
            roots,
        }
    }
}
