use std::fmt::Display;

use egg::{Analysis, EGraph, Language};
use hashbrown::HashMap as HashBrownMap;
use hashbrown::HashSet as HashBrownSet;
use pyo3::pyclass;
use rayon::prelude::*;
use serde::Serialize;

use crate::eqsat::ClassId;

/// Flattened representation of an [`EGraph`] for consumption by a neural network.
/// Extraction should happen in here as well.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FlatGraph {
    /// Roots of the [`FlatGraph`], if any
    pub roots: HashBrownSet<ClassId>,
    /// Vector of Vertices that could be [`EClass`] or [`ENode`]
    ///
    /// [`EClass`]: Vertex::EClass
    /// [`ENode`]: Vertex::ENode
    pub vertices: Vec<Vertex>,
    /// [`edges`] contains the directed edges between the vertices of the
    /// graph.
    ///     Key:    From where the edges are going
    ///     Value:  To which (0-n) vertices the edges are going.
    ///             The order here is important!
    ///
    /// [`edges`]: FlatGraph::edges
    pub edges: HashBrownMap<usize, Vec<usize>>,
    /// Mapping the Ids of the or [`egg::EClass`] in the original [`EGraph`] to
    /// indices into [`vertices`].
    ///
    /// [`vertices`]: FlatGraph::vertices
    pub eclass_ids: HashBrownMap<ClassId, usize>,
}

impl FlatGraph {
    /// Build the [`FlatGraph`] representation of the [`EGraph`] from an [`EGraph`] and
    /// a list of its roots given via their [`ClassId`].
    pub(crate) fn from_egraph<L, N>(egraph: &EGraph<L, N>, roots: &[ClassId]) -> FlatGraph
    where
        L: Language + Display,
        N: Analysis<L>,
        N::Data: Clone,
    {
        let mut vertices = Vec::new();
        let mut edges = HashBrownMap::new();
        let mut eclass_ids = HashBrownMap::new();

        // First add all the eclasses
        for eclass in egraph.classes() {
            let canonical_id = egraph.find(eclass.id);
            if eclass_ids.contains_key(&canonical_id) {
                continue;
            }
            let neural_eclass = Vertex::EClass {
                is_root: roots.contains(&canonical_id),
            };
            let idx = push_get_index(&mut vertices, neural_eclass);
            eclass_ids.insert(canonical_id, idx);
        }

        for (eclass_id, eclass_idx) in &eclass_ids {
            // Iterate over all eclasses to add the connection
            // 1) Pointing from the eclass-node to the enode-nodes
            // 2) Pointing from the nodes contained in this eclass to the other eclasses
            //    these nodes point to.
            let mut enode_indices = Vec::new();
            for enode in &egraph[*eclass_id].nodes {
                let neural_enode = Vertex::ENode {
                    expr: enode.to_string(),
                };
                // Push enode onto vertices vector and get its index
                let enode_idx = push_get_index(&mut vertices, neural_enode);
                // Add the connectino from the parent eclass to the enode in it
                enode_indices.push(enode_idx);

                // Get the the indices of the children eclasses the enode points to
                let children_idx = enode
                    .children()
                    .iter()
                    .map(|child_id| eclass_ids[child_id])
                    .collect();
                edges.insert(enode_idx, children_idx);
            }
            edges.insert(*eclass_idx, enode_indices);
        }

        FlatGraph {
            roots: HashBrownSet::new(),
            vertices,
            edges,
            eclass_ids,
        }
        .set_roots(roots)
    }

    /// Add roots via the [`ClassId`] of the [`egg::EClass`] in the
    /// original [`EGraph`].
    /// By default the graph has no root!
    pub(crate) fn set_roots(mut self, roots: &[ClassId]) -> Self {
        for root in roots {
            self.roots.insert(*root);
        }
        self
    }

    pub(crate) fn remap_costs(
        &self,
        node_costs: &HashBrownMap<usize, f64>,
    ) -> HashBrownMap<ClassId, Vec<f64>> {
        self.eclass_ids
            .par_iter()
            .map(|(eclass_id, eclass_idx)| {
                let node_indices = &self.edges[eclass_idx];
                let costs_in_eclass = node_indices
                    .iter()
                    .map(|node_idx| node_costs[node_idx])
                    .collect();
                (*eclass_id, costs_in_eclass)
            })
            .collect()
    }
}

/// [`Vertex`] in a [`FlatGraph`]
#[pyclass]
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum Vertex {
    EClass { is_root: bool },
    ENode { expr: String },
}
impl Vertex {
    /// Returns `true` if the vertex is [`EClass`].
    ///
    /// [`EClass`]: Vertex::EClass
    #[must_use]
    pub fn is_eclass(&self) -> bool {
        matches!(self, Self::EClass { .. })
    }

    /// Returns true if it a root [`EClass`].
    ///
    /// [`EClass`]: Vertex::EClass
    #[must_use]
    pub fn is_root_eclass(&self) -> bool {
        if let Vertex::EClass { is_root } = self {
            return *is_root;
        }
        false
    }

    /// Returns `true` if the vertex is [`ENode`].
    ///
    /// [`ENode`]: Vertex::ENode
    #[must_use]
    pub fn is_enode(&self) -> bool {
        matches!(self, Self::ENode { .. })
    }

    /// Returns the expression in the [`ENode`].
    /// Otherwise empty String
    ///
    /// [`ENode`]: Vertex::ENode
    #[must_use]
    pub fn enode_content(&self) -> String {
        if let Vertex::ENode { expr } = self {
            return expr.clone();
        }
        String::new()
    }
}

/// Push something onto a vector and get its index in return
#[must_use]
fn push_get_index<T>(v: &mut Vec<T>, item: T) -> usize {
    let idx = v.len();
    v.push(item);
    idx
}
