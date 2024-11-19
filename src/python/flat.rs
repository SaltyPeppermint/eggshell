use std::fmt::{Debug, Display};

use egg::{Analysis, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;
use numpy::{IntoPyArray, Ix1, Ix2, PyArray, PyArray2};
use pyo3::prelude::*;

use super::{RawAst, SymbolTable};
use crate::eqsat::EqsatResult;
use crate::features;
use crate::trs::{TermRewriteSystem, TrsLang};

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

impl From<&RawAst> for FlatAst {
    fn from(root: &RawAst) -> Self {
        fn rec(
            parent_idx: usize,
            root_distance: usize,
            node: &RawAst,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
        ) {
            nodes.push(FlatNode {
                name: node.name().to_owned(),
                root_distance,
                features: node.features().expect("Cannot transform into non-feature"),
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            match node {
                RawAst::Active | RawAst::Open | RawAst::Any => (),
                RawAst::Node { children, .. } => {
                    for c in children {
                        rec(current_idx, root_distance + 1, c, nodes, edges);
                    }
                }
                RawAst::Or(children) => {
                    rec(current_idx, root_distance + 1, &children[0], nodes, edges);
                    rec(current_idx, root_distance + 1, &children[1], nodes, edges);
                }
                RawAst::Contains(child) => {
                    rec(current_idx, root_distance + 1, child, nodes, edges);
                }
            }
        }

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        rec(0, 0, root, &mut nodes, &mut edges);
        FlatAst { nodes, edges }
    }
}

fn vars_in_expr<L: TrsLang>(expr: &RecExpr<L>) -> usize {
    expr.as_ref().iter().filter(|n| n.is_var()).count()
}

impl<L: TrsLang> From<&RecExpr<L>> for FlatAst {
    #[expect(clippy::too_many_arguments)]
    fn from(expr: &RecExpr<L>) -> Self {
        fn rec<IL: TrsLang>(
            node_id: Id,
            expr: &RecExpr<IL>,
            parent_idx: usize,
            root_distance: usize,
            nodes: &mut Vec<FlatNode>,
            edges: &mut Vec<(usize, usize)>,
            pos: usize,
            vars_in_expr: usize,
        ) {
            let node = &expr[node_id];
            nodes.push(FlatNode {
                name: node.to_string(),
                root_distance,
                features: [
                    features::symbol_features(node, vars_in_expr),
                    features::additional(node, pos, root_distance),
                ]
                .concat(),
            });
            let current_idx = nodes.len() - 1;
            edges.push((parent_idx, current_idx));
            for (c_pos, c_id) in node.children().iter().enumerate() {
                rec(
                    *c_id,
                    expr,
                    current_idx,
                    root_distance + 1,
                    nodes,
                    edges,
                    c_pos,
                    vars_in_expr,
                );
            }
        }

        let vars_in_expr = vars_in_expr(expr);
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let root = (expr.as_ref().len() - 1).into();
        rec(root, expr, 0, 0, &mut nodes, &mut edges, 0, vars_in_expr);
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
    #[pyo3(get)]
    features: Vec<f64>,
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
    vertices: Vec<FlatVertex>,
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

    pub fn nodes_to_numpy<'py>(
        &self,
        py: Python<'py>,
        additional_features: usize,
        int_bounds: (f32, f32),
        lut: &SymbolTable,
    ) -> PyResult<Bound<'py, PyArray<f32, Ix2>>> {
        // symbol_table: &HashMap<String, Symbol>

        let v_iter = self
            .vertices
            .iter()
            .map(|v| v.to_feature_vec(lut, additional_features, int_bounds))
            .collect::<Result<Vec<Vec<f32>>, PyErr>>()?;

        Ok(PyArray2::from_vec2_bound(py, &v_iter)?)
    }
}

impl<R: TermRewriteSystem> From<&EqsatResult<R>> for FlatEGraph {
    fn from(eqsat_result: &EqsatResult<R>) -> Self {
        (eqsat_result.egraph(), eqsat_result.roots()).into()
    }
}

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub enum FlatVertex {
    EClass { id: usize, analysis: String },
    ENode { symbol: String, position: usize },
}

impl FlatVertex {
    pub fn to_feature_vec(
        &self,
        lut: &SymbolTable,
        additional_features: usize,
        int_bounds: (f32, f32),
    ) -> Result<Vec<f32>, PyErr> {
        // Int encoding (included in len) + EClass hot-one-encoding + visited marker
        let n_symbols = lut.len();
        let f_vec_len = n_symbols + 1 + additional_features;
        match self {
            FlatVertex::EClass { .. } => {
                let mut v = vec![0.0; f_vec_len];
                v[n_symbols + 1] = 1.0;
                Ok(v)
            }
            FlatVertex::ENode { symbol, .. } => lut
                .get_symbol(symbol.as_str())
                .map(|s| s.to_feature_vec(f_vec_len, n_symbols, int_bounds)),
        }
    }
}

#[pymethods]
impl FlatVertex {
    pub fn to_feature_np<'py>(
        &self,
        py: Python<'py>,
        lut: &SymbolTable,
        additional_features: usize,
        bounds: (f32, f32),
    ) -> PyResult<Bound<'py, PyArray<f32, Ix1>>> {
        self.to_feature_vec(lut, additional_features, bounds)
            .map(|v| v.into_pyarray_bound(py))
    }

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
            vertices: &mut Vec<FlatVertex>,
            edges: &mut Vec<(usize, usize)>,
            visited_classes: &mut HashMap<Id, usize>,
        ) where
            L: Language + Display,
            N: Analysis<L>,
            N::Data: Debug,
        {
            let eclass_idx =
                insert_eclass(vertices, id, egraph, edges, parent_idx, visited_classes);

            for (position, node) in egraph[id].nodes.iter().enumerate() {
                let node_idx = insert_enode(vertices, node, position, edges, eclass_idx);

                for child_id in node.children() {
                    if let Some(child_eclass_idx) = visited_classes.get(child_id) {
                        edges.push((node_idx, *child_eclass_idx));
                    } else {
                        rec(parent_idx, id, egraph, vertices, edges, visited_classes);
                    }
                }
            }
        }

        fn insert_eclass<L, N>(
            vertices: &mut Vec<FlatVertex>,
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
            vertices.push(FlatVertex::EClass {
                id: id.into(),
                analysis: format!("{:?}", egraph[id].data),
            });
            let eclass_idx = vertices.len() - 1;
            edges.push((parent_idx, eclass_idx));
            visited_classes.insert(id, eclass_idx);
            eclass_idx
        }

        fn insert_enode<L: Language + Display>(
            vertices: &mut Vec<FlatVertex>,
            node: &L,
            position: usize,
            edges: &mut Vec<(usize, usize)>,
            eclass_idx: usize,
        ) -> usize {
            vertices.push(FlatVertex::ENode {
                symbol: node.to_string(),
                position,
            });
            let node_idx = vertices.len() - 1;
            edges.push((eclass_idx, node_idx));
            node_idx
        }

        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        let mut roots = Vec::new();

        let mut visited_classes = HashMap::new();
        for root_id in value.1 {
            let root = *root_id;
            vertices.push(FlatVertex::EClass {
                id: root.into(),
                analysis: format!("{:?}", value.0[root].data),
            });
            let root_idx = vertices.len() - 1;
            roots.push(root_idx);
            rec(
                root_idx,
                root,
                value.0,
                &mut vertices,
                &mut edges,
                &mut visited_classes,
            );
        }

        FlatEGraph {
            vertices,
            edges,
            roots,
        }
    }
}
