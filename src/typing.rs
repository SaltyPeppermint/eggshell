use std::fmt::Display;

use egg::{Id, Language, RecExpr};
use hashbrown::HashMap;
use petgraph::{graph::NodeIndex, Graph};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TypingError {
    #[error("Incomparable types {0} {1}")]
    Incomparable(String, String),
    #[error("Type constraint {constraint} incomparable with childrens type : {found}")]
    ConstraintViolation { constraint: String, found: String },
}

create_exception!(
    eggshell,
    TypingException,
    PyException,
    "Eggshell internal error."
);

impl From<TypingError> for PyErr {
    fn from(err: TypingError) -> PyErr {
        TypingException::new_err(err.to_string())
    }
}

pub trait Typeable: Language + Display {
    type Type: Type;

    /// Returns the typing information associated with that specific type
    fn type_info(&self) -> TypingInfo<Self::Type>;

    /// Checks and infers the node type
    ///
    /// # Errors
    /// If a typing error ooccurs, it is propagated upwards meaning the
    /// Expression is wrongly typed
    fn type_node(&self, expr: &RecExpr<Self>) -> Result<Self::Type, TypingError> {
        let child_type = check_child_coherence(self.children(), expr)?;

        let typing_info = self.type_info();
        let checked_child_type = typing_info.child_constraint.subtype(child_type)?;

        // In case the flag is on the children have no influence on the return type
        if typing_info.infer_return_type {
            typing_info.return_constraint.subtype(checked_child_type)
        } else {
            Ok(typing_info.return_constraint)
        }
    }

    /// Gather node types
    ///
    /// # Errors
    /// If a typing error ooccurs, it is propagated upwards meaning the
    /// Expression is wrongly typed
    fn collect_node_types(
        &self,
        id: Id,
        node_types: &mut HashMap<Id, Self::Type>,
        expr: &RecExpr<Self>,
    ) -> Result<Self::Type, TypingError> {
        let child_type = collect_child_coherence(node_types, self.children(), expr)?;

        let typing_info = self.type_info();
        let checked_child_type = typing_info.child_constraint.subtype(child_type)?;

        // In case the flag is on the children have no influence on the return type
        if typing_info.infer_return_type {
            let return_type = typing_info.return_constraint.subtype(checked_child_type)?;
            node_types.insert(id, return_type);
            Ok(return_type)
        } else {
            node_types.insert(id, typing_info.return_constraint);
            Ok(typing_info.return_constraint)
        }
    }
}

/// Checks if the children have compatible subtypes (i.e. their types c
/// an be ordered) and returns the highest common subtype
///
/// # Errors
/// If the types are not ordered and no lowest subtype (except Bottom)
/// can be inferred an error is returned
fn check_child_coherence<L: Typeable>(
    children: &[Id],
    expr: &RecExpr<L>,
) -> Result<L::Type, TypingError> {
    children
        .iter()
        .map(|id| &expr[*id])
        .map(|c| c.type_node(expr))
        .try_fold(L::Type::top(), move |acc, r| {
            let t = r?;
            let ordering = acc
                .partial_cmp(&t)
                .ok_or(TypingError::Incomparable(acc.to_string(), t.to_string()))?;
            Ok(match ordering {
                std::cmp::Ordering::Greater => t,
                _ => acc,
            })
        })
}

/// Check and collect the [`Typeable::Type`] of children of a node in an [`RecExpr`] of an
/// into a [`HashMap`]. Convenient for debugging.
fn collect_child_coherence<L: Typeable>(
    node_types: &mut HashMap<Id, L::Type>,
    children: &[Id],
    expr: &RecExpr<L>,
) -> Result<L::Type, TypingError> {
    children
        .iter()
        .map(|id| (id, &expr[*id]))
        .map(|(id, c)| c.collect_node_types(*id, node_types, expr))
        .try_fold(L::Type::top(), move |acc, r| {
            let t = r?;
            let ordering = acc
                .partial_cmp(&t)
                .ok_or(TypingError::Incomparable(acc.to_string(), t.to_string()))?;
            Ok(match ordering {
                std::cmp::Ordering::Greater => t,
                _ => acc,
            })
        })
}

/// Checks if an [`RecExpr`] is properly typed and
/// returns the roots [`Typeable::Type`]
///
/// # Errors
///
/// Errors is a typing error coccurs
///
/// # Panics
///
/// Panics on an empty [`RecExpr`]
pub fn typecheck_expr<L: Typeable>(expr: &RecExpr<L>) -> Result<L::Type, TypingError> {
    let root = expr
        .as_ref()
        .last()
        .expect("Can't typecheck an empty recexpr");
    root.type_node(expr)
}

/// Checks if the [`RecExpr`] is properly typed and
/// returns the roots [`Typeable::Type`]
///
/// # Errors
///
/// Errors is a typing error coccurs
///
/// # Panics
///
/// Panics if given [`Id`] is not in the [`RecExpr`]
pub fn typecheck_node<L: Typeable>(expr: &RecExpr<L>, node_id: Id) -> Result<L::Type, TypingError> {
    expr[node_id].type_node(expr)
}

/// Collects the types of an and returns a [`HashMap`] of the
/// [`Id`] with their [`Typeable::Type`]
///
/// # Panics
///
/// Panics if given an empty [`RecExpr`]
///
/// # Errors
///
/// Errors is a typing error coccurs
pub fn collect_expr_types<L: Typeable>(
    expr: &RecExpr<L>,
) -> Result<HashMap<Id, L::Type>, TypingError> {
    let root = expr
        .as_ref()
        .last()
        .expect("Can't typecheck an empty recexpr");

    let mut map = HashMap::new();
    root.collect_node_types(Id::from(expr.as_ref().len() - 1), &mut map, expr)?;
    Ok(map)
}

/// Turn a [`RecExpr`] with a [`HashMap`] of its [`Typeable::Type`]
/// into a Dot compatible [`Graph`]
#[must_use]
pub fn dot_typed_ast<L: Typeable>(
    root_id: Id,
    expr: &RecExpr<L>,
    map: &HashMap<Id, L::Type>,
) -> Graph<String, ()> {
    let mut graph = Graph::<String, ()>::new();
    let root = &expr[root_id];
    let root_string = format!("(ROOT) {}: {}", root, map[&root_id]);
    let root_nidx = graph.add_node(root_string);
    let children = root.children();
    build_dot_graph(children, expr, map, root_nidx, &mut graph);
    graph
}

fn build_dot_graph<L: Typeable>(
    children: &[Id],
    expr: &RecExpr<L>,
    map: &HashMap<Id, L::Type>,
    parent_nidx: NodeIndex,
    graph: &mut Graph<String, ()>,
) {
    for child_id in children {
        let child_node = &expr[*child_id];
        let node_string = format!("({}) {}: {}", child_node, child_id, map[child_id]);
        let child_nidx = graph.add_node(node_string);
        graph.add_edge(parent_nidx, child_nidx, ());
        build_dot_graph(child_node.children(), expr, map, child_nidx, graph);
    }
}
pub trait Type:
    Clone + Copy + Serialize + PartialEq + Eq + PartialOrd + std::hash::Hash + Display
{
    /// Returns the top type of that type system
    fn top() -> Self;
    /// Returns the bottom type of that type system
    fn bottom() -> Self;

    /// Checks the subtyping relationship and returns the subtype
    ///
    /// # Errors
    /// If no common subtype (except the Bottom Type) exists, an error is returned
    fn subtype(self, other: Self) -> Result<Self, TypingError> {
        let ordering = self
            .partial_cmp(&other)
            .ok_or(TypingError::ConstraintViolation {
                constraint: self.to_string(),
                found: other.to_string(),
            })?;
        match ordering {
            std::cmp::Ordering::Equal | std::cmp::Ordering::Less => Ok(self),
            std::cmp::Ordering::Greater => Ok(other),
        }
    }
}

#[derive(Debug, Clone, std::hash::Hash, Serialize)]
pub struct TypingInfo<T: Type> {
    return_constraint: T,
    child_constraint: T,
    infer_return_type: bool,
}

impl<T: Type> TypingInfo<T> {
    pub fn new(return_constraint: T, child_constraint: T) -> Self {
        Self {
            return_constraint,
            child_constraint,
            infer_return_type: false,
        }
    }

    #[must_use]
    pub fn infer_return_type(self) -> Self {
        Self {
            return_constraint: self.return_constraint,
            child_constraint: self.child_constraint,
            infer_return_type: true,
        }
    }
}
