use std::fmt::Display;

use egg::{Id, Language, RecExpr};
use pyo3::{create_exception, exceptions::PyException, PyErr};

use thiserror::Error;

pub trait Typeable: Language {
    type Type: PartialOrd + Display + Default;

    /// Returns the type of a node
    ///
    /// # Errors
    /// If a typing error ooccurs, it is propagated upwards meaning the
    /// Expression is wrongly typed
    fn type_node(&self, expr: &RecExpr<Self>) -> Result<Self::Type, TypingError>;

    /// Checks if the child types are subtypes of the parents type constraint
    /// and returns that subtype
    ///
    /// # Errors
    /// If the types are incompatible, an error is returned
    fn check_type_constraints(
        highest_allowed_type: Self::Type,
        child_type: Self::Type,
    ) -> Result<Self::Type, TypingError> {
        let ordering = highest_allowed_type.partial_cmp(&child_type).ok_or(
            TypingError::ConstraintViolation {
                constraint: highest_allowed_type.to_string(),
                found: child_type.to_string(),
            },
        )?;
        match ordering {
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => Ok(child_type),
            std::cmp::Ordering::Less => Ok(highest_allowed_type),
        }
    }

    /// Checks if the children have compatible subtypes (i.e. their types c
    /// an be ordered) and returns the lowest subtype
    ///
    /// # Errors
    /// If the types are not ordered and no lowest subtype can be inferred an
    ///  error is returned
    fn check_child_coherence(
        children: &[Id],
        expr: &RecExpr<Self>,
    ) -> Result<Self::Type, TypingError> {
        children
            .iter()
            .map(|id| &expr[*id])
            .map(|c| c.type_node(expr))
            .try_fold(Self::Type::default(), move |acc, r| {
                let t = r?;
                let ordering = acc
                    .partial_cmp(&t)
                    .ok_or(TypingError::Incomparable(acc.to_string(), t.to_string()))?;
                Ok(match ordering {
                    std::cmp::Ordering::Equal | std::cmp::Ordering::Less => acc,
                    std::cmp::Ordering::Greater => t,
                })
            })
    }
}

/// Checks if the expression is properly typed and returns the roots type
///
/// # Panics
///
/// Panics if given an empty [`RecExpr`]
///
/// # Errors
///
/// Errors is a typing error coccurs
///
pub fn typecheck_expr<L: Typeable>(expr: &RecExpr<L>) -> Result<L::Type, TypingError> {
    let root = expr
        .as_ref()
        .last()
        .expect("Can't typecheck an empty recexpr");
    root.type_node(expr)
}

/// Checks if the expression is properly typed and returns the roots type
///
/// # Panics
///
/// Panics if given [`Id`] is not in the [`RecExpr`]
///
/// # Errors
///
/// Errors is a typing error coccurs
///
pub fn typecheck_node<L: Typeable>(expr: &RecExpr<L>, node_id: Id) -> Result<L::Type, TypingError> {
    expr[node_id].type_node(expr)
}

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
