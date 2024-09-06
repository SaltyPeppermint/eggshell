use std::fmt::Display;

use egg::{Analysis, FromOp, Id, Language, RecExpr, Rewrite};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::Serialize;

pub mod arithmetic;
pub mod halide;
pub mod simple;

pub use arithmetic::Arithmetic;
pub use halide::Halide;
pub use simple::Simple;
use thiserror::Error;

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`Trs::rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait Trs: Serialize {
    type Language: Display + Serialize + FromOp + Typeable<Type: PartialOrd + Eq>;
    type Analysis: Analysis<Self::Language, Data: Serialize + Clone> + Clone + Serialize + Default;
    type Rulesets: TryFrom<String>;

    fn rules(ruleset_class: &Self::Rulesets) -> Vec<Rewrite<Self::Language, Self::Analysis>>;
}

pub trait Typeable: Language {
    type Type: PartialOrd + Default + Display;

    /// Returns the type of a node
    ///
    /// # Errors
    /// If a typing error ooccurs, it is propagated upwards meaning the
    /// Expression is wrongly typed
    fn type_node(&self, expr: &RecExpr<Self>) -> Result<Self::Type, TrsError>;

    /// Checks if the child types are subtypes of the parents type
    /// returns the subtype then
    ///
    /// # Errors
    /// If the types are incompatible, that error is returned
    fn infer_node_type(
        parent_type: Self::Type,
        children: &[Id],
        expr: &RecExpr<Self>,
    ) -> Result<Self::Type, TrsError> {
        let child_type = Typeable::infer_child_type(children, expr)?;
        if let Some(ord) = child_type.partial_cmp(&parent_type) {
            match ord {
                std::cmp::Ordering::Equal | std::cmp::Ordering::Less => Ok(child_type),
                std::cmp::Ordering::Greater => Ok(parent_type),
            }
        } else {
            Err(TrsError::TypingError(format!(
                "Children have type incompatible to parent: {child_type} {parent_type}"
            )))
        }
    }

    /// Returns the inferred type of a node based on its childrens type
    ///
    /// # Errors
    /// If a typing error ooccurs in the children, it is propagated upwards
    fn infer_child_type(children: &[Id], expr: &RecExpr<Self>) -> Result<Self::Type, TrsError> {
        children
            .iter()
            .map(|id| &expr[*id])
            .map(|c| c.type_node(expr))
            .try_fold(Self::Type::default(), |acc, r| {
                let t = r?;
                let ordering = acc.partial_cmp(&t).ok_or(TrsError::TypingError(format!(
                    "Incompatible child types: {acc} {t}"
                )))?;
                Ok(match ordering {
                    std::cmp::Ordering::Equal | std::cmp::Ordering::Less => acc,
                    std::cmp::Ordering::Greater => t,
                })
            })
    }

    /// Checks if the expression is properly typed
    fn typecheck(&self, expr: &RecExpr<Self>) -> bool {
        self.type_node(expr).is_ok()
    }
}

// /// [`EGraph`] parameterized by the Trs
// pub(crate) type TrsEGraph<R> = EGraph<<R as Trs>::Language, <R as Trs>::Analysis>;
#[derive(Debug, Error)]
pub enum TrsError {
    #[error("Wrong number of children: {0}")]
    BadAnalysis(String),
    #[error("Bad ruleset name: {0}")]
    BadRulesetName(String),
    #[error("Could not type: {0}")]
    TypingError(String),
}

create_exception!(
    eggshell,
    TrsException,
    PyException,
    "Eggshell internal error."
);

impl From<TrsError> for PyErr {
    fn from(err: TrsError) -> PyErr {
        TrsException::new_err(err.to_string())
    }
}
