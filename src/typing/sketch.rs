use std::fmt::Display;

use egg::{Id, Language, RecExpr};

use super::{Typeable, TypingError};

pub trait SketchTypable<L: Language>: Language {
    type Type: PartialOrd + Display + Default;
    type InnerLanguage: Typeable;

    /// Returns the type of a node
    ///
    /// # Errors
    /// If a typing error ooccurs, it is propagated upwards meaning the
    /// Expression is wrongly typed
    fn type_lang_node(&self, expr: &RecExpr<Self>) -> Result<Self::Type, TypingError>;

    fn type_sketch_node(&self, expr: &RecExpr<Self>) -> Result<Self::Type, TypingError>;

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
    #[expect(clippy::redundant_closure)]
    fn check_child_coherence(
        children: &[Id],
        expr: &RecExpr<Self>,
    ) -> Result<Self::Type, TypingError> {
        children
            .iter()
            .map(|id| &expr[*id])
            .map(|c| c.type_lang_node(expr))
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
