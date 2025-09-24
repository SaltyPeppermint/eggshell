mod containment;
mod extraction;

use std::fmt::{Display, Formatter};

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use containment::{contains, eclass_contains};
pub use extraction::eclass_extract;

/// Simple alias
pub type Sketch<L> = RecExpr<SketchLang<L>>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize)]
/// The language of [`Sketch`]es.
///
pub enum SketchLang<L> {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any,
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node(L),
    /// Programs that contain *at least one* sub-program satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains(Id),
    /// Programs that *only* contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(onlyContains s)` syntax.
    OnlyContains(Id),
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or(Box<[Id]>),
}

#[derive(Debug, Error)]
pub enum SketchError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error(transparent)]
    BadChildren(#[from] egg::FromOpError),
    #[error(transparent)]
    BadOp(L::Error),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum SketchDiscriminant<L: Language> {
    Any,
    Node(L::Discriminant),
    Contains,
    OnlyContains,
    Or,
}

impl<L: Language> Language for SketchLang<L> {
    type Discriminant = SketchDiscriminant<L>;

    #[inline(always)]
    fn discriminant(&self) -> Self::Discriminant {
        match self {
            SketchLang::Any => SketchDiscriminant::Any,
            SketchLang::Node(n) => SketchDiscriminant::Node(n.discriminant()),
            SketchLang::Contains(_) => SketchDiscriminant::Contains,
            SketchLang::OnlyContains(_) => SketchDiscriminant::OnlyContains,
            SketchLang::Or(_) => SketchDiscriminant::Or,
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Should never call this")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Any => &[],
            Self::Node(n) => n.children(),
            Self::Contains(s) => std::slice::from_ref(s),
            Self::OnlyContains(s) => std::slice::from_ref(s),
            Self::Or(ss) => ss,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Any => &mut [],
            Self::Node(n) => n.children_mut(),
            Self::Contains(s) => std::slice::from_mut(s),
            Self::OnlyContains(s) => std::slice::from_mut(s),
            Self::Or(ss) => ss,
        }
    }
}

impl<L: Language + Display> Display for SketchLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "?"),
            Self::Node(node) => write!(f, "{node}"),
            Self::Contains(_) => write!(f, "contains"),
            Self::OnlyContains(_) => write!(f, "onlyContains"),
            Self::Or(_) => write!(f, "or"),
        }
    }
}

impl<L> egg::FromOp for SketchLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = SketchError<L>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "?" | "any" | "ANY" | "Any" => {
                if children.is_empty() {
                    Ok(Self::Any)
                } else {
                    Err(SketchError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "contains" | "CONTAINS" | "Contains" => {
                if children.len() == 1 {
                    Ok(Self::Contains(children[0]))
                } else {
                    Err(SketchError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "onlyContains" | "ONLYCONTAINS" | "only_contains" | "ONLY_CONTAINS"
            | "OnlyContains" => {
                if children.len() == 1 {
                    Ok(Self::OnlyContains(children[0]))
                } else {
                    Err(SketchError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "or" | "OR" | "Or" => {
                if !children.is_empty() {
                    Ok(Self::Or(children.into_boxed_slice()))
                } else {
                    Err(SketchError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            _ => L::from_op(op, children)
                .map(Self::Node)
                .map_err(|e| SketchError::BadOp(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use egg::{RecExpr, SymbolLang};

    use super::*;

    #[test]
    fn parse_and_print() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();

        let mut sketch_ref = RecExpr::default();
        let any = sketch_ref.add(SketchLang::Any);
        let f = sketch_ref.add(SketchLang::Node(SymbolLang::new("f", vec![any])));
        let _ = sketch_ref.add(SketchLang::Contains(f));

        assert_eq!(sketch, sketch_ref);
        assert_eq!(sketch.to_string(), string);
    }
}
