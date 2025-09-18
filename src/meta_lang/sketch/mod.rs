mod error;
mod extract;
mod guide;

use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use crate::rewrite_system::{LangExtras, SymbolInfo, SymbolType};

pub use error::SketchError;
pub use extract::{eclass_extract, eclass_satisfies_sketch, satisfies_sketch};
pub use guide::SketchGuide;

/// Simple alias
pub type Sketch<L> = RecExpr<SketchLang<L>>;

#[derive(
    Debug,
    Hash,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    EnumDiscriminants,
    EnumCount,
)]
#[strum_discriminants(derive(EnumIter))]
pub enum SketchLang<L: Language> {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any,
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains(Id),
    /// Programs that satisfy any of these sketches.
    ///
    /// Important change from the guided equality saturation: Or can only contain a pair.
    /// This doesnt hamper the expressivity (or or or chains are possible)
    /// but makes life much easier
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or([Id; 2]),
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node(L),
}

impl<L: Language> Language for SketchLang<L> {
    type Discriminant = (Discriminant<Self>, Option<L::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            SketchLang::Node(x) => (discr, Some(x.discriminant())),
            _ => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Any => &[],
            Self::Node(n) => n.children(),
            Self::Contains(s) => std::slice::from_ref(s),
            Self::Or(ss) => ss.as_slice(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Any => &mut [],
            Self::Node(n) => n.children_mut(),
            Self::Contains(s) => std::slice::from_mut(s),
            Self::Or(ss) => ss.as_mut_slice(),
        }
    }
}

impl<L: Language + LangExtras> LangExtras for SketchLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        if let SketchLang::Node(l) = self {
            l.symbol_info()
        } else {
            let id = SketchLangDiscriminants::iter()
                .position(|x| x == self.into())
                .unwrap();
            SymbolInfo::new(id + L::NUM_SYMBOLS, SymbolType::MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["?", "contains", "or"];
        s.extend(L::operators());
        s
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = { if L::MAX_ARITY > 2 { L::MAX_ARITY } else { 2 } };
}

impl<L: Language + Display> Display for SketchLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "?"),
            Self::Node(node) => write!(f, "{node}"),
            Self::Contains(_) => write!(f, "contains"),
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
            "or" | "OR" | "Or" => {
                if children.len() == 2 {
                    Ok(Self::Or([children[0], children[1]]))
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
