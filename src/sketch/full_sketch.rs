use std::fmt::{Display, Formatter};
use std::mem::{discriminant, Discriminant};

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};

use super::SketchParseError;
use crate::features::{AsFeatures, Featurizer, SymbolType};
use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type Sketch<L> = RecExpr<SketchNode<L>>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SketchNode<L: Language> {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any,
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node(L),
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
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub enum SNDiscr<L: Language> {
    This(Discriminant<SketchNode<L>>),
    Inner(Discriminant<L>),
}

impl<L: Language> Language for SketchNode<L> {
    type Discriminant = SNDiscr<L>;

    fn discriminant(&self) -> Self::Discriminant {
        match self {
            SketchNode::Node(x) => SNDiscr::Inner(discriminant(x)),
            _ => SNDiscr::This(discriminant(self)),
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

impl<L: Typeable> Typeable for SketchNode<L> {
    type Type = L::Type;

    fn type_info(&self) -> crate::typing::TypingInfo<Self::Type> {
        match self {
            Self::Any => TypingInfo::new(Self::Type::top(), Self::Type::top()).infer_return_type(),
            Self::Node(t) => t.type_info(),
            Self::Contains(_) => TypingInfo::new(Self::Type::top(), Self::Type::top()),
            Self::Or(_) => {
                TypingInfo::new(Self::Type::top(), Self::Type::top()).infer_return_type()
            }
        }
    }
}

impl<L: Language + AsFeatures> AsFeatures for SketchNode<L> {
    fn featurizer(variable_names: Vec<String>) -> Featurizer<Self> {
        L::featurizer(variable_names).into_meta_lang(|l| SketchNode::Node(l))
    }

    fn symbol_type(&self) -> SymbolType {
        match self {
            SketchNode::Node(l) => l.symbol_type(),
            _ => SymbolType::MetaSymbol,
        }
    }

    fn into_symbol(name: String) -> Self {
        SketchNode::Node(L::into_symbol(name))
    }
}

impl<L: Language + Display> Display for SketchNode<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => write!(f, "?"),
            Self::Node(node) => write!(f, "{node}"),
            Self::Contains(_) => write!(f, "contains"),
            Self::Or(_) => write!(f, "or"),
        }
    }
}

impl<L> egg::FromOp for SketchNode<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "?" | "any" | "ANY" | "Any" => {
                if children.is_empty() {
                    Ok(Self::Any)
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "contains" | "CONTAINS" | "Contains" => {
                if children.len() == 1 {
                    Ok(Self::Contains(children[0]))
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "or" | "OR" | "Or" => {
                if children.len() == 2 {
                    Ok(Self::Or([children[0], children[1]]))
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            _ => L::from_op(op, children)
                .map(Self::Node)
                .map_err(SketchParseError::BadOp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{RecExpr, SymbolLang};

    #[test]
    fn parse_and_print() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();

        let mut sketch_ref = RecExpr::default();
        let any = sketch_ref.add(SketchNode::Any);
        let f = sketch_ref.add(SketchNode::Node(SymbolLang::new("f", vec![any])));
        let _ = sketch_ref.add(SketchNode::Contains(f));

        assert_eq!(sketch, sketch_ref);
        assert_eq!(sketch.to_string(), string);
    }
}
