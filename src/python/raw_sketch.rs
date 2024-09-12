use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::Language;
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use super::PySketch;
use crate::sketch::{PartialSketch, PartialSketchNode, Sketch, SketchNode};
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RawSketch {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any,
    /// In case the sketch is unfinished, there are still open slots to be filled
    ///
    /// This is an inactive open
    Open,
    /// In case the sketch is unfinished, there are still open slots to be filled
    ///
    /// This is an active open being currently worked on
    Active,
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node {
        lang_node: String,
        children: Vec<RawSketch>,
    },
    // /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    // ///
    // /// Corresponds to a single leaf node in the underlying language syntax.
    // Leaf { s: String },
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains(Box<RawSketch>),
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or(Box<[RawSketch; 2]>),
}

impl RawSketch {
    pub fn new(node_type: &str, mut children: Vec<RawSketch>) -> Result<Self, RawSketchError> {
        match (node_type, children.len()) {
            ("any" | "ANY" | "Any" | "?", 0) => Ok(RawSketch::Any),
            ("any" | "ANY" | "Any" | "?", n) => Err(RawSketchError::BadNewChildren("?".into(), n)),

            ("[open]" | "[OPEN]" | "[Open]", 0) => Ok(RawSketch::Open),
            ("[open]" | "[OPEN]" | "[Open]", n) => {
                Err(RawSketchError::BadNewChildren("[open]".into(), n))
            }

            ("[active]" | "[ACTIVE]" | "[Active]", 0) => Ok(RawSketch::Active),
            ("[active]" | "[ACTIVE]" | "[Active]", n) => {
                Err(RawSketchError::BadNewChildren("[active]".into(), n))
            }

            ("or" | "OR" | "Or", 2) => {
                let child_1 = children.pop().expect("Safe cause length 2");
                let child_0 = children.pop().expect("Safe cause length 2");
                Ok(RawSketch::Or(Box::new([child_0, child_1])))
            }
            ("or" | "OR" | "Or", n) => Err(RawSketchError::BadNewChildren("or".into(), n)),

            ("contains" | "CONTAINS" | "Contains", 1) => Ok(RawSketch::Contains(Box::new(
                children.pop().expect("Safe cause len = 1"),
            ))),
            ("contains" | "CONTAINS" | "Contains", n) => {
                Err(RawSketchError::BadNewChildren("contains".into(), n))
            }

            (s, _) => Ok(RawSketch::Node {
                lang_node: s.to_owned(),
                children,
            }),
        }
    }

    /// Replace active with new node
    /// Returns true if successfull
    fn replace_active(&mut self, new_pick: &mut Option<RawSketch>) -> bool {
        match self {
            // Replace active with new node
            RawSketch::Active => {
                *self = new_pick.take().expect("Only one Active in any ast");
                true
            }
            // Any over empty iterators defaults to false
            _ => self
                .children_mut()
                .iter_mut()
                .any(|c| c.replace_active(new_pick)),
        }
    }

    /// Turns a [open] into a new [active]
    /// Returns true if successful, meaning there were open [open]s
    fn new_active(&mut self) -> bool {
        match self {
            RawSketch::Open => {
                *self = RawSketch::Active;
                // Return true if a new active could be found
                true
            }
            // Any over empty iterators defaults to false
            _ => self.children_mut().iter_mut().any(|c| c.new_active()),
        }
    }

    /// Appends at the current [active] node and turns an open [open]
    /// into a new [active]
    /// Returns if the sketch is finished.
    pub fn append(&mut self, new_child: Self) -> bool {
        self.replace_active(&mut Some(new_child));
        self.new_active()
    }

    pub fn finished(&self) -> bool {
        !self.unfinished()
    }

    fn unfinished(&self) -> bool {
        match self {
            RawSketch::Active => true,
            _ => self.children().iter().any(|c| c.unfinished()),
        }
    }
}

impl Tree for RawSketch {
    fn children(&self) -> &[Self] {
        match self {
            RawSketch::Any | RawSketch::Open | RawSketch::Active => &[],
            RawSketch::Node {
                lang_node: _,
                children,
            } => children.as_slice(),
            RawSketch::Contains(child) => std::slice::from_ref(child),
            RawSketch::Or(children) => children.as_slice(),
        }
    }

    fn children_mut(&mut self) -> &mut [Self] {
        match self {
            RawSketch::Any | RawSketch::Open | RawSketch::Active => &mut [],
            RawSketch::Node {
                lang_node: _,
                children,
            } => children.as_mut_slice(),
            RawSketch::Contains(child) => std::slice::from_mut(child),
            RawSketch::Or(children) => children.as_mut_slice(),
        }
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PySketch`].
#[derive(Debug, Error)]
pub enum RawSketchError {
    /// New Error for the sketch
    #[error("Wrong number of children: {0}")]
    BadNewChildren(String, usize),

    /// An error occurred while parsing raw sketch via an the s-expressio
    #[error(transparent)]
    BadSexp(RawSketchParseError),
}

impl Display for RawSketch {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RawSketch::Any => write!(f, "?"),
            RawSketch::Open => write!(f, "[open]"),
            RawSketch::Active => write!(f, "[active]"),
            RawSketch::Contains(node) => write!(f, "(contains {node})"),
            RawSketch::Or(children) => {
                write!(f, "(or {} {})", children[0], children[1])
            }
            RawSketch::Node {
                lang_node,
                children,
            } => {
                if children.is_empty() {
                    write!(f, "{lang_node}")
                } else {
                    let inner = children
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    write!(f, "({lang_node} {inner})")
                }
            }
        }
    }
}

impl<L: Language + Display> From<&PartialSketch<L>> for PySketch {
    fn from(sketch: &PartialSketch<L>) -> Self {
        let raw_sketch = sketch.into();
        PySketch(raw_sketch)
    }
}

impl<L: Language + Display> From<&Sketch<L>> for PySketch {
    fn from(sketch: &Sketch<L>) -> Self {
        let raw_sketch = sketch.into();
        PySketch(raw_sketch)
    }
}

impl FromStr for RawSketch {
    type Err = RawSketchParseError;

    fn from_str(s: &str) -> Result<Self, RawSketchParseError> {
        fn rec(sexp: &Sexp) -> Result<RawSketch, RawSketchParseError> {
            match sexp {
                Sexp::Empty => Err(RawSketchParseError::EmptySexp),
                Sexp::String(s) => match s.as_str() {
                    "?" | "any" | "ANY" | "Any" => Ok(RawSketch::Any),
                    "[open]" | "[OPEN]" | "[Open]" => Ok(RawSketch::Open),
                    "[active]" | "[ACTIVE]" | "[Active]" => Ok(RawSketch::Active),
                    "or" | "OR" | "Or" => Err(RawSketchParseError::BadTerminalOr(sexp.to_owned())),
                    "contains" | "CONTAINS" | "Contains" => {
                        Err(RawSketchParseError::BadTerminalContains(sexp.to_owned()))
                    }
                    _ => Ok(RawSketch::Node {
                        lang_node: s.to_string(),
                        children: vec![],
                    }),
                },
                Sexp::List(list) if list.is_empty() => Err(RawSketchParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(RawSketchParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => match (s.as_str(), list.len()) {
                        ("contains" | "CONTAINS" | "Contains", 2) => {
                            let inner = rec(&list[1])?;
                            Ok(RawSketch::Contains(Box::new(inner)))
                        }
                        ("contains" | "CONTAINS" | "Contains", _) => {
                            Err(RawSketchParseError::BadChildrenContains(list.to_owned()))
                        }

                        ("or" | "OR" | "Or", 3) => {
                            let child_0 = rec(&list[1])?;
                            let child_1 = rec(&list[2])?;
                            Ok(RawSketch::Or(Box::new([child_0, child_1])))
                        }
                        ("or" | "OR" | "Or", _) => {
                            Err(RawSketchParseError::BadChildrenOr(list.to_owned()))
                        }
                        _ => Ok(RawSketch::Node {
                            lang_node: s.to_owned(),
                            children: list[1..].iter().map(rec).collect::<Result<_, _>>()?,
                        }),
                    },
                },
            }
        }

        let sexp = symbolic_expressions::parser::parse_str(s.trim())
            .map_err(RawSketchParseError::BadSexp)?;
        rec(&sexp)
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PySketch`].
#[derive(Debug, Error)]
pub enum RawSketchParseError {
    /// An empty s-expression was found. Usually this is caused by an
    /// empty list "()" somewhere in the input.
    #[error("Found empty s-expression")]
    EmptySexp,

    /// A list was found where an operator was expected. This is caused by
    /// s-expressions of the form "((a b c) d e f)."
    #[error("Found a list in the head position: {0}")]
    HeadList(Sexp),

    /// A or expression was found where with less or more han 2 children.
    #[error("Found an 'or' with less or more than 2 children: {0:?}")]
    BadChildrenOr(Vec<Sexp>),

    /// A or expression was found in a terminal position.
    #[error("Found an or in a terminal position: {0}")]
    BadTerminalOr(Sexp),

    /// A contains expression was found where with more or less than 1 children.
    #[error("Found an 'contains' with more or less than 1 child: {0:?}")]
    BadChildrenContains(Vec<Sexp>),

    /// A contains expression was found in a terminal position.
    #[error("Found an or in a terminal position: {0}")]
    BadTerminalContains(Sexp),

    /// An error occurred while parsing the s-expression itself, generally
    /// because the input had an invalid structure (e.g. unpaired parentheses).
    #[error(transparent)]
    BadSexp(SexpError),
}

impl<L: Language + Display> From<&Sketch<L>> for RawSketch {
    fn from(sketch: &Sketch<L>) -> Self {
        fn rec<L: Language + Display>(node: &SketchNode<L>, sketch: &Sketch<L>) -> RawSketch {
            match node {
                SketchNode::Any => RawSketch::Any,
                SketchNode::Node(lang_node) => RawSketch::Node {
                    lang_node: lang_node.to_string(),
                    children: lang_node
                        .children()
                        .iter()
                        .map(|child_id| rec(&sketch[*child_id], sketch))
                        .collect(),
                },
                SketchNode::Contains(id) => RawSketch::Contains(rec(&sketch[*id], sketch).into()),
                SketchNode::Or(ids) => {
                    let child_0 = rec(&sketch[ids[0]], sketch);
                    let child_1 = rec(&sketch[ids[1]], sketch);
                    RawSketch::Or(Box::new([child_0, child_1]))
                }
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = sketch.as_ref().last().unwrap();
        rec(root, sketch)
    }
}

impl<L: Language + Display> From<&PartialSketch<L>> for RawSketch {
    fn from(sketch: &PartialSketch<L>) -> Self {
        fn rec<L: Language + Display>(
            node: &PartialSketchNode<L>,
            sketch: &PartialSketch<L>,
        ) -> RawSketch {
            match node {
                PartialSketchNode::Open => RawSketch::Open,
                PartialSketchNode::Active => RawSketch::Active,
                PartialSketchNode::Finished(SketchNode::Any) => RawSketch::Any,
                PartialSketchNode::Finished(SketchNode::Contains(id)) => {
                    RawSketch::Contains(rec(&sketch[*id], sketch).into())
                }
                PartialSketchNode::Finished(SketchNode::Or(ids)) => {
                    let child_0 = rec(&sketch[ids[0]], sketch);
                    let child_1 = rec(&sketch[ids[1]], sketch);
                    RawSketch::Or(Box::new([child_0, child_1]))
                }
                PartialSketchNode::Finished(SketchNode::Node(lang_node)) => RawSketch::Node {
                    lang_node: lang_node.to_string(),
                    children: lang_node
                        .children()
                        .iter()
                        .map(|child_id| rec(&sketch[*child_id], sketch))
                        .collect(),
                },
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = sketch.as_ref().last().unwrap();
        rec(root, sketch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::SymbolLang;

    #[test]
    fn parse_and_print_contains() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawSketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_or() {
        let expr = "(or f ?)";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawSketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawSketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_pysketch_vs_sketch() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawSketch = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn pysketch_from_str() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawSketch = (&sketch).into();
        let pysketch_b = RawSketch::from_str(expr).unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn parse_pysketch_vs_sketch_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawSketch = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn bad_children_or() {
        let expr = "(or f)";
        let parse_error = expr.parse::<RawSketch>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(RawSketchParseError::BadChildrenOr(_))
        ));
    }

    #[test]
    fn bad_terminal_or() {
        let expr = "(f or)";
        let parse_error = expr.parse::<RawSketch>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(RawSketchParseError::BadTerminalOr(_))
        ));
    }

    #[test]
    fn bad_children_contains() {
        let expr = "(contains f g)";
        let parse_error = expr.parse::<RawSketch>();
        assert!(matches!(
            parse_error,
            Err(RawSketchParseError::BadChildrenContains(_))
        ));
    }

    #[test]
    fn bad_terminal_contains() {
        let expr = "(f contains)";
        let parse_error = expr.parse::<RawSketch>();
        assert!(matches!(
            parse_error,
            Err(RawSketchParseError::BadTerminalContains(_))
        ));
    }
}
