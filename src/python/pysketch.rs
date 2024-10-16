use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::Language;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use super::FlatAst;
use crate::sketch::{PartialSketch, PartialSketchNode, Sketch, SketchNode};
use crate::utils::Tree;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PySketch(pub(crate) RawSketch);

#[pymethods]
impl PySketch {
    /// This always generates a new node that has [open] as its children
    #[new]
    fn new(node: &str, arity: usize) -> PyResult<Self> {
        let new_children = vec![RawSketch::Open; arity];
        let raw_sketch = RawSketch::new(node, new_children)?;
        Ok(PySketch(raw_sketch))
    }

    /// Generate a new root with an [active] node
    #[staticmethod]
    pub fn new_root() -> Self {
        PySketch(RawSketch::Active)
    }

    /// Parse from string
    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let raw_sketch = s_expr_str.parse().map_err(RawSketchError::BadSexp)?;
        Ok(PySketch(raw_sketch))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    /// Appends at the current [active] node and turns an open [open]
    /// into a new [active]
    /// Returns if the sketch is finished
    pub fn append(&mut self, new_child: Self) -> bool {
        self.0.append(new_child.0)
    }

    /// Returns a flat representation of itself
    pub fn flat(&self) -> FlatAst {
        (&self.0).into()
    }

    /// Returns the number of nodes in the sketch
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns the maximum AST depth in the sketch
    pub fn depth(&self) -> usize {
        self.0.depth()
    }

    /// Checks if sketch has open [active]
    pub fn finished(&self) -> bool {
        self.0.finished()
    }

    /// Checks if sketch has open [active]
    pub fn sketch_symbols(&self) -> usize {
        self.0.sketch_symbols()
    }
}

impl From<RawSketch> for PySketch {
    fn from(value: RawSketch) -> Self {
        PySketch(value)
    }
}

impl FromStr for PySketch {
    type Err = RawSketchParseError;

    fn from_str(s: &str) -> Result<Self, RawSketchParseError> {
        let raw_sketch = s.parse()?;
        Ok(PySketch(raw_sketch))
    }
}

create_exception!(
    eggshell,
    PySketchException,
    PyException,
    "Error dealing with a PySketch."
);

impl From<RawSketchError> for PyErr {
    fn from(err: RawSketchError) -> PyErr {
        PySketchException::new_err(err.to_string())
    }
}

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
        children: Box<[RawSketch]>,
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
                children: children.into_boxed_slice(),
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
        // Need to invert since this returns True if there are open spots
        !self.new_active()
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

    pub fn name(&self) -> &str {
        match self {
            RawSketch::Any => "?",
            RawSketch::Open => "[open]",
            RawSketch::Active => "[active]",
            RawSketch::Node {
                lang_node,
                children: _,
            } => lang_node,
            RawSketch::Contains(_) => "contains",
            RawSketch::Or(_) => "or",
        }
    }

    pub fn sketch_symbols(&self) -> usize {
        match self {
            RawSketch::Open | RawSketch::Active => 0,
            RawSketch::Node {
                lang_node: _,
                children,
            } => children.iter().map(|c| c.sketch_symbols()).sum::<usize>(),
            RawSketch::Any | RawSketch::Contains(_) | RawSketch::Or(_) => {
                1 + self
                    .children()
                    .iter()
                    .map(|c| c.sketch_symbols())
                    .sum::<usize>()
            }
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
            } => children,
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
            } => children,
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
                        children: Box::new([]),
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
