use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{Id, Language, RecExpr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use crate::sketch::{PartialSketch, PartialSketchNode, Sketch, SketchNode};
use crate::trs::TrsLang;
use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RawAst {
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
        children: Box<[RawAst]>,
    },
    // /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    // ///
    // /// Corresponds to a single leaf node in the underlying language syntax.
    // Leaf { s: String },
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains(Box<RawAst>),
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or(Box<[RawAst; 2]>),
}

impl RawAst {
    pub fn new(node_type: &str, mut children: Vec<RawAst>) -> Result<Self, RawAstError> {
        match (node_type, children.len()) {
            ("any" | "ANY" | "Any" | "?", 0) => Ok(RawAst::Any),
            ("any" | "ANY" | "Any" | "?", n) => Err(RawAstError::BadNewChildren("?".into(), n)),

            ("[open]" | "[OPEN]" | "[Open]", 0) => Ok(RawAst::Open),
            ("[open]" | "[OPEN]" | "[Open]", n) => {
                Err(RawAstError::BadNewChildren("[open]".into(), n))
            }

            ("[active]" | "[ACTIVE]" | "[Active]", 0) => Ok(RawAst::Active),
            ("[active]" | "[ACTIVE]" | "[Active]", n) => {
                Err(RawAstError::BadNewChildren("[active]".into(), n))
            }

            ("or" | "OR" | "Or", 2) => {
                let child_1 = children.pop().expect("Safe cause length 2");
                let child_0 = children.pop().expect("Safe cause length 2");
                Ok(RawAst::Or(Box::new([child_0, child_1])))
            }
            ("or" | "OR" | "Or", n) => Err(RawAstError::BadNewChildren("or".into(), n)),

            ("contains" | "CONTAINS" | "Contains", 1) => Ok(RawAst::Contains(Box::new(
                children.pop().expect("Safe cause len = 1"),
            ))),
            ("contains" | "CONTAINS" | "Contains", n) => {
                Err(RawAstError::BadNewChildren("contains".into(), n))
            }

            (s, _) => Ok(RawAst::Node {
                lang_node: s.to_owned(),
                children: children.into_boxed_slice(),
            }),
        }
    }

    /// Replace active with new node
    /// Returns true if successfull
    fn replace_active(&mut self, new_pick: &mut Option<RawAst>) -> bool {
        match self {
            // Replace active with new node
            RawAst::Active => {
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
    fn find_new_active(&mut self) -> bool {
        match self {
            RawAst::Open => {
                *self = RawAst::Active;
                // Return true if a new active could be found
                true
            }
            // Any over empty iterators defaults to false
            _ => self.children_mut().iter_mut().any(|c| c.find_new_active()),
        }
    }

    /// Appends at the current [active] node and turns an open [open]
    /// into a new [active]
    /// Returns if the sketch is finished.
    pub fn append(&mut self, new_child: Self) -> bool {
        self.replace_active(&mut Some(new_child));
        // Need to invert since this returns True if there are open spots
        !self.find_new_active()
    }

    pub fn finished(&self) -> bool {
        !self.unfinished()
    }

    pub fn unfinished(&self) -> bool {
        match self {
            RawAst::Active => true,
            _ => self.children().iter().any(|c| c.unfinished()),
        }
    }

    pub fn is_sketch(&self) -> bool {
        match self {
            RawAst::Node { children, .. } => children.iter().any(|c| c.is_sketch()),
            _ => true,
        }
    }

    pub fn is_partial_sketch(&self) -> bool {
        match self {
            RawAst::Active | RawAst::Open => true,
            _ => self.children().iter().any(|c| c.is_partial_sketch()),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            RawAst::Any => "?",
            RawAst::Open => "[open]",
            RawAst::Active => "[active]",
            RawAst::Node { lang_node, .. } => lang_node,
            RawAst::Contains(_) => "contains",
            RawAst::Or(_) => "or",
        }
    }

    pub fn features(&self) -> Option<Vec<f64>> {
        todo!()
    }

    pub fn sketch_symbols(&self) -> usize {
        match self {
            RawAst::Open | RawAst::Active => 0,
            RawAst::Node { children, .. } => {
                children.iter().map(|c| c.sketch_symbols()).sum::<usize>()
            }
            RawAst::Any | RawAst::Contains(_) | RawAst::Or(_) => {
                1 + self
                    .children()
                    .iter()
                    .map(|c| c.sketch_symbols())
                    .sum::<usize>()
            }
        }
    }
}

impl Tree for RawAst {
    fn children(&self) -> &[Self] {
        match self {
            RawAst::Any | RawAst::Open | RawAst::Active => &[],
            RawAst::Node { children, .. } => children,
            RawAst::Contains(child) => std::slice::from_ref(child),
            RawAst::Or(children) => children.as_slice(),
        }
    }

    fn children_mut(&mut self) -> &mut [Self] {
        match self {
            RawAst::Any | RawAst::Open | RawAst::Active => &mut [],
            RawAst::Node { children, .. } => children,
            RawAst::Contains(child) => std::slice::from_mut(child),
            RawAst::Or(children) => children.as_mut_slice(),
        }
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PySketch`].
#[derive(Debug, Error)]
pub enum RawAstError {
    /// Bad child found during parsing
    #[error("Wrong number of children: {0}")]
    BadNewChildren(String, usize),

    /// Tried parsing a `Sketch` or `PartialSketch` as a `RecExpr`
    #[error("Tried parsing a Sketch or Partial Sketch as a RecExpr: {0}")]
    NotARecExpr(String),

    /// Tried using a `RecExpr` or `PartialSketch` as a `Sketch`
    #[error("Tried using a RecExpr or Partial Sketch as a Sketch: {0}")]
    NotASketchExpr(String),

    /// Tried parsing a `RecExpr` or `Sketch` as a `PartialSketch`
    #[error("Tried parsing a RecExpr or Sketch as a Partial Sketch: {0}")]
    NotAPartialSketchExpr(String),

    /// Error converting a node to a L
    #[error("FromOp Error during parsing: {0}")]
    FromOp(String),

    /// An error occurred while parsing raw sketch via an the s-expression
    #[error(transparent)]
    BadSexp(RawAstParseError),
}

impl Display for RawAst {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RawAst::Any => write!(f, "?"),
            RawAst::Open => write!(f, "[open]"),
            RawAst::Active => write!(f, "[active]"),
            RawAst::Contains(node) => write!(f, "(contains {node})"),
            RawAst::Or(children) => {
                write!(f, "(or {} {})", children[0], children[1])
            }
            RawAst::Node {
                lang_node,
                children,
                ..
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

impl FromStr for RawAst {
    type Err = RawAstParseError;

    fn from_str(s: &str) -> Result<Self, RawAstParseError> {
        fn rec(sexp: &Sexp) -> Result<RawAst, RawAstParseError> {
            match sexp {
                Sexp::Empty => Err(RawAstParseError::EmptySexp),
                Sexp::String(s) => match s.as_str() {
                    "?" | "any" | "ANY" | "Any" => Ok(RawAst::Any),
                    "[open]" | "[OPEN]" | "[Open]" => Ok(RawAst::Open),
                    "[active]" | "[ACTIVE]" | "[Active]" => Ok(RawAst::Active),
                    "or" | "OR" | "Or" => Err(RawAstParseError::BadTerminalOr(sexp.to_owned())),
                    "contains" | "CONTAINS" | "Contains" => {
                        Err(RawAstParseError::BadTerminalContains(sexp.to_owned()))
                    }
                    _ => Ok(RawAst::Node {
                        lang_node: s.to_string(),
                        children: Box::new([]),
                    }),
                },
                Sexp::List(list) if list.is_empty() => Err(RawAstParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(RawAstParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => match (s.as_str(), list.len()) {
                        ("contains" | "CONTAINS" | "Contains", 2) => {
                            let inner = rec(&list[1])?;
                            Ok(RawAst::Contains(Box::new(inner)))
                        }
                        ("contains" | "CONTAINS" | "Contains", _) => {
                            Err(RawAstParseError::BadChildrenContains(list.to_owned()))
                        }

                        ("or" | "OR" | "Or", 3) => {
                            let child_0 = rec(&list[1])?;
                            let child_1 = rec(&list[2])?;
                            Ok(RawAst::Or(Box::new([child_0, child_1])))
                        }
                        ("or" | "OR" | "Or", _) => {
                            Err(RawAstParseError::BadChildrenOr(list.to_owned()))
                        }
                        _ => Ok(RawAst::Node {
                            lang_node: s.to_owned(),
                            children: list[1..].iter().map(rec).collect::<Result<_, _>>()?,
                        }),
                    },
                },
            }
        }

        let sexp =
            symbolic_expressions::parser::parse_str(s.trim()).map_err(RawAstParseError::BadSexp)?;
        rec(&sexp)
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PySketch`].
#[derive(Debug, Error)]
pub enum RawAstParseError {
    /// An empty s-expression was found. Usually this is caused by an
    /// empty list "()" somewhere in the input.
    #[error("Found empty s-expression")]
    EmptySexp,

    /// A list was found where an operator was expected. This is caused by
    /// s-expressions of the form "((a b c) d e f)."
    #[error("Found a list in the head position: {0}")]
    HeadList(Sexp),

    /// A or expression was found where with less or more than 2 children.
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

impl<L: TrsLang> From<&RecExpr<L>> for RawAst {
    fn from(expr: &RecExpr<L>) -> Self {
        fn rec<L: TrsLang>(node: &L, expr: &RecExpr<L>) -> RawAst {
            RawAst::Node {
                lang_node: node.to_string(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| rec(&expr[*child_id], expr))
                    .collect(),
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = expr.as_ref().last().unwrap();
        rec(root, expr)
    }
}

impl<L: Language + Display> From<&Sketch<L>> for RawAst {
    fn from(sketch: &Sketch<L>) -> Self {
        fn rec<L: Language + Display>(node: &SketchNode<L>, sketch: &Sketch<L>) -> RawAst {
            match node {
                SketchNode::Any => RawAst::Any,
                SketchNode::Node(lang_node) => RawAst::Node {
                    lang_node: lang_node.to_string(),
                    children: lang_node
                        .children()
                        .iter()
                        .map(|child_id| rec(&sketch[*child_id], sketch))
                        .collect(),
                },
                SketchNode::Contains(id) => RawAst::Contains(rec(&sketch[*id], sketch).into()),
                SketchNode::Or(ids) => {
                    let child_0 = rec(&sketch[ids[0]], sketch);
                    let child_1 = rec(&sketch[ids[1]], sketch);
                    RawAst::Or(Box::new([child_0, child_1]))
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

impl<L: Language + Display> From<&PartialSketch<L>> for RawAst {
    fn from(sketch: &PartialSketch<L>) -> Self {
        fn rec<L: Language + Display>(
            node: &PartialSketchNode<L>,
            sketch: &PartialSketch<L>,
        ) -> RawAst {
            match node {
                PartialSketchNode::Open => RawAst::Open,
                PartialSketchNode::Active => RawAst::Active,
                PartialSketchNode::Finished(SketchNode::Any) => RawAst::Any,
                PartialSketchNode::Finished(SketchNode::Contains(id)) => {
                    RawAst::Contains(rec(&sketch[*id], sketch).into())
                }
                PartialSketchNode::Finished(SketchNode::Or(ids)) => {
                    let child_0 = rec(&sketch[ids[0]], sketch);
                    let child_1 = rec(&sketch[ids[1]], sketch);
                    RawAst::Or(Box::new([child_0, child_1]))
                }
                PartialSketchNode::Finished(SketchNode::Node(lang_node)) => RawAst::Node {
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

impl<L: TrsLang> TryFrom<&RawAst> for RecExpr<L> {
    type Error = RawAstError;

    fn try_from(value: &RawAst) -> Result<Self, Self::Error> {
        fn rec<L: TrsLang>(ast: &RawAst, rec_expr: &mut RecExpr<L>) -> Result<Id, RawAstError> {
            if let RawAst::Node {
                lang_node,
                children,
                ..
            } = ast
            {
                let c_ids = children
                    .iter()
                    .map(|c| rec(c, rec_expr))
                    .collect::<Result<_, _>>()?;
                let node = L::from_op(lang_node, c_ids)
                    .map_err(|e| RawAstError::FromOp(format!("{e:?}")))?;
                Ok(rec_expr.add(node))
            } else {
                Err(RawAstError::NotARecExpr(ast.to_string()))
            }
        }

        let mut rec_expr = RecExpr::<L>::default();
        rec(value, &mut rec_expr)?;
        Ok(rec_expr)
    }
}

#[cfg(test)]
mod tests {
    use egg::SymbolLang;

    use super::*;

    #[test]
    fn parse_and_print_contains() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawAst = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_or() {
        let expr = "(or f ?)";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawAst = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: RawAst = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_pysketch_vs_sketch() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawAst = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn pysketch_from_str() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawAst = (&sketch).into();
        let pysketch_b = RawAst::from_str(expr).unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn parse_pysketch_vs_sketch_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: RawAst = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn bad_children_or() {
        let expr = "(or f)";
        let parse_error = expr.parse::<RawAst>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(RawAstParseError::BadChildrenOr(_))
        ));
    }

    #[test]
    fn bad_terminal_or() {
        let expr = "(f or)";
        let parse_error = expr.parse::<RawAst>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(RawAstParseError::BadTerminalOr(_))
        ));
    }

    #[test]
    fn bad_children_contains() {
        let expr = "(contains f g)";
        let parse_error = expr.parse::<RawAst>();
        assert!(matches!(
            parse_error,
            Err(RawAstParseError::BadChildrenContains(_))
        ));
    }

    #[test]
    fn bad_terminal_contains() {
        let expr = "(f contains)";
        let parse_error = expr.parse::<RawAst>();
        assert!(matches!(
            parse_error,
            Err(RawAstParseError::BadTerminalContains(_))
        ));
    }
}
