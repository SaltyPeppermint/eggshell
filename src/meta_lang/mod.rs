// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

pub mod extract;
mod partial;
mod sketch_lang;

use std::collections::VecDeque;
use std::fmt::Debug;
use std::fmt::Display;

use egg::FromOp;
use egg::Id;
use egg::Language;
use egg::RecExpr;
use pyo3::{PyErr, create_exception, exceptions::PyException};
use thiserror::Error;

pub use partial::PartialLang;
pub use partial::PartialTerm;
pub use partial::{count_expected_tokens, lower_meta_level, partial_parse};
pub use sketch_lang::Sketch;
pub use sketch_lang::SketchLang;

#[derive(Debug, Error)]
pub enum MetaLangError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error("Wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error("Tried to parse a partial sketch into a full sketch")]
    PartialSketch,
    #[error("Cannot lower into an easier language: {0:?}")]
    NoLowering(String),
    #[error("No open positions in partial sketch to fill: {0:?}")]
    NoOpenPositions(egg::RecExpr<PartialLang<L>>),
    #[error("Max arity reached while trying to parse partial term: {1}: {0}")]
    MaxArity(String, usize),
    #[error(transparent)]
    BadOp(L::Error),
}

create_exception!(
    eggshell,
    SketchParseException,
    PyException,
    "Error parsing a Sketch."
);

impl<L> From<MetaLangError<L>> for PyErr
where
    L: FromOp,
    L::Error: Display,
{
    fn from(err: MetaLangError<L>) -> PyErr {
        SketchParseException::new_err(format!("{err:?}"))
    }
}

#[derive(Debug, Clone)]
struct TempNode<L: Language> {
    node: PartialLang<L>,
    children: Vec<TempNode<L>>,
}

impl<L: Language> TempNode<L> {
    fn find_next_open(&mut self) -> Option<&mut Self> {
        let mut queue = VecDeque::new();
        queue.push_back(self);
        while let Some(x) = queue.pop_front() {
            if matches!(x.node, PartialLang::Pad) {
                return Some(x);
            }
            for c in &mut x.children {
                queue.push_back(c);
            }
        }
        None
    }

    fn new_empty() -> Self {
        Self {
            node: PartialLang::Pad,
            children: vec![],
        }
    }
}

impl<L: Language> From<TempNode<L>> for RecExpr<PartialLang<L>> {
    fn from(root: TempNode<L>) -> Self {
        fn rec<LL: Language>(mut curr: TempNode<LL>, vec: &mut Vec<PartialLang<LL>>) -> Id {
            let c = curr.children.into_iter().map(|c| rec(c, vec));
            for (dummy_id, c_id) in curr.node.children_mut().iter_mut().zip(c) {
                *dummy_id = c_id;
            }
            let id = Id::from(vec.len());
            vec.push(curr.node);
            id
        }
        let mut stack = Vec::new();
        rec(root, &mut stack);
        RecExpr::from(stack)
    }
}
