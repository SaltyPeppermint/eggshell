mod error;
mod lang;

use std::collections::VecDeque;
use std::fmt::{Debug, Display};

use egg::{FromOp, Id, Language, RecExpr};
use error::PartialError;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter};

use crate::node::Node;
use crate::trs::MetaInfo;

/// Simple alias
pub type PartialRecExpr<L> = RecExpr<PartialLang<L>>;

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
pub enum PartialLang<L: Language> {
    /// Finshed parts represented by [`SketchNode`]
    Pad,
    Finished(L),
}

impl<L: Language> PartialLang<L> {
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Self::Pad)
    }
}

impl<L: Language, T> Node<PartialLang<L>, Option<T>> {
    fn new_empty() -> Self {
        Node::new(PartialLang::Pad, Vec::new(), None)
    }

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
}

impl<L: Language, T> From<Node<PartialLang<L>, T>> for RecExpr<PartialLang<L>> {
    fn from(root: Node<PartialLang<L>, T>) -> Self {
        fn rec<LL: Language, TT>(
            mut curr: Node<PartialLang<LL>, TT>,
            vec: &mut Vec<PartialLang<LL>>,
        ) -> Id {
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

impl<L, T> TryFrom<Node<PartialLang<L>, Option<T>>> for (RecExpr<PartialLang<L>>, Vec<T>)
where
    L::Error: Display,
    L: Language + FromOp,
{
    type Error = PartialError<PartialLang<L>>;

    fn try_from(root: Node<PartialLang<L>, Option<T>>) -> Result<Self, Self::Error> {
        fn rec<LL, TT>(
            mut curr: Node<PartialLang<LL>, Option<TT>>,
            stack: &mut Vec<PartialLang<LL>>,
            probs_stack: &mut Vec<TT>,
        ) -> Result<Id, PartialError<PartialLang<LL>>>
        where
            LL::Error: Display,
            LL: Language + FromOp,
        {
            let c = curr
                .children
                .into_iter()
                .map(|c| rec(c, stack, probs_stack));
            for (dummy_id, c_id) in curr.node.children_mut().iter_mut().zip(c) {
                *dummy_id = c_id?;
            }
            let id = Id::from(stack.len());
            let additional_data = curr
                .additional_data
                .ok_or_else(|| PartialError::NoProbability(format!("{:?}", &curr.node)))?;
            stack.push(curr.node);
            probs_stack.push(additional_data);
            Ok(id)
        }

        let mut stack = Vec::new();
        let mut probs_stack = Vec::new();
        let _ = rec(root, &mut stack, &mut probs_stack)?;
        Ok((RecExpr::from(stack), probs_stack))
    }
}

/// Tries to parse a list of tokens into a list of nodes in the language
///
/// # Errors
///
/// This function will return an error if it cannot be parsed as a partial term
pub fn partial_parse<L, S>(
    tokens: &[S],
    probs: Option<&[f64]>,
) -> Result<(Node<PartialLang<L>, Option<f64>>, usize), PartialError<L>>
where
    S: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    // If this is empty, return the empty RecExpr
    if tokens.is_empty() {
        return Err(PartialError::NoTokens);
    }

    let mut ast = Node::new_empty();
    for (index, token) in tokens.iter().enumerate() {
        let mut children_ids = vec![Id::from(0); L::MAX_ARITY];
        let (node, arity) = loop {
            if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
                break (node, children_ids.len());
            }
            children_ids.pop();
        };

        if let Some(position) = ast.find_next_open() {
            *position = Node::new(
                PartialLang::Finished(node),
                vec![Node::new_empty(); arity],
                probs.map(|v| v[index]),
            );
        } else {
            return Ok((ast.into(), index));
        }
    }

    Ok((ast.into(), tokens.len()))
}

/// Count how many nodes are expected to be there, including to be generated nodes atm
///
/// # Errors
///
/// This function will return an error if it cannot parse any tokens
pub fn count_expected_tokens<L, T>(filtered_tokens: &[T]) -> Result<usize, PartialError<L>>
where
    T: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    filtered_tokens.iter().try_fold(1, |acc, token| {
        (0..=L::MAX_ARITY)
            // Need to start with the biggest possible arity
            .rev()
            .find_map(|i| L::from_op(token.as_ref(), vec![Id::from(0); i]).ok())
            .map(|l| l.children().len())
            // Get the first find and add it to the count
            .map(|n_children| acc + n_children)
            // Otherwise error out
            .ok_or_else(|| PartialError::MaxArity(token.as_ref().to_owned(), L::MAX_ARITY))
    })
}

/// Lower the meta level of this partial lang to the underlying lang
///
/// # Errors
///
/// This function will return an error if it cant be lowered because meta lang nodes are still contained
pub fn lower_meta_level<L>(higher: RecExpr<PartialLang<L>>) -> Result<RecExpr<L>, PartialError<L>>
where
    L: FromOp + Display,
    L::Error: Display,
{
    higher
        .into_iter()
        .map(|partial_node| match partial_node {
            PartialLang::Finished(node) => Ok(node),
            PartialLang::Pad => Err(PartialError::NoLowering(partial_node.to_string())),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|v| v.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::python::data::TreeData;
    use crate::sketch::SketchLang;
    use crate::trs::rise::RiseLang;
    use crate::trs::{MetaInfo, halide::HalideLang};

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 <pad>) ?))"
            .parse::<PartialRecExpr<SketchLang<HalideLang>>>()
            .unwrap();
        assert_eq!(&sketch.to_string(), "(contains (max (min 1 <pad>) ?))");
    }

    #[test]
    fn operators() {
        let known_operators = vec![
            "<pad>", "?", "contains", "or", "+", "-", "*", "/", "%", "max", "min", "<", ">", "!",
            "<=", ">=", "==", "!=", "||", "&&",
        ];
        assert_eq!(
            PartialLang::<SketchLang<HalideLang>>::operators(),
            known_operators
        );
    }

    #[test]
    fn partial_parse_1_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "7", "2"];
        let (partial_node, _) = super::partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();
        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Number(7)),
                PartialLang::Finished(HalideLang::Sub([0.into(), 1.into()])),
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Pad,
                PartialLang::Finished(HalideLang::Add([3.into(), 4.into()])),
                PartialLang::Finished(HalideLang::Mul([2.into(), 5.into()]))
            ]
        );
        assert_eq!(&partial_rec_expr.to_string(), "(* (- v1 7) (+ 2 <pad>))");
    }

    #[test]
    fn partial_parse_2_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "2", "-", "v2"];

        let (partial_node, _) = super::partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Finished(HalideLang::Sub([0.into(), 1.into()])),
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(HalideLang::Sub([3.into(), 4.into()])),
                PartialLang::Finished(HalideLang::Symbol("v2".into())),
                PartialLang::Finished(HalideLang::Add([5.into(), 6.into()])),
                PartialLang::Finished(HalideLang::Mul([2.into(), 7.into()]))
            ]
        );
        assert_eq!(
            &partial_rec_expr.to_string(),
            "(* (- v1 2) (+ (- <pad> <pad>) v2))"
        );
    }

    #[test]
    fn empty_tokens() {
        let tokens = Vec::<&str>::new();
        let partial_error: PartialError<HalideLang> =
            super::partial_parse(tokens.as_slice(), None).unwrap_err();
        assert!(matches!(partial_error, PartialError::NoTokens))
    }

    #[test]
    fn filler_tokens_long() {
        let tokens = vec![
            "lam",
            ">>",
            "lam",
            ">>",
            "transpose",
            "[variable]",
            "lam",
            "[variable]",
            "transpose",
            "[variable]",
            "lam",
            "[variable]",
            "[variable]",
            "[variable]",
            "lam",
            "[variable]",
            "app",
            "app",
            "app",
            "map",
            "var",
            "lam",
            "var",
            "[variable]",
            "[variable]",
            "app",
            "[variable]",
            "app",
            "app",
            "app",
            "map",
            "var",
            "lam",
            "var",
            "[variable]",
            "[variable]",
            "app",
            "[variable]",
            "app",
            "app",
            "map",
            "var",
            "app",
            "var",
            "[variable]",
            "map",
            "var",
            "[variable]",
            "[variable]",
        ];
        let (partial_node, used_tokens) = super::partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        let _treedata: TreeData = (&partial_rec_expr).into();
        assert_eq!(
            partial_rec_expr.as_ref().last().unwrap(),
            &PartialLang::Finished(RiseLang::Lambda([4.into(), 11.into()])),
        );
        assert_eq!(used_tokens, 13);
        assert_eq!(
            &partial_rec_expr.to_string(),
            "(lam (>> (>> [variable] transpose) transpose) (lam [variable] (lam [variable] (lam [variable] [variable]))))"
        );
    }

    #[test]
    fn lam_overflow() {
        let tokens = vec!["lam"];
        let (partial_node, _) = super::partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        let _treedata: TreeData = (&partial_rec_expr).into();
        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(RiseLang::Lambda([0.into(), 1.into()])),
            ]
        );
    }

    #[test]
    fn partial_parse_recexpr() {
        let control: RecExpr<PartialLang<HalideLang>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let tokens = vec!["*", "-", "+", "2", "v1", "-", "v2"];
        let (partial_node, _) =
            super::partial_parse::<HalideLang, _>(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        assert_eq!(control.to_string(), partial_rec_expr.to_string());
        assert_eq!(
            &partial_rec_expr.to_string(),
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))"
        );
    }
}
