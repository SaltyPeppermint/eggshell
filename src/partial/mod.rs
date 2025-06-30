mod error;

use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOp, Id, Language, RecExpr};
use error::PartialError;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use crate::trs::SymbolType;
use crate::trs::{MetaInfo, SymbolInfo};

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

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialLang::Finished(x) => (discr, Some(x.discriminant())),
            Self::Pad => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Finished(n) => n.children(),
            Self::Pad => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Finished(n) => n.children_mut(),
            Self::Pad => &mut [],
        }
    }
}

impl<L: Language + MetaInfo> MetaInfo for PartialLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        if let PartialLang::Finished(l) = self {
            l.symbol_info()
        } else {
            let position = PartialLangDiscriminants::iter()
                .position(|x| x == self.into())
                .unwrap();
            SymbolInfo::new(position + L::NUM_SYMBOLS, SymbolType::MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["<pad>"];
        s.extend(L::operators());
        s
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
            Self::Pad => write!(f, "<pad>"),
        }
    }
}

impl<L> egg::FromOp for PartialLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = PartialError<L>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "[pad]" | "[PAD]" | "[Pad]" | "<pad>" | "<PAD>" | "<Pad>" => {
                if children.is_empty() {
                    Ok(Self::Pad)
                } else {
                    Err(PartialError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }

            _ => L::from_op(op, children)
                .map(Self::Finished)
                .map_err(PartialError::BadOp),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartialNode<L: Language> {
    node: PartialLang<L>,
    children: Vec<PartialNode<L>>,
    prob: Option<f64>,
}

impl<L: Language> PartialNode<L> {
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
            prob: None,
        }
    }
}

impl<L: Language> From<PartialNode<L>> for RecExpr<PartialLang<L>> {
    fn from(root: PartialNode<L>) -> Self {
        fn rec<LL: Language>(mut curr: PartialNode<LL>, vec: &mut Vec<PartialLang<LL>>) -> Id {
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

impl<L> TryFrom<PartialNode<L>> for (RecExpr<PartialLang<L>>, Vec<f64>)
where
    L::Error: Display,
    L: Language + FromOp,
{
    type Error = PartialError<PartialLang<L>>;

    fn try_from(root: PartialNode<L>) -> Result<Self, Self::Error> {
        fn rec<LL>(
            mut curr: PartialNode<LL>,
            stack: &mut Vec<PartialLang<LL>>,
            probs_stack: &mut Vec<f64>,
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
            let prob = curr
                .prob
                .ok_or_else(|| PartialError::NoProbability(format!("{:?}", &curr.node)))?;
            stack.push(curr.node);
            probs_stack.push(prob);
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
pub fn partial_parse<L, T>(
    tokens: &[T],
    probs: Option<&[f64]>,
) -> Result<(PartialNode<L>, usize), PartialError<L>>
where
    T: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    // If this is empty, return the empty RecExpr
    if tokens.is_empty() {
        return Err(PartialError::NoTokens);
    }

    let mut ast = PartialNode::new_empty();
    for (index, token) in tokens.iter().enumerate() {
        let mut children_ids = vec![Id::from(0); L::MAX_ARITY];
        let (node, arity) = loop {
            if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
                break (node, children_ids.len());
            }
            children_ids.pop();
        };

        if let Some(position) = ast.find_next_open() {
            *position = PartialNode {
                node: PartialLang::Finished(node),
                children: vec![PartialNode::new_empty(); arity],
                prob: probs.map(|v| v[index]),
            };
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
    L: FromOp + MetaInfo,
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
