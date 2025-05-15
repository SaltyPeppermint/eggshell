use std::fmt::{Debug, Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{MetaLangError, TempNode};
use crate::trs::SymbolType;
use crate::trs::{MetaInfo, SymbolInfo};

/// Simple alias
pub type PartialTerm<L> = RecExpr<PartialLang<L>>;

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
    type Error = MetaLangError<L>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "[pad]" | "[PAD]" | "[Pad]" | "<pad>" | "<PAD>" | "<Pad>" => {
                if children.is_empty() {
                    Ok(Self::Pad)
                } else {
                    Err(MetaLangError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }

            _ => L::from_op(op, children)
                .map(Self::Finished)
                .map_err(MetaLangError::BadOp),
        }
    }
}

/// Tries to parse a list of tokens into a list of nodes in the language
///
/// # Errors
///
/// This function will return an error if it cannot be parsed as a partial term
pub fn partial_parse<L, T>(
    tokens: &[T],
) -> Result<(RecExpr<PartialLang<L>>, usize), MetaLangError<L>>
where
    T: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    // If this is empty, return the empty RecExpr
    if tokens.is_empty() {
        return Ok((RecExpr::default(), 0));
    }

    let mut ast = TempNode::new_empty();
    for (used_tokens, token) in tokens.iter().enumerate() {
        let mut children_ids = vec![Id::from(0); L::MAX_ARITY];
        let (node, arity) = loop {
            if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
                break (node, children_ids.len());
            }
            children_ids.pop();
        };

        if let Some(position) = ast.find_next_open() {
            *position = TempNode {
                node: PartialLang::Finished(node),
                children: vec![TempNode::new_empty(); arity],
            };
        } else {
            return Ok((ast.into(), used_tokens));
        }
    }

    Ok((ast.into(), tokens.len()))
    // First determine the number of placeholders starting with the root, which should always be there
    // let expected_tokens = count_expected_tokens::<L, _>(tokens)?;
    // dbg!(expected_tokens);
    // dbg!(tokens.len());

    // let mut nodes = vec![PartialLang::Pad; expected_tokens - tokens.len()];
    // let mut used_pointer = 0;

    // for token in tokens.iter().rev() {
    //     // Cant absorb more nodes than in those that are not yet used up
    //     let end_pointer = usize::min(used_pointer + L::MAX_ARITY, nodes.len());
    //     let mut children_ids = (used_pointer..end_pointer)
    //         .rev()
    //         .map(Id::from)
    //         .collect::<Vec<_>>();
    //     let r = loop {
    //         // Try to absorb all children on the stack with the new token
    //         if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
    //             break PartialLang::Finished(node);
    //         }
    //         if children_ids.pop().is_none() {
    //             return Err(MetaLangError::<L::Error>::MaxArity(
    //                 token.as_ref().to_owned(),
    //                 L::MAX_ARITY,
    //             ));
    //         }
    //     };
    //     nodes.push(r);
    //     used_pointer += children_ids.len();
    // }
    // Ok(RecExpr::from(nodes))
}

/// Count how many nodes are expected to be there, including to be generated nodes atm
///
/// # Errors
///
/// This function will return an error if it cannot parse any tokens
pub fn count_expected_tokens<L, T>(filtered_tokens: &[T]) -> Result<usize, MetaLangError<L>>
where
    T: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let expected_tokens = filtered_tokens.iter().try_fold(1, |acc, token| {
        (0..=L::MAX_ARITY)
            // Need to start with the biggest possible arity
            .rev()
            .find_map(|i| L::from_op(token.as_ref(), vec![Id::from(0); i]).ok())
            .map(|l| l.children().len())
            // Get the first find and add it to the count
            .map(|n_children| acc + n_children)
            // Otherwise error out
            .ok_or_else(|| MetaLangError::MaxArity(token.as_ref().to_owned(), L::MAX_ARITY))
    })?;
    Ok(expected_tokens)
}

/// Lower the meta level of this partial lang to the underlying lang
///
/// # Errors
///
/// This function will return an error if it cant be lowered because meta lang nodes are still contained
pub fn lower_meta_level<L>(higher: &RecExpr<PartialLang<L>>) -> Result<RecExpr<L>, MetaLangError<L>>
where
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let nodes = higher
        .iter()
        .map(|partial_node| match partial_node {
            PartialLang::Finished(node) => Ok(node.to_owned()),
            PartialLang::Pad => Err(MetaLangError::NoLowering(partial_node.to_string())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(nodes.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_lang::SketchLang;
    use crate::python::data::TreeData;
    use crate::trs::rise::RiseLang;
    use crate::trs::{MetaInfo, halide::HalideLang};

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 <pad>) ?))"
            .parse::<PartialTerm<SketchLang<HalideLang>>>()
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
        let (l, _) = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
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
        assert_eq!(&l.to_string(), "(* (- v1 7) (+ 2 <pad>))");
    }

    #[test]
    fn partial_parse_2_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "2", "-", "v2"];

        let (l, _) = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
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
        assert_eq!(&l.to_string(), "(* (- v1 2) (+ (- <pad> <pad>) v2))");
    }

    #[test]
    fn empty_tokens() {
        let tokens = Vec::<&str>::new();
        let (parsed, used_tokens) = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(used_tokens, 0);
        assert_eq!(RecExpr::<PartialLang<HalideLang>>::default(), parsed);
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
        let (parsed, used_tokens) = super::partial_parse(tokens.as_slice()).unwrap();
        let _treedata: TreeData = (&parsed).into();
        assert_eq!(
            parsed.as_ref().last().unwrap(),
            &PartialLang::Finished(RiseLang::Lambda([4.into(), 11.into()])),
        );
        assert_eq!(used_tokens, 13);
        assert_eq!(
            &parsed.to_string(),
            "(lam (>> (>> [variable] transpose) transpose) (lam [variable] (lam [variable] (lam [variable] [variable]))))"
        );
    }

    #[test]
    fn lam_overflow() {
        let tokens = vec!["lam"];
        let (parsed, _) = super::partial_parse(tokens.as_slice()).unwrap();
        let _treedata: TreeData = (&parsed).into();
        assert_eq!(
            parsed.as_ref().to_vec(),
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
        let (l, _) = super::partial_parse::<HalideLang, _>(tokens.as_slice()).unwrap();
        assert_eq!(control.to_string(), l.to_string());
        assert_eq!(&l.to_string(), "(* (- 2 v1) (+ (- <pad> <pad>) v2))");
    }
}
