// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Debug, Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::MetaLangError;
use crate::trs::SymbolType::MetaSymbol;
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
    Start,
    End,
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
            _ => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Finished(n) => n.children(),
            _ => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Finished(n) => n.children_mut(),
            _ => &mut [],
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
            SymbolInfo::new(position + L::NUM_SYMBOLS, MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["<pad>", "<s>", "</s>"];
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
            Self::Start => write!(f, "<s>"),
            Self::End => write!(f, "</s>"),
        }
    }
}

impl<L> egg::FromOp for PartialLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = MetaLangError<L::Error>;

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
            "[s]" | "[S]" | "<s>" | "<S>" => {
                if children.is_empty() {
                    Ok(Self::Start)
                } else {
                    Err(MetaLangError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "[/s]" | "[/S]" | "</s>" | "</S>" => {
                if children.is_empty() {
                    Ok(Self::End)
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
pub fn partial_parse<L, T>(tokens: &[T]) -> Result<RecExpr<PartialLang<L>>, MetaLangError<L::Error>>
where
    T: AsRef<str> + Debug,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let start = tokens.iter().position(|x| {
        !PartialLang::<L>::from_op(x.as_ref(), vec![])
            .is_ok_and(|l| !matches!(l, PartialLang::Finished(_)))
    });
    let end = tokens.iter().rev().position(|x| {
        !PartialLang::<L>::from_op(x.as_ref(), vec![])
            .is_ok_and(|l| !matches!(l, PartialLang::Finished(_)))
    });
    let filtered_tokens = if let (Some(s), Some(e)) = (start, end) {
        &tokens[s..tokens.len() - e]
    } else {
        return Ok(RecExpr::default());
    };

    // First determine the number of placeholders starting with the root, which should always be there
    let mut expected_tokens = 1;
    for token in filtered_tokens {
        // Need to start with the biggest possible arity
        for i in (0..=L::MAX_ARITY).rev() {
            if let Ok(l) = L::from_op(token.as_ref(), vec![Id::from(0); i]) {
                // Count children
                expected_tokens += l.children().len();
                break;
            }
            if i == 0 {
                return Err(MetaLangError::<L::Error>::MaxArity(
                    token.as_ref().to_owned(),
                    L::MAX_ARITY,
                ));
            }
        }
    }

    let mut nodes = vec![PartialLang::Pad; expected_tokens - filtered_tokens.len()];
    let mut used_pointer = 0;

    for token in filtered_tokens.iter().rev() {
        // Cant absorb more nodes than in those that are not yet used up
        let end_pointer = usize::min(used_pointer + L::MAX_ARITY, nodes.len());
        let mut children_ids = (used_pointer..end_pointer)
            .rev()
            .map(Id::from)
            .collect::<Vec<_>>();
        let r = loop {
            // Try to absorb all children on the stack with the new token

            if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
                break PartialLang::Finished(node);
            }
            if children_ids.pop().is_none() {
                return Err(MetaLangError::<L::Error>::MaxArity(
                    token.as_ref().to_owned(),
                    L::MAX_ARITY,
                ));
            }
        };
        nodes.push(r);
        used_pointer += children_ids.len();
    }
    Ok(RecExpr::from(nodes))
}

/// Lower the meta level of this partial lang to the underlying lang
///
/// # Errors
///
/// This function will return an error if it cant be lowered because meta lang nodes are still contained
pub fn lower_meta_level<L>(
    higher: &RecExpr<PartialLang<L>>,
) -> Result<RecExpr<L>, MetaLangError<L::Error>>
where
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let nodes = higher
        .iter()
        .map(|partial_node| match partial_node {
            PartialLang::Finished(node) => Ok(node.to_owned()),
            _ => Err(MetaLangError::NoLowering(partial_node.to_string())),
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
            "<pad>", "<s>", "</s>", "?", "contains", "or", "+", "-", "*", "/", "%", "max", "min",
            "<", ">", "!", "<=", ">=", "==", "!=", "||", "&&",
        ];
        assert_eq!(
            PartialLang::<SketchLang<HalideLang>>::operators(),
            known_operators
        );
    }

    #[test]
    fn partial_parse_1_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "7", "2"];
        let l = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
            vec![
                PartialLang::Pad,
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Finished(HalideLang::Number(7)),
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Add([1.into(), 0.into()])),
                PartialLang::Finished(HalideLang::Sub([3.into(), 2.into()])),
                PartialLang::Finished(HalideLang::Mul([5.into(), 4.into()]))
            ]
        );
        assert_eq!(l.to_string(), "(* (- v1 7) (+ 2 <pad>))".to_owned());
    }

    #[test]
    fn partial_parse_2_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "2", "-", "v2"];

        let l = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
            vec![
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(HalideLang::Symbol("v2".into())),
                PartialLang::Finished(HalideLang::Sub([1.into(), 0.into()])),
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Add([3.into(), 2.into()])),
                PartialLang::Finished(HalideLang::Sub([5.into(), 4.into()])),
                PartialLang::Finished(HalideLang::Mul([7.into(), 6.into()]))
            ]
        );
        assert_eq!(
            l.to_string(),
            "(* (- v1 2) (+ (- <pad> <pad>) v2))".to_owned()
        );
    }

    #[test]
    fn prefix_postfix_strip() {
        let tokens = vec!["<s>", "*", "-", "+", "v1", "2", "-", "v2", "<pad>", "<pad>"];
        let tokens_stripped = vec!["*", "-", "+", "v1", "2", "-", "v2"];

        let parsed = super::partial_parse(tokens.as_slice()).unwrap();
        let parsed_stripped = super::partial_parse(tokens_stripped.as_slice()).unwrap();
        assert_eq!(parsed, parsed_stripped);
        assert_eq!(
            parsed_stripped.as_ref()[2],
            PartialLang::Finished(HalideLang::Symbol("v2".into()))
        );
    }

    #[test]
    fn empty_tokens() {
        let tokens = Vec::<&str>::new();
        let parsed = super::partial_parse(tokens.as_slice()).unwrap();
        assert_eq!(RecExpr::<PartialLang<HalideLang>>::default(), parsed);
    }

    #[test]
    fn filler_tokens() {
        let tokens = vec![
            "<s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
            "<pad>",
        ];
        let parsed = super::partial_parse(tokens.as_slice()).unwrap();
        let _treedata: TreeData = (&parsed).into();
        assert_eq!(RecExpr::<PartialLang<RiseLang>>::default(), parsed);
    }

    #[test]
    fn lam_overflow() {
        let tokens = vec!["lam"];
        let parsed = super::partial_parse(tokens.as_slice()).unwrap();
        let _treedata: TreeData = (&parsed).into();
        assert_eq!(
            parsed.as_ref().to_vec(),
            vec![
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(RiseLang::Lambda([1.into(), 0.into()])),
            ]
        );
    }

    #[test]
    fn partial_parse_recexpr() {
        let control: RecExpr<PartialLang<HalideLang>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let tokens = vec!["*", "-", "+", "2", "v1", "-", "v2"];
        let l = super::partial_parse::<HalideLang, _>(tokens.as_slice()).unwrap();
        assert_eq!(control.to_string(), l.to_string());
        assert_eq!(
            l.to_string(),
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".to_owned()
        );
    }
}
