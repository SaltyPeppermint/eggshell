// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Display, Formatter};
use std::iter;
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
    Finished(L),
    Placeholder,
}

impl<L: Language> PartialLang<L> {
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Self::Placeholder)
    }
}

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialLang::Finished(x) => (discr, Some(x.discriminant())),
            PartialLang::Placeholder => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Placeholder => &[],
            Self::Finished(n) => n.children(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Placeholder => &mut [],
            Self::Finished(n) => n.children_mut(),
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
        let mut s = vec!["[placeholder]"];
        s.extend(L::operators());
        s
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placeholder => write!(f, "[placeholder]"),
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
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
            "[placeholder]" | "[PLACEHOLDER]" | "[Placeholder]" => {
                if children.is_empty() {
                    Ok(Self::Placeholder)
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
pub fn partial_parse<L, T>(value: &[T]) -> Result<RecExpr<PartialLang<L>>, MetaLangError<L::Error>>
where
    T: AsRef<str>,
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let mut children_ids = Vec::new();
    let mut nodes = Vec::new();
    for token in value {
        let nothings_to_add = L::MAX_ARITY.saturating_sub(children_ids.len());

        nodes.extend(iter::repeat_n(PartialLang::Placeholder, nothings_to_add));
        children_ids.extend((0..nothings_to_add).map(|x| Id::from(x + nodes.len() - 1)));

        let mut skipped_ids = 0;
        // First the parent case where all children on the stack are "absorbed"
        let r = loop {
            if let Ok(node) = L::from_op(token.as_ref(), children_ids[skipped_ids..].to_owned()) {
                children_ids.truncate(skipped_ids);
                break Ok(PartialLang::Finished(node));
            }
            if nodes.pop_if(|n| n.is_placeholder()).is_some() {
                children_ids.pop();
            } else if skipped_ids < children_ids.len() {
                skipped_ids += 1;
            } else {
                break Err(MetaLangError::<L::Error>::MaxArity(
                    token.as_ref().to_owned(),
                    L::MAX_ARITY,
                ));
            }
        }?;
        nodes.push(r);
        children_ids.push(Id::from(nodes.len() - 1));
    }
    Ok(RecExpr::from(nodes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_lang::SketchLang;
    use crate::trs::{MetaInfo, halide::HalideLang};

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 [placeholder]) ?))"
            .parse::<PartialTerm<SketchLang<HalideLang>>>()
            .unwrap();
        assert_eq!(
            &sketch.to_string(),
            "(contains (max (min 1 [placeholder]) ?))"
        );
    }

    #[test]
    fn operators() {
        let known_operators = vec![
            "[placeholder]",
            "?",
            "contains",
            "or",
            "+",
            "-",
            "*",
            "/",
            "%",
            "max",
            "min",
            "<",
            ">",
            "!",
            "<=",
            ">=",
            "==",
            "!=",
            "||",
            "&&",
        ];
        assert_eq!(
            PartialLang::<SketchLang<HalideLang>>::operators(),
            known_operators
        );
    }

    #[test]
    fn partial_parse_1_placeholder() {
        let str_list = vec!["7", "+", "2", "v1", "-", "*"];
        let l = super::partial_parse(str_list.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
            vec![
                PartialLang::Finished(HalideLang::Number(7)),
                PartialLang::Placeholder,
                PartialLang::Finished(HalideLang::Add([0.into(), 1.into()])),
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Sub([3.into(), 4.into()])),
                PartialLang::Finished(HalideLang::Mul([2.into(), 5.into()]))
            ]
        );
    }

    #[test]
    fn partial_parse_2_placeholder() {
        let str_list = vec!["v2", "-", "+", "2", "v1", "-", "*"];
        let l = super::partial_parse(str_list.as_slice()).unwrap();
        assert_eq!(
            l.as_ref().to_vec(),
            vec![
                PartialLang::Finished(HalideLang::Symbol("v2".into())),
                PartialLang::Placeholder,
                PartialLang::Finished(HalideLang::Sub([0.into(), 1.into()])),
                PartialLang::Placeholder,
                PartialLang::Finished(HalideLang::Add([2.into(), 3.into()])),
                PartialLang::Finished(HalideLang::Number(2)),
                PartialLang::Finished(HalideLang::Symbol("v1".into())),
                PartialLang::Finished(HalideLang::Sub([5.into(), 6.into()])),
                PartialLang::Finished(HalideLang::Mul([4.into(), 7.into()]))
            ]
        );
    }

    #[test]
    fn partial_parse_recexpr() {
        let control: RecExpr<PartialLang<HalideLang>> =
            "(* (+ (- v2 [placeholder]) [placeholder]) (- 2 v1))"
                .parse()
                .unwrap();
        let str_list = vec!["v2", "-", "+", "2", "v1", "-", "*"];
        let l = super::partial_parse::<HalideLang, _>(str_list.as_slice()).unwrap();
        assert_eq!(control, l);
        assert_eq!(
            l.to_string(),
            "(* (+ (- v2 [placeholder]) [placeholder]) (- 2 v1))".to_owned()
        );
    }
}
