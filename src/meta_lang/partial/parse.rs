use std::fmt::{Debug, Display};

use egg::{FromOp, Id};

use super::PartialLang;
use super::error::PartialError;

use crate::meta_lang::ProbabilisticLang;
use crate::node::RecNode;
use crate::rewrite_system::LangExtras;

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
                .map(|l| Self::Finished(l))
                .map_err(PartialError::BadOp),
        }
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
) -> Result<(RecNode<PartialLang<ProbabilisticLang<L>>>, usize), PartialError<L>>
where
    S: AsRef<str> + Debug,
    L: FromOp + LangExtras,
    L::Error: Display,
{
    // If this is empty, return the empty RecExpr
    if tokens.is_empty() {
        return Err(PartialError::NoTokens);
    }

    let mut ast = RecNode::new_empty();
    let v = match probs {
        Some(v) => v.iter().map(|p| Some(*p)).collect(),
        None => vec![None; tokens.len()],
    };
    for (index, (token, prob)) in tokens.iter().zip(v).enumerate() {
        let mut children_ids = vec![Id::from(0); L::MAX_ARITY];
        let (node, arity) = loop {
            if let Ok(node) = L::from_op(token.as_ref(), children_ids.clone()) {
                break (node, children_ids.len());
            }
            children_ids.pop();
        };

        if let Some(position) = ast.find_next_open() {
            *position = RecNode::new(
                PartialLang::Finished(ProbabilisticLang::new(node, prob)),
                vec![RecNode::new_empty(); arity],
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
    L: FromOp + LangExtras,
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

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use super::*;
    use crate::meta_lang::SketchLang;
    use crate::meta_lang::partial::PartialRecExpr;
    use crate::rewrite_system::halide::HalideLang;
    use crate::rewrite_system::rise::RiseLang;
    use crate::tree_data::TreeData;

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 <pad>) ?))"
            .parse::<PartialRecExpr<ProbabilisticLang<SketchLang<HalideLang>>>>()
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
        let (partial_node, _) = partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();
        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Symbol("v1".into()))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Number(7))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Sub([
                    0.into(),
                    1.into()
                ]))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Number(2))),
                PartialLang::Pad,
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Add([
                    3.into(),
                    4.into()
                ]))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Mul([
                    2.into(),
                    5.into()
                ]))),
            ]
        );
        assert_eq!(&partial_rec_expr.to_string(), "(* (- v1 7) (+ 2 <pad>))");
    }

    #[test]
    fn partial_parse_2_placeholder() {
        let tokens = vec!["*", "-", "+", "v1", "2", "-", "v2"];

        let (partial_node, _) = partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Symbol("v1".into()))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Number(2))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Sub([
                    0.into(),
                    1.into()
                ]))),
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Sub([
                    3.into(),
                    4.into()
                ]))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Symbol("v2".into()))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Add([
                    5.into(),
                    6.into()
                ]))),
                PartialLang::Finished(ProbabilisticLang::NoProb(HalideLang::Mul([
                    2.into(),
                    7.into()
                ]))),
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
            partial_parse(tokens.as_slice(), None).unwrap_err();
        assert!(matches!(partial_error, PartialError::NoTokens))
    }

    #[test]
    fn lam_overflow() {
        let tokens = vec!["lam"];
        let (partial_node, _) = partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        let _treedata: TreeData = (&partial_rec_expr).into();
        assert_eq!(
            partial_rec_expr.as_ref().to_vec(),
            vec![
                PartialLang::Pad,
                PartialLang::Pad,
                PartialLang::Finished(ProbabilisticLang::NoProb(RiseLang::Lambda([
                    0.into(),
                    1.into()
                ]))),
            ]
        );
    }

    #[test]
    fn partial_parse_recexpr() {
        let control: RecExpr<PartialLang<ProbabilisticLang<HalideLang>>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let tokens = vec!["*", "-", "+", "2", "v1", "-", "v2"];
        let (partial_node, _) = partial_parse::<HalideLang, _>(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        assert_eq!(control.to_string(), partial_rec_expr.to_string());
        assert_eq!(
            &partial_rec_expr.to_string(),
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))"
        );
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
        let (partial_node, used_tokens) = partial_parse(tokens.as_slice(), None).unwrap();
        let partial_rec_expr: RecExpr<_> = partial_node.into();

        let _treedata: TreeData = (&partial_rec_expr).into();
        assert_eq!(
            partial_rec_expr.as_ref().last().unwrap(),
            &PartialLang::Finished(ProbabilisticLang::NoProb(RiseLang::Lambda([
                4.into(),
                11.into()
            ]))),
        );
        assert_eq!(used_tokens, 13);
        assert_eq!(
            &partial_rec_expr.to_string(),
            "(lam (>> (>> [variable] transpose) transpose) (lam [variable] (lam [variable] (lam [variable] [variable]))))"
        );
    }
}
