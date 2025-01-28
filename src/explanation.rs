use std::fmt::Display;

use egg::{Analysis, EGraph, Explanation, FlatTerm, FromOp, Language, RecExpr};
use log::debug;
use serde::Serialize;

use crate::io::sampling::ExplanationData;

pub fn explain_equivalence<L: Language + FromOp + Display, N: Analysis<L>>(
    egraph: &mut EGraph<L, N>,
    from: &RecExpr<L>,
    to: &RecExpr<L>,
) -> ExplanationData<L> {
    debug!("Constructing explanation of \"{from} == {to}\"...");
    let mut expl = egraph.explain_equivalence(from, to);
    let expl_chain = explanation_chain(&mut expl);
    let flat_string = expl.get_flat_string();
    debug!("Explanation constructed!");
    ExplanationData::new(flat_string, expl_chain)
}

#[derive(Debug, PartialEq, Serialize, Clone)]
pub struct IntermediateTerms<L: Language + FromOp + Display> {
    pub rec_expr: RecExpr<L>,
    pub applied_rules: Vec<String>,
}

fn explanation_chain<L: Language + FromOp + Display>(
    explanation: &mut Explanation<L>,
) -> Vec<IntermediateTerms<L>> {
    let flat_expl = explanation.make_flat_explanation();
    flat_expl
        .iter()
        .map(|flat_term| {
            let mut applied_rules = vec![];
            find_used_rules(&mut applied_rules, flat_term);
            IntermediateTerms {
                rec_expr: flat_term.get_recexpr(),
                applied_rules,
            }
        })
        .collect()
}

fn find_used_rules<L: Language + FromOp + Display>(
    found_expl: &mut Vec<String>,
    flat_term: &FlatTerm<L>,
) {
    if let Some(rule) = flat_term.forward_rule {
        found_expl.push(rule.to_string());
    }
    for child in &flat_term.children {
        find_used_rules(found_expl, child);
    }
}
