use std::fmt::Display;

use egg::{Analysis, EGraph, Explanation, FlatTerm, FromOp, Language, RecExpr};
use log::debug;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct ExplanationData<L: Language + FromOp + Display> {
    flat_string: String,
    explanation_chain: Vec<IntermediateTerms<L>>,
}

impl<L: Language + FromOp + Display> ExplanationData<L> {
    #[must_use]
    pub fn new(flat_string: String, explanation_chain: Vec<IntermediateTerms<L>>) -> Self {
        Self {
            flat_string,
            explanation_chain,
        }
    }
}

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
            let rec_expr = flat_term.get_recexpr();
            let mut applied_rules = Vec::new();
            rec_applied_rules(&mut applied_rules, flat_term);
            IntermediateTerms {
                rec_expr,
                applied_rules,
            }
        })
        .collect()
}

fn rec_applied_rules<L: Language>(rules: &mut Vec<String>, expl: &FlatTerm<L>) {
    if let Some(rule) = expl.forward_rule {
        rules.push(rule.to_string());
    }
    for child in &expl.children {
        rec_applied_rules(rules, child);
    }
}
