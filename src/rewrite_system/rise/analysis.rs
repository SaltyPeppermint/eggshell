use egg::{Analysis, DidMerge, EGraph, Language, RecExpr};
use hashbrown::HashSet;

use super::{Index, Rise};

#[derive(Default, Debug)]
pub struct RiseAnalysis;

#[derive(Default, Debug)]
pub struct AnalysisData {
    pub free: HashSet<Index>,
    pub beta_extract: RecExpr<Rise>,
}

impl Analysis<Rise> for RiseAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, to: &mut AnalysisData, from: AnalysisData) -> DidMerge {
        let before_len = to.free.len();
        to.free.extend(from.free);
        let mut did_change = before_len != to.free.len();
        if !from.beta_extract.as_ref().is_empty()
            && (to.beta_extract.as_ref().is_empty()
                || to.beta_extract.as_ref().len() > from.beta_extract.as_ref().len())
        {
            to.beta_extract = from.beta_extract;
            did_change = true;
        }
        DidMerge(did_change, true) // TODO: more precise second bool
    }

    fn make(egraph: &mut EGraph<Rise, RiseAnalysis>, enode: &Rise) -> AnalysisData {
        let mut free = HashSet::default();
        match enode {
            Rise::Var(v) => {
                free.insert(*v);
            }
            Rise::Lambda(a) => {
                free.extend(
                    egraph[*a]
                        .data
                        .free
                        .iter()
                        .copied()
                        .filter(|&idx| idx != Index::zero())
                        .map(|idx| idx - 1),
                );
            }
            _ => {
                enode.for_each(|c| free.extend(&egraph[c].data.free));
            }
        }
        let empty = enode.any(|id| egraph[id].data.beta_extract.as_ref().is_empty());
        let beta_extract = if empty {
            // vec![].into()
            RecExpr::default()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };
        AnalysisData { free, beta_extract }
    }
}
