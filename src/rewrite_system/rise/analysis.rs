use egg::{Analysis, AstSize, CostFunction, DidMerge, EGraph, Language, RecExpr};
use hashbrown::HashSet;

use crate::rewrite_system::rise::kind::Kindable;

use super::nat::Math;
use super::{Index, Rise};

#[derive(Default, Debug)]
pub struct RiseAnalysis(HashSet<(RecExpr<Math>, RecExpr<Math>)>);

impl RiseAnalysis {
    pub fn new() -> Self {
        Self(HashSet::default())
    }

    pub fn get_mut_term_bank(&mut self) -> &mut HashSet<(RecExpr<Math>, RecExpr<Math>)> {
        &mut self.0
    }
}

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
        let did_change = before_len != to.free.len();

        if !from.beta_extract.is_empty()
            && (to.beta_extract.is_empty() || to.beta_extract.len() > from.beta_extract.len())
        {
            to.beta_extract = from.beta_extract;
            return DidMerge(true, true);
        }
        DidMerge(did_change, true) // TODO: more precise second bool
    }

    fn make(egraph: &mut EGraph<Rise, RiseAnalysis>, enode: &Rise) -> AnalysisData {
        let free = match enode {
            Rise::Var(v) => [*v].into(),
            Rise::Lambda(e)
            | Rise::NatLambda(e)
            | Rise::DataLambda(e)
            | Rise::AddrLambda(e)
            | Rise::NatNatLambda(e) => egraph[*e]
                .data
                .free
                .iter()
                .filter(|idx| !idx.is_zero() && idx.kind() == enode.kind())
                .map(|idx| idx.downshifted())
                .collect(),
            _ => enode
                .children()
                .iter()
                .flat_map(|c| egraph[*c].data.free.iter())
                .copied()
                .collect(),
        };
        let empty = enode.any(|id| egraph[id].data.beta_extract.as_ref().is_empty());
        let beta_extract = if empty {
            RecExpr::default()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };
        AnalysisData { free, beta_extract }
    }
}
