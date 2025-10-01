use std::fmt::Debug;
use std::usize;

use egg::{Id, Language, RecExpr};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::Serialize;

use super::SampleError;

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PartialRecExpr<L: Language> {
    Filled {
        pick: L,
        children: Box<[PartialRecExpr<L>]>,
    },
    Open(Id),
}

pub struct Trace(Vec<usize>);

impl<L: Language> PartialRecExpr<L> {
    // Returns position of next open
    pub fn select_next_open<R: Rng>(&self, rng: &mut R) -> Option<(Id, Trace)> {
        match self {
            PartialRecExpr::Filled { children, .. } => {
                let weights = children.iter().map(|c| c.n_open()).collect::<Vec<_>>();
                if weights.iter().sum::<usize>() == 0 {
                    return None;
                }
                let this_key = WeightedIndex::new(weights).unwrap().sample(rng);
                let (id, mut trace) = children[this_key].select_next_open(rng)?;
                trace.0.push(this_key);
                Some((id, trace))
            }
            PartialRecExpr::Open(id) => Some((*id, Trace(vec![]))),
        }
    }

    fn get_slot_by_trace(&mut self, trace: &[usize]) -> &mut Self {
        match self {
            PartialRecExpr::Filled { children, .. } => {
                let (idx, rest_trace) = trace.split_last().unwrap();
                children[*idx].get_slot_by_trace(rest_trace)
            }
            PartialRecExpr::Open(_) => self,
        }
    }

    pub fn fill_next(&mut self, trace: Trace, pick: &L) -> Result<(), SampleError> {
        // Check if there is an open position to be filled
        let slot = self.get_slot_by_trace(&trace.0);
        if matches!(slot, Self::Filled { .. }) {
            return Err(SampleError::DoublePick);
        };
        let children = pick
            .children()
            .iter()
            .map(|c_id| PartialRecExpr::Open(*c_id))
            .collect();

        *slot = PartialRecExpr::Filled {
            pick: pick.to_owned(),
            children,
        };
        Ok(())
    }

    pub fn n_chosen(&self) -> usize {
        match self {
            PartialRecExpr::Filled { children, .. } => {
                1 + children.iter().map(|c| c.n_chosen()).sum::<usize>()
            }
            PartialRecExpr::Open(_) => 0,
        }
    }

    pub fn n_open(&self) -> usize {
        match self {
            PartialRecExpr::Filled { children, .. } => children.iter().map(|c| c.n_open()).sum(),
            PartialRecExpr::Open(_) => 1,
        }
    }

    pub fn open_ids(&self) -> Vec<Id> {
        match self {
            PartialRecExpr::Filled { children, .. } => {
                children.iter().flat_map(|c| c.open_ids()).collect()
            }
            PartialRecExpr::Open(id) => vec![*id],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PartialRecExpr::Filled { children, .. } => {
                1 + children.iter().map(|c| c.len()).sum::<usize>()
            }
            PartialRecExpr::Open(_) => 1,
        }
    }
}

impl<L: Language> From<Id> for PartialRecExpr<L> {
    fn from(id: Id) -> Self {
        PartialRecExpr::Open(id)
    }
}

impl<L: Language> TryFrom<PartialRecExpr<L>> for RecExpr<L> {
    type Error = SampleError;

    fn try_from(partial_rec_expr: PartialRecExpr<L>) -> Result<Self, Self::Error> {
        fn rec<LL: Language>(
            partial_rec_expr: PartialRecExpr<LL>,
            rec_expr: &mut RecExpr<LL>,
        ) -> Result<Id, SampleError> {
            let PartialRecExpr::Filled {
                children, mut pick, ..
            } = partial_rec_expr
            else {
                return Err(SampleError::UnfinishedChoice);
            };

            let new_c_ids = children
                .into_iter()
                .map(|c| rec(c, rec_expr))
                .collect::<Result<Vec<_>, _>>()?;
            for (old_id, new_id) in pick.children_mut().iter_mut().zip(new_c_ids) {
                *old_id = new_id;
            }
            Ok(rec_expr.add(pick))
        }

        let mut rec_expr = RecExpr::default();
        rec(partial_rec_expr, &mut rec_expr)?;
        Ok(rec_expr)
    }
}
