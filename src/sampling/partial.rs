use core::panic;
use std::fmt::Debug;
use std::usize;

use egg::{Id, Language, RecExpr};
use hashbrown::HashSet;
use rand::prelude::*;

use super::SampleError;

#[derive(Debug)]
pub struct PartialRecExpr<L: Language> {
    slots: Vec<Slot<L>>,
    open_slot_idx: HashSet<Key>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(usize);

#[derive(Debug)]
enum Slot<L: Language> {
    Open(Id),
    Picked(L),
    Dummy,
}

impl<L: Language> Default for Slot<L> {
    fn default() -> Self {
        Slot::Dummy
    }
}

impl<L: Language> PartialRecExpr<L> {
    // Returns position of next open
    pub fn select_next_open<R: Rng>(&self, rng: &mut R) -> Option<(Id, Key)> {
        self.open_slot_idx.iter().choose(rng).map(|key| {
            let Slot::Open(id) = self.slots[key.0] else {
                panic!("Picked a non open slot")
            };
            (id, *key)
        })
    }

    pub fn fill_next(&mut self, key: Key, pick: &L) -> Result<(), SampleError> {
        // Check if there is an open position to be filled
        // let slot = self.get_slot_by_trace(&trace.0);
        if matches!(self.slots[key.0], Slot::Picked(_)) {
            return Err(SampleError::DoublePick);
        };
        let old_len = self.slots.len();
        self.slots
            .extend(pick.children().iter().map(|id| Slot::Open(*id)));

        let mut pick_owned = pick.to_owned();
        for (c, idx) in pick_owned
            .children_mut()
            .iter_mut()
            .zip(old_len..self.slots.len())
        {
            *c = Id::from(idx);
        }

        self.slots[key.0] = Slot::Picked(pick_owned);
        self.open_slot_idx.remove(&key);

        self.open_slot_idx
            .extend((old_len..self.slots.len()).map(|i| Key(i)));
        Ok(())
    }

    pub fn n_open(&self) -> usize {
        self.open_slot_idx.len()
    }

    pub fn open_ids(&self) -> impl Iterator<Item = Id> {
        self.open_slot_idx
            .iter()
            .map(|key| match &self.slots[key.0] {
                Slot::Open(id) => *id,
                _ => panic!("noooo"),
            })
    }

    pub fn n_chosen(&self) -> usize {
        self.len() - self.n_open()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

impl<L: Language> From<Id> for PartialRecExpr<L> {
    fn from(id: Id) -> Self {
        PartialRecExpr {
            slots: vec![Slot::Open(id)],
            open_slot_idx: HashSet::from([Key(0)]),
        }
    }
}

impl<L: Language> TryFrom<PartialRecExpr<L>> for RecExpr<L> {
    type Error = SampleError;

    fn try_from(mut partial_rec_expr: PartialRecExpr<L>) -> Result<Self, Self::Error> {
        fn rec<LL: Language>(
            partial_rec_expr: &mut PartialRecExpr<LL>,
            idx: usize,
            rec_expr: &mut RecExpr<LL>,
        ) -> Result<Id, SampleError> {
            let Slot::Picked(mut node) = std::mem::take(&mut partial_rec_expr.slots[idx]) else {
                return Err(SampleError::UnfinishedChoice);
            };

            for child_id in node.children_mut() {
                *child_id = rec(partial_rec_expr, usize::from(*child_id), rec_expr)?;
            }

            Ok(rec_expr.add(node))
        }

        let mut rec_expr = RecExpr::default();
        rec(&mut partial_rec_expr, 0, &mut rec_expr)?;
        debug_assert!(
            partial_rec_expr
                .slots
                .iter()
                .all(|s| matches!(s, Slot::Dummy))
        );
        debug_assert!(partial_rec_expr.open_slot_idx.is_empty());
        Ok(rec_expr)
    }
}

// #[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
// pub enum PartialRecExpr<L: Language> {
//     Filled {
//         pick: L,
//         children: Box<[PartialRecExpr<L>]>,
//     },
//     Open(Id),
// }

// pub struct Trace(Vec<usize>);

// impl<L: Language> PartialRecExpr<L> {
//     // Returns position of next open
//     pub fn select_next_open<R: Rng>(&mut self, rng: &mut R) -> Option<(Id, &mut Self)> {
//         match self {
//             PartialRecExpr::Filled { children, .. } => {
//                 let weights = children.iter().map(|c| c.n_open()).collect::<Vec<_>>();
//                 if weights.iter().sum::<usize>() == 0 {
//                     return None;
//                 }
//                 let this_key = WeightedIndex::new(weights).unwrap().sample(rng);
//                 let (id, slot) = children[this_key].select_next_open(rng)?;
//                 Some((id, slot))
//             }
//             PartialRecExpr::Open(id) => Some((*id, self)),
//         }
//     }

//     fn get_slot_by_trace(&mut self, trace: &[usize]) -> &mut Self {
//         match self {
//             PartialRecExpr::Filled { children, .. } => {
//                 let (idx, rest_trace) = trace.split_last().unwrap();
//                 children[*idx].get_slot_by_trace(rest_trace)
//             }
//             PartialRecExpr::Open(_) => self,
//         }
//     }

//     pub fn fill_next(&mut self, pick: &L) -> Result<(), SampleError> {
//         // Check if there is an open position to be filled
//         // let slot = self.get_slot_by_trace(&trace.0);
//         if matches!(self, Self::Filled { .. }) {
//             return Err(SampleError::DoublePick);
//         };
//         let children = pick
//             .children()
//             .iter()
//             .map(|c_id| PartialRecExpr::Open(*c_id))
//             .collect();

//         *self = PartialRecExpr::Filled {
//             pick: pick.to_owned(),
//             children,
//         };
//         Ok(())
//     }

//     pub fn n_chosen(&self) -> usize {
//         match self {
//             PartialRecExpr::Filled { children, .. } => {
//                 1 + children.iter().map(|c| c.n_chosen()).sum::<usize>()
//             }
//             PartialRecExpr::Open(_) => 0,
//         }
//     }

//     pub fn n_open(&self) -> usize {
//         match self {
//             PartialRecExpr::Filled { children, .. } => children.iter().map(|c| c.n_open()).sum(),
//             PartialRecExpr::Open(_) => 1,
//         }
//     }

//     pub fn open_ids(&self) -> Vec<Id> {
//         match self {
//             PartialRecExpr::Filled { children, .. } => {
//                 children.iter().flat_map(|c| c.open_ids()).collect()
//             }
//             PartialRecExpr::Open(id) => vec![*id],
//         }
//     }

//     pub fn len(&self) -> usize {
//         match self {
//             PartialRecExpr::Filled { children, .. } => {
//                 1 + children.iter().map(|c| c.len()).sum::<usize>()
//             }
//             PartialRecExpr::Open(_) => 1,
//         }
//     }
// }

// impl<L: Language> From<Id> for PartialRecExpr<L> {
//     fn from(id: Id) -> Self {
//         PartialRecExpr::Open(id)
//     }
// }

// impl<L: Language> TryFrom<PartialRecExpr<L>> for RecExpr<L> {
//     type Error = SampleError;

//     fn try_from(partial_rec_expr: PartialRecExpr<L>) -> Result<Self, Self::Error> {
//         fn rec<LL: Language>(
//             partial_rec_expr: PartialRecExpr<LL>,
//             rec_expr: &mut RecExpr<LL>,
//         ) -> Result<Id, SampleError> {
//             let PartialRecExpr::Filled {
//                 children, mut pick, ..
//             } = partial_rec_expr
//             else {
//                 return Err(SampleError::UnfinishedChoice);
//             };

//             let new_c_ids = children
//                 .into_iter()
//                 .map(|c| rec(c, rec_expr))
//                 .collect::<Result<Vec<_>, _>>()?;
//             for (old_id, new_id) in pick.children_mut().iter_mut().zip(new_c_ids) {
//                 *old_id = new_id;
//             }
//             Ok(rec_expr.add(pick))
//         }

//         let mut rec_expr = RecExpr::default();
//         rec(partial_rec_expr, &mut rec_expr)?;
//         Ok(rec_expr)
//     }
// }
