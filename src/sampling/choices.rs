use std::collections::BTreeSet;

use egg::{Id, Language, RecExpr};
use rand::Rng;
use rand::seq::IteratorRandom;
use serde::Serialize;

use super::SampleError;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct PartialRecExpr<L: Language> {
    slots: Vec<Slot<L>>,
    // We need something other than hashbrown or std HashSet cause
    // they DO NOT produce stable iterators between program runs
    open_slots: BTreeSet<usize>,
    next_to_fill: Option<usize>,
}

impl<L: Language> PartialRecExpr<L> {
    // Returns position of next open
    pub fn select_next_open<R: Rng>(&mut self, rng: &mut R) -> Option<Id> {
        if let Some(next_position) = self.open_slots.iter().choose(rng) {
            self.next_to_fill = Some(*next_position);
            // dbg!(&self.choices[*next_position]);
            let id = self.slots[*next_position].eclass_id();
            Some(id)
        } else {
            self.next_to_fill = None;
            None
        }
    }

    pub fn fill_next(&mut self, pick: &L) -> Result<(), SampleError> {
        // Check if there is an open position to be filled
        let position = self.next_to_fill.ok_or(SampleError::UnfinishedChoice)?;
        debug_assert!(matches!(self.slots[position].pick, Pick::Open));

        let mut owned_pick = pick.to_owned();

        // Create new open slots for children with the eclass ids of the children
        let new_open_slots = owned_pick.children().iter().map(|child_id| Slot {
            eclass_id: *child_id,
            parent: Some(position),
            pick: Pick::Open,
        });

        let old_len = self.slots.len();
        self.slots.extend(new_open_slots);

        // Calculate and add new positions to open positions hashmap
        for (i, child) in owned_pick.children_mut().iter_mut().enumerate() {
            let child_position = old_len + i;
            *child = Id::from(child_position);
            self.open_slots.insert(child_position);
        }

        // Remove spot about to be filled from open positions
        self.open_slots.remove(&position);
        self.slots[position].pick = Pick::Picked(owned_pick);
        Ok(())
    }

    pub fn other_open_slots(&self) -> impl Iterator<Item = Id> + use<'_, L> {
        self.open_slots
            .iter()
            // The next_to_fill position is not included if it exists
            .filter(|p| self.next_to_fill != Some(**p))
            .map(|p| self.slots[*p].eclass_id())
    }

    pub fn n_chosen(&self) -> usize {
        self.slots
            .iter()
            .filter(|c| matches!(c.pick, Pick::Picked(_)))
            .count()
    }

    pub fn n_open(&self) -> usize {
        self.slots
            .iter()
            .filter(|c| matches!(c.pick, Pick::Open))
            .count()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

impl<L: Language> From<Id> for PartialRecExpr<L> {
    fn from(id: Id) -> Self {
        PartialRecExpr {
            slots: vec![Slot {
                eclass_id: id,
                parent: None,
                pick: Pick::Open,
            }],
            open_slots: BTreeSet::from([0]),
            next_to_fill: Some(0),
        }
    }
}

impl<L: Language> IntoIterator for PartialRecExpr<L> {
    type Item = Slot<L>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.slots.into_iter()
    }
}

impl<L: Language> TryFrom<PartialRecExpr<L>> for RecExpr<L> {
    type Error = super::SampleError;

    fn try_from(partial_rec_expr: PartialRecExpr<L>) -> Result<Self, Self::Error> {
        // let mut expr = RecExpr::default();
        let len_1 = partial_rec_expr.slots.len() - 1;

        // Reversing so we get a sensible insertion order for RecExpr
        let rec_expr = partial_rec_expr
            .slots
            .into_iter()
            .rev()
            .map(|choice| {
                let mut pick = choice.pick().ok_or(super::SampleError::UnfinishedChoice)?;
                for id in pick.children_mut() {
                    *id = (len_1 - usize::from(*id)).into();
                }
                Ok(pick)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into();
        Ok(rec_expr)
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Slot<L: Language> {
    eclass_id: Id,
    parent: Option<usize>,
    pick: Pick<L>,
}

impl<L: Language> Slot<L> {
    pub fn eclass_id(&self) -> Id {
        self.eclass_id
    }

    fn pick(self) -> Option<L> {
        match self.pick {
            Pick::Open => None,
            Pick::Picked(pick) => Some(pick),
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pick<L: Language> {
    Open,
    Picked(L),
}
