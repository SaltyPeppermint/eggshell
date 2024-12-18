use std::collections::BTreeSet;

use egg::{Id, Language, RecExpr};
use hashbrown::HashMap;
use rand::seq::IteratorRandom;
use rand::Rng;
use serde::Serialize;

use super::SampleError;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ChoiceList<L: Language + Send + Sync> {
    slots: Vec<Slot<L>>,
    // We need something other than hashbrown or std HashSet cause
    // they DO NOT produce stable iterators between program runs
    open_slots: BTreeSet<usize>,
    next_to_fill: Option<usize>,
}

impl<L: Language + Send + Sync> ChoiceList<L> {
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
        let position = self.next_to_fill.ok_or(SampleError::ChoiceError)?;
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

    pub fn check_ancestors<'a>(
        &'a self,
        slot: &'a Slot<L>,
        taboo: &mut HashMap<(Id, &'a L), usize>,
        limit: usize,
    ) -> bool {
        if let Some(slot_l) = slot.pick_ref() {
            if let Some(count) = taboo.get_mut(&(slot.eclass_id, slot_l)) {
                *count += 1;
                if *count > limit {
                    return false;
                }
            }
        }
        if let Some(parent_position) = slot.parent {
            self.check_ancestors(&self.slots[parent_position], taboo, limit)
        } else {
            true
        }
    }
    pub fn check_single_in_ancestors<'a>(
        &'a self,
        slot: &'a Slot<L>,
        taboo_eclass: Id,
        taboo_node: &'a L,
        mut count: usize,
        limit: usize,
    ) -> bool {
        if let Some(slot_l) = slot.pick_ref() {
            if slot_l == taboo_node && slot.eclass_id == taboo_eclass {
                count += 1;
                if count > limit {
                    return false;
                }
            }
        }
        if let Some(parent_position) = slot.parent {
            self.check_single_in_ancestors(
                &self.slots[parent_position],
                taboo_eclass,
                taboo_node,
                count,
                limit,
            )
        } else {
            true
        }
    }

    pub fn other_open_slots(&self) -> impl Iterator<Item = Id> + use<'_, L> {
        self.open_slots
            .iter()
            // The next_to_fill position is not included if it exists
            .filter(|p| self.next_to_fill.map_or(true, |x| x != **p))
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

impl<L: Language + Send + Sync> From<Id> for ChoiceList<L> {
    fn from(id: Id) -> Self {
        ChoiceList {
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

impl<L: Language + Send + Sync> IntoIterator for ChoiceList<L> {
    type Item = Slot<L>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.slots.into_iter()
    }
}

impl<L: Language + Send + Sync> TryFrom<ChoiceList<L>> for RecExpr<L> {
    type Error = super::SampleError;

    fn try_from(choice_list: ChoiceList<L>) -> Result<Self, Self::Error> {
        // let mut expr = RecExpr::default();
        let len_1 = choice_list.slots.len() - 1;

        // Reversing so we get a sensible insertion order for RecExpr
        let rec_expr = choice_list
            .slots
            .into_iter()
            .rev()
            .map(|choice| {
                let mut pick = choice.pick().ok_or(super::SampleError::ChoiceError)?;
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
pub struct Slot<L: Language + Send + Sync> {
    eclass_id: Id,
    parent: Option<usize>,
    pick: Pick<L>,
}

impl<L: Language + Send + Sync> Slot<L> {
    pub fn eclass_id(&self) -> Id {
        self.eclass_id
    }

    fn pick(self) -> Option<L> {
        match self.pick {
            Pick::Open => None,
            Pick::Picked(pick) => Some(pick),
        }
    }

    fn pick_ref(&self) -> Option<&L> {
        match &self.pick {
            Pick::Open => None,
            Pick::Picked(pick) => Some(pick),
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pick<L: Language + Send + Sync> {
    Open,
    Picked(L),
}
