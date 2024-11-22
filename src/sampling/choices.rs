use std::fmt::Debug;
use std::iter::IntoIterator;

use egg::{Id, Language, RecExpr};
use hashbrown::HashSet;
use rand::{seq::IteratorRandom, Rng};
use serde::Serialize;

use super::SampleError;

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ChoiceList<L: Language> {
    choices: Vec<Choice<L>>,
    open_positions: HashSet<usize>,
    next_to_fill: Option<usize>,
}

impl<L: Language> ChoiceList<L> {
    // Returns position of next open
    pub fn next_open<R: Rng>(&mut self, rng: &mut R) -> Option<Id> {
        if let Some(next_position) = self.open_positions.iter().choose(rng) {
            self.next_to_fill = Some(*next_position);
            // dbg!(&self.choices[*next_position]);
            let id = self.choices[*next_position].eclass_id().unwrap();
            Some(id)
        } else {
            self.next_to_fill = None;
            None
        }
    }

    pub fn fill_next(&mut self, pick: &L) -> Result<(), SampleError> {
        // Check if there is an open position to be filled
        let position = self.next_to_fill.ok_or(SampleError::ChoiceError)?;
        assert!(matches!(self.choices[position], Choice::Open(_)));

        let mut owned_pick = pick.clone();

        // Create new open choices for children with the eclass ids of the children
        let new_open_choices = owned_pick
            .children()
            .iter()
            .map(|child_id| Choice::Open(*child_id));

        let old_len = self.choices.len();
        self.choices.extend(new_open_choices);

        // Calculate and add new positions to open positions hashmap
        for (i, child) in owned_pick.children_mut().iter_mut().enumerate() {
            let child_position = old_len + i;
            *child = Id::from(child_position);
            self.open_positions.insert(child_position);
        }

        // Remove spot about to be filled from open positions
        self.open_positions.remove(&position);
        self.choices[position] = Choice::Picked(owned_pick);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }
}

impl<L: Language> From<Id> for ChoiceList<L> {
    fn from(id: Id) -> Self {
        ChoiceList {
            choices: vec![Choice::Open(id)],
            open_positions: HashSet::from([0]),
            next_to_fill: Some(0),
        }
    }
}

impl<L: Language> IntoIterator for ChoiceList<L> {
    type Item = Choice<L>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.choices.into_iter()
    }
}

impl<L: Language> TryFrom<ChoiceList<L>> for RecExpr<L> {
    type Error = super::SampleError;

    fn try_from(choice_list: ChoiceList<L>) -> Result<Self, Self::Error> {
        // let mut expr = RecExpr::default();
        let len_1 = choice_list.choices.len() - 1;

        // Reversing so we get a sensible insertion order for RecExpr
        let rec_expr = choice_list
            .choices
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
pub enum Choice<L: Language> {
    Open(Id),
    Picked(L),
}

impl<L: Language> Choice<L> {
    pub fn eclass_id(&self) -> Option<Id> {
        match self {
            Choice::Open(eclass_id) => Some(*eclass_id),
            Choice::Picked(_) => None,
        }
    }

    fn pick(self) -> Option<L> {
        match self {
            Choice::Open(_) => None,
            Choice::Picked(pick) => Some(pick),
        }
    }
}
