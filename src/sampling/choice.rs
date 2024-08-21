use std::fmt::Debug;
use std::iter::IntoIterator;

use egg::{Id, Language, RecExpr};
use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChoiceList<L: Language> {
    choices: Vec<Choice<L>>,
    open_idx: usize,
}

impl<L: Language> ChoiceList<L> {
    pub fn new(choices: Vec<Choice<L>>) -> Self {
        Self {
            choices,
            open_idx: 0,
        }
    }

    pub fn next_open(&mut self) -> Option<Id> {
        if self.choices.len() <= self.open_idx {
            None
        } else {
            Some(self.choices[self.open_idx].eclass_id())
        }
    }

    pub fn fill_next(&mut self, pick: &L) {
        let mut owned_pick = pick.clone();

        let new_open_choices = owned_pick
            .children()
            .iter()
            .map(|child_id| Choice::Open(*child_id));

        let old_len = self.choices.len();

        self.choices.extend(new_open_choices);

        for (i, child) in owned_pick.children_mut().iter_mut().enumerate() {
            *child = Id::from(old_len + i);
        }

        self.choices[self.open_idx] = Choice::Picked {
            eclass_id: Id::from(self.open_idx),
            pick: owned_pick,
        };

        self.open_idx += 1;
    }
}

impl<L: Language> From<Id> for ChoiceList<L> {
    fn from(id: Id) -> Self {
        ChoiceList {
            choices: vec![Choice::Open(id)],
            open_idx: 0,
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
        Ok(choice_list
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
            .into())
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Choice<L: Language> {
    Open(Id),
    Picked { eclass_id: Id, pick: L },
}

impl<L: Language> Choice<L> {
    pub fn eclass_id(&self) -> Id {
        match self {
            Choice::Picked { eclass_id, .. } | Choice::Open(eclass_id) => *eclass_id,
        }
    }

    fn pick(self) -> Option<L> {
        match self {
            Choice::Open(_) => None,
            Choice::Picked { pick, .. } => Some(pick),
        }
    }
}
