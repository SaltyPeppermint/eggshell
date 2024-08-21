use core::panic;
use std::fmt::Debug;
use std::iter::IntoIterator;

use egg::{Id, Language, RecExpr};
use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Choice<'a, L: Language> {
    Open(Id),
    Picked {
        eclass_id: Id,
        pick: &'a L,
        children: Vec<Choice<'a, L>>,
    },
}

impl<'a, L: Language> Choice<'a, L> {
    pub fn eclass_id(&self) -> Id {
        match self {
            Choice::Picked { eclass_id, .. } | Choice::Open(eclass_id) => *eclass_id,
        }
    }

    fn children(&self) -> &'a [Choice<L>] {
        match self {
            Choice::Open(_) => &[],
            Choice::Picked { children, .. } => children,
        }
    }

    fn collect_children(self, all: &mut Vec<L>) {
        match self {
            Choice::Open(_) => {
                panic!("Calling collect on an unfinished tree makes no sense")
            }
            Choice::Picked { pick, children, .. } => {
                for child in children {
                    child.collect_children(all);
                }
                all.push(pick.clone());
            }
        }
    }

    fn all_choices(&'a self, all: &mut Vec<&'a Choice<'a, L>>) {
        all.push(self);
        for child in self.children() {
            child.all_choices(all);
        }
    }

    pub fn next_open(&mut self) -> Option<&mut Self> {
        match self {
            Choice::Open(_) => Some(self),
            Choice::Picked { children, .. } => children.iter_mut().find_map(|c| c.next_open()),
        }
    }
}

impl<'a, L: Language> IntoIterator for &'a Choice<'a, L> {
    type Item = &'a Choice<'a, L>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut all_children = Vec::new();
        self.all_choices(&mut all_children);
        all_children.into_iter()
    }
}

impl<'a, L: Language> From<Choice<'a, L>> for RecExpr<L> {
    fn from(choices: Choice<'a, L>) -> Self {
        let mut picks = Vec::new();

        choices.collect_children(&mut picks);

        let mut expr = RecExpr::default();

        for (id_counter, mut node) in picks.into_iter().enumerate() {
            {
                for (idx, child_id) in node.children_mut().iter_mut().enumerate() {
                    *child_id = Id::from(id_counter - idx - 1);
                }

                expr.add(node);
            }
        }
        expr
    }
}
