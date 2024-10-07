mod term_weighted;
mod uniform;

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::{rngs::StdRng, seq::IteratorRandom};

use super::{choices::ChoiceList, SampleConf};

pub use uniform::Uniform;

pub trait Strategy<'a, L, N>
where
    L: Language + 'a,
    N: Analysis<L> + 'a,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>) -> &'c L;

    fn start_new(&mut self);

    fn egraph(&self) -> &'a EGraph<L, N>;

    fn rng_mut(&mut self) -> &mut StdRng;

    fn sample_term(&mut self, root_eclass: &EClass<L, N::Data>) -> RecExpr<L> {
        let egraph = self.egraph();
        let choice_list = ChoiceList::from(root_eclass.id);
        let mut choices: ChoiceList<L> = choice_list;
        // let mut visited = HashSet::from([root_eclass.id]);

        while let Some(next_open_id) = choices.next_open() {
            let eclass_id = egraph.find(next_open_id);
            let eclass = &egraph[eclass_id];
            let pick = self.pick(eclass);
            choices.fill_next(pick);
        }
        self.start_new();
        choices.try_into().expect("No open choices should be left")
    }

    fn sample_root(&mut self, conf: &SampleConf, root: Id) -> HashSet<RecExpr<L>> {
        let root_eclass = &self.egraph()[root];
        (0..conf.samples_per_eclass)
            .map(|_| self.sample_term(root_eclass))
            .collect()
    }

    fn sample(&mut self, conf: &SampleConf) -> HashMap<Id, HashSet<RecExpr<L>>> {
        self.egraph()
            .classes()
            .choose_multiple(self.rng_mut(), conf.samples_per_egraph)
            .into_iter()
            .map(|eclass| {
                let exprs = (0..conf.samples_per_eclass)
                    .map(|_| self.sample_term(eclass))
                    .collect();
                (eclass.id, exprs)
            })
            .collect()
    }
}
