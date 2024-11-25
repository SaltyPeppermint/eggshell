mod cost;
mod expr_count;

use std::fmt::{Debug, Display};

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use log::{debug, info, warn};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;

use super::SampleError;
use super::{choices::ChoiceList, SampleConf};

pub use cost::CostWeighted;
pub use expr_count::SizeCountLutWeighted;
pub use expr_count::SizeCountWeighted;

pub trait Strategy<'a, L, N>: Debug
where
    L: Language + Display + Debug + 'a,
    N: Analysis<L> + Debug + 'a,
    N::Data: Debug,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>, current_size: usize) -> &'c L;

    fn egraph(&self) -> &'a EGraph<L, N>;

    fn rng_mut(&mut self) -> &mut StdRng;

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    fn extractable(&self, id: Id) -> Result<(), SampleError>;

    /// .
    ///
    /// # Panics
    ///
    /// Panics if .
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong during sampling.
    fn sample_expr(&mut self, root_eclass: &EClass<L, N::Data>) -> Result<RecExpr<L>, SampleError> {
        let egraph = self.egraph();

        let canonical_root_id = egraph.find(root_eclass.id);
        self.extractable(canonical_root_id)?;

        let mut choices = ChoiceList::from(canonical_root_id);

        while let Some(id) = choices.next_open(self.rng_mut()) {
            let eclass_id = egraph.find(id);
            let eclass = &egraph[eclass_id];
            let pick = self.pick(eclass, choices.len());
            choices.fill_next(pick)?;
            if choices.len() > 10000 && choices.len() % 100 == 0 {
                warn!("Building very large sample with {} entries!", choices.len());
            }
        }
        let expr: RecExpr<L> = choices.try_into().expect("No open choices should be left");
        debug!(
            "Sampled expression of size {}: {} ",
            expr.as_ref().len(),
            expr
        );
        Ok(expr)
    }

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if the precondition for a successful sampling of the eclass are not fullfilled.
    fn sample_eclass(
        &mut self,
        conf: &SampleConf,
        root: Id,
    ) -> Result<HashSet<RecExpr<L>>, SampleError> {
        let root_eclass = &self.egraph()[root];
        let mut samples = HashSet::with_capacity(conf.samples_per_eclass);
        for i in 0..conf.samples_per_eclass {
            let sample = self.sample_expr(root_eclass)?;
            if i % 1000 == 0 {
                info!("Sampled {i} expressions from eclass {}", root_eclass.id);
            }
            samples.insert(sample);
        }
        Ok(samples)
    }

    /// .
    ///
    /// # Panics
    ///
    /// Panics if .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    fn sample(
        &mut self,
        conf: &SampleConf,
    ) -> Result<HashMap<Id, HashSet<RecExpr<L>>>, SampleError> {
        self.egraph()
            .classes()
            .choose_multiple(self.rng_mut(), conf.samples_per_egraph)
            .into_iter()
            .map(|eclass| {
                let mut samples = HashSet::with_capacity(conf.samples_per_eclass);
                for i in 0..conf.samples_per_eclass {
                    let sample = self.sample_expr(eclass)?;
                    if i % 1000 == 0 {
                        info!("Sampled {i} expressions from eclass {}", eclass.id);
                    }
                    samples.insert(sample);
                }
                Ok((eclass.id, samples))
            })
            .collect()
    }
}
