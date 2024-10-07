use egg::{Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language};
use hashbrown::HashMap;
use rand::{rngs::StdRng, seq::SliceRandom};

use super::Strategy;

pub struct Uniform<'a, 'b, L, N, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Into<usize>,
{
    egraph: &'a EGraph<L, N>,
    extractor: Extractor<'a, CF, L, N>,
    rng: &'b mut StdRng,
    loop_count: usize,
    raw_weights_memo: HashMap<Id, HashMap<&'a L, usize>>,
    loop_limit: usize,
}

impl<'a, 'b, L, N, CF> Strategy<'a, L, N> for Uniform<'a, 'b, L, N, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Into<usize>,
{
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>) -> &'c L {
        let pick = if self.loop_limit > self.loop_count {
            eclass
                .nodes
                .choose(&mut self.rng)
                .expect("Each class contains at least one enode.")
        } else {
            let raw_weights = self
                .raw_weights_memo
                .entry(eclass.id)
                .or_insert_with(|| calc_weights(eclass, &self.extractor));

            let urgency = (self.loop_count - self.loop_limit) as i32;
            // println!("Urgency: {urgency}");
            // println!("{raw_weights:?}");
            let pick = if urgency < 32 {
                eclass
                    .nodes
                    .choose_weighted(&mut self.rng, |node| {
                        (raw_weights[node] as f64).powi(urgency)
                    })
                    .expect("Infallible weight calculation.")
            } else {
                eclass
                    .nodes
                    .iter()
                    .max_by_key(|node| raw_weights[node])
                    .unwrap()
            };
            pick
        };
        self.loop_count += 1;
        pick
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }

    fn rng_mut(&mut self) -> &mut StdRng {
        self.rng
    }

    fn start_new(&mut self) {
        self.loop_count = 0;
    }
}

impl<'a, 'b, L, N, CF> Uniform<'a, 'b, L, N, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Into<usize>,
{
    pub fn new(
        rng: &'b mut StdRng,
        egraph: &'a EGraph<L, N>,
        cost_fn: CF,
        loop_limit: usize,
    ) -> Self {
        Uniform {
            egraph,
            extractor: Extractor::new(egraph, cost_fn),
            rng,
            loop_count: 0,
            loop_limit,
            raw_weights_memo: HashMap::new(),
        }
    }
}

// fn fun_name<'a, L, N, CF, X>(
//     sample_conf: &SampleConf,
//     loop_count: usize,
//     eclass: &'a EClass<L, <N as Analysis<L>>::Data>,
//     rng: &mut StdRng,
//     raw_weights_memo: &'a mut HashMap<Id, HashMap<&'a L, usize>>,
//     extractor: &'a Extractor<'_, CF, L, N>,
// ) -> &'a L
// where
//     L: Language,
//     N: Analysis<L>,
//     CF: CostFunction<L>,
//     CF::Cost: Sum + SampleBorrow<X> + Into<usize>,
//     X: SampleUniform + for<'x> AddAssign<&'x X> + PartialOrd<X> + Clone + Default,
// {
//     let pick = if sample_conf.loop_limit > loop_count {
//         eclass
//             .nodes
//             .choose(rng)
//             .expect("Each class contains at least one enode.")
//     } else {
//         let raw_weights = raw_weights_memo
//             .entry(eclass.id)
//             .or_insert_with(|| calc_weights(eclass, extractor));

//         let urgency = (loop_count - sample_conf.loop_limit) as i32;
//         // println!("Urgency: {urgency}");
//         // println!("{raw_weights:?}");
//         let pick = if urgency < 32 {
//             eclass
//                 .nodes
//                 .choose_weighted(rng, |node| (raw_weights[node] as f64).powi(urgency))
//                 .expect("Infallible weight calculation.")
//         } else {
//             eclass
//                 .nodes
//                 .iter()
//                 .max_by_key(|node| raw_weights[node])
//                 .unwrap()
//         };
//         pick
//     };
//     pick
// }

fn calc_weights<'a, L, N, CF>(
    eclass: &'a EClass<L, N::Data>,
    extractor: &Extractor<CF, L, N>,
) -> HashMap<&'a L, usize>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Into<usize>,
{
    let costs = eclass.nodes.iter().map(|node| {
        node.children()
            .iter()
            .map(|c| extractor.find_best_cost(*c).into())
            .sum()
    });
    let max = costs.clone().max().unwrap_or(0);

    costs
        .zip(&eclass.nodes)
        .map(move |(cost, node)| (node, max - cost + 1))
        .collect()
}
