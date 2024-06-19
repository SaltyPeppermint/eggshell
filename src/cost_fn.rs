use std::fmt::{Debug, Display};

use egg::{Analysis, EGraph, Language};
use hashbrown::HashMap as HashBrownMap;
use ordered_float::NotNan;

use crate::eqsat::ClassId;
use crate::utils::MapGet;

pub type Cost = NotNan<f64>;

pub const EPSILON_ALLOWANCE: f64 = 0.00001;

pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(f64::INFINITY) };

pub trait CostFn {
    fn cost<L, N>(&self, eclass: &EGraph<L, N>, class_id: ClassId, node: &L) -> Cost
    where
        L: Language + Display + Debug,
        N: Analysis<L>;

    fn node_sum_cost<'a, M, N, L>(
        &self,
        egraph: &EGraph<L, N>,
        class_id: ClassId,
        node: &L,
        choices: &'a M,
    ) -> Cost
    where
        L: Language + Display + 'a,
        M: MapGet<ClassId, (&'a L, Cost)>,
        N: Analysis<L>,
    {
        self.cost(egraph, class_id, node)
            + node
                .children()
                .iter()
                .map(|class_id| {
                    let class_id = egraph.find(*class_id);
                    match choices.get(&class_id) {
                        Some((_, cost)) => cost,
                        None => &INFINITY,
                    }
                })
                .sum::<Cost>()
    }
}

/// A [`CostFn`] that gets provided with a lookup table for each node
/// The index of the cost in the vector corresponds ot the index of the endoe in the
/// class
#[derive(Debug)]
pub struct LookupCost {
    cost_map: HashBrownMap<ClassId, Vec<f64>>,
}

impl LookupCost {
    #[must_use]
    pub fn new(cost_map: HashBrownMap<ClassId, Vec<f64>>) -> Self {
        Self { cost_map }
    }
}

impl CostFn for LookupCost {
    fn cost<L, N>(&self, egraph: &EGraph<L, N>, class_id: ClassId, node: &L) -> Cost
    where
        L: Language + Display + Debug,
        N: Analysis<L>,
    {
        let canonical_id = egraph.find(class_id);

        let costs = &self.cost_map[&canonical_id];
        let eclass = &egraph[canonical_id];
        // let ad_hoc_analysis = Analysis::make(egraph, node);
        // dbg!(node);
        // dbg!(ad_hoc_analysis);
        // dbg!(&eclass.data);
        // dbg!(node.children());

        // // for n in eclass.nodes {
        // //     if let
        // // }

        // // dbg!(&eclass.id);
        // dbg!(&eclass.nodes);

        // this is an ugly handling of the constant analysis
        // It doesnt change anythinig for all other cases
        // (If the costs vector only has one entry, the eclass has only
        // one entry so it is by default the first entry in the costs
        // vector.)
        // if costs.len() == 1 {
        //     dbg!("Returning from the costs_len = 1");
        //     return unsafe { NotNan::new_unchecked(costs[0]) };
        // }

        let node_position = eclass.nodes.iter().position(|x| cmp_node(x, node)).unwrap();

        dbg!(&eclass.nodes[node_position]);
        dbg!(node_position);
        let node_cost = costs[node_position];
        unsafe { NotNan::new_unchecked(node_cost) }
    }
}

fn cmp_node<L>(lhs: &L, rhs: &L) -> bool
where
    L: Language,
{
    // let children_l_test = lhs.children().iter().map(|id| egraph.find(*id));
    // let children_r_test = rhs.children().iter().map(|id| egraph.find(*id));

    // Seems like the children were already canonicalized
    let children_l = lhs.children(); //;.iter().map(|id| egraph.find(*id));
    let children_r = rhs.children(); //.iter().map(|id| egraph.find(*id));

    let children_match = children_l.eq(children_r);

    // if children_match && children_l_test.eq(children_r_test) {
    //     println!("OOOOOPS WE HAVE TO CHECK THE CHILDREN");
    // }

    lhs.matches(rhs) && children_match
}

///  A simple dummy [`CostFn`] that just returns 1.0 as the cost for every node.
#[derive(Debug)]
pub struct ExprSize;
impl CostFn for ExprSize {
    fn cost<L, N>(&self, _: &EGraph<L, N>, _: ClassId, _: &L) -> Cost
    where
        L: Language + Display + Debug,
        N: Analysis<L>,
    {
        NotNan::from(1)
    }
}

///  A simple very dumb [`CostFn`] that returns the printable lenght of a symbol.
#[derive(Debug)]
pub struct StringSize;
impl CostFn for StringSize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn cost<L, N>(&self, _: &EGraph<L, N>, _: ClassId, node: &L) -> Cost
    where
        L: Language + Display + Debug,
        N: Analysis<L>,
    {
        let string_len = node.to_string().len();
        NotNan::from(string_len as i32)
    }
}
