use std::cmp::Ordering;
use std::fmt::Debug;

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;

pub struct LutCost<'a, L: Language, N: Analysis<L>> {
    table: HashMap<(Id, usize), f64>,
    egraph: &'a EGraph<L, N>,
}

impl<'a, L: Language, N: Analysis<L>> LutCost<'a, L, N> {
    pub fn new(table: HashMap<(Id, usize), f64>, egraph: &'a EGraph<L, N>) -> Self {
        Self { table, egraph }
    }
}

impl<'a, L: Language, N: Analysis<L>> ClassCostFunction<L> for LutCost<'a, L, N> {
    type Cost = f64;

    fn cost<C>(&self, class_id: Id, enode: &L, costs: C) -> Self::Cost
    where
        C: Fn(Id) -> Self::Cost,
    {
        let node_position = self.egraph[class_id]
            .nodes
            .iter()
            .position(|n| n.matches(enode))
            .expect("Node is in eclass");
        let node_cost = *self
            .table
            .get(&(class_id, node_position))
            .expect("Each node item must have a cost in the LUT");
        enode.fold(node_cost, |sum, id| sum + costs(id))
    }
}

pub trait ClassCostFunction<L: Language> {
    /// The `Cost` type. It only requires `PartialOrd` so you can use
    /// floating point types, but failed comparisons (`NaN`s) will
    /// result in a panic.
    type Cost: PartialOrd + Debug + Clone;

    /// Calculates the cost of an enode whose children are `Cost`s.
    ///
    /// For this to work properly, your cost function should be
    /// _monotonic_, i.e. `cost` should return a `Cost` greater than
    /// any of the child costs of the given enode.
    fn cost<C>(&self, class_id: Id, enode: &L, costs: C) -> Self::Cost
    where
        C: Fn(Id) -> Self::Cost;
}

#[derive(Debug)]
pub struct ClassExtractor<'a, CF: ClassCostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    costs: HashMap<Id, (CF::Cost, L)>,
    egraph: &'a EGraph<L, N>,
}

fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    // None is high
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a_i), Some(b_i)) => a_i.partial_cmp(b_i).unwrap(),
    }
}

impl<'a, CF, L, N> ClassExtractor<'a, CF, L, N>
where
    CF: ClassCostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    /// Create a new `ClassExtractor` given an `EGraph` and a
    /// `ClassCostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let costs = HashMap::default();
        let mut extractor = ClassExtractor {
            cost_function,
            costs,
            egraph,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&self, eclass: Id) -> (CF::Cost, RecExpr<L>) {
        let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (cost, expr)
    }

    /// Find the cheapest e-node in the given e-class.
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)].1
    }

    fn node_total_cost(&mut self, eclass_id: Id, node: &L) -> Option<CF::Cost> {
        let eg = &self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        if node.all(has_cost) {
            let costs = &self.costs;
            let cost_f = |id| costs[&eg.find(id)].0.clone();
            Some(self.cost_function.cost(eclass_id, node, cost_f))
        } else {
            None
        }
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                );
            }
        }
    }

    fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> Option<(CF::Cost, L)> {
        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(eclass.id, n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {eclass:#?}"));
        cost.map(|c| (c, node.clone()))
    }
}
