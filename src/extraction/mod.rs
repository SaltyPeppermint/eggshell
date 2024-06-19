pub mod bottom_up;

use std::fmt::Display;

use egg::{Analysis, EGraph, Language, RecExpr};
use serde::Serialize;

use crate::cost_fn::{Cost, CostFn};
use crate::eqsat::ClassId;
pub use bottom_up::BottomUp;

pub trait Extractor<'a, C, L, N>
where
    C: CostFn,
    L: Language + Display + Sync,
    N: Analysis<L> + Sync,
{
    fn new(cost_fn: C, egraph: &'a EGraph<L, N>) -> Self;

    fn extract(self, roots: &[ClassId]) -> Vec<ExtractResult<L>>;
}

#[derive(Clone, Debug, Serialize)]
pub struct ExtractResult<L>
where
    L: Language + Display,
{
    pub expr: RecExpr<L>,
    pub cost: f64,
}

impl<L> From<(RecExpr<L>, Cost)> for ExtractResult<L>
where
    L: Language + Display,
{
    fn from(value: (RecExpr<L>, Cost)) -> Self {
        ExtractResult {
            expr: value.0,
            cost: value.1.into(),
        }
    }
}

impl<L> ExtractResult<L>
where
    L: Language + Display,
{
    #[must_use]
    pub fn new(expr: RecExpr<L>, cost: f64) -> Self {
        Self { expr, cost }
    }

    #[must_use]
    pub fn expr_as_strign(&self) -> String {
        self.expr.to_string()
    }

    #[must_use]
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;
    use more_asserts as ma;

    use super::*;

    use crate::cost_fn::{ExprSize, LookupCost, StringSize};
    use crate::eqsat::Eqsat;
    use crate::trs::halide::{Halide, Math};

    fn count_nodes<L: Language>(expr: &RecExpr<L>) -> usize {
        expr.as_ref().len()
    }

    /// Passes if we are better or equal compared to the original Caviar paper
    /// That difference may arise if we give more ressources
    #[test]
    fn simplify_ast_size() {
        let start_stmt: RecExpr<Math> = "( == ( * ( / ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) 4 ) 4 ) ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) )".parse().unwrap();
        let baseline_simplified: RecExpr<Math> = "(== 0 (% (max (+ -1388 (* 128 v2)) (+ (* 8 (max -45 (min 204 v1))) (* (max 124 (min 373 v1)) -8))) 4))".parse().unwrap();

        let mut eqsat: Eqsat<Halide> = Eqsat::new(0).unwrap();
        let _ = eqsat.run_simplify_once(&start_stmt);
        let extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
        let r = extractor.extract(eqsat.last_roots().unwrap());
        ma::assert_le!(count_nodes(&r[0].expr), count_nodes(&baseline_simplified));
    }

    /// Passes if we are better or equal compared to the original Caviar paper
    /// That difference may arise if we give more ressources
    #[test]
    fn simplify_str_size() {
        let start_stmt: RecExpr<Math> = "( == ( * ( / ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) 4 ) 4 ) ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) )".parse().unwrap();
        let baseline_simplified: RecExpr<Math> = "(== 0 (% (max (+ -1388 (* 128 v2)) (+ (* 8 (max -45 (min 204 v1))) (* (max 124 (min 373 v1)) -8))) 4))".parse().unwrap();

        let mut eqsat: Eqsat<Halide> = Eqsat::new(0).unwrap();
        let _ = eqsat.run_simplify_once(&start_stmt);
        let extractor = BottomUp::new(StringSize, eqsat.last_egraph().unwrap());
        let r = extractor.extract(eqsat.last_roots().unwrap());
        ma::assert_le!(
            r[0].expr.to_string().len(),
            baseline_simplified.to_string().len()
        );
    }

    /// We are using a dummy cost table where every nodes cost is 2 so the cost should be
    /// strictly double that of a `ExprSize` extraction times 2.
    #[test]

    fn simplify_simulate_nn_extraction() {
        // We will construct a dumb lookup table where each node costs x.
        // Basically an inflated Astsize
        const NODE_LOOKUP_COST: f64 = 4.0;

        let start_stmt: RecExpr<Math> = "( == ( * ( / ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) 4 ) 4 ) ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) )".parse().unwrap();

        let mut eqsat: Eqsat<Halide> = Eqsat::new(0).unwrap();
        let flat_graph = eqsat.run_simplify_once(&start_stmt);

        let dummy_map = (0..flat_graph.vertices.len())
            .map(|i| (i, NODE_LOOKUP_COST))
            .collect();
        let dummy_map = flat_graph.remap_costs(&dummy_map);
        let lookup_extractor =
            BottomUp::new(LookupCost::new(dummy_map), eqsat.last_egraph().unwrap());
        let r_lookup = lookup_extractor.extract(eqsat.last_roots().unwrap());
        let lookup_cost = r_lookup[0].cost;

        let astsize_extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
        let r_astsize = astsize_extractor.extract(eqsat.last_roots().unwrap());
        let adjusted_astsize_cost = NODE_LOOKUP_COST * r_astsize[0].cost;

        assert!((adjusted_astsize_cost - lookup_cost).abs() < f64::EPSILON);
    }

    /// Check if we are unstable in the ast size extracted
    #[test]
    fn simplify_ast_size_value() {
        let start_stmt: RecExpr<Math> = "( == ( * ( / ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) 4 ) 4 ) ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) )".parse().unwrap();

        let mut eqsat: Eqsat<Halide> = Eqsat::new(0).unwrap();
        let _ = eqsat.run_simplify_once(&start_stmt);
        let extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
        let r = extractor.extract(eqsat.last_roots().unwrap());
        assert!((r[0].cost - 25.0).abs() < f64::EPSILON);
    }

    /// Check if we are unstable in the ast size extracted
    #[test]
    fn simplify_str_size_value() {
        let start_stmt: RecExpr<Math> = "( == ( * ( / ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) 4 ) 4 ) ( * ( min ( + ( * v2 128 ) ( * ( max ( min v1 373 ) 124 ) 8 ) ) ( + ( * ( min ( + ( * v2 64 ) ( * ( max ( min v1 373 ) 124 ) 4 ) ) ( + ( * ( max ( min v1 204 ) -45 ) 4 ) 693 ) ) 2 ) 2 ) ) -1 ) )".parse().unwrap();

        let mut eqsat: Eqsat<Halide> = Eqsat::new(0).unwrap();
        let _ = eqsat.run_simplify_once(&start_stmt);
        let extractor = BottomUp::new(StringSize, eqsat.last_egraph().unwrap());
        let r = extractor.extract(eqsat.last_roots().unwrap());
        assert!((r[0].cost - 51.0).abs() < f64::EPSILON);
    }
}
