use egg::{Applier, EGraph, Id, Pattern, PatternAst, Subst, Symbol, Var};

use super::{Rise, RiseAnalysis};

// pub struct ComputeNat<A: Applier<Rise, RiseAnalysis>> {
//     var: Var,
//     nat_pattern: Pattern<Rise>,
//     applier: A,
// }

// impl<A: Applier<Rise, RiseAnalysis>> ComputeNat<A> {
//     #[expect(dead_code)]
//     pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
//         ComputeNat {
//             var: var.parse().unwrap(),
//             nat_pattern: nat_pattern.parse().unwrap(),
//             applier,
//         }
//     }
// }

// impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ComputeNat<A> {
//     fn apply_one(
//         &self,
//         egraph: &mut EGraph<Rise, RiseAnalysis>,
//         eclass: Id,
//         subst: &Subst,
//         searcher_ast: Option<&PatternAst<Rise>>,
//         rule_name: Symbol,
//     ) -> Vec<Id> {
//         let Some(nat_matches) = self.nat_pattern.search_eclass(egraph, subst[self.var]) else {
//             return Vec::new();
//         };
//         let Some(expr) = &egraph[subst[self.var]].data.small_repr(egraph) else {
//             return Vec::new();
//         };
//         let simplified_nat = super::try_simplify(expr).unwrap();
//         let mut new_subst = subst.clone();
//         let added_expr_id = egraph.add_expr(&simplified_nat);
//         new_subst.insert(self.var, added_expr_id);

//         self.applier
//             .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
//     }
// }

pub struct ComputeNatCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    nat_pattern: Pattern<Rise>,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ComputeNatCheck<A> {
    pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
        ComputeNatCheck {
            var: var.parse().unwrap(),
            nat_pattern: nat_pattern.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ComputeNatCheck<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(expected) = egraph[subst[self.var]].data.small_repr(egraph) else {
            return Vec::new();
        };
        let Some(extracted) = super::extract_small(egraph, &self.nat_pattern, subst) else {
            return Vec::new();
        };

        let a = &mut egraph.analysis;
        if super::check_equivalence(a, &expected, &extracted) {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}
