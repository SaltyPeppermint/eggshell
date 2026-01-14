use std::cmp::{Reverse, max};
use std::collections::BinaryHeap;

use egg::{EGraph, Id, Language, PatternAst, RecExpr, Runner, SearchMatches, Searcher};
use hashbrown::HashMap;

use super::{Rise, RiseAnalysis};

#[derive(Clone)]
pub struct RisePredicate<S: Searcher<Rise, RiseAnalysis>> {
    max_array_dim: usize,
    max_ast_size: usize,
    searcher: S,
}

impl<S: Searcher<Rise, RiseAnalysis>> Searcher<Rise, RiseAnalysis> for RisePredicate<S> {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, Rise>> {
        if self.check_limits(egraph, eclass) {
            return self
                .searcher
                .search_eclass_with_limit(egraph, eclass, limit);
        }
        println!("PREDICATED SAID NAH\n");
        None
    }

    fn vars(&self) -> Vec<egg::Var> {
        self.searcher.vars()
    }

    fn get_pattern_ast(&self) -> Option<&PatternAst<Rise>> {
        self.searcher.get_pattern_ast()
    }
}

impl<S: Searcher<Rise, RiseAnalysis>> RisePredicate<S> {
    #[must_use]
    pub fn new(searcher: S) -> Self {
        Self {
            max_array_dim: 8,
            max_ast_size: 1024,
            searcher,
        }
    }

    #[must_use]
    pub fn with_max_array_dim(mut self, max_array_dim: usize) -> Self {
        self.max_array_dim = max_array_dim;
        self
    }

    #[must_use]
    pub fn with_max_ast_size(mut self, max_ast_size: usize) -> Self {
        self.max_ast_size = max_ast_size;
        self
    }

    fn check_limits(&self, egraph: &EGraph<Rise, RiseAnalysis>, id: Id) -> bool {
        let upstream_sizes = egraph.analysis.min_upstream_size().unwrap();
        check_array_dim(egraph, id, self.max_array_dim)
            && check_ast_size(egraph, id, upstream_sizes, self.max_ast_size)
    }
}

fn check_array_dim(egraph: &EGraph<Rise, RiseAnalysis>, id: Id, limit: usize) -> bool {
    fn rec(ty_expr: &RecExpr<Rise>, id: Id) -> usize {
        match ty_expr[id] {
            Rise::NatFun(c_id)
            | Rise::DataFun(c_id)
            | Rise::AddrFun(c_id)
            | Rise::NatNatFun(c_id) => rec(ty_expr, c_id),
            Rise::FunType([in_ty, out_ty]) => {
                max(rec(ty_expr, in_ty), rec(ty_expr, out_ty))
            }
            Rise::ArrType([_, c_id]) => 1 + rec(ty_expr, c_id), // Only thing we really count
            Rise::PairType([fst_id, snd_id]) => {
                max(rec(ty_expr, fst_id), rec(ty_expr, snd_id))
            }
            // Ignore other data types
            Rise::IndexType(_)
            | Rise::VecType(_)
            | Rise::NatType
            | Rise::F32
            // Ignore natexpr
            | Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_)
            | Rise::Integer(_) => 0,
            _ => panic!("This should not be found in a ty_expr {}", ty_expr[id]),
        }
    }
    let Some(repr) = egraph[id].data.small_repr(egraph) else {
        return true;
    };
    let Rise::TypeOf([_, ty_id]) = repr[repr.root()] else {
        return true;
    };
    let array_dim = rec(&repr, ty_id);
    println!("Array dim is {array_dim}");
    array_dim < limit
}

/// Step 1: Compute the upstream sizes for the whole graph (or reachable parts).
pub fn compute_upstream_sizes(
    egraph: &EGraph<Rise, RiseAnalysis>,
    roots: &[Id],
) -> HashMap<Id, usize> {
    let mut min_upstream_size = HashMap::new();
    let mut todo = BinaryHeap::new();

    // Initialize roots
    for &root in roots {
        let canon_root = egraph.find(root);
        min_upstream_size.insert(canon_root, 0);
        todo.push(Reverse((0, canon_root)));
    }

    while let Some(Reverse((max_up_size, id))) = todo.pop() {
        // Lazily skip if we found a better path previously
        if let Some(&best) = min_upstream_size.get(&id)
            && max_up_size > best
        {
            continue;
        }

        for node in &egraph[id].nodes {
            // Get downstream sizes (using the existing analysis)
            let Some(children_max_down) = node
                .children()
                .iter()
                .map(|c_id| egraph[*c_id].data.optimal.as_ref().map(|o| o.size()))
                .collect::<Option<Vec<_>>>()
            else {
                // Check if any child was essentially infinite (unreachable bottom)
                continue;
            };

            let node_max_down = 1 + children_max_down.iter().sum::<usize>();

            // Propagate upstream cost to children
            for (i, &child) in node.children().iter().enumerate() {
                let child_id = egraph.find(child);
                let child_max_down = children_max_down[i];

                // Logic: Upstream + (Total Node Size - This Child Size)
                let child_max_up = max_up_size + (node_max_down - child_max_down);

                let is_better = min_upstream_size
                    .get(&child_id)
                    .is_none_or(|&old| child_max_up < old);

                if is_better {
                    min_upstream_size.insert(child_id, child_max_up);
                    todo.push(Reverse((child_max_up, child_id)));
                }
            }
        }
    }
    min_upstream_size
}

/// Hook to compute the upstream size needed for the predicates
///
/// # Errors
///
/// Never Errors.
pub fn compute_upstream_sizes_hook(runner: &mut Runner<Rise, RiseAnalysis>) -> Result<(), String> {
    let min_upstream_size = compute_upstream_sizes(&runner.egraph, &runner.roots);
    runner
        .egraph
        .analysis
        .set_min_upstream_size(Some(min_upstream_size));
    Ok(())
}

/// Step 2: The check function.
pub fn check_ast_size(
    egraph: &EGraph<Rise, RiseAnalysis>,
    id: Id,
    upstream_sizes: &HashMap<Id, usize>,
    limit: usize,
) -> bool {
    let canon_id = egraph.find(id);

    // If id is not in the map, it means it wasn't reachable from roots,
    // effectively making its upstream size infinite.
    let Some(mus) = upstream_sizes.get(&canon_id) else {
        return false;
    };

    let Some(mds) = egraph[canon_id].data.optimal.as_ref() else {
        return false;
    };

    let min_ast_size = mus + mds.size();
    println!("Ast size is {min_ast_size}");
    min_ast_size < limit
}
