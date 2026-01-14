mod applier;
mod monomial;
mod polynomial;
mod rational;

use std::num::TryFromIntError;

use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, RecExpr, Subst};
use thiserror::Error;

use crate::utils;

use super::{Rise, RiseAnalysis};
use monomial::Monomial;
use polynomial::Polynomial;

// Todo Fixme
pub use applier::ComputeNatCheck; // ComputeNat,
pub use rational::RationalFunction;

type Ratio = num::rational::Ratio<i64>;

// ============================================================================
// Helper Functions
// ============================================================================

pub fn try_simplify(nat_expr: &RecExpr<Rise>) -> Result<RecExpr<Rise>, NatSolverError> {
    let rf: RationalFunction = nat_expr.try_into()?;
    Ok(rf.simplified()?.into())
}

fn check_equivalence<'a, 'b: 'a>(
    cache: &'b mut RiseAnalysis,
    lhs: &RecExpr<Rise>,
    rhs: &RecExpr<Rise>,
) -> bool {
    // check cache
    if cache.check_cache_equiv(lhs, rhs) {
        return true;
    }

    let rf_lhs: RationalFunction = lhs.try_into().unwrap();
    let rf_rhs: RationalFunction = rhs.try_into().unwrap();
    if rf_lhs == rf_rhs {
        cache.add_pair_to_cache(lhs, rhs);
        return true;
    }
    false
}

fn extract_small(
    egraph: &EGraph<Rise, RiseAnalysis>,
    pattern: &Pattern<Rise>,
    subst: &Subst,
) -> Option<RecExpr<Rise>> {
    fn rec(
        ast: &PatternAst<Rise>,
        id: Id,
        subst: &Subst,
        egraph: &EGraph<Rise, RiseAnalysis>,
    ) -> Option<RecExpr<Rise>> {
        match &ast[id] {
            ENodeOrVar::Var(w) => egraph[subst[*w]].data.small_repr(egraph),
            ENodeOrVar::ENode(e) => {
                let mut expr = RecExpr::default();
                let mut new_node = e.clone();
                for c_id in new_node.children_mut() {
                    *c_id = utils::add_expr(&mut expr, rec(ast, *c_id, subst, egraph)?);
                }
                expr.add(new_node);
                Some(expr)
            }
        }
    }
    rec(&pattern.ast, pattern.ast.root(), subst, egraph)
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum NatSolverError {
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Cannot invert multi-term polynomial: {0}")]
    NonMonomialInversion(Polynomial),
    #[error("Result is not a polynomial (has non-trivial denominator): {0}")]
    NotAPolynomial(RationalFunction),
    #[error("Unsupported Rise node type: {0}")]
    UnsupportedNode(Rise),
    #[error("Exponent must be an integer constant, got: {0}")]
    NonIntegerExponent(Rise),
    #[error("Integer conversion failed: {0}")]
    IntConversionFailure(#[from] TryFromIntError),
}
