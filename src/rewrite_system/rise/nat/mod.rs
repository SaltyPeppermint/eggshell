mod applier;
mod monomial;
mod polynomial;
mod rational;

use std::num::TryFromIntError;

use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, RecExpr, Subst};
use thiserror::Error;

use super::{Rise, RiseAnalysis};
use monomial::Monomial;
use polynomial::Polynomial;

// Todo Fixme
#[expect(unused_imports)]
pub use applier::{ComputeNat, ComputeNatCheck};
pub use rational::RationalFunction;

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
    // println!();
    // println!("{lhs}");
    // println!("{rhs}");

    let rf_lhs: RationalFunction = lhs.try_into().unwrap();
    let rf_rhs: RationalFunction = rhs.try_into().unwrap();

    // println!("{rf_lhs}");
    // println!("{rf_rhs}");
    // println!("{}", rf_lhs == rf_rhs);

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
) -> RecExpr<Rise> {
    fn rec(
        ast: &PatternAst<Rise>,
        id: Id,
        subst: &Subst,
        egraph: &EGraph<Rise, RiseAnalysis>,
    ) -> RecExpr<Rise> {
        match &ast[id] {
            ENodeOrVar::Var(w) => egraph[subst[*w]].data.beta_extract.clone(),
            ENodeOrVar::ENode(e) => {
                let new_e = e.clone();
                new_e.join_recexprs(|i| rec(ast, i, subst, egraph))
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
