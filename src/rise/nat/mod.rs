mod monomial;
mod polynomial;
mod rational;

use std::num::TryFromIntError;

use egg::RecExpr;
use thiserror::Error;

use super::{Rise, RiseAnalysis};
use monomial::Monomial;
use polynomial::Polynomial;

// Todo Fixme
pub use rational::RationalFunction;

type Ratio = num::rational::Ratio<i64>;

// ============================================================================
// Helper Functions
// ============================================================================

pub fn try_simplify(nat_expr: &RecExpr<Rise>) -> Result<RecExpr<Rise>, NatSolverError> {
    let rf: RationalFunction = nat_expr.try_into()?;
    Ok(rf.simplified()?.into())
}

pub fn check_equivalence<'a, 'b: 'a>(
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
