mod applier;
mod monomial;
mod polynomial;
mod rational;

use std::num::TryFromIntError;

use egg::RecExpr;
use num::rational::Ratio;
use num_traits::{Signed, Zero};
use thiserror::Error;

use super::{Rise, RiseAnalysis};
use monomial::Monomial;
use polynomial::Polynomial;

pub use applier::{ComputeNat, ComputeNatCheck};
pub use rational::RationalFunction;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute GCD of two rational numbers
fn gcd_ratio(a: Ratio<i32>, b: Ratio<i32>) -> Ratio<i32> {
    if b.is_zero() {
        return a.abs();
    }
    // For rationals a/b and c/d, gcd = gcd(a*d, c*b) / (b*d)
    // Simplified: we work with the absolute values
    let a_abs = a.abs();
    let b_abs = b.abs();

    // Use Euclidean algorithm on rationals
    let (mut x, mut y) = if a_abs >= b_abs {
        (a_abs, b_abs)
    } else {
        (b_abs, a_abs)
    };

    while !y.is_zero() {
        let remainder = x - (x / y).trunc() * y;
        x = y;
        y = remainder;
    }

    x
}

pub fn try_simplify(nat_expr: &RecExpr<Rise>) -> Result<RecExpr<Rise>, NatSolverError> {
    let polynomial: RationalFunction = nat_expr.try_into()?;
    Ok(polynomial.simplified().into())
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

    let poly_lhs: RationalFunction = lhs.try_into().unwrap();
    let poly_rhs: RationalFunction = rhs.try_into().unwrap();

    if poly_lhs == poly_rhs {
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
