use egg::RecExpr;
use num::rational::Ratio;
use num_traits::{One, Zero};

use super::{Monomial, Polynomial, Rise};
use crate::rewrite_system::rise::Index;

// ============================================================================
// RecExpr Conversions
// ============================================================================

impl From<&Polynomial> for RecExpr<Rise> {
    fn from(p: &Polynomial) -> Self {
        let mut expr = RecExpr::default();

        if p.is_zero() {
            expr.add(Rise::Integer(0));
            return expr;
        }

        let mut result_id = None;

        for (monomial, coeff) in p.sorted_monomials() {
            if coeff.is_zero() {
                continue;
            }

            // Get the monomial expression
            let monomial_id = monomial.append_to_expr(&mut expr);

            // Handle rational coefficients
            let term_id = if coeff.is_one() {
                monomial_id
            } else if coeff.is_integer() {
                // Integer coefficient
                let coeff_id = expr.add(Rise::Integer(*coeff.numer()));
                expr.add(Rise::NatMul([coeff_id, monomial_id]))
            } else {
                // Rational coefficient: represent as (numer/denom) * monomial
                let numer_id = expr.add(Rise::Integer(*coeff.numer()));
                let denom_id = expr.add(Rise::Integer(*coeff.denom()));
                let frac_id = expr.add(Rise::NatDiv([numer_id, denom_id]));
                expr.add(Rise::NatMul([frac_id, monomial_id]))
            };

            // Add to running sum
            result_id = Some(match result_id {
                None => term_id,
                Some(prev_id) => expr.add(Rise::NatAdd([prev_id, term_id])),
            });
        }

        // result_id.unwrap();
        expr
    }
}

impl From<Polynomial> for RecExpr<Rise> {
    fn from(p: Polynomial) -> Self {
        (&p).into()
    }
}

// ============================================================================
// From Simple Types
// ============================================================================

/// Create a `Polynomial` from an integer constant
impl From<i32> for Polynomial {
    fn from(n: i32) -> Self {
        Self::new().add_term(n.into(), Monomial::new())
    }
}
/// Create a `Polynomial` from an integer constant
impl From<Ratio<i32>> for Polynomial {
    fn from(r: Ratio<i32>) -> Self {
        Self::new().add_term(r, Monomial::new())
    }
}

/// Create a `Polynomial` from with a single variable
impl From<Index> for Polynomial {
    fn from(index: Index) -> Self {
        Self::new().add_term(Ratio::one(), Monomial::new().with_var(index, 1))
    }
}
