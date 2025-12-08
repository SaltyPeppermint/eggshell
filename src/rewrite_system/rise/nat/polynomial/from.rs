use egg::RecExpr;
use num::rational::Ratio;
use num_traits::One;

use super::{Monomial, Polynomial, Rise};
use crate::rewrite_system::rise::Index;

// ============================================================================
// RecExpr Conversions
// ============================================================================

impl From<&Polynomial> for RecExpr<Rise> {
    fn from(p: &Polynomial) -> Self {
        if p.is_zero() {
            let mut expr = RecExpr::default();
            expr.add(Rise::Integer(0));
            return expr;
        }

        p.sorted_monomials()
            .iter()
            .fold(RecExpr::default(), |mut expr, (monomial, coeff)| {
                let prev_id = expr.ids().last();

                let term_id = if monomial.is_constant() {
                    // Handle constant
                    expr.add(Rise::Integer(*coeff.numer()))
                } else {
                    // Get the monomial expression
                    let monomial_id = monomial.append_to_expr(&mut expr);
                    if coeff.is_one() {
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
                    }
                };

                // Add to running sum
                match prev_id {
                    None => term_id,
                    Some(id) => expr.add(Rise::NatAdd([id, term_id])),
                };
                expr
            })
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
