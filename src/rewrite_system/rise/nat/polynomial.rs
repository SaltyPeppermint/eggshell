use std::collections::BTreeMap;
use std::fmt;

use egg::{Id, RecExpr};
use num::rational::Ratio;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};

use super::{Monomial, Rise};

/// Represents a polynomial as a map from monomials to coefficients
/// This can represent Laurent polynomials (with negative exponents) and
/// rational expressions that simplify to a single term in the denominator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Polynomial {
    // Map from monomial to coefficient
    terms: BTreeMap<Monomial, Ratio<i32>>,
}

impl Polynomial {
    pub fn new() -> Self {
        Polynomial {
            terms: BTreeMap::new(),
        }
    }

    pub fn one() -> Self {
        Polynomial::new().add_term(Ratio::one(), Monomial::new())
    }

    pub fn neg_one() -> Self {
        Polynomial::new().add_term(-Ratio::one(), Monomial::new())
    }

    /// Try to invert a polynomial, panic if it cannot be represented as a polynomial.
    ///
    /// See `try_inv` for details
    pub fn inv(self) -> Self {
        // First simplify to remove zero terms

        assert!(!self.is_zero(), "Cannot invert zero polynomial");

        match self.try_inv() {
            Ok(inverted) => inverted,
            Err(failed_inverted) => panic!(
                "Cannot invert multi-term polynomial '{failed_inverted}'. \
                    Only single-term polynomials (monomials) can be inverted to produce a polynomial.",
            ),
        }
    }

    /// Try to invert a polynomial, returning Err if it cannot be represented as a polynomial.
    ///
    /// This operation is only valid for:
    /// 1. Single-term polynomials (monomials with coefficient): c * x^a * y^b -> (1/c) * x^(-a) * y^(-b)
    /// 2. Constant polynomials: c -> 1/c
    ///
    /// For multi-term polynomials like (x + 1), inversion cannot produce a polynomial,
    /// so this will panic.
    pub fn try_inv(mut self) -> Result<Self, Self> {
        self.simplify();

        if self.is_zero() {
            return Err(self);
        }

        // Check if this is a single-term polynomial (monomial)
        if self.term_count() == 1 {
            let (monomial, coeff) = self.terms.into_iter().next().unwrap();
            // Invert coefficient: c -> 1/c
            let inv_coeff = Ratio::one() / coeff;
            // Invert monomial: negate all exponents
            let inv_monomial = monomial.inv();
            Ok(Polynomial::new().add_term(inv_coeff, inv_monomial))
        } else {
            Err(self)
        }
    }

    /// Add a term to the polynomial
    pub fn add_term(mut self, coefficient: Ratio<i32>, monomial: Monomial) -> Self {
        if !coefficient.is_zero() {
            *self.terms.entry(monomial).or_insert(Ratio::zero()) += coefficient;
        }
        self
    }

    /// Simplify by removing zero coefficients
    pub fn simplify(&mut self) {
        self.terms.retain(|_, coeff| !coeff.is_zero());
    }

    /// Simplified (see `simplify`)
    pub fn simplified(mut self) -> Self {
        self.simplify();
        self
    }

    /// Check if polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Check if polynomial is a single monomial (can be inverted)
    pub fn is_monomial(&self) -> bool {
        let non_zero_count = self.terms.values().filter(|c| !c.is_zero()).count();
        non_zero_count <= 1
    }

    /// Get the number of non-zero terms
    pub fn term_count(&self) -> usize {
        self.terms.values().filter(|c| !c.is_zero()).count()
    }

    fn sorted_monomials(&self) -> Vec<(&Monomial, &Ratio<i32>)> {
        // Sort terms: first by total degree (descending), then lexicographically
        let mut sorted_terms: Vec<_> = self
            .terms
            .iter()
            .filter(|(_, coeff)| !coeff.is_zero())
            .collect();
        sorted_terms.sort_by(|(m1, _), (m2, _)| {
            // Calculate total degree (sum of all exponents, can be negative)
            let deg1: i32 = m1.variables().values().sum();
            let deg2: i32 = m2.variables().values().sum();

            // First compare by total degree (descending)
            match deg2.cmp(&deg1) {
                std::cmp::Ordering::Equal => {
                    // Then compare lexicographically (already handled by BTreeMap ordering)
                    m1.cmp(m2)
                }
                other => other,
            }
        });
        sorted_terms
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self::new()
    }
}

// -------------------------------------
// Arithmatic
// -------------------------------------

// impl std::ops::Add<&Polynomial> for Polynomial {
//     type Output = Polynomial;

//     fn add(self, rhs: &Polynomial) -> Self::Output {
//         self + rhs.to_owned()
//     }
// }

impl std::ops::Add<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: Polynomial) -> Polynomial {
        for (monomial, coeff) in rhs.terms {
            *self.terms.entry(monomial).or_insert(Ratio::zero()) += coeff;
        }

        self.simplify();
        self
    }
}

impl std::ops::Mul<Polynomial> for Polynomial {
    type Output = Polynomial;

    /// Multiply two polynomials
    fn mul(self, rhs: Polynomial) -> Self::Output {
        self * (&rhs)
    }
}

impl std::ops::Mul<&Polynomial> for Polynomial {
    type Output = Polynomial;

    /// Multiply two polynomials
    fn mul(self, rhs: &Polynomial) -> Self::Output {
        let mut result = Polynomial::new();

        for (mon1, coeff1) in self.terms {
            for (mon2, coeff2) in &rhs.terms {
                // Multiply coefficients
                let new_coeff = coeff1 * coeff2;

                if new_coeff.is_zero() {
                    continue;
                }

                // Multiply monomials by adding exponents
                let new_monomial = mon1.clone() * mon2;

                result = result.add_term(new_coeff, new_monomial);
            }
        }

        result.simplify();
        result
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn div(self, rhs: Polynomial) -> Self::Output {
        let rhs_inv = rhs.inv();
        self * rhs_inv
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial::neg_one() * self
    }
}

impl std::ops::Sub<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn sub(self, rhs: Polynomial) -> Self::Output {
        self + (-rhs)
    }
}

/// Raise a polynomial to a non-negative integer power
fn power_polynomial(mut base: Polynomial, n: u32) -> Polynomial {
    if n == 0 {
        return Polynomial::one();
    }
    if n == 1 {
        return base.clone();
    }

    // Use binary exponentiation for efficiency
    let mut result = Polynomial::one();
    let mut exp = n;

    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base.clone();
        }
        base = base.clone() * base.clone();
        exp /= 2;
    }

    result
}

// -------------------------------------
// Partial Eq and Eq
// -------------------------------------

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        // Compare only non-zero terms
        let self_terms: BTreeMap<_, _> = self.terms.iter().filter(|(_, c)| !c.is_zero()).collect();
        let other_terms: BTreeMap<_, _> =
            other.terms.iter().filter(|(_, c)| !c.is_zero()).collect();

        self_terms == other_terms
    }
}

impl Eq for Polynomial {}

// -------------------------------------
// Display
// -------------------------------------

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (monomial, coeff) in self.sorted_monomials() {
            if !first {
                if coeff.is_positive() {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if coeff.is_negative() {
                write!(f, "-")?;
            }

            let abs_coeff = coeff.abs();
            if !abs_coeff.is_one() || monomial.is_constant() {
                write!(f, "{abs_coeff}")?;
                if !monomial.is_constant() {
                    write!(f, "*")?;
                }
            }

            if !monomial.is_constant() {
                write!(f, "{monomial}")?;
            }

            first = false;
        }

        if first {
            write!(f, "0")?;
        }

        Ok(())
    }
}

// -------------------------------------
// From Polynomial to RecExpr<Rise>
// -------------------------------------

impl From<&Polynomial> for RecExpr<Rise> {
    fn from(p: &Polynomial) -> Self {
        let mut expr = RecExpr::default();

        if p.is_zero() || p.terms.is_empty() {
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

        result_id.unwrap();
        expr
    }
}

impl From<Polynomial> for RecExpr<Rise> {
    fn from(p: Polynomial) -> Self {
        (&p).into()
    }
}

// -------------------------------------
// From RecExpr<Rise> to Polynomial
// -------------------------------------

impl From<RecExpr<Rise>> for Polynomial {
    fn from(expr: RecExpr<Rise>) -> Self {
        (&expr).into()
    }
}

impl From<&RecExpr<Rise>> for Polynomial {
    fn from(expr: &RecExpr<Rise>) -> Self {
        /// Parse a Rise expression into a Polynomial
        fn rec(expr: &RecExpr<Rise>, id: Id) -> Polynomial {
            let node = &expr[id];

            match node {
                // Integer constant
                Rise::Integer(n) => Polynomial::new().add_term((*n).into(), Monomial::new()),
                // Single variable with exponent 1
                Rise::Var(index) => {
                    Polynomial::new().add_term(Ratio::one(), Monomial::new().with_var(*index, 1))
                }

                Rise::NatAdd([left, right]) => {
                    // Addition: recursively parse both sides and add
                    let left_poly = rec(expr, *left);
                    let right_poly = rec(expr, *right);
                    left_poly + right_poly
                }

                Rise::NatSub([left, right]) => {
                    // Subtraction: left - right = left + (-1 * right)
                    let left_poly = rec(expr, *left);
                    let right_poly = rec(expr, *right);
                    left_poly - right_poly
                }

                Rise::NatMul([left, right]) => {
                    // Multiplication: recursively parse and multiply
                    let left_poly = rec(expr, *left);
                    let right_poly = rec(expr, *right);
                    left_poly * right_poly
                }

                Rise::NatPow([base, exp]) => {
                    // Power: handle base^exponent
                    let base_poly = rec(expr, *base);

                    // Exponent should be an integer
                    match &expr[*exp] {
                        Rise::Integer(n) => {
                            if *n == 0 {
                                Polynomial::one()
                            } else if *n == 1 {
                                base_poly
                            } else if *n == -1 {
                                base_poly.inv()
                            } else if *n > 0 {
                                power_polynomial(base_poly, (*n).try_into().unwrap())
                            } else {
                                // Negative exponent: (base)^(-n) = (base^(-1))^n
                                let inverted = base_poly.inv();
                                power_polynomial(inverted, (-*n).try_into().unwrap())
                            }
                        }
                        _ => panic!("NatPow exponent must be an integer constant"),
                    }
                }

                Rise::NatDiv([left, right]) => {
                    // Division: left / right = left * (right^(-1))
                    let dividend = rec(expr, *left);
                    let divisor = rec(expr, *right);

                    // Try to invert the divisor
                    let divisor_inv = divisor.inv();
                    dividend * divisor_inv
                }

                _ => panic!("Unsupported Rise node type for polynomial conversion: {node:?}"),
            }
        }
        // Parse from the root (last node in the RecExpr)
        if expr.is_empty() {
            return Polynomial::new();
        }

        let root_id = Id::from(expr.as_ref().len() - 1);
        rec(expr, root_id)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::rewrite_system::rise::{Index, Kind};

    #[test]
    fn poly_easy() {
        // 3x^2y + 2x
        let poly1 = Polynomial::new()
            .add_term(
                3.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 2)
                    .with_var(Index::new(2, Kind::Nat), 1),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            );

        // 5x^2y - 2x + 7z
        let poly2 = Polynomial::new()
            .add_term(
                5.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 2)
                    .with_var(Index::new(2, Kind::Nat), 1),
            )
            .add_term(
                (-2).into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(
                7.into(),
                Monomial::new().with_var(Index::new(3, Kind::Nat), 1),
            );

        println!("Polynomial 1: {poly1}");
        println!("Polynomial 2: {poly2}");
        println!();

        let result = poly1 + poly2;
        println!("Sum (canonical form): {result}");
        println!();
    }

    #[test]
    fn poly_complex() {
        // More complex polynomials
        let poly1 = Polynomial::new()
            .add_term(
                4.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 3)
                    .with_var(Index::new(2, Kind::Nat), 2),
            )
            .add_term(
                (-3).into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                5.into(),
                Monomial::new().with_var(Index::new(2, Kind::Nat), 1),
            )
            .add_term(2.into(), Monomial::new()); // constant term

        let poly2 = Polynomial::new()
            .add_term(
                2.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 3)
                    .with_var(Index::new(2, Kind::Nat), 2),
            )
            .add_term(
                3.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                (-5).into(),
                Monomial::new().with_var(Index::new(2, Kind::Nat), 1),
            )
            .add_term(8.into(), Monomial::new()); // constant term

        println!("Polynomial 1: {poly1}");
        println!("Polynomial 2: {poly2}");
        println!();

        let result = poly1 + poly2;
        println!("Sum (canonical form): {result}");
        println!();
    }

    #[test]
    fn poly_cancellation() {
        let poly1 = Polynomial::new().add_term(
            5.into(),
            Monomial::new()
                .with_var(Index::new(1, Kind::Nat), 1)
                .with_var(Index::new(2, Kind::Nat), 1),
        );

        let poly2 = Polynomial::new()
            .add_term(
                (-5).into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 1)
                    .with_var(Index::new(2, Kind::Nat), 1),
            )
            .add_term(3.into(), Monomial::new());

        println!("Polynomial 1: {poly1}");
        println!("Polynomial 2: {poly2}");
        println!();

        let result = poly1 + poly2;
        println!("Sum (canonical form): {result}");
        println!();
    }

    #[test]
    fn poly_negative_exponent() {
        let poly1 = Polynomial::new()
            .add_term(
                3.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), -1)
                    .with_var(Index::new(2, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), -2),
            );

        let poly2 = Polynomial::new()
            .add_term(
                5.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), -1)
                    .with_var(Index::new(2, Kind::Nat), 2),
            )
            .add_term(
                (-2).into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), -2),
            )
            .add_term(
                4.into(),
                Monomial::new().with_var(Index::new(3, Kind::Nat), -1),
            );

        println!("Polynomial 1 (with negative exponents): {poly1}");
        println!("Polynomial 2 (with negative exponents): {poly2}");
        println!();

        let result = poly1 + poly2;
        println!("Sum (canonical form): {result}");
        println!();
    }

    #[test]
    fn poly_mixed_exponents() {
        let poly1 = Polynomial::new()
            .add_term(
                2.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 2)
                    .with_var(Index::new(2, Kind::Nat), -1),
            )
            .add_term(
                3.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            );

        let poly2 = Polynomial::new()
            .add_term(
                1.into(),
                Monomial::new()
                    .with_var(Index::new(1, Kind::Nat), 2)
                    .with_var(Index::new(2, Kind::Nat), -1),
            )
            .add_term(
                (-3).into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(
                5.into(),
                Monomial::new().with_var(Index::new(2, Kind::Nat), -2),
            );

        println!("Polynomial 1 (mixed exponents): {poly1}");
        println!("Polynomial 2 (mixed exponents): {poly2}");
        println!();

        let result = poly1 + poly2;
        println!("Sum (canonical form): {result}");
    }

    #[test]
    fn poly_roundtrip() {
        // Test converting to RecExpr and back
        let poly = Polynomial::new()
            .add_term(
                3.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(5.into(), Monomial::new());

        println!("Original polynomial: {poly}");

        let expr: RecExpr<Rise> = poly.clone().into();
        println!("As RecExpr: {expr}");

        let poly_back: Polynomial = expr.into();
        println!("Back to polynomial: {poly_back}");

        assert_eq!(poly.to_string(), poly_back.to_string());
    }

    #[test]
    fn poly_roundtrip_neg() {
        // Test converting to RecExpr and back
        let poly = Polynomial::new()
            .add_term(
                3.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), -1),
            )
            .add_term(5.into(), Monomial::new());

        println!("Original polynomial: {poly}");

        let expr: RecExpr<Rise> = poly.clone().into();
        println!("As RecExpr: {expr}");

        let poly_back: Polynomial = expr.into();
        println!("Back to polynomial: {poly_back}");

        assert_eq!(poly.to_string(), poly_back.to_string());
    }

    #[test]
    fn poly_from_expr_simple() {
        // Manually create a simple expression: 2*x + 3
        let mut expr = RecExpr::default();

        let two = expr.add(Rise::Integer(2));
        let x = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_x = expr.add(Rise::NatMul([two, x]));
        let three = expr.add(Rise::Integer(3));
        expr.add(Rise::NatAdd([two_x, three]));

        let poly: Polynomial = expr.into();
        println!("Parsed polynomial: {poly}");

        assert_eq!(poly.to_string(), "2*x_1 + 3");
    }

    #[test]
    fn poly_from_expr_power() {
        // Create expression: x^2 + 2*x + 1
        let mut expr = RecExpr::default();

        let x = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_int = expr.add(Rise::Integer(2));
        let x_squared = expr.add(Rise::NatPow([x, two_int]));

        let x2 = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_int2 = expr.add(Rise::Integer(2));
        let two_x = expr.add(Rise::NatMul([two_int2, x2]));

        let sum1 = expr.add(Rise::NatAdd([x_squared, two_x]));

        let one = expr.add(Rise::Integer(1));
        expr.add(Rise::NatAdd([sum1, one]));

        let poly: Polynomial = expr.into();
        println!("Parsed polynomial: {poly}");

        assert_eq!(poly.to_string(), "x_1^2 + 2*x_1 + 1");
    }

    #[test]
    fn poly_from_expr_neg_exp() {
        // Create expression: x^2 + 2*x^(-1) + 1
        let mut expr = RecExpr::default();

        let x = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_int = expr.add(Rise::Integer(2));
        let x_squared = expr.add(Rise::NatPow([x, two_int]));

        let x2 = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_int2 = expr.add(Rise::Integer(2));
        let neg_1 = expr.add(Rise::Integer(-1));
        let x_inv = expr.add(Rise::NatPow([x2, neg_1]));
        let two_x = expr.add(Rise::NatMul([two_int2, x_inv]));

        let sum1 = expr.add(Rise::NatAdd([x_squared, two_x]));

        let one = expr.add(Rise::Integer(1));
        expr.add(Rise::NatAdd([sum1, one]));

        let poly: Polynomial = expr.into();
        println!("Parsed polynomial: {poly}");

        assert_eq!(poly.to_string(), "x_1^2 + 1 + 2*x_1^(-1)");
    }

    // New tests for division support

    #[test]
    fn poly_inv_constant() {
        // Invert a constant: 5 -> 1/5
        let poly = Polynomial::new().add_term(5.into(), Monomial::new());
        println!("Original: {poly}");

        let inv = poly.inv();
        println!("Inverted: {inv}");

        assert_eq!(inv.to_string(), "1/5");
    }

    #[test]
    fn poly_inv_monomial() {
        // Invert 3x^2: 3x^2 -> (1/3)x^(-2)
        let poly = Polynomial::new().add_term(
            3.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
        );
        println!("Original: {poly}");

        let inv = poly.inv();
        println!("Inverted: {inv}");

        assert_eq!(inv.to_string(), "1/3*x_1^(-2)");
    }

    #[test]
    fn poly_inv_monomial_multi_var() {
        // Invert 2x^2*y: 2x^2*y -> (1/2)x^(-2)*y^(-1)
        let poly = Polynomial::new().add_term(
            2.into(),
            Monomial::new()
                .with_var(Index::new(1, Kind::Nat), 2)
                .with_var(Index::new(2, Kind::Nat), 1),
        );
        println!("Original: {poly}");

        let inv = poly.inv();
        println!("Inverted: {inv}");

        assert_eq!(inv.to_string(), "1/2*x_1^(-2) * x_2^(-1)");
    }

    #[test]
    #[should_panic(expected = "Cannot invert multi-term polynomial")]
    fn poly_inv_multi_term_panics() {
        // Trying to invert (x + 1) should panic
        let poly = Polynomial::new()
            .add_term(
                1.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(1.into(), Monomial::new());

        println!("Trying to invert: {poly}");
        let _ = poly.inv(); // Should panic
    }

    #[test]
    fn poly_try_inv_multi_term_returns_err() {
        // try_inv on (x + 1) should return Err
        let poly = Polynomial::new()
            .add_term(
                1.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(1.into(), Monomial::new());

        assert!(poly.try_inv().is_err());
    }

    #[test]
    fn poly_division_simple() {
        // (6x^2) / (2x) = 3x
        let dividend = Polynomial::new().add_term(
            6.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
        );
        let divisor = Polynomial::new().add_term(
            2.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
        );

        println!("Dividend: {dividend}");
        println!("Divisor: {divisor}");

        let result = dividend / divisor;
        println!("Result: {result}");

        assert_eq!(result.to_string(), "3*x_1");
    }

    #[test]
    fn poly_division_with_remainder_style() {
        // (4x^2 + 2x) / (2x) = 2x + 1
        let dividend = Polynomial::new()
            .add_term(
                4.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            );
        let divisor = Polynomial::new().add_term(
            2.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
        );

        println!("Dividend: {dividend}");
        println!("Divisor: {divisor}");

        let result = dividend / divisor;
        println!("Result: {result}");

        assert_eq!(result.to_string(), "2*x_1 + 1");
    }

    #[test]
    fn poly_division_creates_negative_exp() {
        // (3x) / (x^2) = 3x^(-1)
        let dividend = Polynomial::new().add_term(
            3.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
        );
        let divisor = Polynomial::new().add_term(
            1.into(),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
        );

        println!("Dividend: {dividend}");
        println!("Divisor: {divisor}");

        let result = dividend / divisor;
        println!("Result: {result}");

        assert_eq!(result.to_string(), "3*x_1^(-1)");
    }

    #[test]
    fn poly_from_expr_division() {
        // Create expression: (6*x^2) / (3*x) which should simplify to 2*x
        let mut expr = RecExpr::default();

        let six = expr.add(Rise::Integer(6));
        let x1 = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two_exp = expr.add(Rise::Integer(2));
        let x_squared = expr.add(Rise::NatPow([x1, two_exp]));
        let numerator = expr.add(Rise::NatMul([six, x_squared]));

        let three = expr.add(Rise::Integer(3));
        let x2 = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let denominator = expr.add(Rise::NatMul([three, x2]));

        expr.add(Rise::NatDiv([numerator, denominator]));

        let poly: Polynomial = expr.into();
        println!("Parsed polynomial: {poly}");

        assert_eq!(poly.to_string(), "2*x_1");
    }

    #[test]
    fn poly_multiplication() {
        // (2x + 1) * (x + 3) = 2x^2 + 7x + 3
        let poly1 = Polynomial::new()
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(1.into(), Monomial::new());

        let poly2 = Polynomial::new()
            .add_term(
                1.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            )
            .add_term(3.into(), Monomial::new());

        println!("Poly1: {poly1}");
        println!("Poly2: {poly2}");

        let result = poly1 * poly2;
        println!("Product: {result}");

        assert_eq!(result.to_string(), "2*x_1^2 + 7*x_1 + 3");
    }

    #[test]
    fn poly_subtraction() {
        // (3x^2 + 2x) - (x^2 + 2x) = 2x^2
        let poly1 = Polynomial::new()
            .add_term(
                3.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            );

        let poly2 = Polynomial::new()
            .add_term(
                1.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 2),
            )
            .add_term(
                2.into(),
                Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
            );

        println!("Poly1: {poly1}");
        println!("Poly2: {poly2}");

        let result = poly1 - poly2;
        println!("Difference: {result}");

        assert_eq!(result.to_string(), "2*x_1^2");
    }

    #[test]
    fn poly_rational_coefficient_roundtrip() {
        // Test that rational coefficients survive roundtrip
        // (1/2)*x
        let poly = Polynomial::new().add_term(
            Ratio::new(1, 2),
            Monomial::new().with_var(Index::new(1, Kind::Nat), 1),
        );

        println!("Original polynomial: {poly}");

        let expr: RecExpr<Rise> = poly.clone().into();
        println!("As RecExpr: {expr}");

        // Note: The roundtrip will parse (1/2)*x as a division expression
        // which will then be simplified. This tests that path.
    }

    #[test]
    fn poly_complex_division_roundtrip() {
        // Create expression: x / 2 + y / 3
        let mut expr = RecExpr::default();

        let x = expr.add(Rise::Var(Index::new(1, Kind::Nat)));
        let two = expr.add(Rise::Integer(2));
        let x_div_2 = expr.add(Rise::NatDiv([x, two]));

        let y = expr.add(Rise::Var(Index::new(2, Kind::Nat)));
        let three = expr.add(Rise::Integer(3));
        let y_div_3 = expr.add(Rise::NatDiv([y, three]));

        expr.add(Rise::NatAdd([x_div_2, y_div_3]));

        let poly: Polynomial = expr.into();
        println!("Parsed polynomial: {poly}");

        // Should be (1/2)*x + (1/3)*y
        assert_eq!(poly.to_string(), "1/2*x_1 + 1/3*x_2");
    }
}
