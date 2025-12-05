use std::collections::BTreeMap;
use std::fmt;

use egg::RecExpr;
use num::rational::Ratio;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};

use super::{Monomial, NatSolverError, Rise};
use crate::rewrite_system::rise::Index;

// ============================================================================
// Polynomial
// ============================================================================

/// Represents a polynomial as a map from monomials to coefficients
/// This can represent Laurent polynomials (with negative exponents) and
/// rational expressions that simplify to a single term in the denominator.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Polynomial {
    // Map from monomial to coefficient
    terms: BTreeMap<Monomial, Ratio<i32>>,
}

impl Polynomial {
    /// Create the empty polynomial
    pub fn new() -> Self {
        Self::default()
    }

    /// Create the 1 polynomial
    pub fn one() -> Self {
        Self::new().add_term(Ratio::one(), Monomial::new())
    }

    /// Create a polynomial from a single variable
    pub fn var(index: Index) -> Self {
        index.into()
    }

    /// Create a polynomial from a constant integer
    pub fn constant(n: i32) -> Self {
        n.into()
    }

    /// Create a polynomial from a constant ratio
    pub fn constant_ratio(r: Ratio<i32>) -> Self {
        r.into()
    }

    /// Try to invert a polynomial, returning Err if it cannot be represented as a polynomial.
    ///
    /// This operation is only valid for:
    /// 1. Single-term polynomials (monomials with coefficient): c * x^a * y^b -> (1/c) * x^(-a) * y^(-b)
    /// 2. Constant polynomials: c -> 1/c
    ///
    /// For multi-term polynomials like (x + 1), inversion cannot produce a polynomial,
    /// so this will return an error.
    pub fn try_inv(mut self) -> Result<Self, NatSolverError> {
        self.simplify();

        if self.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }

        // Check if this is a single-term polynomial (monomial)
        if self.term_count() == 1 {
            let (monomial, coeff) = self.terms.into_iter().next().unwrap();
            // Invert coefficient: c -> 1/c
            let inv_coeff = Ratio::one() / coeff;
            // Invert monomial: negate all exponents
            let inv_monomial = monomial.inv();
            Ok(Self::new().add_term(inv_coeff, inv_monomial))
        } else {
            Err(NatSolverError::NonMonomialInversion(self))
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

    /// Check if polynomial equals one
    pub fn is_one(&self) -> bool {
        if self.term_count() == 1
            && let Some((mon, coeff)) = self.terms.iter().next()
        {
            return mon.is_constant() && coeff.is_one();
        }
        false
    }

    /// Check if polynomial is a single monomial (can be inverted)
    pub fn is_monomial(&self) -> bool {
        self.term_count() <= 1
    }

    /// Check if polynomial is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        self.terms
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .all(|(m, _)| m.is_constant())
    }

    /// Get the constant value if this is a constant polynomial
    pub fn as_constant(&self) -> Option<Ratio<i32>> {
        self.is_constant().then(|| {
            self.terms
                .iter()
                .filter(|(m, _)| m.is_constant())
                .map(|(_, c)| *c)
                .fold(Ratio::zero(), |a, b| a + b)
        })
    }

    /// Get terms iterator
    fn terms(&self) -> impl Iterator<Item = (&Monomial, &Ratio<i32>)> {
        self.terms.iter().filter(|(_, c)| !c.is_zero())
    }

    /// Get the number of non-zero terms
    pub fn term_count(&self) -> usize {
        self.terms().count()
    }

    pub fn sorted_monomials(&self) -> Vec<(&Monomial, &Ratio<i32>)> {
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

    /// Compute GCD of all coefficients in a polynomial
    pub fn coefficient_gcd(&self) -> Ratio<i32> {
        let mut result: Option<Ratio<i32>> = None;
        for (_, coeff) in self.terms() {
            let abs_coeff = coeff.abs();
            result = Some(match result {
                None => abs_coeff,
                Some(g) => super::gcd_ratio(g, abs_coeff),
            });
        }
        result.unwrap_or_else(Ratio::zero)
    }

    /// Divide all coefficients in a polynomial by a constant
    pub fn divide_coefficients(&self, divisor: Ratio<i32>) -> Self {
        let mut result = Self::new();
        for (mon, coeff) in self.terms() {
            result = result.add_term(*coeff / divisor, mon.clone());
        }
        result
    }

    /// Raise polynomial to a non-negative integer power
    pub fn pow(mut self, n: u32) -> Self {
        if n == 0 {
            return Self::one();
        }
        if n == 1 {
            return self;
        }

        // Use binary exponentiation for efficiency
        let mut result = Self::one();
        let mut exp = n;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result * self.clone();
            }
            self = self.clone() * self.clone();
            exp /= 2;
        }

        result
    }
}

// ============================================================================
// Polynomial Arithmetic
// ============================================================================

impl std::ops::Add for Polynomial {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        for (monomial, coeff) in rhs.terms {
            *self.terms.entry(monomial).or_insert(Ratio::zero()) += coeff;
        }

        self.simplified()
    }
}

impl std::ops::Add<&Self> for Polynomial {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        self + rhs.clone()
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * (&rhs)
    }
}

impl std::ops::Mul<&Self> for Polynomial {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        let mut result = Self::new();

        for (mon1, coeff1) in self.terms {
            for (mon2, coeff2) in &rhs.terms {
                let new_coeff = coeff1 * coeff2;

                if new_coeff.is_zero() {
                    continue;
                }

                let new_monomial = mon1.clone() * mon2;
                result = result.add_term(new_coeff, new_monomial);
            }
        }

        result.simplified()
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new().add_term(-Ratio::one(), Monomial::new()) * self
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<&Self> for Polynomial {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

// ============================================================================
// Polynomial Equality
// ============================================================================

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        let self_terms: BTreeMap<_, _> = self.terms.iter().filter(|(_, c)| !c.is_zero()).collect();
        let other_terms: BTreeMap<_, _> =
            other.terms.iter().filter(|(_, c)| !c.is_zero()).collect();

        self_terms == other_terms
    }
}

// TODO: Check if i can do this via a derive macro
impl Eq for Polynomial {}

// ============================================================================
// Polynomial Display
// ============================================================================

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

// ============================================================================
// RecExpr Conversions
// ============================================================================

// -------------------------------------
// From Polynomial to RecExpr<Rise>
// -------------------------------------

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

        result_id.unwrap();
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

/// Create a polynomial from an integer constant
impl From<i32> for Polynomial {
    fn from(n: i32) -> Self {
        Self::new().add_term(n.into(), Monomial::new())
    }
}
/// Create a polynomial from an integer constant
impl From<Ratio<i32>> for Polynomial {
    fn from(r: Ratio<i32>) -> Self {
        Self::new().add_term(r, Monomial::new())
    }
}

impl From<Index> for Polynomial {
    fn from(index: Index) -> Self {
        Self::new().add_term(Ratio::one(), Monomial::new().with_var(index, 1))
    }
}
