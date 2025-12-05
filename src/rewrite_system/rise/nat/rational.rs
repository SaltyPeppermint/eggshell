use std::fmt;

use egg::{Id, Language, RecExpr};
use num::rational::Ratio;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};

use super::{NatSolverError, Polynomial, Rise};
use crate::rewrite_system::rise::Index;

// ============================================================================
// RationalFunction
// ============================================================================

/// Represents a rational function as a ratio of two polynomials: numerator / denominator
///
/// This can represent any rational expression including:
/// - Simple polynomials: `x^2 + 2x + 1` (denominator = 1)
/// - Rational expressions: `1 / (x + 1)`
/// - Complex ratios: `(x^2 + 1) / (x^2 - 1)`
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RationalFunction {
    numerator: Polynomial,
    denominator: Polynomial,
}

impl RationalFunction {
    /// Create a new rational function, simplifying if possible
    pub fn new(numerator: Polynomial, denominator: Polynomial) -> Result<Self, NatSolverError> {
        if denominator.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }
        Ok(RationalFunction {
            numerator,
            denominator,
        }
        .simplified())
    }

    /// Create a rational function from just a polynomial (denominator = 1)
    pub fn from_polynomial(p: Polynomial) -> Self {
        p.into()
    }

    /// Create the zero rational function
    pub fn zero() -> Self {
        RationalFunction {
            numerator: Polynomial::new(),
            denominator: Polynomial::one(),
        }
    }

    /// Create the one rational function
    pub fn one() -> Self {
        RationalFunction {
            numerator: Polynomial::one(),
            denominator: Polynomial::one(),
        }
    }

    /// Check if this rational function is zero
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Check if this rational function equals one
    pub fn is_one(&self) -> bool {
        self.numerator == self.denominator && !self.numerator.is_zero()
    }

    /// Check if this is effectively a polynomial (denominator is 1)
    pub fn is_polynomial(&self) -> bool {
        self.denominator.is_one()
    }

    /// Check if this is a constant (both numerator and denominator are constants)
    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominator.is_constant()
    }

    /// Get the constant value if this is a constant rational function
    pub fn as_constant(&self) -> Option<Ratio<i32>> {
        if let (Some(n), Some(d)) = (self.numerator.as_constant(), self.denominator.as_constant()) {
            if d.is_zero() { None } else { Some(n / d) }
        } else {
            None
        }
    }

    /// Simplify the rational function
    ///
    /// Currently handles:
    /// 1. If denominator is a monomial, absorb it into numerator using negative exponents
    /// 2. Normalize signs (if denominator has negative leading coefficient, negate both)
    /// 3. Cancel common constant factors
    pub fn simplified(mut self) -> Self {
        self.numerator.simplify();
        self.denominator.simplify();

        // Handle zero numerator
        if self.numerator.is_zero() {
            return RationalFunction {
                numerator: Polynomial::new(),
                denominator: Polynomial::one(),
            };
        }

        // If denominator is one, we're done
        if self.denominator.is_one() {
            return self;
        }

        // If denominator is a monomial, we can absorb it into the numerator
        if self.denominator.is_monomial()
            && let Ok(denom_inv) = self.denominator.clone().try_inv()
        {
            self.numerator = self.numerator * denom_inv;
            self.denominator = Polynomial::one();
            return self;
        }

        // Try to cancel common constant factors from coefficients
        self.cancel_constant_factors();
        // Normalize: if denominator's leading coefficient is negative, negate both
        self.normalize_sign();

        self
    }

    /// Cancel common constant factors between numerator and denominator
    fn cancel_constant_factors(&mut self) {
        // Get GCD of all coefficients in numerator
        let numer_gcd = self.numerator.coefficient_gcd();
        let denom_gcd = self.denominator.coefficient_gcd();

        if numer_gcd.is_zero() || denom_gcd.is_zero() {
            return;
        }

        // Find GCD of the two GCDs
        let common_gcd = super::gcd_ratio(numer_gcd, denom_gcd);

        if !common_gcd.is_one() && !common_gcd.is_zero() {
            // Divide all coefficients by the common GCD
            self.numerator = self.numerator.divide_coefficients(common_gcd);
            self.denominator = self.denominator.divide_coefficients(common_gcd);
        }
    }

    /// Normalize sign: ensure denominator's leading coefficient is positive
    fn normalize_sign(&mut self) {
        let denom_sorted = self.denominator.sorted_monomials();
        if let Some((_, coeff)) = denom_sorted.first()
            && coeff.is_negative()
        {
            self.numerator = -std::mem::take(&mut self.numerator);
            self.denominator = -std::mem::take(&mut self.denominator);
        }
    }

    /// Invert the rational function: (n/d) -> (d/n)
    pub fn inv(self) -> Result<Self, NatSolverError> {
        if self.numerator.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }
        Ok(RationalFunction {
            numerator: self.denominator,
            denominator: self.numerator,
        }
        .simplified())
    }

    /// Raise to an integer power (can be negative)
    pub fn pow(self, n: i32) -> Result<Self, NatSolverError> {
        if n == 0 {
            return Ok(RationalFunction::one());
        }

        if n > 0 {
            let exp = n.try_into()?;
            Ok(RationalFunction {
                numerator: self.numerator.pow(exp),
                denominator: self.denominator.pow(exp),
            }
            .simplified())
        } else {
            // Negative exponent: invert first, then raise to positive power
            let exp = (-n).try_into()?;
            let inverted = self.inv()?;
            Ok(RationalFunction {
                numerator: inverted.numerator.pow(exp),
                denominator: inverted.denominator.pow(exp),
            }
            .simplified())
        }
    }
}

impl Default for RationalFunction {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// RationalFunction Arithmetic
// ============================================================================

impl std::ops::Add<RationalFunction> for RationalFunction {
    type Output = RationalFunction;

    /// Add two rational functions: a/b + c/d = (ad + bc) / bd
    fn add(self, rhs: RationalFunction) -> Self::Output {
        // Optimization: if denominators are equal, just add numerators
        if self.denominator == rhs.denominator {
            return RationalFunction {
                numerator: self.numerator + rhs.numerator,
                denominator: self.denominator,
            }
            .simplified();
        }

        let numerator =
            self.numerator.clone() * &rhs.denominator + rhs.numerator.clone() * &self.denominator;
        let denominator = self.denominator * rhs.denominator;
        RationalFunction {
            numerator,
            denominator,
        }
        .simplified()
    }
}

impl std::ops::Sub<RationalFunction> for RationalFunction {
    type Output = RationalFunction;

    fn sub(self, rhs: RationalFunction) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul<RationalFunction> for RationalFunction {
    type Output = RationalFunction;

    /// Multiply two rational functions: (a/b) * (c/d) = ac / bd
    fn mul(self, rhs: RationalFunction) -> Self::Output {
        RationalFunction {
            numerator: self.numerator * rhs.numerator,
            denominator: self.denominator * rhs.denominator,
        }
        .simplified()
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl std::ops::Div<RationalFunction> for RationalFunction {
    type Output = Result<RationalFunction, NatSolverError>;

    /// Divide two rational functions: (a/b) / (c/d) = (a/b) * (d/c) = ad / bc
    fn div(self, rhs: RationalFunction) -> Self::Output {
        if rhs.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }
        Ok(self * rhs.inv()?)
    }
}

impl std::ops::Neg for RationalFunction {
    type Output = RationalFunction;

    fn neg(self) -> Self::Output {
        RationalFunction {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
        .simplified()
    }
}

// ============================================================================
// RationalFunction Equality
// ============================================================================

impl PartialEq for RationalFunction {
    fn eq(&self, other: &Self) -> bool {
        // Two rational functions are equal if their cross products are equal
        // a/b == c/d iff ad == bc
        let lhs = self.numerator.clone() * &other.denominator;
        let rhs = other.numerator.clone() * &self.denominator;
        lhs == rhs
    }
}

// TODO: Check if i can do this via a derive macro
impl Eq for RationalFunction {}

// ============================================================================
// RationalFunction Display
// ============================================================================

impl fmt::Display for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else if self.numerator.term_count() <= 1 && self.denominator.term_count() <= 1 {
            // Simple case: single terms, no parens needed for numerator
            write!(f, "{} / {}", self.numerator, self.denominator)
        } else {
            // Complex case: use parentheses
            let numer_str = if self.numerator.term_count() > 1 {
                format!("({})", self.numerator)
            } else {
                format!("{}", self.numerator)
            };
            let denom_str = if self.denominator.term_count() > 1 {
                format!("({})", self.denominator)
            } else {
                format!("{}", self.denominator)
            };
            write!(f, "{numer_str} / {denom_str}")
        }
    }
}

// ============================================================================
// Conversions: Polynomial <-> RationalFunction
// ============================================================================

impl From<Polynomial> for RationalFunction {
    fn from(p: Polynomial) -> Self {
        RationalFunction::from_polynomial(p)
    }
}

impl TryFrom<RationalFunction> for Polynomial {
    type Error = NatSolverError;

    fn try_from(rf: RationalFunction) -> Result<Self, Self::Error> {
        {
            let simplified = rf.simplified();
            if simplified.is_polynomial() {
                Ok(simplified.numerator)
            } else {
                Err(NatSolverError::NotAPolynomial(simplified))
            }
        }
    }
}

// ============================================================================
// RecExpr <-> RationalFunction
// ============================================================================

// -------------------------------------
// RationalFunction -> RecExpr<Rise>
// -------------------------------------

impl From<&RationalFunction> for RecExpr<Rise> {
    fn from(rf: &RationalFunction) -> Self {
        let mut expr = RecExpr::default();

        // If it's just a polynomial, convert directly
        if rf.denominator.is_one() {
            return (&rf.numerator).into();
        }

        // Build numerator expression
        let numer_expr: RecExpr<Rise> = (&rf.numerator).into();
        let numer_nodes: Vec<_> = numer_expr.as_ref().to_vec();

        // Add numerator nodes to our expression, tracking the ID mapping
        let mut numer_id_map: Vec<Id> = Vec::with_capacity(numer_nodes.len());
        for node in &numer_nodes {
            let new_node = node
                .clone()
                .map_children(|old_id| numer_id_map[usize::from(old_id)]);
            let new_id = expr.add(new_node);
            numer_id_map.push(new_id);
        }
        let numer_root = *numer_id_map.last().unwrap();

        // Build denominator expression
        let denom_expr: RecExpr<Rise> = (&rf.denominator).into();
        let denom_nodes: Vec<_> = denom_expr.as_ref().to_vec();

        // Add denominator nodes to our expression
        let mut denom_id_map: Vec<Id> = Vec::with_capacity(denom_nodes.len());
        for node in &denom_nodes {
            let new_node = node
                .clone()
                .map_children(|old_id| denom_id_map[usize::from(old_id)]);
            let new_id = expr.add(new_node);
            denom_id_map.push(new_id);
        }
        let denom_root = *denom_id_map.last().unwrap();

        // Create division node
        expr.add(Rise::NatDiv([numer_root, denom_root]));

        expr
    }
}

impl From<RationalFunction> for RecExpr<Rise> {
    fn from(rf: RationalFunction) -> Self {
        (&rf).into()
    }
}

// -------------------------------------
// RecExpr<Rise> -> RationalFunction
// -------------------------------------

impl TryFrom<RecExpr<Rise>> for RationalFunction {
    type Error = NatSolverError;

    fn try_from(expr: RecExpr<Rise>) -> Result<Self, Self::Error> {
        (&expr).try_into()
    }
}

impl TryFrom<&RecExpr<Rise>> for RationalFunction {
    type Error = NatSolverError;

    fn try_from(expr: &RecExpr<Rise>) -> Result<Self, Self::Error> {
        /// Parse a Rise expression into a `RationalFunction`
        fn rec(expr: &RecExpr<Rise>, id: Id) -> Result<RationalFunction, NatSolverError> {
            match &expr[id] {
                // Integer constant
                Rise::Integer(n) => Ok((*n).into()),
                // Single variable with exponent 1
                Rise::Var(index) => Ok((*index).into()),
                Rise::NatAdd([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    Ok(left_rf + right_rf)
                }
                Rise::NatSub([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    Ok(left_rf - right_rf)
                }
                Rise::NatMul([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    Ok(left_rf * right_rf)
                }
                Rise::NatDiv([left, right]) => {
                    let dividend = rec(expr, *left)?;
                    let divisor = rec(expr, *right)?;
                    Ok((dividend / divisor)?)
                }
                Rise::NatPow([base, exp]) => {
                    let base_rf = rec(expr, *base)?;
                    // Exponent should be an integer
                    match &expr[*exp] {
                        Rise::Integer(n) => base_rf.pow(*n),
                        node => Err(NatSolverError::NonIntegerExponent(node.clone())),
                    }
                }

                node => Err(NatSolverError::UnsupportedNode(node.clone())),
            }
        }

        // Parse from the root (last node in the RecExpr)
        if expr.is_empty() {
            return Ok(Self::zero());
        }

        let root_id = Id::from(expr.as_ref().len() - 1);
        rec(expr, root_id)
    }
}

// ============================================================================
// From Simple Types
// ============================================================================

impl From<Index> for RationalFunction {
    fn from(index: Index) -> Self {
        Self::from_polynomial(Polynomial::var(index))
    }
}

impl From<i32> for RationalFunction {
    fn from(n: i32) -> RationalFunction {
        Self::from_polynomial(n.into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Monomial;
    use super::*;

    use crate::rewrite_system::rise::{Index, Kind};

    // Helper to create an Index
    fn idx(n: u32) -> Index {
        Index::new(n, Kind::Nat)
    }

    // -------------------------------------
    // Basic RationalFunction tests
    // -------------------------------------

    #[test]
    fn rf_from_polynomial() {
        let poly = Polynomial::new()
            .add_term(3.into(), Monomial::new().with_var(idx(1), 2))
            .add_term(2.into(), Monomial::new().with_var(idx(1), 1));

        let rf = RationalFunction::from_polynomial(poly.clone());
        println!("3x^2 + 2x: {rf}");

        assert!(rf.is_polynomial());
        assert_eq!(rf.numerator, poly);
        assert_eq!(rf.to_string(), "3*x_1^2 + 2*x_1");
    }

    #[test]
    fn rf_simple_division() {
        // (6x^2) / (2x) = 3x
        let numer = Polynomial::new().add_term(6.into(), Monomial::new().with_var(idx(1), 2));
        let denom = Polynomial::new().add_term(2.into(), Monomial::new().with_var(idx(1), 1));

        let rf = RationalFunction::new(numer, denom).unwrap();
        println!("(6x^2) / (2x) = {rf}");

        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "3*x_1");
    }

    #[test]
    fn rf_non_simplifiable() {
        // 1 / (x + 1) cannot be simplified to a polynomial
        let numer = Polynomial::one();
        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(numer, denom).unwrap();
        println!("1 / (x + 1) = {rf}");

        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "1 / (x_1 + 1)");
    }

    #[test]
    fn rf_complex_fraction() {
        // (x^2 + 2x + 1) / (x + 1) - this would simplify if we had polynomial factoring
        // For now, it stays as a fraction
        let numer = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 2))
            .add_term(2.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(numer, denom).unwrap();
        println!("(x^2 + 2x + 1) / (x + 1) = {rf}");

        // Without polynomial factoring, this won't simplify to (x + 1)
        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "(x_1^2 + 2*x_1 + 1) / (x_1 + 1)");
    }

    // -------------------------------------
    // Arithmetic tests
    // -------------------------------------

    #[test]
    fn rf_addition_same_denom() {
        // (1/x) + (2/x) = 3/x = 3*x^(-1)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(2.into(), Polynomial::var(idx(1))).unwrap();

        let result = rf1 + rf2;
        println!("(1/x) + (2/x) = {result}");

        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "3*x_1^(-1)");
    }

    #[test]
    fn rf_addition_different_denom() {
        // (1/x) + (1/y) = (y + x) / (xy)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(2))).unwrap();

        let result = rf1 + rf2;
        println!("(1/x) + (1/y) = {result}");

        // This simplifies to x^(-1) + y^(-1) as a polynomial
        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "x_1^(-1) + x_2^(-1)");
    }

    #[test]
    fn rf_addition_poly_denom() {
        // 1/(x+1) + 1/(x+1) = 2/(x+1)
        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf1 = RationalFunction::new(Polynomial::one(), denom.clone()).unwrap();
        let rf2 = RationalFunction::new(Polynomial::one(), denom).unwrap();

        let result = rf1 + rf2;
        println!("1/(x+1) + 1/(x+1) = {result}");

        assert!(!result.is_polynomial());
        assert_eq!(result.to_string(), "2 / (x_1 + 1)");
    }

    #[test]
    fn rf_multiplication() {
        // (1/x) * (x/y) = 1/y = y^(-1)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(Polynomial::var(idx(1)), Polynomial::var(idx(2))).unwrap();

        let result = rf1 * rf2;
        println!("(1/x) * (x/y) = {result}");

        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "x_2^(-1)");
    }

    #[test]
    fn rf_division() {
        // (x/y) / (x/z) = (x/y) * (z/x) = z/y
        let rf1 = RationalFunction::new(Polynomial::var(idx(1)), Polynomial::var(idx(2))).unwrap();

        let rf2 = RationalFunction::new(Polynomial::var(idx(1)), Polynomial::var(idx(3))).unwrap();

        let result = (rf1 / rf2).unwrap();
        println!("(x/y) / (x/z) = {result}");

        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "x_2^(-1) * x_3");
    }

    #[test]
    fn rf_power_positive() {
        // (1/(x+1))^2 = 1/(x+1)^2
        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(Polynomial::one(), denom).unwrap();
        let result = rf.pow(2).unwrap();
        println!("(1/(x+1))^2 = {result}");

        assert!(!result.is_polynomial());
        assert_eq!(result.to_string(), "1 / (x_1^2 + 2*x_1 + 1)");
    }

    #[test]
    fn rf_power_negative() {
        // (1/(x+1))^(-1) = (x+1)/1 = x+1
        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(Polynomial::one(), denom).unwrap();
        let result = rf.pow(-1).unwrap();
        println!("(1/(x+1))^(-1) = {result}");

        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "x_1 + 1");
    }

    // -------------------------------------
    // RecExpr conversion tests
    // -------------------------------------

    #[test]
    fn rf_roundtrip_polynomial() {
        // Test roundtrip for a simple polynomial
        let poly = Polynomial::new()
            .add_term(3.into(), Monomial::new().with_var(idx(1), 2))
            .add_term(2.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(5.into(), Monomial::new());

        let rf = RationalFunction::from_polynomial(poly);
        println!("Original RF: {rf}");

        let expr: RecExpr<Rise> = rf.clone().into();
        println!("As RecExpr: {expr}");

        let rf_back: RationalFunction = expr.try_into().unwrap();
        println!("Back to RF: {rf_back}");

        assert_eq!(rf, rf_back);
    }

    #[test]
    fn rf_roundtrip_fraction() {
        // Test roundtrip for a rational function
        let numer = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 2))
            .add_term(1.into(), Monomial::new());

        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(numer, denom).unwrap();
        println!("Original RF: {rf}");

        let expr: RecExpr<Rise> = rf.clone().into();
        println!("As RecExpr: {expr}");

        let rf_back: RationalFunction = expr.try_into().unwrap();
        println!("Back to RF: {rf_back}");

        assert_eq!(rf, rf_back);
    }

    #[test]
    fn rf_parse_division_expr() {
        // Parse: x / (x + 1)
        let mut expr = RecExpr::default();

        let x1 = expr.add(Rise::Var(idx(1)));
        let x2 = expr.add(Rise::Var(idx(1)));
        let one = expr.add(Rise::Integer(1));
        let x_plus_1 = expr.add(Rise::NatAdd([x2, one]));
        expr.add(Rise::NatDiv([x1, x_plus_1]));

        let rf: RationalFunction = expr.try_into().unwrap();
        println!("Parsed: x / (x + 1) = {rf}");

        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "x_1 / (x_1 + 1)");
    }

    #[test]
    fn rf_parse_complex() {
        // Parse: (x^2 + 1) / (x - 1)
        let mut expr = RecExpr::default();

        // Numerator: x^2 + 1
        let x1 = expr.add(Rise::Var(idx(1)));
        let two = expr.add(Rise::Integer(2));
        let x_sq = expr.add(Rise::NatPow([x1, two]));
        let one1 = expr.add(Rise::Integer(1));
        let numer = expr.add(Rise::NatAdd([x_sq, one1]));

        // Denominator: x - 1
        let x2 = expr.add(Rise::Var(idx(1)));
        let one2 = expr.add(Rise::Integer(1));
        let denom = expr.add(Rise::NatSub([x2, one2]));

        expr.add(Rise::NatDiv([numer, denom]));

        let rf: RationalFunction = expr.try_into().unwrap();
        println!("Parsed: (x^2 + 1) / (x - 1) = {rf}");

        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "(x_1^2 + 1) / (x_1 - 1)");
    }

    #[test]
    fn rf_parse_negative_power() {
        // Parse: (x + 1)^(-1)
        let mut expr = RecExpr::default();

        let x = expr.add(Rise::Var(idx(1)));
        let one = expr.add(Rise::Integer(1));
        let x_plus_1 = expr.add(Rise::NatAdd([x, one]));
        let neg_one = expr.add(Rise::Integer(-1));
        expr.add(Rise::NatPow([x_plus_1, neg_one]));

        let rf: RationalFunction = expr.try_into().unwrap();
        println!("Parsed: (x + 1)^(-1) = {rf}");

        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "1 / (x_1 + 1)");
    }

    #[test]
    fn rf_parse_negative_power_squared() {
        // Parse: (x + 1)^(-2)
        let mut expr = RecExpr::default();

        let x = expr.add(Rise::Var(idx(1)));
        let one = expr.add(Rise::Integer(1));
        let x_plus_1 = expr.add(Rise::NatAdd([x, one]));
        let neg_two = expr.add(Rise::Integer(-2));
        expr.add(Rise::NatPow([x_plus_1, neg_two]));

        let rf: RationalFunction = expr.try_into().unwrap();
        println!("Parsed: (x + 1)^(-2) = {rf}");

        assert!(!rf.is_polynomial());
        assert_eq!(rf.to_string(), "1 / (x_1^2 + 2*x_1 + 1)");
    }

    // -------------------------------------
    // Edge cases
    // -------------------------------------

    #[test]
    fn rf_zero() {
        let rf = RationalFunction::zero();
        assert!(rf.is_zero());
        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "0");
    }

    #[test]
    fn rf_one() {
        let rf = RationalFunction::one();
        assert!(rf.is_one());
        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "1");
    }

    #[test]
    fn rf_divide_by_zero() {
        let rf = RationalFunction::one();
        let zero = RationalFunction::zero();
        assert!((rf / zero).is_err());
    }

    #[test]
    fn rf_invert_zero() {
        let rf = RationalFunction::zero();
        assert!(rf.inv().is_err());
    }

    #[test]
    fn rf_constant_simplification() {
        // 2/4 should simplify to 1/2
        let rf = RationalFunction::new(2.into(), 4.into()).unwrap();

        println!("2/4 simplified = {rf}");
        assert!(rf.is_polynomial());

        if let Some(c) = rf.as_constant() {
            assert_eq!(c, Ratio::new(1, 2));
        } else {
            panic!("Expected constant");
        }
    }

    #[test]
    fn rf_sign_normalization() {
        // 1/(-x) should normalize to (-1)/x or -x^(-1)
        let rf = RationalFunction::new(
            Polynomial::one(),
            Polynomial::constant(-1) * Polynomial::var(idx(1)),
        )
        .unwrap();

        println!("1/(-x) = {rf}");

        // After simplification, should be -x^(-1)
        assert!(rf.is_polynomial());
    }

    // -------------------------------------
    // Equality tests
    // -------------------------------------

    #[test]
    fn rf_equality_simple() {
        let rf1 = RationalFunction::new(2.into(), 4.into()).unwrap();
        let rf2 = RationalFunction::new(1.into(), 2.into()).unwrap();
        assert_eq!(rf1, rf2);
    }

    #[test]
    fn rf_equality_cross_multiply() {
        // x/(x+1) == x/(x+1)
        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf1 = RationalFunction::new(Polynomial::var(idx(1)), denom.clone()).unwrap();
        let rf2 = RationalFunction::new(Polynomial::var(idx(1)), denom).unwrap();
        assert_eq!(rf1, rf2);
    }
}
