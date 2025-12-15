mod from;
mod ops;

use num::rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::{NatSolverError, Polynomial, Rise, polynomial};

// ============================================================================
// RationalFunction
// ============================================================================

/// Represents a rational function as a ratio of two polynomials: numerator / denominator
///
/// This can represent any rational expression including:
/// - Simple polynomials: `x^2 + 2x + 1` (denominator = 1)
/// - Rational expressions: `1 / (x + 1)`
/// - Complex ratios: `(x^2 + 1) / (x^2 - 1)`
#[derive(Debug, Clone)]
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
        RationalFunction {
            numerator,
            denominator,
        }
        .simplified()
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

    /// Simplify by computing GCD of numerator and denominator
    pub fn simplified(mut self) -> Result<Self, NatSolverError> {
        self.numerator.simplify();
        self.denominator.simplify();

        // Handle trivial cases
        if self.numerator.is_zero() {
            return Ok(super::RationalFunction {
                numerator: Polynomial::new(),
                denominator: Polynomial::one(),
            });
        }

        if self.denominator.is_one() {
            return Ok(self);
        }

        // If denominator is a monomial, absorb it
        if self.denominator.is_monomial()
            && let Ok(denom_inv) = self.denominator.clone().try_inv()
        {
            self.numerator = self.numerator * denom_inv;
            self.denominator = Polynomial::one();
            return Ok(self);
        }

        // Compute GCD and cancel
        let gcd = self.numerator.gcd(&self.denominator)?;

        if !gcd.is_one() && !gcd.is_zero() {
            let vars = polynomial::collect_variables(&[&self.numerator, &self.denominator]);
            self.numerator = Polynomial::exact_divide(&self.numerator, &gcd, &vars)?;
            self.denominator = Polynomial::exact_divide(&self.denominator, &gcd, &vars)?;
        }

        // Clean up constants and sign
        self.cancel_constant_factors();
        self.normalize_sign();

        Ok(self)
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
        let common_gcd = polynomial::gcd_ratio(numer_gcd, denom_gcd);

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
        RationalFunction {
            numerator: self.denominator,
            denominator: self.numerator,
        }
        .simplified()
    }

    /// Raise to an integer power (can be negative)
    pub fn pow(self, n: i32) -> Result<Self, NatSolverError> {
        if n == 0 {
            return Ok(RationalFunction::one());
        }

        if n > 0 {
            let exp = n.try_into()?;
            RationalFunction {
                numerator: self.numerator.pow(exp),
                denominator: self.denominator.pow(exp),
            }
            .simplified()
        } else {
            // Negative exponent: invert first, then raise to positive power
            let exp = (-n).try_into()?;
            let inverted = self.inv()?;
            RationalFunction {
                numerator: inverted.numerator.pow(exp),
                denominator: inverted.denominator.pow(exp),
            }
            .simplified()
        }
    }
}

impl Default for RationalFunction {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use super::super::Monomial;
    use super::*;

    use crate::rewrite_system::rise::{DBIndex, Kind};

    // Helper to create an Index
    fn idx(n: u32) -> DBIndex {
        DBIndex::new(Kind::Nat, n)
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
        // (x^2 + 2x + 1) / (x + 1) = x_1 + 1
        let numer = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 2))
            .add_term(2.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let denom = Polynomial::new()
            .add_term(1.into(), Monomial::new().with_var(idx(1), 1))
            .add_term(1.into(), Monomial::new());

        let rf = RationalFunction::new(numer, denom).unwrap();
        println!("(x^2 + 2x + 1) / (x + 1) = {rf}");

        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "x_1 + 1");
    }

    // -------------------------------------
    // Arithmetic tests
    // -------------------------------------

    #[test]
    fn rf_addition_same_denom() {
        // (1/x) + (2/x) = 3/x = 3*x^(-1)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(2.into(), Polynomial::var(idx(1))).unwrap();

        let result = (rf1 + rf2).unwrap();
        println!("(1/x) + (2/x) = {result}");

        assert!(result.is_polynomial());
        assert_eq!(result.to_string(), "3*x_1^(-1)");
    }

    #[test]
    fn rf_addition_different_denom() {
        // (1/x) + (1/y) = (y + x) / (xy)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(2))).unwrap();

        let result = (rf1 + rf2).unwrap();
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

        let result = (rf1 + rf2).unwrap();
        println!("1/(x+1) + 1/(x+1) = {result}");

        assert!(!result.is_polynomial());
        assert_eq!(result.to_string(), "2 / (x_1 + 1)");
    }

    #[test]
    fn rf_multiplication() {
        // (1/x) * (x/y) = 1/y = y^(-1)
        let rf1 = RationalFunction::new(Polynomial::one(), Polynomial::var(idx(1))).unwrap();

        let rf2 = RationalFunction::new(Polynomial::var(idx(1)), Polynomial::var(idx(2))).unwrap();

        let result = (rf1 * rf2).unwrap();
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
            Polynomial::from_i32(-1) * Polynomial::var(idx(1)),
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
