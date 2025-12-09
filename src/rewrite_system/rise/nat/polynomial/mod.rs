mod from;
mod ops;

use std::collections::{BTreeMap, BTreeSet};

use num::rational::Ratio;
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Serialize};

use super::{Monomial, NatSolverError, Rise};
use crate::rewrite_system::rise::DBIndex;

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
    pub fn var(index: DBIndex) -> Self {
        index.into()
    }

    /// Create a polynomial from a constant integer
    pub fn from_i32(n: i32) -> Self {
        n.into()
    }

    /// Create a polynomial from a constant ratio
    pub fn from_ratio(r: Ratio<i32>) -> Self {
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
            let (monomial, coeff) = self.terms.pop_first().unwrap();
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
            && let Some((mon, coeff)) = self.terms.first_key_value()
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
                Some(g) => gcd_ratio(g, abs_coeff),
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

    /// Compute the GCD of two multivariate polynomials
    /// Uses the recursive variable-by-variable approach
    pub fn gcd(&self, other: &Polynomial) -> Result<Polynomial, NatSolverError> {
        let mut a = self.clone().simplified();
        let mut b = other.clone().simplified();

        // Handle trivial cases
        if a.is_zero() {
            return Ok(b);
        }
        if b.is_zero() {
            return Ok(a);
        }
        if let Some(a_const) = a.as_constant()
            && let Some(b_const) = b.as_constant()
        {
            // GCD of constants
            let gcd = gcd_ratio(a_const, b_const);
            return Ok((gcd).into());
        }

        // Get all variables in both polynomials
        let vars = collect_variables(&[&a, &b]);

        if vars.is_empty() {
            // Both are constants
            let gcd = gcd_ratio(
                a.as_constant().unwrap_or(Ratio::one()),
                b.as_constant().unwrap_or(Ratio::one()),
            );
            return Ok((gcd).into());
        }

        // Factor out content (GCD of coefficients) first
        let content_a = a.content(&vars);
        let content_b = b.content(&vars);
        let content_gcd = gcd_ratio(content_a, content_b);

        if !content_a.is_zero() && !content_a.is_one() {
            a = a.divide_coefficients(content_a);
        }
        if !content_b.is_zero() && !content_b.is_one() {
            b = b.divide_coefficients(content_b);
        }

        // Compute primitive GCD recursively
        let primitive_gcd = Polynomial::multivariate_gcd_recursive(&a, &b, &vars)?;

        // Multiply back the content GCD
        if content_gcd.is_one() {
            Ok(primitive_gcd)
        } else {
            Ok(primitive_gcd * Polynomial::from_ratio(content_gcd))
        }
    }

    /// Compute the content of a polynomial (GCD of all coefficients)
    fn content(&self, _vars: &[DBIndex]) -> Ratio<i32> {
        self.coefficient_gcd()
    }

    /// Check if polynomial divides another exactly
    pub fn divides(&self, other: &Polynomial) -> Result<bool, NatSolverError> {
        if self.is_zero() {
            return Ok(other.is_zero());
        }

        let gcd = self.gcd(other)?;

        // self divides other iff gcd(self, other) is associate to self
        // i.e., self = c * gcd for some constant c
        self.is_associate(&gcd)
    }

    /// Check if two polynomials are associates (differ by a constant factor)
    fn is_associate(&self, other: &Polynomial) -> Result<bool, NatSolverError> {
        if self.is_zero() && other.is_zero() {
            return Ok(true);
        }
        if self.is_zero() || other.is_zero() {
            return Ok(false);
        }

        // They're associates if self/other is a constant
        let vars = collect_variables(&[self, other]);
        if vars.is_empty() {
            return Ok(true); // Both constants
        }

        let main_var = vars[0];
        let self_uni = UnivariateView::from_polynomial(self, main_var)?;
        let other_uni = UnivariateView::from_polynomial(other, main_var)?;

        if self_uni.degree() != other_uni.degree() {
            return Ok(false);
        }

        // Check if ratio of leading coefficients is constant and same for all terms
        let self_lc = self_uni.leading_coefficient();
        let other_lc = other_uni.leading_coefficient();

        if self_lc.is_zero() || other_lc.is_zero() {
            return Ok(self_lc.is_zero() && other_lc.is_zero());
        }

        // For associates, self = c * other, so divide and check if constant
        let quotient = self.gcd(other)?;
        let q1 = Polynomial::exact_divide(self, &quotient, &vars)?;
        let q2 = Polynomial::exact_divide(other, &quotient, &vars)?;

        Ok(q1.is_constant() && q2.is_constant())
    }

    /// Recursive multivariate GCD using the "recursive representation" approach
    /// Treats the polynomial as univariate in the first variable with coefficients
    /// that are polynomials in the remaining variables
    fn multivariate_gcd_recursive(
        a: &Polynomial,
        b: &Polynomial,
        vars: &[DBIndex],
    ) -> Result<Polynomial, NatSolverError> {
        if vars.is_empty() {
            // Base case: both are constants
            let const_a = a.as_constant().unwrap_or(Ratio::one());
            let const_b = b.as_constant().unwrap_or(Ratio::one());
            return Ok((gcd_ratio(const_a, const_b)).into());
        }

        let main_var = vars[0];
        let remaining_vars = &vars[1..];

        // Convert to univariate representation over the main variable
        let a_uni = UnivariateView::from_polynomial(a, main_var)?;
        let b_uni = UnivariateView::from_polynomial(b, main_var)?;

        // Compute GCD of contents (coefficients as polynomials in remaining vars)
        let content_gcd = UnivariateView::content_gcd(&a_uni, &b_uni, remaining_vars)?;

        // Make primitive (divide by content)
        let a_primitive = a_uni.make_primitive(remaining_vars)?;
        let b_primitive = b_uni.make_primitive(remaining_vars)?;

        // Compute primitive part GCD using subresultant PRS
        let primitive_gcd = UnivariateView::univariate_gcd_subresultant(
            &a_primitive,
            &b_primitive,
            main_var,
            remaining_vars,
        )?;

        // Result = content_gcd * primitive_gcd
        Ok(content_gcd * primitive_gcd)
    }

    /// Exact polynomial division (assumes divisibility)
    pub fn exact_divide(
        a: &Polynomial,
        b: &Polynomial,
        vars: &[DBIndex],
    ) -> Result<Polynomial, NatSolverError> {
        if b.is_one() {
            return Ok(a.clone());
        }
        if b.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }

        // For constants, do direct division
        if a.is_constant() && b.is_constant() {
            let ca = a.as_constant().unwrap();
            let cb = b.as_constant().unwrap();
            return Ok((ca / cb).into());
        }

        if vars.is_empty() {
            let ca = a.as_constant().unwrap_or(Ratio::one());
            let cb = b.as_constant().unwrap_or(Ratio::one());
            return Ok((ca / cb).into());
        }

        // Use polynomial long division
        let main_var = vars[0];
        let a_uni = UnivariateView::from_polynomial(a, main_var)?;
        let b_uni = UnivariateView::from_polynomial(b, main_var)?;

        let (quotient, remainder) = UnivariateView::polynomial_divide(&a_uni, &b_uni, &vars[1..])?;

        debug_assert!(
            remainder.is_zero(),
            "exact_divide: non-zero remainder in exact division"
        );

        quotient.to_polynomial(main_var)
    }
}

// ============================================================================
// Univariate View - treats polynomial as univariate in one variable
// ============================================================================

/// View a multivariate polynomial as univariate in one variable
/// with coefficients being polynomials in the remaining variables
#[derive(Debug, Clone, Default)]
pub struct UnivariateView {
    /// Map from degree to coefficient polynomial
    coefficients: BTreeMap<usize, Polynomial>,
}

impl UnivariateView {
    fn new() -> Self {
        Self::default()
    }

    fn zero() -> Self {
        Self::new()
    }

    fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.values().all(|c| c.is_zero())
    }

    fn is_constant_view(&self) -> bool {
        self.coefficients.keys().all(|&d| d == 0)
    }

    /// Convert a polynomial to univariate view in the given variable
    pub fn from_polynomial(poly: &Polynomial, var: DBIndex) -> Result<Self, NatSolverError> {
        let mut result = Self::new();

        for (monomial, coeff) in poly.terms() {
            // Extract the degree of the main variable
            let degree = monomial.variables().get(&var).copied().unwrap_or(0);

            // Create the remaining monomial (without the main variable)
            let remaining_monomial = Monomial::new().with_variables_except(monomial, var);

            // Add to the coefficient polynomial for this degree
            let coeff_poly = result.coefficients.entry(degree.try_into()?).or_default();

            *coeff_poly = std::mem::take(coeff_poly).add_term(*coeff, remaining_monomial);
        }

        result.normalize();
        Ok(result)
    }

    /// Convert back to a polynomial
    fn to_polynomial(&self, var: DBIndex) -> Result<Polynomial, NatSolverError> {
        let mut result = Polynomial::new();

        for (&degree, coeff_poly) in &self.coefficients {
            for (monomial, coeff) in coeff_poly.terms() {
                // Add the main variable with its degree
                let full_monomial = monomial.clone().with_var(var, degree.try_into()?);
                result = result.add_term(*coeff, full_monomial);
            }
        }

        Ok(result.simplified())
    }

    /// Get the degree (highest power of the main variable)
    pub fn degree(&self) -> usize {
        self.coefficients
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(&d, _)| d)
            .max()
            .unwrap_or(0)
    }

    /// Get the leading coefficient (coefficient of highest degree term)
    pub fn leading_coefficient(&self) -> Polynomial {
        let deg = self.degree();
        self.coefficients
            .get(&deg)
            .cloned()
            .unwrap_or_else(Polynomial::new)
    }

    /// Set coefficient for a given degree
    fn set_coefficient(&mut self, degree: usize, coeff: Polynomial) {
        if coeff.is_zero() {
            self.coefficients.remove(&degree);
        } else {
            self.coefficients.insert(degree, coeff);
        }
    }

    /// Subtract from coefficient at given degree
    fn subtract_from_coefficient(&mut self, degree: usize, subtract: &Polynomial) {
        let current = self.coefficients.entry(degree).or_default();
        *current = std::mem::take(current) - subtract.clone();
    }

    /// Multiply all coefficients by a polynomial
    fn multiply_coefficients_by(&self, factor: &Polynomial) -> Self {
        let mut result = Self::new();
        for (&deg, coeff) in &self.coefficients {
            let new_coeff = coeff.clone() * factor;
            if !new_coeff.is_zero() {
                result.coefficients.insert(deg, new_coeff);
            }
        }
        result
    }

    /// Divide all coefficients by a polynomial (assumes exact division)
    fn divide_coefficients_by(
        &self,
        divisor: &Polynomial,
        remaining_vars: &[DBIndex],
    ) -> Result<Self, NatSolverError> {
        let mut result = Self::new();
        for (&deg, coeff) in &self.coefficients {
            if !coeff.is_zero() {
                let new_coeff = Polynomial::exact_divide(coeff, divisor, remaining_vars)?;
                if !new_coeff.is_zero() {
                    result.coefficients.insert(deg, new_coeff);
                }
            }
        }
        Ok(result)
    }

    /// Make primitive (divide by content)
    fn make_primitive(&self, remaining_vars: &[DBIndex]) -> Result<Self, NatSolverError> {
        let content = self.content(remaining_vars)?;
        if content.is_one() || content.is_zero() {
            return Ok(self.clone());
        }
        self.divide_coefficients_by(&content, remaining_vars)
    }

    /// Compute content (GCD of all coefficients)
    fn content(&self, remaining_vars: &[DBIndex]) -> Result<Polynomial, NatSolverError> {
        let coeffs: Vec<&Polynomial> = self.coefficients.values().collect();

        if coeffs.is_empty() {
            return Ok(Polynomial::one());
        }

        let mut result = coeffs[0].clone();
        for coeff in coeffs.iter().skip(1) {
            if remaining_vars.is_empty() {
                let c1 = result.as_constant().unwrap_or(Ratio::one());
                let c2 = coeff.as_constant().unwrap_or(Ratio::one());
                result = (gcd_ratio(c1, c2)).into();
            } else {
                result = Polynomial::multivariate_gcd_recursive(&result, coeff, remaining_vars)?;
            }

            if result.is_one() {
                break;
            }
        }

        if result.is_zero() {
            Ok(Polynomial::one())
        } else {
            Ok(result)
        }
    }

    /// Compute GCD of the contents of two univariate views
    fn content_gcd(
        a: &UnivariateView,
        b: &UnivariateView,
        remaining_vars: &[DBIndex],
    ) -> Result<Polynomial, NatSolverError> {
        // Todo: Fixme
        // let mut result = Polynomial::new();
        let mut result;

        // Collect all coefficients
        let all_coeffs: Vec<&Polynomial> = a
            .coefficients
            .values()
            .chain(b.coefficients.values())
            .collect();

        if all_coeffs.is_empty() {
            return Ok(Polynomial::one());
        }

        // Compute GCD of all coefficients recursively
        result = all_coeffs[0].clone();
        for coeff in all_coeffs.iter().skip(1) {
            if remaining_vars.is_empty() {
                // Constants
                let c1 = result.as_constant().unwrap_or(Ratio::one());
                let c2 = coeff.as_constant().unwrap_or(Ratio::one());
                result = (gcd_ratio(c1, c2)).into();
            } else {
                result = Polynomial::multivariate_gcd_recursive(&result, coeff, remaining_vars)?;
            }

            if result.is_one() {
                break; // GCD is 1, can't get smaller
            }
        }

        if result.is_zero() {
            Ok(Polynomial::one())
        } else {
            Ok(result)
        }
    }

    /// Univariate GCD using subresultant PRS for better coefficient control
    fn univariate_gcd_subresultant(
        a: &UnivariateView,
        b: &UnivariateView,
        main_var: DBIndex,
        remaining_vars: &[DBIndex],
    ) -> Result<Polynomial, NatSolverError> {
        if a.is_zero() {
            return b.to_polynomial(main_var);
        }
        if b.is_zero() {
            return a.to_polynomial(main_var);
        }

        // Ensure deg(a) >= deg(b)
        let (mut r0, mut r1) = if a.degree() >= b.degree() {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };

        if r1.is_zero() {
            return r0.to_polynomial(main_var);
        }

        // Subresultant PRS
        let mut g = Polynomial::one(); // g_i
        let mut h = Polynomial::one(); // h_i

        loop {
            let delta = r0.degree().saturating_sub(r1.degree());

            // Pseudo-remainder
            let (_, mut rem) = UnivariateView::pseudo_divide(&r0, &r1, remaining_vars)?;

            if rem.is_zero() {
                // r1 divides r0, so r1 is the GCD (after making primitive)
                let result = r1.make_primitive(remaining_vars)?;
                return result.to_polynomial(main_var);
            }

            // If remainder is constant (degree 0), GCD is 1
            if rem.degree() == 0 || rem.is_constant_view() {
                return Ok(Polynomial::one());
            }

            // Update for subresultant PRS
            // rem = rem / (g * h^delta)
            let divisor = g.clone() * h.clone().pow(delta.try_into()?);
            rem = rem.divide_coefficients_by(&divisor, remaining_vars)?;

            // g = lc(r1)
            g = r1.leading_coefficient();

            // h = h^(1-delta) * g^delta  (when delta > 0)
            // For delta = 1: h = g
            // For delta > 1: h = g^delta / h^(delta-1)
            if delta == 0 {
                // h stays the same
            } else if delta == 1 {
                h = g.clone();
            } else {
                // h = g^delta / h^(delta-1)
                let g_pow = g.clone().pow(delta.try_into()?);
                let h_pow = h.clone().pow((delta - 1).try_into()?);
                h = Polynomial::exact_divide(&g_pow, &h_pow, remaining_vars)?;
            }

            r0 = r1;
            r1 = rem;
        }
    }

    /// Pseudo-division of univariate polynomials
    /// Returns (quotient, remainder) such that lc(b)^(deg(a)-deg(b)+1) * a = q*b + r
    fn pseudo_divide(
        a: &UnivariateView,
        b: &UnivariateView,
        // main_var: Index,
        remaining_vars: &[DBIndex],
    ) -> Result<(UnivariateView, UnivariateView), NatSolverError> {
        if b.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }

        let mut remainder = a.clone();
        let b_degree = b.degree();
        let b_lc = b.leading_coefficient();

        if a.degree() < b_degree {
            return Ok((UnivariateView::zero(), remainder));
        }

        let mut quotient = UnivariateView::zero();
        let iterations = a.degree() - b_degree + 1;

        // Multiply remainder by lc(b)^iterations upfront for pseudo-division
        let multiplier = b_lc.clone().pow(iterations.try_into()?);
        remainder = remainder.multiply_coefficients_by(&multiplier);

        for _ in 0..iterations {
            if remainder.degree() < b_degree {
                break;
            }

            let degree_diff = remainder.degree() - b_degree;
            let r_lc = remainder.leading_coefficient();

            // q_term = (lc(r) / lc(b)) * x^(degree_diff)
            // But since we multiplied by lc(b)^iterations, we use lc(r) directly
            let q_coeff = Polynomial::exact_divide(&r_lc, &b_lc, remaining_vars)?;

            quotient.set_coefficient(degree_diff, q_coeff.clone());

            // r = r - q_term * b
            for (deg, coeff) in &b.coefficients {
                let target_deg = deg + degree_diff;
                let subtract = q_coeff.clone() * coeff.clone();
                remainder.subtract_from_coefficient(target_deg, &subtract);
            }
        }

        remainder.normalize();
        quotient.normalize();

        Ok((quotient, remainder))
    }

    /// Standard polynomial long division
    fn polynomial_divide(
        a: &UnivariateView,
        b: &UnivariateView,
        // main_var: Index,
        remaining_vars: &[DBIndex],
    ) -> Result<(UnivariateView, UnivariateView), NatSolverError> {
        if b.is_zero() {
            return Err(NatSolverError::DivisionByZero);
        }

        let mut remainder = a.clone();
        let b_degree = b.degree();
        let b_lc = b.leading_coefficient();

        if a.degree() < b_degree {
            return Ok((UnivariateView::zero(), remainder));
        }

        let mut quotient = UnivariateView::zero();

        while !remainder.is_zero() && remainder.degree() >= b_degree {
            let degree_diff = remainder.degree() - b_degree;
            let r_lc = remainder.leading_coefficient();

            // q_coeff = lc(r) / lc(b)
            let q_coeff = Polynomial::exact_divide(&r_lc, &b_lc, remaining_vars)?;

            quotient.set_coefficient(degree_diff, q_coeff.clone());

            // r = r - q_term * b
            for (deg, coeff) in &b.coefficients {
                let target_deg = deg + degree_diff;
                let subtract = q_coeff.clone() * coeff.clone();
                remainder.subtract_from_coefficient(target_deg, &subtract);
            }

            remainder.normalize();
        }

        Ok((quotient, remainder))
    }

    /// Remove zero coefficients
    fn normalize(&mut self) {
        self.coefficients.retain(|_, c| !c.is_zero());
    }
}

/// Collect all variables from a set of polynomials
pub fn collect_variables(polys: &[&Polynomial]) -> Vec<DBIndex> {
    let mut vars: BTreeSet<DBIndex> = BTreeSet::new();
    for poly in polys {
        for (monomial, coeff) in poly.terms() {
            if !coeff.is_zero() {
                for var in monomial.variables().keys() {
                    vars.insert(*var);
                }
            }
        }
    }
    vars.into_iter().collect()
}

/// Compute GCD of two rational numbers
pub fn gcd_ratio(a: Ratio<i32>, b: Ratio<i32>) -> Ratio<i32> {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_system::rise::{DBIndex, Kind};

    fn idx(n: u32) -> DBIndex {
        DBIndex::new(n, Kind::Nat)
    }

    fn x() -> Polynomial {
        Polynomial::var(idx(1))
    }

    fn y() -> Polynomial {
        Polynomial::var(idx(2))
    }

    fn z() -> Polynomial {
        Polynomial::var(idx(3))
    }

    fn constant(n: i32) -> Polynomial {
        Polynomial::from_i32(n)
    }

    #[test]
    fn gcd_constants() {
        let a = constant(12);
        let b = constant(8);
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(12, 8) = {gcd}");
        assert_eq!(gcd.as_constant(), Some(Ratio::from(4)));
    }

    #[test]
    fn gcd_univariate_simple() {
        // gcd(x^2, x) = x
        let a = x() * x();
        let b = x();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(x^2, x) = {gcd}");
        assert_eq!(gcd.to_string(), "x_1");
    }

    #[test]
    fn gcd_univariate_linear() {
        // gcd(x^2 - 1, x - 1) = x - 1
        // x^2 - 1 = (x-1)(x+1)
        let a = x() * x() - constant(1);
        let b = x() - constant(1);
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(x^2 - 1, x - 1) = {gcd}");
        // Should be x - 1 (or 1 - x, which is associate)
        assert!(gcd.to_string() == "x_1 - 1" || gcd.to_string() == "-x_1 + 1");
    }

    #[test]
    fn gcd_univariate_quadratic() {
        // gcd(x^2 + 2x + 1, x + 1) = x + 1
        // x^2 + 2x + 1 = (x+1)^2
        let a = x() * x() + constant(2) * x() + constant(1);
        let b = x() + constant(1);
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(x^2 + 2x + 1, x + 1) = {gcd}");
        assert!(gcd.to_string() == "x_1 + 1" || gcd.to_string() == "-x_1 - 1");
    }

    #[test]
    fn gcd_coprime() {
        // gcd(x + 1, x + 2) = 1
        let a = x() + constant(1);
        let b = x() + constant(2);
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(x + 1, x + 2) = {gcd}");
        assert!(gcd.is_constant());
    }

    #[test]
    fn gcd_bivariate_simple() {
        // gcd(xy, x) = x
        let a = x() * y();
        let b = x();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(xy, x) = {gcd}");
        assert_eq!(gcd.to_string(), "x_1");
    }

    #[test]
    fn gcd_bivariate_common_factor() {
        // gcd(x^2*y, x*y^2) = xy
        let a = x() * x() * y();
        let b = x() * y() * y();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(x^2*y, x*y^2) = {gcd}");
        // Should contain xy
        assert!(gcd.to_string().contains("x_1") && gcd.to_string().contains("x_2"));
    }

    #[test]
    fn gcd_bivariate_polynomial() {
        // gcd((x+y)^2, (x+y)) = (x+y)
        let xy = x() + y();
        let a = xy.clone() * xy.clone();
        let b = xy.clone();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd((x+y)^2, (x+y)) = {gcd}");
        // Should be associate to (x + y)
        assert_eq!(gcd.term_count(), 2);
    }

    #[test]
    fn gcd_trivariate() {
        // gcd(xyz, xy) = xy
        let a = x() * y() * z();
        let b = x() * y();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(xyz, xy) = {gcd}");
        assert!(gcd.to_string().contains("x_1") && gcd.to_string().contains("x_2"));
        assert!(!gcd.to_string().contains("x_3"));
    }

    #[test]
    fn gcd_with_coefficients() {
        // gcd(6x^2, 4x) = 2x
        let a = constant(6) * x() * x();
        let b = constant(4) * x();
        let gcd = a.gcd(&b).unwrap();
        println!("gcd(6x^2, 4x) = {gcd}");
        assert!(gcd.to_string().contains("x_1"));
    }

    #[test]
    fn rational_function_simplification() {
        // (x^2 - 1) / (x - 1) should simplify to (x + 1)
        let numer = x() * x() - constant(1);
        let denom = x() - constant(1);

        let rf = super::super::RationalFunction::new(numer, denom)
            .unwrap()
            .simplified()
            .unwrap();

        println!("(x^2 - 1) / (x - 1) = {rf}");
        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "x_1 + 1");
    }

    #[test]
    fn rational_function_perfect_square() {
        // (x^2 + 2x + 1) / (x + 1) should simplify to (x + 1)
        let numer = x() * x() + constant(2) * x() + constant(1);
        let denom = x() + constant(1);

        let rf = super::super::RationalFunction::new(numer, denom)
            .unwrap()
            .simplified()
            .unwrap();

        println!("(x^2 + 2x + 1) / (x + 1) = {rf}");
        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "x_1 + 1");
    }

    #[test]
    fn rational_function_no_simplification() {
        // (x + 1) / (x + 2) cannot be simplified
        let numer = x() + constant(1);
        let denom = x() + constant(2);

        let rf = super::super::RationalFunction::new(numer, denom)
            .unwrap()
            .simplified()
            .unwrap();

        println!("(x + 1) / (x + 2) = {rf}");
        assert!(!rf.is_polynomial());
    }

    #[test]
    fn rational_function_bivariate() {
        // (x*y + x) / (y + 1) = x * (y + 1) / (y + 1) = x
        let numer = x() * y() + x();
        let denom = y() + constant(1);

        let rf = super::super::RationalFunction::new(numer, denom)
            .unwrap()
            .simplified()
            .unwrap();

        println!("(xy + x) / (y + 1) = {rf}");
        assert!(rf.is_polynomial());
        assert_eq!(rf.to_string(), "x_1");
    }
}
