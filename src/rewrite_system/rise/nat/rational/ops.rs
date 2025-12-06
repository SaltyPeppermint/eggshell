use std::fmt;

use super::{NatSolverError, RationalFunction};

// ============================================================================
// RationalFunction Arithmetic
// ============================================================================

impl std::ops::Add<RationalFunction> for RationalFunction {
    type Output = Result<Self, NatSolverError>;

    /// Add two rational functions: a/b + c/d = (ad + bc) / bd
    fn add(self, rhs: RationalFunction) -> Self::Output {
        // Small Optimization: if denominators are equal, just add numerators
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
    type Output = Result<Self, NatSolverError>;

    fn sub(self, rhs: RationalFunction) -> Self::Output {
        self + (-rhs)?
    }
}

impl std::ops::Mul<RationalFunction> for RationalFunction {
    type Output = Result<Self, NatSolverError>;

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
        self * rhs.inv()?
    }
}

impl std::ops::Neg for RationalFunction {
    type Output = Result<Self, NatSolverError>;

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
