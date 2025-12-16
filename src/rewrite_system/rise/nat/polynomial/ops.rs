use core::fmt;
use std::collections::BTreeMap;

use num_traits::{One, Signed, Zero};

use super::{Monomial, Polynomial, Ratio};

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
