use std::{collections::BTreeMap, fmt};

use egg::{Id, RecExpr};
use serde::{Deserialize, Serialize};

use super::Rise;
use crate::rewrite_system::rise::Index;

/// Represents a monomial term's variables and their exponents
/// We use `BTreeMap` to keep variables sorted for canonical form
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct Monomial {
    // Map from dbindex to exponent (can be negative for rational expressions)
    variables: BTreeMap<Index, i32>,
}

impl Monomial {
    pub fn new() -> Self {
        Monomial {
            variables: BTreeMap::new(),
        }
    }

    pub fn variables(&self) -> &BTreeMap<Index, i32> {
        &self.variables
    }

    pub fn with_var(mut self, dbindex: Index, exponent: i32) -> Self {
        if exponent != 0 {
            self.variables.insert(dbindex, exponent);
        }
        self
    }

    pub fn is_constant(&self) -> bool {
        self.variables.is_empty()
    }

    pub fn inv(mut self) -> Self {
        for exp in self.variables.values_mut() {
            *exp = -*exp;
        }
        self
    }

    /// Convert monomial to egg `RecExpr`
    /// Returns the Id of the root node representing this monomial
    pub fn append_to_expr(&self, expr: &mut RecExpr<Rise>) -> Id {
        // Build the product of all variables with their exponents
        let mut result_id = None;

        for (dbindex, exponent) in &self.variables {
            // Create variable node
            let var_id = expr.add(Rise::Var(*dbindex));
            // Handle the exponent
            let term_id = if *exponent == 1 {
                // x^1 = x
                var_id
            } else {
                // Positive or negative exponent: use pow
                let exp_id = expr.add(Rise::Integer(*exponent));
                expr.add(Rise::NatPow([var_id, exp_id]))
            };
            // Multiply with previous terms
            result_id = Some(match result_id {
                None => term_id,
                Some(prev_id) => expr.add(Rise::NatMul([prev_id, term_id])),
            });
        }
        // If we have never set it to something other than none, the self.var were empty
        // => Constant monomial (1)
        result_id.unwrap_or_else(|| expr.add(Rise::Integer(1)))
    }
}

impl Default for Monomial {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.variables.is_empty() {
            return write!(f, "1");
        }

        let mut first = true;
        for (symbol, exp) in &self.variables {
            if !first {
                write!(f, " * ")?;
            }
            first = false;

            write!(f, "x_{}", symbol.value())?;
            if *exp < 0 {
                write!(f, "^({exp})")?;
            } else if *exp != 1 {
                write!(f, "^{exp}")?;
            }
        }
        Ok(())
    }
}

impl std::ops::Mul<Monomial> for Monomial {
    type Output = Monomial;

    /// Multiply two monomials by adding their exponents
    fn mul(self, rhs: Monomial) -> Self::Output {
        self * (&rhs)
    }
}

#[expect(clippy::suspicious_arithmetic_impl)]
impl std::ops::Mul<&Monomial> for Monomial {
    type Output = Monomial;

    /// Multiply two monomials by adding their exponents
    fn mul(mut self, rhs: &Monomial) -> Self::Output {
        for (var, exp) in &rhs.variables {
            *self.variables.entry(*var).or_insert(0) += exp;
        }
        // Remove any variables with zero exponent
        self.variables.retain(|_, exp| *exp != 0);
        self
    }
}

/// Convert monomial to egg `RecExpr`
/// Returns the Id of the root node representing this monomial
impl From<&Monomial> for RecExpr<Rise> {
    fn from(mon: &Monomial) -> RecExpr<Rise> {
        let mut new = RecExpr::default();
        mon.append_to_expr(&mut new);
        new
    }
}
