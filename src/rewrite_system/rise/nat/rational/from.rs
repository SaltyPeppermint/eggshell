use egg::{Id, RecExpr};
use num::rational::Ratio;

use super::{NatSolverError, Polynomial, RationalFunction, Rise};
use crate::rewrite_system::rise::kind::{Kind, Kindable};
use crate::rewrite_system::rise::{DBIndex, add_expr};

// ============================================================================
// Conversions: Polynomial <-> RationalFunction
// ============================================================================

impl From<Polynomial> for RationalFunction {
    fn from(p: Polynomial) -> Self {
        RationalFunction {
            numerator: p,
            denominator: Polynomial::one(),
        }
    }
}

impl TryFrom<RationalFunction> for Polynomial {
    type Error = NatSolverError;

    fn try_from(rf: RationalFunction) -> Result<Self, Self::Error> {
        {
            let simplified = rf.simplified()?;
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

// TODO: i think this has a bug but not sure
impl From<&RationalFunction> for RecExpr<Rise> {
    fn from(rf: &RationalFunction) -> Self {
        // If it's just a polynomial, convert directly
        if rf.denominator.is_one() {
            return (&rf.numerator).into();
        }

        let mut expr = RecExpr::default();
        // Build numerator expression
        let numer_expr: RecExpr<Rise> = (&rf.numerator).into();
        let numer_root = add_expr(&mut expr, numer_expr);

        // Build denominator expression
        let denom_expr: RecExpr<Rise> = (&rf.denominator).into();
        let denom_root = add_expr(&mut expr, denom_expr);

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
                Rise::Var(index) if index.kind() == Kind::Nat => Ok((*index).into()),
                Rise::NatAdd([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    left_rf + right_rf
                }
                Rise::NatSub([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    left_rf - right_rf
                }
                Rise::NatMul([left, right]) => {
                    let left_rf = rec(expr, *left)?;
                    let right_rf = rec(expr, *right)?;
                    left_rf * right_rf
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
        rec(expr, root_id)?.simplified()
    }
}

// ============================================================================
// From Simple Types
// ============================================================================

/// Create a `RationalFunction` from an integer constant
impl From<i32> for RationalFunction {
    fn from(n: i32) -> RationalFunction {
        Polynomial::from_i32(n).into()
    }
}

/// Create a `RationalFunction` from an integer constant
impl From<Ratio<i32>> for RationalFunction {
    fn from(r: Ratio<i32>) -> Self {
        Polynomial::from_ratio(r).into()
    }
}

/// Create a `RationalFunction` from with a single variable
impl From<DBIndex> for RationalFunction {
    fn from(index: DBIndex) -> Self {
        (Polynomial::var(index)).into()
    }
}
