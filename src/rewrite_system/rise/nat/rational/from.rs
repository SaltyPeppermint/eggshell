use egg::{Id, Language, RecExpr};
use num::rational::Ratio;

use super::{NatSolverError, Polynomial, RationalFunction, Rise};
use crate::rewrite_system::rise::Index;

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
impl From<Index> for RationalFunction {
    fn from(index: Index) -> Self {
        (Polynomial::var(index)).into()
    }
}
