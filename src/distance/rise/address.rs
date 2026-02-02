//! Memory address spaces in Rise.

use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp};

use super::ParseError;
use super::label::RiseLabel;
use crate::distance::tree::TreeNode;

/// Memory address spaces in Rise (for GPU programming).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Address {
    /// Variable: $a<index>
    Var(usize),
    /// Global memory
    Global,
    /// Local/shared memory
    Local,
    /// Private/register memory
    Private,
    /// Constant memory
    Constant,
}

impl Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::Var(i) => write!(f, "$a{i}"),
            Address::Global => write!(f, "global"),
            Address::Local => write!(f, "local"),
            Address::Private => write!(f, "private"),
            Address::Constant => write!(f, "constant"),
        }
    }
}

impl IntoSexp for Address {
    fn into_sexp(&self) -> Sexp {
        Sexp::String(self.to_string())
    }
}

impl Address {
    /// Convert this address to a `RiseLabel`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            Address::Var(i) => RiseLabel::AddrVar(*i),
            Address::Global => RiseLabel::Global,
            Address::Local => RiseLabel::Local,
            Address::Private => RiseLabel::Private,
            Address::Constant => RiseLabel::Constant,
        }
    }

    /// Convert this address to a `TreeNode<RiseLabel>`.
    #[must_use]
    pub fn to_tree(&self) -> TreeNode<RiseLabel> {
        TreeNode::leaf(self.to_label())
    }
}

/// Parse an address from an S-expression.
pub fn parse_address(sexp: &Sexp) -> Result<Address, ParseError> {
    match sexp {
        Sexp::String(s) => parse_address_atom(s),
        _ => Err(ParseError("expected address atom".to_owned())),
    }
}

fn parse_address_atom(s: &str) -> Result<Address, ParseError> {
    // Address variable: $a<index>
    if let Some(rest) = s.strip_prefix("$a") {
        let idx = rest
            .parse::<usize>()
            .map_err(|e| ParseError(format!("invalid variable index: {s} ({e})")))?;
        return Ok(Address::Var(idx));
    }

    match s {
        "global" => Ok(Address::Global),
        "local" => Ok(Address::Local),
        "private" => Ok(Address::Private),
        "constant" => Ok(Address::Constant),
        _ => Err(ParseError(format!("unknown address: {s}"))),
    }
}
