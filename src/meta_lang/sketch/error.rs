use std::fmt::{Debug, Display};

use egg::FromOp;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SketchError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error(transparent)]
    BadChildren(#[from] egg::FromOpError),
    #[error(transparent)]
    BadOp(L::Error),
}
