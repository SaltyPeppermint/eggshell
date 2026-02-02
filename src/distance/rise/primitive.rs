//! Built-in primitives in Rise.

use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp};

/// Built-in primitives in Rise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Primitive {
    // Core
    Map,
    Reduce,
    Zip,
    Fst,
    Snd,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Not,
    // Comparison
    Gt,
    Lt,
    Equal,
    Select,
    // Identity and let
    Id,
    Let,
    // Array operations
    Transpose,
    Join,
    Split,
    Slide,
    Take,
    Drop,
    Concat,
    // Pairs
    MakePair,
    MapFst,
    MapSnd,
    // Sequential operations
    MapSeq,
    MapSeqUnroll,
    ReduceSeq,
    ReduceSeqUnroll,
    ScanSeq,
    Iterate,
    // Memory
    ToMem,
    // Indexing
    Idx,
    Cast,
    Generate,
    Gather,
    Scatter,
    Unzip,
    // Padding
    PadCst,
    PadClamp,
    PadEmpty,
    // Other
    Partition,
    Reorder,
    CircularBuffer,
    RotateValues,
    // Vectorization
    AsVector,
    AsVectorAligned,
    AsScalar,
    VectorFromScalar,
    // Nat/Index conversions
    IndexAsNat,
    NatAsIndex,
    // Dependent types
    DepJoin,
    DepMapSeq,
    DepSlide,
    DepTile,
    DepZip,
    Dmatch,
    // Streaming
    IterateStream,
    MapStream,
    MakeDepPair,
    // Other primitive (fallback for unrecognized names)
    Other(String),
}

impl Primitive {
    /// Parse a primitive from its name.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "map" => Primitive::Map,
            "reduce" => Primitive::Reduce,
            "zip" => Primitive::Zip,
            "fst" => Primitive::Fst,
            "snd" => Primitive::Snd,
            "add" => Primitive::Add,
            "sub" => Primitive::Sub,
            "mul" => Primitive::Mul,
            "div" => Primitive::Div,
            "mod" => Primitive::Mod,
            "neg" => Primitive::Neg,
            "not" => Primitive::Not,
            "gt" => Primitive::Gt,
            "lt" => Primitive::Lt,
            "equal" => Primitive::Equal,
            "select" => Primitive::Select,
            "id" => Primitive::Id,
            "let" => Primitive::Let,
            "transpose" => Primitive::Transpose,
            "join" => Primitive::Join,
            "split" => Primitive::Split,
            "slide" => Primitive::Slide,
            "take" => Primitive::Take,
            "drop" => Primitive::Drop,
            "concat" => Primitive::Concat,
            "makePair" => Primitive::MakePair,
            "mapFst" => Primitive::MapFst,
            "mapSnd" => Primitive::MapSnd,
            "mapSeq" => Primitive::MapSeq,
            "mapSeqUnroll" => Primitive::MapSeqUnroll,
            "reduceSeq" => Primitive::ReduceSeq,
            "reduceSeqUnroll" => Primitive::ReduceSeqUnroll,
            "scanSeq" => Primitive::ScanSeq,
            "iterate" => Primitive::Iterate,
            "toMem" => Primitive::ToMem,
            "idx" => Primitive::Idx,
            "cast" => Primitive::Cast,
            "generate" => Primitive::Generate,
            "gather" => Primitive::Gather,
            "scatter" => Primitive::Scatter,
            "unzip" => Primitive::Unzip,
            "padCst" => Primitive::PadCst,
            "padClamp" => Primitive::PadClamp,
            "padEmpty" => Primitive::PadEmpty,
            "partition" => Primitive::Partition,
            "reorder" => Primitive::Reorder,
            "circularBuffer" => Primitive::CircularBuffer,
            "rotateValues" => Primitive::RotateValues,
            "asVector" => Primitive::AsVector,
            "asVectorAligned" => Primitive::AsVectorAligned,
            "asScalar" => Primitive::AsScalar,
            "vectorFromScalar" => Primitive::VectorFromScalar,
            "indexAsNat" => Primitive::IndexAsNat,
            "natAsIndex" => Primitive::NatAsIndex,
            "depJoin" => Primitive::DepJoin,
            "depMapSeq" => Primitive::DepMapSeq,
            "depSlide" => Primitive::DepSlide,
            "depTile" => Primitive::DepTile,
            "depZip" => Primitive::DepZip,
            "dmatch" => Primitive::Dmatch,
            "iterateStream" => Primitive::IterateStream,
            "mapStream" => Primitive::MapStream,
            "makeDepPair" => Primitive::MakeDepPair,
            other => Primitive::Other(other.to_owned()),
        }
    }

    /// Get the name of this primitive.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Primitive::Map => "map",
            Primitive::Reduce => "reduce",
            Primitive::Zip => "zip",
            Primitive::Fst => "fst",
            Primitive::Snd => "snd",
            Primitive::Add => "add",
            Primitive::Sub => "sub",
            Primitive::Mul => "mul",
            Primitive::Div => "div",
            Primitive::Mod => "mod",
            Primitive::Neg => "neg",
            Primitive::Not => "not",
            Primitive::Gt => "gt",
            Primitive::Lt => "lt",
            Primitive::Equal => "equal",
            Primitive::Select => "select",
            Primitive::Id => "id",
            Primitive::Let => "let",
            Primitive::Transpose => "transpose",
            Primitive::Join => "join",
            Primitive::Split => "split",
            Primitive::Slide => "slide",
            Primitive::Take => "take",
            Primitive::Drop => "drop",
            Primitive::Concat => "concat",
            Primitive::MakePair => "makePair",
            Primitive::MapFst => "mapFst",
            Primitive::MapSnd => "mapSnd",
            Primitive::MapSeq => "mapSeq",
            Primitive::MapSeqUnroll => "mapSeqUnroll",
            Primitive::ReduceSeq => "reduceSeq",
            Primitive::ReduceSeqUnroll => "reduceSeqUnroll",
            Primitive::ScanSeq => "scanSeq",
            Primitive::Iterate => "iterate",
            Primitive::ToMem => "toMem",
            Primitive::Idx => "idx",
            Primitive::Cast => "cast",
            Primitive::Generate => "generate",
            Primitive::Gather => "gather",
            Primitive::Scatter => "scatter",
            Primitive::Unzip => "unzip",
            Primitive::PadCst => "padCst",
            Primitive::PadClamp => "padClamp",
            Primitive::PadEmpty => "padEmpty",
            Primitive::Partition => "partition",
            Primitive::Reorder => "reorder",
            Primitive::CircularBuffer => "circularBuffer",
            Primitive::RotateValues => "rotateValues",
            Primitive::AsVector => "asVector",
            Primitive::AsVectorAligned => "asVectorAligned",
            Primitive::AsScalar => "asScalar",
            Primitive::VectorFromScalar => "vectorFromScalar",
            Primitive::IndexAsNat => "indexAsNat",
            Primitive::NatAsIndex => "natAsIndex",
            Primitive::DepJoin => "depJoin",
            Primitive::DepMapSeq => "depMapSeq",
            Primitive::DepSlide => "depSlide",
            Primitive::DepTile => "depTile",
            Primitive::DepZip => "depZip",
            Primitive::Dmatch => "dmatch",
            Primitive::IterateStream => "iterateStream",
            Primitive::MapStream => "mapStream",
            Primitive::MakeDepPair => "makeDepPair",
            Primitive::Other(s) => s,
        }
    }
}

impl Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl IntoSexp for Primitive {
    fn into_sexp(&self) -> Sexp {
        Sexp::String(self.name().to_owned())
    }
}
