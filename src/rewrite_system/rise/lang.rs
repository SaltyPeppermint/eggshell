use colored::{ColoredString, Colorize};
use egg::{Id, Language, RecExpr};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use super::{Index, Kind, Kindable};

egg::define_language! {
  #[derive(Serialize, Deserialize)]
  pub enum Rise {
    Var(Index),
    "app" = App([Id; 2]),
    "natApp" = NatApp([Id; 2]),
    "dataApp" = DataApp([Id; 2]),
    "addrApp" = AddrApp([Id; 2]),
    "natNatApp" = NatNatApp([Id; 2]),


    "lam" = Lambda(Id),
    "natLam" = NatLambda(Id),
    "dataLam" = DataLambda(Id),
    "addrLam" = AddrLambda(Id),
    "natNatLam" = NatNatLambda(Id),

    "natFun" = NatFun(Id),
    "dataFun" = DataFun(Id),
    "addrFun" = AddrFun(Id),
    "natNatFun" = NatNatFun(Id),

    "let" = Let,

    "typeOf" = TypeOf([Id; 2]),
    "fun" = FunType([Id; 2]),

    "arrT" = ArrType([Id; 2]),
    "vecT" = VecType([Id; 2]),
    "pairT" = PairType([Id; 2]),
    "idxT" = IndexType(Id),
    "natT" = NatType,

    "f32" = F32,

    // Natural number operations
    "natAdd" = NatAdd([Id; 2]),
    "natSub" = NatSub([Id; 2]),
    "natMul" = NatMul([Id; 2]),
    "natDiv" = NatDiv([Id; 2]),
    "natPow" = NatPow([Id; 2]),

    "asVector" = AsVector,
    "asScalar" = AsScalar,
    "vectorFromScalar" = VectorFromScalar,

    "snd" = Snd,
    "fst" = Fst,
    "add" = Add,
    "mul" = Mul,

    "toMem" = ToMem,
    "split" = Split,
    "join" = Join,

    "generate" = Generate,
    "transpose" = Transpose,

    "zip" = Zip,
    "unzip" = Unzip,

    "map" = Map,
    "mapPar" = MapPar,

    "reduce" = Reduce,
    "reduceSeq" = ReduceSeq,
    "reduceSeqUnroll" = ReduceSeqUnroll,

    // to implement explicit substitution:
    // "sig" = Sigma([Id; 3]),
    // "phi" = Phi([Id; 3]),

    Integer(i32),
    Float(NotNan<f32>),
    // Double(f64),
    // Symbol(Symbol),
  }
}

impl Rise {
    pub fn kind(&self) -> Option<Kind> {
        Some(match self {
            Rise::Var(index) => index.kind()?,
            Rise::App(_)
            | Rise::Lambda(_)
            | Rise::Let
            | Rise::AsVector
            | Rise::AsScalar
            | Rise::VectorFromScalar
            | Rise::Snd
            | Rise::Fst
            | Rise::Add
            | Rise::Mul
            | Rise::ToMem
            | Rise::Split
            | Rise::Join
            | Rise::Generate
            | Rise::Transpose
            | Rise::Zip
            | Rise::Unzip
            | Rise::Map
            | Rise::MapPar
            | Rise::Reduce
            | Rise::ReduceSeq
            | Rise::ReduceSeqUnroll
            | Rise::Float(_) => Kind::Expr,
            Rise::NatApp(_)
            | Rise::NatLambda(_)
            | Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_) => Kind::Nat,
            Rise::DataApp(_)
            | Rise::DataLambda(_)
            | Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            | Rise::IndexType(_)
            | Rise::NatType
            | Rise::F32 => Kind::Data,
            Rise::AddrApp(_) | Rise::AddrLambda(_) => Kind::Addr,
            Rise::TypeOf(_) | Rise::NatNatApp(_) | Rise::NatNatLambda(_) | Rise::Integer(_) => {
                return None;
            }
        })
    }
}

pub trait PrettyPrint {
    fn pp(self, skip_wrapper: bool);
}

impl PrettyPrint for &RecExpr<Rise> {
    fn pp(self, skip_wrapper: bool) {
        let tree = PPNode::new(self, self.root(), skip_wrapper);
        tree.pp();
    }
}

struct PPNode {
    children: Box<[PPNode]>,
    expr: ColoredString,
    ty: ColoredString,
}

impl PPNode {
    fn pp(&self) {
        fn rec(node: &PPNode, indent: &mut String) {
            if let Some(shortened) = indent.strip_suffix(' ') {
                println!("{shortened}┗━{}: {}", node.expr, node.ty);
            } else if let Some(shortened) = indent.strip_suffix('┃') {
                println!("{shortened}┣━{}: {}", node.expr, node.ty);
            } else {
                println!("{}: {}", node.expr, node.ty);
            }

            let Some((last, rest)) = node.children.split_last() else {
                return;
            };
            indent.push_str("   ┃");
            for r in rest {
                rec(r, indent);
            }
            indent.pop();
            indent.push(' ');
            rec(last, indent);
            indent.truncate(indent.len() - 4);
        }
        rec(self, &mut String::new());
    }

    fn new(expr: &RecExpr<Rise>, id: Id, skip_wrapper: bool) -> PPNode {
        let Rise::TypeOf([expr_id, ty_id]) = &expr[id] else {
            return Self {
                children: expr[id]
                    .children()
                    .iter()
                    .map(|c_id| Self::new(expr, *c_id, skip_wrapper))
                    .collect(),
                expr: expr[id].to_string().into(),
                ty: ("NO TYPE INFO AVAILABLE").red(),
            };
        };
        let node = &expr[*expr_id];
        let colored_string = match node {
            Rise::Var(index) => index.to_string().magenta(),
            Rise::App(_) | Rise::Lambda(_) => node.to_string().red(),
            Rise::NatApp(_) | Rise::DataApp(_) | Rise::AddrApp(_) | Rise::NatNatApp(_) => {
                node.to_string().cyan()
            }
            Rise::NatLambda(c_id)
            | Rise::DataLambda(c_id)
            | Rise::AddrLambda(c_id)
            | Rise::NatNatLambda(c_id) => {
                if skip_wrapper {
                    return Self::new(expr, *c_id, skip_wrapper);
                }
                node.to_string().cyan()
            }
            Rise::FunType(_)
            | Rise::NatFun(_)
            | Rise::DataFun(_)
            | Rise::AddrFun(_)
            | Rise::NatNatFun(_)
            | Rise::TypeOf(_)
            | Rise::ArrType(_)
            | Rise::VecType(_)
            | Rise::PairType(_)
            | Rise::IndexType(_)
            | Rise::NatType
            | Rise::F32 => panic!("Should not see types here: {node}"),
            Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_) => {
                panic!("NatExpr should only appear in types: {node}")
            } // node.to_string().white()
            Rise::Let
            | Rise::AsVector
            | Rise::AsScalar
            | Rise::VectorFromScalar
            | Rise::Snd
            | Rise::Fst
            | Rise::Add
            | Rise::Mul
            | Rise::ToMem
            | Rise::Split
            | Rise::Join
            | Rise::Generate
            | Rise::Transpose
            | Rise::Zip
            | Rise::Unzip
            | Rise::Map
            | Rise::MapPar
            | Rise::Reduce
            | Rise::ReduceSeq
            | Rise::ReduceSeqUnroll
            | Rise::Float(_) => node.to_string().yellow(),
            Rise::Integer(i) => format!("int{i}").purple(),
        };
        Self {
            children: node
                .children()
                .iter()
                .map(|c_id| Self::new(expr, *c_id, skip_wrapper))
                .collect(),
            expr: colored_string,
            ty: pp_ty(expr, *ty_id, false),
        }
    }
}

fn pp_ty(expr: &RecExpr<Rise>, id: Id, fn_brackets: bool) -> ColoredString {
    let node = &expr[id];
    match node {
        Rise::Var(index) => index.to_string().green(),
        Rise::FunType([i, o]) => if fn_brackets {
            format!("({} -> {})", pp_ty(expr, *i, true), pp_ty(expr, *o, true))
        } else {
            format!("{} -> {}", pp_ty(expr, *i, true), pp_ty(expr, *o, true))
        }
        .blue(),
        Rise::ArrType([n, ty]) => format!(
            "[{}: {}]",
            pp_ty(expr, *ty, fn_brackets),
            pp_ty(expr, *n, fn_brackets)
        )
        .blue(),
        Rise::VecType([n, ty]) => format!(
            "Vec[{}: {}]",
            pp_ty(expr, *ty, fn_brackets),
            pp_ty(expr, *n, fn_brackets)
        )
        .blue(),
        Rise::PairType([fst, snd]) => format!(
            "({}, {})",
            pp_ty(expr, *fst, fn_brackets),
            pp_ty(expr, *snd, fn_brackets)
        )
        .blue(),
        Rise::IndexType(c) => format!("Idx[{}]", pp_ty(expr, *c, fn_brackets)).blue(),
        Rise::NatType => "nat".to_owned().blue(),
        Rise::F32 => "f32".to_owned().blue(),
        Rise::NatFun(c) => format!("NatFun[{}]", pp_ty(expr, *c, fn_brackets)).blue(),
        Rise::DataFun(c) => format!("DataFun[{}]", pp_ty(expr, *c, fn_brackets)).blue(),
        Rise::AddrFun(c) => format!("AddrFun[{}]", pp_ty(expr, *c, fn_brackets)).blue(),
        Rise::NatNatFun(c) => format!("NatNatFun[{}]", pp_ty(expr, *c, fn_brackets)).blue(),
        Rise::NatAdd([a, b]) => format!(
            "({} + {})",
            pp_ty(expr, *a, fn_brackets),
            pp_ty(expr, *b, fn_brackets)
        )
        .white(),
        Rise::NatSub([a, b]) => format!(
            "({} - {})",
            pp_ty(expr, *a, fn_brackets),
            pp_ty(expr, *b, fn_brackets)
        )
        .white(),
        Rise::NatMul([a, b]) => format!(
            "({} * {})",
            pp_ty(expr, *a, fn_brackets),
            pp_ty(expr, *b, fn_brackets)
        )
        .white(),
        Rise::NatDiv([a, b]) => format!(
            "({} / {})",
            pp_ty(expr, *a, fn_brackets),
            pp_ty(expr, *b, fn_brackets)
        )
        .white(),
        Rise::NatPow([a, b]) => format!(
            "({} ^ {})",
            pp_ty(expr, *a, fn_brackets),
            pp_ty(expr, *b, fn_brackets)
        )
        .white(),
        _ => panic!("only for types but found {node}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::{BASELINE_GOAL, MM};

    #[test]
    fn pp_mm() {
        let mm: RecExpr<Rise> = MM.parse().unwrap();
        mm.pp(true);
        mm.pp(false);
    }

    #[test]
    fn pp_baseline() {
        let baseline: RecExpr<Rise> = BASELINE_GOAL.parse().unwrap();
        baseline.pp(true);
        baseline.pp(false);
    }
}
