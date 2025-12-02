use colored::{ColoredString, Colorize};
use egg::{Id, Language, RecExpr};

use super::Rise;

pub trait PrettyPrint {
    fn pp(self, skip_wrapper: bool);
}

impl PrettyPrint for &RecExpr<Rise> {
    fn pp(self, skip_wrapper: bool) {
        let tree: PPNode = self.into();
        tree.pp(skip_wrapper);
    }
}

struct PPNode {
    children: Box<[PPNode]>,
    expr: ColoredString,
    ty: ColoredString,
    wrapper: bool,
}

impl PPNode {
    fn pp(&self, skip_wrapper: bool) {
        fn rec(node: &PPNode, indent: &mut String, skip_wrapper: bool) {
            if node.wrapper && skip_wrapper {
                for c in &node.children {
                    c.pp(skip_wrapper);
                }
                return;
            }
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
                rec(r, indent, skip_wrapper);
            }
            indent.pop();
            indent.push(' ');
            rec(last, indent, skip_wrapper);
            indent.truncate(indent.len() - 4);
        }
        rec(self, &mut String::new(), skip_wrapper);
    }
}

impl From<&RecExpr<Rise>> for PPNode {
    fn from(value: &RecExpr<Rise>) -> PPNode {
        fn rec(expr: &RecExpr<Rise>, id: Id) -> PPNode {
            let Rise::TypeOf([expr_id, ty_id]) = &expr[id] else {
                return PPNode {
                    children: expr[id]
                        .children()
                        .iter()
                        .map(|c_id| rec(expr, *c_id))
                        .collect(),
                    expr: expr[id].to_string().into(),
                    ty: ("NO TYPE INFO AVAILABLE").red(),
                    wrapper: false,
                };
            };
            let node = &expr[*expr_id];
            let mut is_wrapper = false;
            let colored_string = match node {
                Rise::Var(index) => index.to_string().magenta(),
                Rise::App(_) | Rise::Lambda(_) => node.to_string().red(),
                Rise::NatApp(_) | Rise::DataApp(_) | Rise::AddrApp(_) | Rise::NatNatApp(_) => {
                    node.to_string().cyan()
                }
                Rise::NatLambda(_)
                | Rise::DataLambda(_)
                | Rise::AddrLambda(_)
                | Rise::NatNatLambda(_) => {
                    is_wrapper = true;
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
            PPNode {
                children: node
                    .children()
                    .iter()
                    .map(|c_id| rec(expr, *c_id))
                    .collect(),
                expr: colored_string,
                ty: pp_ty(expr, *ty_id, false),
                wrapper: is_wrapper,
            }
        }
        rec(value, value.root())
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
