use colored::{ColoredString, Colorize};
use egg::{ENodeOrVar, Id, Language, RecExpr};

use super::{Kind, Kindable, Rise};

pub trait PrettyPrint {
    fn pp(self, skip_wrapper: bool);
}

// ============================================================================
// Shared Styling Logic
// ============================================================================

/// Returns the colored string representation and a boolean indicating if
/// the node is a "wrapper" (like a Lambda) that impacts indentation logic.
fn get_rise_style(node: &Rise) -> (ColoredString, bool) {
    match node {
        Rise::Var(index) => (index.to_string().magenta(), false),
        Rise::App(a, _) => {
            if a.kind() == Kind::Expr {
                (node.to_string().red(), false)
            } else {
                (node.to_string().cyan(), true)
            }
        }
        Rise::Lambda(l, _) => {
            if l.kind() == Kind::Expr {
                (node.to_string().red(), false)
            } else {
                (node.to_string().cyan(), true)
            }
        }
        // Rise::App(x) | Rise::Lambda(_) => (node.to_string().red(), false),
        // Rise::NatApp(_) | Rise::DataApp(_) | Rise::AddrApp(_) | Rise::NatNatApp(_) => {}
        // Rise::NatLambda(_) | Rise::DataLambda(_) | Rise::AddrLambda(_) | Rise::NatNatLambda(_) => {
        //     (node.to_string().cyan(), true)
        // } // is_wrapper = true

        // Primitive Types inside Expr position (Panic)
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
        | Rise::F32 => panic!("Should not see types in expression position: {node}"),

        // NatExprs inside Expr position (Panic)
        Rise::NatAdd(_) | Rise::NatSub(_) | Rise::NatMul(_) | Rise::NatDiv(_) | Rise::NatPow(_) => {
            panic!("NatExpr should only appear in types: {node}")
        }

        // Standard Opcodes
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
        | Rise::Float(_) => (node.to_string().yellow(), false),

        Rise::Integer(i) => (format!("{i}int").cyan(), false),
    }
}

/// Helper to format a type node given pre-formatted children strings.
/// This allows us to share the formatting string logic between `Rise` and `egg::ENodeOrVar<Rise>`.
fn fmt_ty_node(node: &Rise, children: &[String], fn_brackets: bool) -> ColoredString {
    match node {
        Rise::Var(index) => index.to_string().green(),
        Rise::FunType(_) => {
            let s = format!("{} -> {}", children[0], children[1]);
            if fn_brackets { format!("({s})") } else { s }.blue()
        }
        Rise::ArrType(_) => format!("[{}: {}]", children[1], children[0]).blue(),
        Rise::VecType(_) => format!("Vec[{}: {}]", children[1], children[0]).blue(),
        Rise::PairType(_) => format!("({}, {})", children[0], children[1]).blue(),
        Rise::IndexType(_) => format!("Idx[{}]", children[0]).blue(),
        Rise::NatType => "nat".to_owned().blue(),
        Rise::F32 => "f32".to_owned().blue(),

        Rise::NatFun(_) => format!("NatFun[{}]", children[0]).blue(),
        Rise::DataFun(_) => format!("DataFun[{}]", children[0]).blue(),
        Rise::AddrFun(_) => format!("AddrFun[{}]", children[0]).blue(),
        Rise::NatNatFun(_) => format!("NatNatFun[{}]", children[0]).blue(),

        Rise::NatAdd(_) => format!("({} + {})", children[0], children[1]).white(),
        Rise::NatSub(_) => format!("({} - {})", children[0], children[1]).white(),
        Rise::NatMul(_) => format!("({} * {})", children[0], children[1]).white(),
        Rise::NatDiv(_) => format!("({} / {})", children[0], children[1]).white(),
        Rise::NatPow(_) => format!("({} ^ {})", children[0], children[1]).white(),
        Rise::Integer(i) => format!("{i}int").cyan(),

        _ => panic!("Expected type node but found {node}"),
    }
}

// ============================================================================
// Tree Structure
// ============================================================================

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

impl PrettyPrint for &RecExpr<Rise> {
    fn pp(self, skip_wrapper: bool) {
        let tree: PPNode = self.into();
        tree.pp(skip_wrapper);
    }
}

// ============================================================================
// Implementation: RecExpr<Rise>
// ============================================================================

impl From<&RecExpr<Rise>> for PPNode {
    fn from(value: &RecExpr<Rise>) -> PPNode {
        fn rec(expr: &RecExpr<Rise>, id: Id) -> PPNode {
            let Rise::TypeOf([expr_id, ty_id]) = &expr[id] else {
                // Fallback if top-level node is not TypeOf
                return PPNode {
                    children: expr[id].children().iter().map(|c| rec(expr, *c)).collect(),
                    expr: expr[id].to_string().into(),
                    ty: "NO TYPE INFO".red(),
                    wrapper: false,
                };
            };

            let node = &expr[*expr_id];
            let (colored_string, is_wrapper) = get_rise_style(node);

            PPNode {
                children: node.children().iter().map(|c| rec(expr, *c)).collect(),
                expr: colored_string,
                ty: pp_ty_rise(expr, *ty_id, false),
                wrapper: is_wrapper,
            }
        }
        rec(value, value.root())
    }
}

fn pp_ty_rise(expr: &RecExpr<Rise>, id: Id, fn_brackets: bool) -> ColoredString {
    let node = &expr[id];
    let children_strs: Vec<String> = node
        .children()
        .iter()
        .map(|&c| pp_ty_rise(expr, c, true).to_string()) // recursive call with brackets=true
        .collect();

    fmt_ty_node(node, &children_strs, fn_brackets)
}

// ============================================================================
// Implementation: RecExpr<ENodeOrVar<Rise>>
// ============================================================================

impl PrettyPrint for &RecExpr<ENodeOrVar<Rise>> {
    fn pp(self, skip_wrapper: bool) {
        let tree: PPNode = self.into();
        tree.pp(skip_wrapper);
    }
}

impl From<&RecExpr<ENodeOrVar<Rise>>> for PPNode {
    fn from(value: &RecExpr<ENodeOrVar<Rise>>) -> PPNode {
        fn rec(expr: &RecExpr<ENodeOrVar<Rise>>, id: Id) -> PPNode {
            // 1. Resolve Root Node (must be TypeOf inside an ENode)
            let inner_node = match &expr[id] {
                ENodeOrVar::ENode(n) => n,
                ENodeOrVar::Var(v) => {
                    return PPNode {
                        children: Box::new([]),
                        expr: v.to_string().magenta(),
                        ty: "PATTERN VAR".normal(),
                        wrapper: false,
                    };
                }
            };

            let Rise::TypeOf([expr_id, ty_id]) = inner_node else {
                return PPNode {
                    children: expr[id].children().iter().map(|c| rec(expr, *c)).collect(),
                    expr: expr[id].to_string().into(),
                    ty: "NO TYPE INFO".red(),
                    wrapper: false,
                };
            };

            // 2. Resolve Actual Expression Node
            match &expr[*expr_id] {
                ENodeOrVar::Var(v) => PPNode {
                    children: Box::new([]),
                    expr: v.to_string().magenta(),
                    ty: pp_ty_enode(expr, *ty_id, false),
                    wrapper: false,
                },
                ENodeOrVar::ENode(node) => {
                    let (colored_string, is_wrapper) = get_rise_style(node);
                    PPNode {
                        children: node.children().iter().map(|c| rec(expr, *c)).collect(),
                        expr: colored_string,
                        ty: pp_ty_enode(expr, *ty_id, false),
                        wrapper: is_wrapper,
                    }
                }
            }
        }
        rec(value, value.root())
    }
}

fn pp_ty_enode(expr: &RecExpr<ENodeOrVar<Rise>>, id: Id, fn_brackets: bool) -> ColoredString {
    match &expr[id] {
        ENodeOrVar::Var(v) => v.to_string().green(),
        ENodeOrVar::ENode(node) => {
            let children_strs = node
                .children()
                .iter()
                .map(|&c| pp_ty_enode(expr, c, true).to_string())
                .collect::<Vec<_>>();
            fmt_ty_node(node, &children_strs, fn_brackets)
        }
    }
}

#[cfg(test)]
mod tests {

    use egg::RecExpr;

    use super::super::MM;
    use super::*;

    #[test]
    fn pp_mm() {
        let mm: RecExpr<Rise> = MM.parse().unwrap();
        mm.pp(true);
    }

    #[test]
    fn pp_reduce_seq_lhs() {
        let r: RecExpr<ENodeOrVar<Rise>> = "(typeOf reduce (fun (fun ?dt0 (fun ?dt0 ?dt0)) (fun ?dt0 (fun (arrT ?n0 ?dt0) ?dt0))))".parse().unwrap();
        r.pp(false);
    }

    #[test]
    fn pp_reduce_seq_rhs() {
        let r: RecExpr<ENodeOrVar<Rise>> = "(typeOf reduceSeq (fun (fun ?dt0 (fun ?dt0 ?dt0)) (fun ?dt0 (fun (arrT ?n0 ?dt0) ?dt0))))".parse().unwrap();
        r.pp(false);
    }

    #[test]
    fn pp_reduce_seq_map_fusion_lhs() {
        let r: RecExpr<ENodeOrVar<Rise>> = "(typeOf (app (typeOf (app (typeOf (app (typeOf reduceSeq ?tAny7) (typeOf ?0 (fun ?dt0 (fun ?dt1 ?dt0)))) ?tAny8) (typeOf ?1 ?dt0)) ?tAny9) (typeOf (app (typeOf (app (typeOf map ?tAny10) (typeOf ?2 (fun ?dt2 ?dt1))) ?tAny11) (typeOf ?3 (arrT ?n0 ?dt2))) (arrT ?n0 ?dt1))) ?dt0)".parse().unwrap();
        r.pp(false);
    }

    #[test]
    fn pp_reduce_seq_map_fusion_rhs() {
        let r: RecExpr<ENodeOrVar<Rise>> = "(typeOf (app (typeOf (app (typeOf (app (typeOf reduceSeq (fun (fun ?dt0 (fun ?dt2 ?dt0)) (fun ?dt0 (fun (arrT ?n0 ?dt2) ?dt0)))) (typeOf (lam (typeOf (lam (typeOf (app (typeOf (app (typeOf ?4 (fun ?dt3 (fun ?dt4 ?dt3))) (typeOf %e0 ?dt3)) (fun ?dt4 ?dt3)) (typeOf (app (typeOf ?5 (fun ?dt5 ?dt4)) (typeOf %e0 ?dt5)) ?dt4)) ?dt3)) (fun ?dt6 ?dt7))) (fun ?dt0 (fun ?dt2 ?dt0)))) (fun ?dt0 (fun (arrT ?n0 ?dt2) ?dt0))) (typeOf ?1 ?dt0)) (fun (arrT ?n0 ?dt2) ?dt0)) (typeOf ?3 (arrT ?n0 ?dt2))) ?dt0)".parse().unwrap();
        r.pp(false);
    }
}
