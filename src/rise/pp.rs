use core::panic;

use colored::{ColoredString, Colorize};
use egg::{ENodeOrVar, Id, Language, RecExpr};

use crate::rise::kind::Kindable;

use super::Rise;

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
        Rise::App(app, _) if app.is_expr() => (node.to_string().red(), false),
        Rise::Lambda(lam, _) if lam.is_expr() => (node.to_string().red(), false),
        Rise::App(_, _) | Rise::Lambda(_, _) => (node.to_string().cyan(), true),
        Rise::Var(index) | Rise::TypedVar(index, _) => (index.to_string().magenta(), false),
        // Primitive Types inside Expr position (Panic)
        Rise::FunType(_)
        | Rise::NatFun(_)
        | Rise::DataFun(_)
        | Rise::AddrFun(_)
        | Rise::NatNatFun(_)
        | Rise::ArrType(_)
        | Rise::VecType(_)
        | Rise::PairType(_)
        | Rise::IndexType(_)
        | Rise::F32
        | Rise::I64 => panic!("Should not see types in expression position: {node}"),
        // NatExprs inside Expr position (Panic)
        Rise::NatAdd(_)
        | Rise::NatSub(_)
        | Rise::NatMul(_)
        | Rise::NatDiv(_)
        | Rise::NatPow(_)
        | Rise::NatCst(_) => {
            panic!("NatExpr should only appear in types: {node}")
        }

        // Standard Opcodes
        Rise::Let(_) | Rise::FloatLit(_, _) => (node.to_string().yellow(), false),
        Rise::Prim(prim, _) => (prim.to_string().yellow(), false),
        Rise::IntLit(_, _) => (node.to_string().cyan(), false),
    }
}

/// Helper to format a type node given pre-formatted children strings.
/// This allows us to share the formatting string logic between `Rise` and `egg::ENodeOrVar<Rise>`.
fn fmt_ty_node(node: &Rise, children: &[String], fn_brackets: bool) -> ColoredString {
    match node {
        Rise::Var(_) => node.to_string().green(),
        Rise::FunType(_) => {
            let s = format!("{} -> {}", children[0], children[1]);
            if fn_brackets { format!("({s})") } else { s }.blue()
        }
        Rise::ArrType(_) => format!("[{}: {}]", children[1], children[0]).blue(),
        Rise::VecType(_) => format!("Vec[{}: {}]", children[1], children[0]).blue(),
        Rise::PairType(_) => format!("({}, {})", children[0], children[1]).blue(),
        Rise::IndexType(_) => format!("Idx[{}]", children[0]).blue(),
        Rise::I64 | Rise::F32 => node.to_string().blue(),

        Rise::NatFun(_) => format!("NatFun[{}]", children[0]).blue(),
        Rise::DataFun(_) => format!("DataFun[{}]", children[0]).blue(),
        Rise::AddrFun(_) => format!("AddrFun[{}]", children[0]).blue(),
        Rise::NatNatFun(_) => format!("NatNatFun[{}]", children[0]).blue(),

        Rise::NatAdd(_) => format!("({} + {})", children[0], children[1]).white(),
        Rise::NatSub(_) => format!("({} - {})", children[0], children[1]).white(),
        Rise::NatMul(_) => format!("({} * {})", children[0], children[1]).white(),
        Rise::NatDiv(_) => format!("({} / {})", children[0], children[1]).white(),
        Rise::NatPow(_) => format!("({} ^ {})", children[0], children[1]).white(),
        Rise::NatCst(i) => format!("{i}").cyan(),

        _ => panic!("Expected type constructor node but found {node}"),
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
            let node = &expr[id];
            let (colored_string, is_wrapper) = get_rise_style(node);

            PPNode {
                children: node
                    .normal_children()
                    .iter()
                    .map(|c| rec(expr, *c))
                    .collect(),
                expr: colored_string,
                ty: node.ty_id().map_or_else(
                    || "NO TYPE INFO".red(),
                    |ty_id| pp_ty_rise(expr, ty_id, false),
                ),
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
            match &expr[id] {
                ENodeOrVar::Var(v) => PPNode {
                    children: Box::new([]),
                    expr: v.to_string().magenta(),
                    ty: "PATTERN VAR".normal(),
                    wrapper: false,
                },
                ENodeOrVar::ENode(node) => {
                    let (colored_string, is_wrapper) = get_rise_style(node);
                    PPNode {
                        children: node
                            .normal_children()
                            .iter()
                            .map(|c| rec(expr, *c))
                            .collect(),
                        expr: colored_string,
                        ty: node.ty_id().map_or_else(
                            || "NO TYPE INFO".red(),
                            |ty_id| pp_ty_enode(expr, ty_id, false),
                        ),
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
