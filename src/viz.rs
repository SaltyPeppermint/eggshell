use std::fmt::Display;

use dot_generator::{attr, edge, id, node, node_id, stmt};
use dot_structures::{Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex};
use egg::{Language, RecExpr};

use graphviz_rust::printer::{DotPrinter, PrinterContext};

pub fn to_dot<L: Language + Display>(rec_expr: &RecExpr<L>, name: &str) -> String {
    fn rec<LL: Language + Display>(
        parent_graph_id: usize,
        curr: &LL,
        rec_expr: &RecExpr<LL>,
        nodes: &mut Vec<String>,
        edges: &mut Vec<(usize, usize)>,
    ) {
        let curr_graph_id = nodes.len();
        nodes.push(curr.to_string());
        edges.push((parent_graph_id, curr_graph_id));
        for c_id in curr.children() {
            rec(curr_graph_id, &rec_expr[*c_id], rec_expr, nodes, edges);
        }
    }
    let root_id = rec_expr.root();
    let mut nodes = vec![rec_expr[root_id].to_string()];
    let mut edges = Vec::new();
    for c in rec_expr[root_id].children() {
        rec(0, &rec_expr[*c], rec_expr, &mut nodes, &mut edges);
    }

    let mut stmts = nodes
        .into_iter()
        .map(|x| format!("\"{x}\""))
        .enumerate()
        .map(|(idx, label)| node!(idx; attr!("label", label)))
        .map(Stmt::Node)
        .chain(
            edges
                .iter()
                .map(|(p_id, c_id)| edge!(node_id!(p_id) => node_id!(c_id)))
                .map(Stmt::Edge),
        )
        .collect::<Vec<_>>();

    let n = format!("\"{name}\"");
    stmts.extend([
        stmt!(attr!("ordering", "out")),
        stmt!(attr!("labelloc", "t")),
        stmt!(attr!("label", n)),
    ]);

    Graph::Graph {
        id: id!(name),
        strict: true,
        stmts,
    }
    .print(&mut PrinterContext::default())
}

pub fn dot_to_svg(dot: &str) -> Vec<u8> {
    let format = graphviz_rust::cmd::Format::Svg;
    graphviz_rust::exec_dot(dot.into(), vec![format.into()]).unwrap()
}

#[cfg(test)]
mod tests {

    use super::*;

    use egg::RecExpr;

    use crate::meta_lang::PartialLang;
    use crate::trs::halide::HalideLang;

    #[test]
    fn simple_ast_dot() {
        let expr: RecExpr<PartialLang<HalideLang>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let dot = to_dot(&expr, "partial_lang_test");
        assert_eq!(
            &dot,
            "strict graph partial_lang_test {\n  0[label=\"*\"]\n  1[label=\"-\"]\n  2[label=\"2\"]\n  3[label=\"v1\"]\n  4[label=\"+\"]\n  5[label=\"-\"]\n  6[label=\"<pad>\"]\n  7[label=\"<pad>\"]\n  8[label=\"v2\"]\n  0 -- 1\n  1 -- 2\n  1 -- 3\n  0 -- 4\n  4 -- 5\n  5 -- 6\n  5 -- 7\n  4 -- 8\n  ordering=out\n  labelloc=t\n  label=\"partial_lang_test\"\n}"
        );
    }
}
