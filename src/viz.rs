use std::fmt::Display;

use dot_generator::{attr, edge, id, node, node_id, stmt};
use dot_structures::{Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex};
use egg::{Language, RecExpr};

use graphviz_rust::printer::{DotPrinter, PrinterContext};

pub fn to_dot<L: Language + Display>(
    rec_expr: &RecExpr<L>,
    probs: Option<&[f64]>,
    name: &str,
    transparent: bool,
) -> String {
    fn rec<LL: Language + Display>(
        parent_graph_id: Option<usize>,
        curr_id: egg::Id,
        rec_expr: &RecExpr<LL>,
        probs: Option<&[f64]>,
        nodes: &mut Vec<String>,
        edges: &mut Vec<(usize, usize)>,
    ) {
        let curr = &rec_expr[curr_id];

        let curr_graph_id = nodes.len();
        let node_name = probs
            .map(|v| v[usize::from(curr_id)])
            .map(|v| format!("{curr}\n{v}"))
            .unwrap_or_else(|| curr.to_string());
        nodes.push(node_name);
        if let Some(p_id) = parent_graph_id {
            edges.push((p_id, curr_graph_id));
        }
        for c_id in curr.children() {
            rec(Some(curr_graph_id), *c_id, rec_expr, probs, nodes, edges);
        }
    }
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    rec(
        None,
        rec_expr.root(),
        rec_expr,
        probs,
        &mut nodes,
        &mut edges,
    );

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

    let escaped_name = format!("\"{name}\"");
    stmts.extend([
        stmt!(attr!("ordering", "out")),
        stmt!(attr!("labelloc", "t")),
        stmt!(attr!(html "label", escaped_name)),
    ]);
    if transparent {
        stmts.push(stmt!(attr!("bgcolor", "transparent")));
    }

    Graph::Graph {
        id: id!(escaped_name),
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

    use crate::partial::PartialLang;
    use crate::trs::halide::HalideLang;
    use crate::trs::rise::RiseLang;

    #[test]
    fn simple_ast_dot() {
        let expr: RecExpr<PartialLang<HalideLang>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let dot = to_dot(&expr, None, "partial_lang_test", false);

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test1.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"partial_lang_test\" {\n  0[label=\"*\"]\n  1[label=\"-\"]\n  2[label=\"2\"]\n  3[label=\"v1\"]\n  4[label=\"+\"]\n  5[label=\"-\"]\n  6[label=\"<pad>\"]\n  7[label=\"<pad>\"]\n  8[label=\"v2\"]\n  0 -- 1\n  1 -- 2\n  1 -- 3\n  0 -- 4\n  4 -- 5\n  5 -- 6\n  5 -- 7\n  4 -- 8\n  ordering=out\n  labelloc=t\n  label=\"partial_lang_test\"\n}"
        );
    }

    #[test]
    fn with_prob_ast_dot() {
        let expr: RecExpr<PartialLang<HalideLang>> =
            "(* (- 2 v1) (+ (- <pad> <pad>) v2))".parse().unwrap();
        let probs = vec![0.2, 0.04, 0.6, 0.9, 0.3, 0.1, 0.3, 0.01, 0.9];
        let dot = to_dot(&expr, Some(&probs), "partial_lang_test_probs", false);

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test1.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"partial_lang_test_probs\" {\n  0[label=\"*\n0.9\"]\n  1[label=\"-\n0.6\"]\n  2[label=\"2\n0.2\"]\n  3[label=\"v1\n0.04\"]\n  4[label=\"+\n0.01\"]\n  5[label=\"-\n0.1\"]\n  6[label=\"<pad>\n0.9\"]\n  7[label=\"<pad>\n0.3\"]\n  8[label=\"v2\n0.3\"]\n  0 -- 1\n  1 -- 2\n  1 -- 3\n  0 -- 4\n  4 -- 5\n  5 -- 6\n  5 -- 7\n  4 -- 8\n  ordering=out\n  labelloc=t\n  label=\"partial_lang_test_probs\"\n}"
        );
    }

    #[test]
    fn longer_ast_dot() {
        let s_expr = "(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))";
        let expr: RecExpr<RiseLang> = s_expr.parse().unwrap();
        let dot = to_dot(&expr, None, s_expr, false);

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test2.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))\" {\n  0[label=\"lam\"]\n  1[label=\"f1\"]\n  2[label=\"lam\"]\n  3[label=\"f2\"]\n  4[label=\"lam\"]\n  5[label=\"f3\"]\n  6[label=\"lam\"]\n  7[label=\"f4\"]\n  8[label=\"lam\"]\n  9[label=\"f5\"]\n  10[label=\"lam\"]\n  11[label=\"x3\"]\n  12[label=\"app\"]\n  13[label=\"app\"]\n  14[label=\"map\"]\n  15[label=\"var\"]\n  16[label=\"f5\"]\n  17[label=\"app\"]\n  18[label=\"lam\"]\n  19[label=\"x2\"]\n  20[label=\"app\"]\n  21[label=\"app\"]\n  22[label=\"map\"]\n  23[label=\"var\"]\n  24[label=\"f4\"]\n  25[label=\"app\"]\n  26[label=\"lam\"]\n  27[label=\"x1\"]\n  28[label=\"app\"]\n  29[label=\"app\"]\n  30[label=\"map\"]\n  31[label=\"var\"]\n  32[label=\"f3\"]\n  33[label=\"app\"]\n  34[label=\"lam\"]\n  35[label=\"x0\"]\n  36[label=\"app\"]\n  37[label=\"app\"]\n  38[label=\"map\"]\n  39[label=\"var\"]\n  40[label=\"f2\"]\n  41[label=\"app\"]\n  42[label=\"app\"]\n  43[label=\"map\"]\n  44[label=\"var\"]\n  45[label=\"f1\"]\n  46[label=\"var\"]\n  47[label=\"x0\"]\n  48[label=\"var\"]\n  49[label=\"x1\"]\n  50[label=\"var\"]\n  51[label=\"x2\"]\n  52[label=\"var\"]\n  53[label=\"x3\"]\n  0 -- 1\n  0 -- 2\n  2 -- 3\n  2 -- 4\n  4 -- 5\n  4 -- 6\n  6 -- 7\n  6 -- 8\n  8 -- 9\n  8 -- 10\n  10 -- 11\n  10 -- 12\n  12 -- 13\n  13 -- 14\n  13 -- 15\n  15 -- 16\n  12 -- 17\n  17 -- 18\n  18 -- 19\n  18 -- 20\n  20 -- 21\n  21 -- 22\n  21 -- 23\n  23 -- 24\n  20 -- 25\n  25 -- 26\n  26 -- 27\n  26 -- 28\n  28 -- 29\n  29 -- 30\n  29 -- 31\n  31 -- 32\n  28 -- 33\n  33 -- 34\n  34 -- 35\n  34 -- 36\n  36 -- 37\n  37 -- 38\n  37 -- 39\n  39 -- 40\n  36 -- 41\n  41 -- 42\n  42 -- 43\n  42 -- 44\n  44 -- 45\n  41 -- 46\n  46 -- 47\n  33 -- 48\n  48 -- 49\n  25 -- 50\n  50 -- 51\n  17 -- 52\n  52 -- 53\n  ordering=out\n  labelloc=t\n  label=\"(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))\"\n}"
        );
    }
}
