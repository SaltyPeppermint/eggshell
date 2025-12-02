use std::fmt::Display;

use dot_generator::{attr, edge, id, node, node_id, stmt};
use dot_structures::{Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex};
use egg::{Language, RecExpr};

use graphviz_rust::cmd::Format;
use graphviz_rust::printer::{DotPrinter, PrinterContext};
use hashbrown::HashSet;

pub fn to_dot<L: Language + Display>(
    rec_expr: &RecExpr<L>,
    name: &str,
    marked_ids: &HashSet<egg::Id>,
    transparent: bool,
) -> String {
    fn rec<LL: Language + Display>(
        parent_graph_id: usize,
        id: egg::Id,
        rec_expr: &RecExpr<LL>,
        nodes: &mut Vec<(String, bool)>,
        edges: &mut Vec<(usize, usize)>,
        marked_ids: &HashSet<egg::Id>,
    ) {
        let node = &rec_expr[id];
        let graph_id = nodes.len();
        nodes.push((node.to_string(), marked_ids.contains(&id)));
        edges.push((parent_graph_id, graph_id));
        for c_id in node.children() {
            rec(graph_id, *c_id, rec_expr, nodes, edges, marked_ids);
        }
    }
    let root = &rec_expr[rec_expr.root()];
    let mut nodes = vec![(root.to_string(), marked_ids.contains(&(rec_expr.root())))];
    let mut edges = Vec::new();
    for c_id in root.children() {
        rec(0, *c_id, rec_expr, &mut nodes, &mut edges, marked_ids);
    }

    let mut stmts = nodes
        .into_iter()
        .enumerate()
        .map(|(idx, (label, marked))| {
            let label_str = format!("\"{label}\"");
            if marked {
                node!(idx; attr!("label", label_str), attr!("color", "red"))
            } else {
                node!(idx; attr!("label", label_str))
            }
        })
        .map(Stmt::Node)
        .chain(
            edges
                .iter()
                .map(|(p_id, c_id)| edge!(node_id!(p_id) => node_id!(c_id)))
                .map(Stmt::Edge),
        )
        .collect::<Vec<_>>();

    // let escaped_name = format!("\\\"{name}\\\"");
    let escaped_name = format!("\"{}\"", name.replace('"', "\\\""));
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

#[expect(clippy::missing_panics_doc)]
#[must_use]
pub fn dot_to_svg(dot: &str) -> Vec<u8> {
    let format = Format::Svg;
    graphviz_rust::exec_dot(dot.into(), vec![format.into()]).unwrap()
}

#[cfg(test)]
mod tests {

    use super::*;

    use egg::RecExpr;

    use crate::rewrite_system::dummy_rise::DummyRiseLang;
    use crate::rewrite_system::halide::HalideLang;

    #[test]
    fn simple_ast_dot() {
        let expr: RecExpr<HalideLang> = "(* (- 2 v1) (+ (- 2 5) v2))".parse().unwrap();
        let dot = to_dot(&expr, "partial_lang_test", &HashSet::new(), false);

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test1.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"partial_lang_test\" {\n  0[label=\"*\"]\n  1[label=\"-\"]\n  2[label=\"2\"]\n  3[label=\"v1\"]\n  4[label=\"+\"]\n  5[label=\"-\"]\n  6[label=\"2\"]\n  7[label=\"5\"]\n  8[label=\"v2\"]\n  0 -- 1\n  1 -- 2\n  1 -- 3\n  0 -- 4\n  4 -- 5\n  5 -- 6\n  5 -- 7\n  4 -- 8\n  ordering=out\n  labelloc=t\n  label=\"partial_lang_test\"\n}"
        );
    }

    #[test]
    fn simple_ast_dot_marked_id() {
        let expr: RecExpr<HalideLang> = "(* (- 2 v1) (+ (- 2 5) v2))".parse().unwrap();
        let dot = to_dot(
            &expr,
            "partial_lang_test",
            &HashSet::from([0.into()]),
            false,
        );

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test1.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"partial_lang_test\" {\n  0[label=\"*\"]\n  1[label=\"-\"]\n  2[label=\"2\",color=red]\n  3[label=\"v1\"]\n  4[label=\"+\"]\n  5[label=\"-\"]\n  6[label=\"2\"]\n  7[label=\"5\"]\n  8[label=\"v2\"]\n  0 -- 1\n  1 -- 2\n  1 -- 3\n  0 -- 4\n  4 -- 5\n  5 -- 6\n  5 -- 7\n  4 -- 8\n  ordering=out\n  labelloc=t\n  label=\"partial_lang_test\"\n}"
        );
    }

    #[test]
    fn longer_ast_dot() {
        let s_expr = "(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))";
        let expr: RecExpr<DummyRiseLang> = s_expr.parse().unwrap();
        let dot = to_dot(&expr, s_expr, &HashSet::new(), false);

        // let svg = crate::viz::dot_to_svg(&dot);
        // let path = std::env::current_dir().unwrap().join("test2.svg");
        // std::fs::write(path, svg).unwrap();

        assert_eq!(
            &dot,
            "strict graph \"(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))\" {\n  0[label=\"lam\"]\n  1[label=\"f1\"]\n  2[label=\"lam\"]\n  3[label=\"f2\"]\n  4[label=\"lam\"]\n  5[label=\"f3\"]\n  6[label=\"lam\"]\n  7[label=\"f4\"]\n  8[label=\"lam\"]\n  9[label=\"f5\"]\n  10[label=\"lam\"]\n  11[label=\"x3\"]\n  12[label=\"app\"]\n  13[label=\"app\"]\n  14[label=\"map\"]\n  15[label=\"var\"]\n  16[label=\"f5\"]\n  17[label=\"app\"]\n  18[label=\"lam\"]\n  19[label=\"x2\"]\n  20[label=\"app\"]\n  21[label=\"app\"]\n  22[label=\"map\"]\n  23[label=\"var\"]\n  24[label=\"f4\"]\n  25[label=\"app\"]\n  26[label=\"lam\"]\n  27[label=\"x1\"]\n  28[label=\"app\"]\n  29[label=\"app\"]\n  30[label=\"map\"]\n  31[label=\"var\"]\n  32[label=\"f3\"]\n  33[label=\"app\"]\n  34[label=\"lam\"]\n  35[label=\"x0\"]\n  36[label=\"app\"]\n  37[label=\"app\"]\n  38[label=\"map\"]\n  39[label=\"var\"]\n  40[label=\"f2\"]\n  41[label=\"app\"]\n  42[label=\"app\"]\n  43[label=\"map\"]\n  44[label=\"var\"]\n  45[label=\"f1\"]\n  46[label=\"var\"]\n  47[label=\"x0\"]\n  48[label=\"var\"]\n  49[label=\"x1\"]\n  50[label=\"var\"]\n  51[label=\"x2\"]\n  52[label=\"var\"]\n  53[label=\"x3\"]\n  0 -- 1\n  0 -- 2\n  2 -- 3\n  2 -- 4\n  4 -- 5\n  4 -- 6\n  6 -- 7\n  6 -- 8\n  8 -- 9\n  8 -- 10\n  10 -- 11\n  10 -- 12\n  12 -- 13\n  13 -- 14\n  13 -- 15\n  15 -- 16\n  12 -- 17\n  17 -- 18\n  18 -- 19\n  18 -- 20\n  20 -- 21\n  21 -- 22\n  21 -- 23\n  23 -- 24\n  20 -- 25\n  25 -- 26\n  26 -- 27\n  26 -- 28\n  28 -- 29\n  29 -- 30\n  29 -- 31\n  31 -- 32\n  28 -- 33\n  33 -- 34\n  34 -- 35\n  34 -- 36\n  36 -- 37\n  37 -- 38\n  37 -- 39\n  39 -- 40\n  36 -- 41\n  41 -- 42\n  42 -- 43\n  42 -- 44\n  44 -- 45\n  41 -- 46\n  46 -- 47\n  33 -- 48\n  48 -- 49\n  25 -- 50\n  50 -- 51\n  17 -- 52\n  52 -- 53\n  ordering=out\n  labelloc=t\n  label=\"(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))\"\n}"
        );
    }
}
