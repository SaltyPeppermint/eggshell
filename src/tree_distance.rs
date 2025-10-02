use egg::{Language, RecExpr};

const DELETE: usize = 1;
const INSERT: usize = 1;
const RELABEL: usize = 1;

#[must_use]
#[expect(clippy::similar_names)]
pub fn distance<L: Language>(left: &RecExpr<L>, right: &RecExpr<L>) -> usize {
    fn inner<LL: Language>(
        left: &TreeDiffData<LL>,
        right: &TreeDiffData<LL>,
        i: usize,
        j: usize,
        td: &mut [Vec<usize>],
    ) -> usize {
        // forestdist
        let mut fdist = vec![vec![0usize; j + 1]; i + 1];

        for i1 in left.l[i - 1]..=i {
            fdist[i1][0] = fdist[i1 - 1][0] + DELETE;
        }
        for j1 in right.l[j - 1]..=j {
            fdist[0][j1] = fdist[0][j1 - 1] + INSERT;
        }
        for i1 in left.l[i - 1]..=i {
            for j1 in right.l[j - 1]..=j {
                let i_temp = if left.l[i - 1] > i1 - 1 { 0 } else { i1 - 1 };
                let j_temp = if right.l[j - 1] > j1 - 1 { 0 } else { j1 - 1 };
                if (left.l[i1 - 1] == left.l[i - 1]) && (right.l[j1 - 1] == right.l[j - 1]) {
                    let cost = if left.labels[i1 - 1].matches(right.labels[j1 - 1]) {
                        0
                    } else {
                        RELABEL
                    };
                    fdist[i1][j1] = (fdist[i_temp][j1] + DELETE)
                        .min(fdist[i1][j_temp] + INSERT)
                        .min(fdist[i_temp][j_temp] + cost);
                    td[i1][j1] = fdist[i1][j1];
                } else {
                    let i1_temp = left.l[i1 - 1] - 1;
                    let j1_temp = right.l[j1 - 1] - 1;

                    let i_temp2 = if left.l[i - 1] > i1_temp { 0 } else { i1_temp };
                    let j_temp2 = if right.l[j - 1] > j1_temp { 0 } else { j1_temp };

                    fdist[i1][j1] = (fdist[i_temp][j1] + DELETE)
                        .min(fdist[i1][j_temp] + INSERT)
                        .min(fdist[i_temp2][j_temp2] + td[i1][j1]);
                }
            }
        }
        fdist[i][j]
    }

    let left_td: TreeDiffData<L> = left.into();
    let right_td: TreeDiffData<L> = right.into();

    let mut td = vec![vec![0usize; right_td.l.len() + 1]; left_td.l.len() + 1];
    for i1 in 1..=left_td.keyroots.len() {
        for j1 in 1..=right_td.keyroots.len() {
            let i = left_td.keyroots[i1 - 1];
            let j = right_td.keyroots[j1 - 1];
            td[i][j] = inner(&left_td, &right_td, i, j, &mut td);
        }
    }

    td[left_td.l.len()][right_td.l.len()]
}

struct TreeDiffData<'a, L: Language> {
    l: Vec<usize>,
    keyroots: Vec<usize>,
    labels: Vec<&'a L>,
}

impl<'a, L: Language> From<&'a RecExpr<L>> for TreeDiffData<'a, L> {
    fn from(value: &'a RecExpr<L>) -> TreeDiffData<'a, L> {
        fn traverse<'a, LL: Language>(
            rec_expr: &'a RecExpr<LL>,
            node_id: egg::Id,
            labels: &mut Vec<&'a LL>,
            index: &mut usize,
            leftmost_table: &mut Vec<usize>,
        ) -> usize {
            let node = &rec_expr[node_id];
            let child_leftmosts = node.children().split_first().map(|(fst_id, tail_ids)| {
                let fst_leftmost = traverse(rec_expr, *fst_id, labels, index, leftmost_table);
                for child_id in tail_ids {
                    traverse(rec_expr, *child_id, labels, index, leftmost_table);
                }
                fst_leftmost
            });

            labels.push(node);
            *index += 1;
            let leftmost = child_leftmosts.unwrap_or(*index);
            leftmost_table.push(leftmost);
            leftmost
        }

        fn mk_keyroots(l: &[usize]) -> Vec<usize> {
            l.iter()
                .enumerate()
                .filter_map(|(i, l_i)| (l[i + 1..]).iter().all(|lj| l_i != lj).then_some(i + 1))
                .collect()
        }

        fn mk_l<LL: Language>(
            rec_expr: &RecExpr<LL>,
            node_id: egg::Id,
            leftmost_table: &[usize],
            index: &mut usize,
            mut leftmost: Vec<usize>,
        ) -> Vec<usize> {
            let node = &rec_expr[node_id];
            for child_id in node.children() {
                leftmost = mk_l(rec_expr, *child_id, leftmost_table, index, leftmost);
            }
            leftmost.push(leftmost_table[*index]);
            *index += 1;
            leftmost
        }

        let mut labels = Vec::new();
        let mut leftmost_table = Vec::new();
        traverse(
            value,
            value.root(),
            &mut labels,
            &mut 0,
            &mut leftmost_table,
        );
        let keyroots = mk_keyroots(&leftmost_table);
        let l = mk_l(value, value.root(), &leftmost_table, &mut 0, Vec::new());

        TreeDiffData {
            l,
            keyroots,
            labels,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::rewrite_system::arithmetic::Math;

    use super::*;

    #[test]
    fn zero_distance() {
        let expr: RecExpr<Math> = "(+ x x)".parse().unwrap();
        assert_eq!(0, distance(&expr, &expr));
    }

    #[test]
    fn one_distance() {
        let left: RecExpr<Math> = "(+ x x)".parse().unwrap();
        let right: RecExpr<Math> = "(+ x y)".parse().unwrap();
        assert_eq!(1, distance(&left, &right));
    }

    #[test]
    fn one_distance2() {
        let left: RecExpr<Math> = "(ln x)".parse().unwrap();
        let right: RecExpr<Math> = "(ln y)".parse().unwrap();
        assert_eq!(1, distance(&left, &right));
    }

    #[test]
    fn two_distance() {
        let left: RecExpr<Math> = "(+ x y)".parse().unwrap();
        let right: RecExpr<Math> = "(+ v w)".parse().unwrap();
        assert_eq!(2, distance(&left, &right));
    }

    #[test]
    fn subtree_distance() {
        let left: RecExpr<Math> = "(+ x y)".parse().unwrap();
        let right: RecExpr<Math> = "(+ (+ v w) y)".parse().unwrap();
        assert_eq!(3, distance(&left, &right));
    }

    #[test]
    fn subtree_distance2() {
        let left: RecExpr<Math> = "(- x z)".parse().unwrap();
        let right: RecExpr<Math> = "(+ (+ v w) y)".parse().unwrap();
        assert_eq!(5, distance(&left, &right));
    }
}
