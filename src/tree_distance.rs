use egg::{Language, RecExpr};

pub struct IdRecNode {
    node_id: egg::Id,
    index: usize,
    children: Vec<IdRecNode>,
}

impl IdRecNode {
    pub fn leftmost(&self) -> &Self {
        self.children.first().map(|c| c.leftmost()).unwrap_or(self)
    }

    pub fn children(&self) -> &[IdRecNode] {
        &self.children
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn node_id(&self) -> egg::Id {
        self.node_id
    }
}

impl<'a, L: Language> From<&RecExpr<L>> for IdRecNode {
    fn from(value: &RecExpr<L>) -> Self {
        fn rec<LL: Language>(rec_expr: &RecExpr<LL>, id: egg::Id, index: &mut usize) -> IdRecNode {
            let curr_index = *index;
            *index += 1;
            let children = rec_expr[id]
                .children()
                .iter()
                .map(|c_id| rec(rec_expr, *c_id, index))
                .collect();

            IdRecNode {
                node_id: id,
                children,
                index: curr_index,
            }
        }
        let mut index = 0;
        rec(value, value.root(), &mut index)
    }
}
const DELETE: usize = 1;
const INSERT: usize = 1;
const RELABEL: usize = 1;

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

        // for (int i1 = l1.get(i - 1); i1 <= i; i1++) {
        for i1 in left.l[i - 1]..=i {
            fdist[i1][0] = fdist[i1 - 1][0] + DELETE;
        }
        // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {

        for j1 in right.l[j - 1]..=j {
            fdist[0][j1] = fdist[0][j1 - 1] + INSERT;
        }
        // for (int i1 = l1.get(i - 1); i1 <= i; i1++) {
        for i1 in left.l[i - 1]..=i {
            // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {
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

    let left: TreeDiffData<L> = left.into();
    let right: TreeDiffData<L> = right.into();

    let mut td = vec![vec![0usize; right.l.len() + 1]; left.l.len() + 1];
    for i1 in 1..=left.keyroots.len() {
        for j1 in 1..=right.keyroots.len() {
            let i = left.keyroots[i1 - 1];
            let j = right.keyroots[j1 - 1];
            td[i][j] = inner(&left, &right, i, j, &mut td);
        }
    }

    td[left.l.len()][right.l.len()]
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
            leftmosts: &mut Vec<usize>,
        ) -> usize {
            let node = &rec_expr[node_id];
            let child_leftmosts = node
                .children()
                .into_iter()
                .map(|child_id| traverse(rec_expr, *child_id, labels, index, leftmosts))
                .collect::<Vec<_>>();

            labels.push(node);
            *index += 1;
            let leftmost = *child_leftmosts.first().unwrap_or(&index);
            leftmosts.push(leftmost);
            leftmost
        }

        fn mk_keyroots(l: &[usize]) -> Vec<usize> {
            let mut keyroots = Vec::new();
            // for (int i = 0; i < l.size(); i++) {
            for i in 0..l.len() {
                let mut flag = true;

                // for (int j = i + 1; j < l.size(); j++) {
                for j in i + 1..l.len() {
                    if l.get(j) == l.get(i) {
                        flag = false;
                    }
                }
                if flag {
                    keyroots.push(i + 1);
                }
            }
            keyroots
        }

        fn mk_l<LL: Language>(
            rec_expr: &RecExpr<LL>,
            node_id: egg::Id,
            leftmosts: &[usize],
            index: &mut usize,
            mut l: Vec<usize>,
        ) -> Vec<usize> {
            let node = &rec_expr[node_id];
            for child_id in node.children() {
                l = mk_l(rec_expr, *child_id, leftmosts, index, l);
            }
            l.push(leftmosts[*index]);
            *index += 1;
            l
        }
        let mut labels = Vec::new();
        let mut leftmosts = Vec::new();
        traverse(&value, value.root(), &mut labels, &mut 0, &mut leftmosts);
        let keyroots = mk_keyroots(&leftmosts);
        let l = mk_l(value, value.root(), &leftmosts, &mut 0, Vec::new());

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
        assert_eq!(0, distance(&expr, &expr))
    }

    #[test]
    fn one_distance() {
        let left: RecExpr<Math> = "(+ x x)".parse().unwrap();
        let right: RecExpr<Math> = "(+ x y)".parse().unwrap();
        assert_eq!(1, distance(&left, &right))
    }

    #[test]
    fn one_distance2() {
        let left: RecExpr<Math> = "(ln x)".parse().unwrap();
        let right: RecExpr<Math> = "(ln y)".parse().unwrap();
        assert_eq!(1, distance(&left, &right))
    }

    #[test]
    fn two_distance() {
        let left: RecExpr<Math> = "(+ x y)".parse().unwrap();
        let right: RecExpr<Math> = "(+ v w)".parse().unwrap();
        assert_eq!(2, distance(&left, &right))
    }

    #[test]
    fn subtree_distance() {
        let left: RecExpr<Math> = "(+ x y)".parse().unwrap();
        let right: RecExpr<Math> = "(+ (+ v w) y)".parse().unwrap();
        assert_eq!(3, distance(&left, &right))
    }

    #[test]
    fn subtree_distance2() {
        let left: RecExpr<Math> = "(- x z)".parse().unwrap();
        let right: RecExpr<Math> = "(+ (+ v w) y)".parse().unwrap();
        assert_eq!(5, distance(&left, &right))
    }
}
