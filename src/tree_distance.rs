use egg::{Language, RecExpr};

use crate::node::BorrowingRecNode;

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
        for i_inner in left.leftmosts[i - 1]..=i {
            fdist[i_inner][0] = fdist[i_inner - 1][0] + DELETE;
        }
        // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {

        for j_inner in right.leftmosts[j - 1]..=j {
            fdist[0][j_inner] = fdist[0][j_inner - 1] + INSERT;
        }
        // for (int i1 = l1.get(i - 1); i1 <= i; i1++) {
        for i_inner in left.leftmosts[i - 1]..=i {
            // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {
            for j_inner in right.leftmosts[j - 1]..=j {
                let i_temp = if left.leftmosts[i - 1] > i_inner - 1 {
                    0
                } else {
                    i_inner - 1
                };
                let j_temp = if right.leftmosts[j - 1] > j_inner - 1 {
                    0
                } else {
                    j_inner - 1
                };
                if (left.leftmosts[i_inner - 1] == left.leftmosts[i - 1])
                    && (right.leftmosts[j_inner - 1] == right.leftmosts[j - 1])
                {
                    let cost = if left.labels[i_inner - 1] == right.labels[j_inner - 1] {
                        0
                    } else {
                        RELABEL
                    };

                    fdist[i_inner][j_inner] = (fdist[i_temp][j_inner] + DELETE)
                        .min(fdist[i_inner][j_temp] + INSERT)
                        .min(fdist[i_temp][j_temp] + cost);
                    td[i_inner][j_inner] = fdist[i_inner][j_inner];
                } else {
                    let i_inner_temp = left.leftmosts[i_inner - 1] - 1;
                    let i_temp2 = if left.leftmosts[i - 1] > i_inner_temp {
                        0
                    } else {
                        i_inner_temp
                    };

                    let j_inner_temp = right.leftmosts[j_inner - 1] - 1;
                    let j_temp2 = if right.leftmosts[j - 1] > j_inner_temp {
                        0
                    } else {
                        j_inner_temp
                    };

                    fdist[i_inner][j_inner] = (fdist[i_temp][j_inner] + DELETE)
                        .min(fdist[i_inner][j_temp] + INSERT)
                        .min(fdist[i_temp2][j_temp2] + td[i_inner][j_inner]);
                }
            }
        }
        fdist[i][j]
    }

    let left: TreeDiffData<L> = left.into();
    let right: TreeDiffData<L> = right.into();

    let mut td = vec![vec![0usize; left.leftmosts.len() + 1]; right.leftmosts.len() + 1];
    for i1 in 1..=left.keyroots.len() {
        for j1 in 1..=right.keyroots.len() {
            let i = left.keyroots[i1 - 1];
            let j = right.keyroots[j1 - 1];
            td[i][j] = inner(&left, &right, i, j, &mut td);
        }
    }

    td[left.leftmosts.len()][right.leftmosts.len()]
}

struct TreeDiffData<L: Language> {
    leftmosts: Vec<usize>,
    keyroots: Vec<usize>,
    labels: Vec<L::Discriminant>,
}

impl<L: Language> From<&RecExpr<L>> for TreeDiffData<L> {
    fn from(value: &RecExpr<L>) -> TreeDiffData<L> {
        fn traverse<'a, LL: Language>(
            node: &'a BorrowingRecNode<'a, LL>,
            labels: &mut Vec<LL::Discriminant>,
            leftmosts: &mut Vec<usize>,
        ) {
            for children in node.children() {
                traverse(children, labels, leftmosts);
            }
            labels.push(node.node().discriminant());
            leftmosts.push(node.leftmost().index());
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

        let root = value.into();
        let mut labels = Vec::new();
        let mut leftmosts = Vec::new();
        traverse(&root, &mut labels, &mut leftmosts);
        let keyroots = mk_keyroots(&leftmosts);

        TreeDiffData {
            leftmosts,
            keyroots,
            labels,
        }
    }
}
