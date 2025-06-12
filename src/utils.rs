use std::hash::Hash;

use egg::{Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct ExprHashCons<L: Hash + Eq> {
    expr: RecExpr<L>,
    memo: HashMap<L, Id>,
}

impl<L: Language> ExprHashCons<L> {
    pub fn new() -> Self {
        ExprHashCons {
            expr: RecExpr::default(),
            memo: HashMap::default(),
        }
    }

    pub(crate) fn add(&mut self, node: L) -> Id {
        if let Some(id) = self.memo.get(&node) {
            *id
        } else {
            self.expr.add(node)
        }
    }

    pub(crate) fn extract(&self, id: Id) -> RecExpr<L> {
        let all = self.expr.as_ref();

        let mut used = HashSet::new();
        used.insert(id);
        for (i, node) in all.iter().enumerate().rev() {
            if used.contains(&Id::from(i)) {
                for c in node.children() {
                    used.insert(*c);
                }
            }
        }

        let mut fresh = RecExpr::default();
        let mut map = HashMap::<Id, Id>::default();
        for (i, node) in all.iter().enumerate() {
            if used.contains(&Id::from(i)) {
                let fresh_node = node.clone().map_children(|c| map[&c]);
                let fresh_id = fresh.add(fresh_node);
                map.insert(Id::from(i), fresh_id);
            }
        }

        fresh
    }
}

struct RecNode<'a, L: Language> {
    node: &'a L,
    index: usize,
    children: Vec<RecNode<'a, L>>,
}

impl<L: Language> RecNode<'_, L> {
    fn leftmost(&self) -> &Self {
        self.children.first().unwrap_or(self)
    }
}

impl<'a, L: Language> From<&'a RecExpr<L>> for RecNode<'a, L> {
    fn from(value: &'a RecExpr<L>) -> Self {
        fn rec<'aa, LL: Language>(
            rec_expr: &'aa RecExpr<LL>,
            id: Id,
            index: &mut usize,
        ) -> RecNode<'aa, LL> {
            let curr_index = *index;
            *index += 1;
            let children = rec_expr[id]
                .children()
                .iter()
                .map(|c_id| rec(rec_expr, *c_id, index))
                .collect();

            RecNode {
                node: &rec_expr[id],
                children,
                index: curr_index,
            }
        }
        let mut index = 0;
        rec(value, value.root(), &mut index)
    }
}

pub(crate) struct TreeDiffData<L: Language> {
    leftmosts: Vec<usize>,
    keyroots: Vec<usize>,
    labels: Vec<L::Discriminant>,
}

impl<L: Language> TreeDiffData<L> {
    pub fn distance(&self, other: &TreeDiffData<L>) -> usize {
        fn treedist<LL: Language>(
            t1: &TreeDiffData<LL>,
            t2: &TreeDiffData<LL>,
            i: usize,
            j: usize,
            td: &mut [Vec<usize>],
        ) -> usize {
            const DELETE: usize = 1;
            const INSERT: usize = 1;
            const RELABEL: usize = 1;

            // forestdist
            let mut fdist = vec![vec![0usize; j + 1]; i + 1];

            // for (int i1 = l1.get(i - 1); i1 <= i; i1++) {
            for i_inner in t1.leftmosts[i - 1]..=i {
                fdist[i_inner][0] = fdist[i_inner - 1][0] + DELETE;
            }
            // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {

            for j_inner in t2.leftmosts[j - 1]..=j {
                fdist[0][j_inner] = fdist[0][j_inner - 1] + INSERT;
            }
            // for (int i1 = l1.get(i - 1); i1 <= i; i1++) {
            for i_inner in t1.leftmosts[i - 1]..=i {
                // for (int j1 = l2.get(j - 1); j1 <= j; j1++) {
                for j_inner in t2.leftmosts[j - 1]..=j {
                    let i_temp = if t1.leftmosts[i - 1] > i_inner - 1 {
                        0
                    } else {
                        i_inner - 1
                    };
                    let j_temp = if t2.leftmosts[j - 1] > j_inner - 1 {
                        0
                    } else {
                        j_inner - 1
                    };
                    if (t1.leftmosts[i_inner - 1] == t1.leftmosts[i - 1])
                        && (t2.leftmosts[j_inner - 1] == t2.leftmosts[j - 1])
                    {
                        let cost = if t1.labels[i_inner - 1] == t2.labels[j_inner - 1] {
                            0
                        } else {
                            RELABEL
                        };

                        fdist[i_inner][j_inner] = (fdist[i_temp][j_inner] + DELETE)
                            .min(fdist[i_inner][j_temp] + INSERT)
                            .min(fdist[i_temp][j_temp] + cost);
                        td[i_inner][j_inner] = fdist[i_inner][j_inner];
                    } else {
                        let i_inner_temp = t1.leftmosts[i_inner - 1] - 1;
                        let i_temp2 = if t1.leftmosts[i - 1] > i_inner_temp {
                            0
                        } else {
                            i_inner_temp
                        };

                        let j_inner_temp = t2.leftmosts[j_inner - 1] - 1;
                        let j_temp2 = if t2.leftmosts[j - 1] > j_inner_temp {
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

        let mut td = vec![vec![0usize; self.leftmosts.len() + 1]; other.leftmosts.len() + 1];
        for i1 in 1..=self.keyroots.len() {
            for j1 in 1..=other.keyroots.len() {
                let i = self.keyroots[i1 - 1];
                let j = other.keyroots[j1 - 1];
                td[i][j] = treedist(self, other, i, j, &mut td);
            }
        }

        td[self.leftmosts.len()][other.leftmosts.len()]
    }
}

impl<'a, L: Language> From<&'a RecExpr<L>> for TreeDiffData<L> {
    fn from(value: &'a RecExpr<L>) -> TreeDiffData<L> {
        fn traverse<'aa, 'bb: 'aa, LL: Language>(
            node: &'aa RecNode<'aa, LL>,
            labels: &mut Vec<LL::Discriminant>,
            leftmosts: &mut Vec<usize>,
        ) {
            for children in &node.children {
                traverse(children, labels, leftmosts);
            }
            labels.push(node.node.discriminant());
            leftmosts.push(node.leftmost().index);
        }

        fn mk_keyroots(l: &[usize]) -> Vec<usize> {
            let mut keyroots = vec![];
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
        let mut labels = vec![];
        let mut leftmosts = vec![];
        traverse(&root, &mut labels, &mut leftmosts);
        let keyroots = mk_keyroots(&leftmosts);

        TreeDiffData {
            leftmosts,
            keyroots,
            labels,
        }
    }
}
