// case class CountProgramsUpToSize(limit: Int) extends CommutativeSemigroupAnalysis {
//     override type Data = HashMap[Int, Long]

//     override def requiredAnalyses(): (Set[Analysis], Set[TypeAnalysis]) = (Set(), Set())

//     override def make(egraph: EGraph, enode: ENode, t: TypeId,
//                       analysisOf: EClassId => Data): Data = {
//       val counts = HashMap.empty[Int, Long]
//       val childrenCounts = enode.children().map(analysisOf).toSeq

//       def rec(remaining: Seq[HashMap[Int, Long]], size: Int, count: Long): Unit = {
//         if (size > limit) {
//           return
//         }
//         remaining match {
//           case Nil =>
//             val total = counts.getOrElse(size, 0L) + count
//             counts += (size -> total)
//           case childCounts +: rest =>
//             childCounts.foreach { case (s, c) =>
//               rec(rest, size + s, count * c)
//             }
//         }
//       }

//       rec(childrenCounts, 1, 1)
//       counts
//     }

//     override def merge(a: HashMap[Int, Long], b: HashMap[Int, Long]): HashMap[Int, Long] = {
//       b.foreach { case (size, count) =>
//         val total = a.getOrElse(size, 0L) + count
//         a += size -> total
//       }
//       a
//     }
//   }

use egg::{Analysis, DidMerge, EGraph, Language};
use hashbrown::HashMap;

use super::SemiLatticeAnalysis;

pub struct TermsUpToSize {
    limit: usize,
}

impl TermsUpToSize {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<L, N> SemiLatticeAnalysis<L, N> for TermsUpToSize
where
    L: Language,
    N: Analysis<L>,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, usize>;

    fn make<'a>(
        &mut self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(egg::Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
        Self: 'a,
    {
        fn rec(
            remaining: &[&HashMap<usize, usize>],
            size: usize,
            count: usize,
            counts: &mut HashMap<usize, usize>,
            limit: usize,
        ) {
            dbg!(size);
            if size > limit {
                return;
            }

            if let Some((head, rest)) = remaining.split_first() {
                for (s, c) in *head {
                    dbg!(s);
                    rec(rest, size + s, count * c, counts, limit);
                }
            } else {
                dbg!("in leaf case");
                let total = counts.get(&size).unwrap_or(&0) + count;
                counts.insert(size, total);
            }
        }
        let children_counts = enode
            .children()
            .iter()
            .map(|c_id| analysis_of(*c_id))
            .collect::<Vec<_>>();
        dbg!(&children_counts);
        let mut counts = HashMap::new();
        rec(&children_counts, 1, 1, &mut counts, self.limit);
        counts
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        for (size, count) in b {
            dbg!("ADDING IN MERGE");
            let total = a.get(&size).unwrap_or(&0) + count;
            a.insert(size, total);
        }
        DidMerge(true, false)
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};

    use super::*;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let mut data = HashMap::new();
        TermsUpToSize::new(7).one_shot_analysis(&egraph, &mut data);

        let root_data = &data[&egraph.find(apb)];
        dbg!(&data);
        assert_eq!(root_data[&5], 1);
    }
}
