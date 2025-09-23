/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::RecExpr as EggRecExpr;
        use pyo3::prelude::*;

        use $crate::eqsat::{self, EqsatConf};
        use $crate::meta_lang::Sketch;
        use $crate::python::err::EggshellError;
        use $crate::rewrite_system::RewriteSystem;

        type L = <$type as RewriteSystem>::Language;

        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct RecExpr(EggRecExpr<L>);

        #[pymethods]
        impl RecExpr {
            /// Parse from string
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(s_expr_str: &str) -> PyResult<RecExpr> {
                let rec_expr = s_expr_str
                    .parse::<egg::RecExpr<L>>()
                    .map_err(|e: egg::RecExprParseError<_>| EggshellError::<L>::from(e))?;
                Ok(RecExpr(rec_expr))
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.0.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[pyo3(signature = (name, path, marked_ids=None, transparent=false))]
            pub fn to_dot(
                &self,
                name: String,
                path: String,
                marked_ids: Option<Vec<usize>>,
                transparent: bool,
            ) {
                let dot = $crate::viz::to_dot(
                    &self.0,
                    &name,
                    &(marked_ids
                        .unwrap_or_default()
                        .into_iter()
                        .map(|id| egg::Id::from(id))
                        .collect()),
                    transparent,
                );
                let svg = $crate::viz::dot_to_svg(&dot);
                let path = std::path::PathBuf::from(path).with_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }

            #[must_use]
            pub fn tree_distance(&self, other: &RecExpr) -> usize {
                $crate::tree_distance::distance(&self.0, &other.0)
            }
        }

        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct Guide(Sketch<L>);

        #[pymethods]
        impl Guide {
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(s_expr_str: String) -> PyResult<Guide> {
                let sketch = s_expr_str
                    .parse::<Sketch<L>>()
                    .map_err(|e: egg::RecExprParseError<_>| EggshellError::<L>::from(e))?;
                Ok(Guide(sketch))
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.0.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[must_use]
            pub fn tree_distance(&self, other: &Guide) -> usize {
                $crate::tree_distance::distance(&self.0, &other.0)
            }

            #[pyo3(signature = (name, path, marked_ids=None, transparent=false))]
            pub fn to_dot(
                &self,
                name: String,
                path: String,
                marked_ids: Option<Vec<usize>>,
                transparent: bool,
            ) {
                let dot = $crate::viz::to_dot(
                    &self.0,
                    &name,
                    &(marked_ids
                        .unwrap_or_default()
                        .into_iter()
                        .map(|id| egg::Id::from(id))
                        .collect()),
                    transparent,
                );
                let svg = $crate::viz::dot_to_svg(&dot);
                let path = std::env::current_dir()
                    .unwrap()
                    .join(path)
                    .with_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }
        }

        #[pyfunction]
        #[must_use]
        #[pyo3(signature = (start, target, iter_limit=None, node_limit=None, time_limit=None))]
        pub fn eqsat_check(
            start: &RecExpr,
            target: &RecExpr,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            time_limit: Option<f64>,
        ) -> (String, bool) {
            let conf = EqsatConf::builder()
                .maybe_iter_limit(iter_limit)
                .maybe_node_limit(node_limit)
                .maybe_time_limit(time_limit.map(std::time::Duration::from_secs_f64))
                .build();

            let eqsat_result = eqsat::eqsat(
                conf,
                (&start.0).into(),
                &<$type as RewriteSystem>::full_rules(),
                Some(target.0.clone()),
                egg::SimpleScheduler,
            );
            if let egg::StopReason::Other(stop_message) = &eqsat_result.report().stop_reason
                && stop_message.contains("Target")
            {
                return (serde_json::to_string(&eqsat_result).unwrap(), true);
            }
            (serde_json::to_string(&eqsat_result).unwrap(), false)
        }

        #[pyfunction]
        #[must_use]
        #[pyo3(signature = (start, guide, target, iter_limit=None, node_limit=None, time_limit=None))]
        pub fn eqsat_guide_check(
            start: &RecExpr,
            guide: &Guide,
            target: &RecExpr,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            time_limit: Option<f64>,
        ) -> (String, Option<(String, String)>, bool) {
            let conf = EqsatConf::builder()
                .maybe_iter_limit(iter_limit)
                .maybe_node_limit(node_limit)
                .maybe_time_limit(time_limit.map(std::time::Duration::from_secs_f64))
                .build();

            let eqsat_result = eqsat::eqsat(
                conf.clone(),
                (&start.0).into(),
                &<$type as RewriteSystem>::full_rules(),
                Some(target.0.clone()),
                egg::SimpleScheduler,
            );
            let first_report_str = serde_json::to_string(&eqsat_result).unwrap();
            if let egg::StopReason::Other(s) = &eqsat_result.report().stop_reason
                && s.contains("Target")
            {
                return (first_report_str, None, false);
            }

            for root in eqsat_result.roots() {
                if let Some((_, extracted_guide)) = $crate::meta_lang::sketch::eclass_extract(
                    &guide.0,
                    egg::AstSize,
                    &eqsat_result.egraph(),
                    *root,
                ) {
                    let eqsat_result_2 = eqsat::eqsat(
                        conf.clone(),
                        (&extracted_guide).into(),
                        &<$type as RewriteSystem>::full_rules(),
                        Some(target.0.clone()),
                        egg::SimpleScheduler,
                    );
                    let second_report_str = serde_json::to_string(&eqsat_result_2).unwrap();

                    if let egg::StopReason::Other(stop_message) =
                        &eqsat_result_2.report().stop_reason
                        && stop_message.contains("Target")
                    {
                        return (
                            first_report_str,
                            Some((second_report_str, extracted_guide.to_string())),
                            true,
                        );
                    } else {
                        return (
                            first_report_str,
                            Some((second_report_str, extracted_guide.to_string())),
                            false,
                        );
                    }
                }
            }
            (first_report_str, None, false)
        }

        #[pyfunction]
        #[must_use]
        #[pyo3(signature = (start, guide, target, iter_limit=None, node_limit=None, time_limit=None))]
        pub fn eqsat_two_guide_check(
            start: &RecExpr,
            guide: &Guide,
            target: &Guide,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            time_limit: Option<f64>,
        ) -> (String, Option<(String, String)>, Option<String>) {
            let conf = EqsatConf::builder()
                .maybe_iter_limit(iter_limit)
                .maybe_node_limit(node_limit)
                .maybe_time_limit(time_limit.map(std::time::Duration::from_secs_f64))
                .build();

            let eqsat_result = eqsat::eqsat(
                conf.clone(),
                (&start.0).into(),
                &<$type as RewriteSystem>::full_rules(),
                None,
                egg::SimpleScheduler,
            );
            let first_report_str = serde_json::to_string(&eqsat_result).unwrap();

            for root in eqsat_result.roots() {
                if let Some((_, extracted_guide)) = $crate::meta_lang::sketch::eclass_extract(
                    &guide.0,
                    egg::AstSize,
                    &eqsat_result.egraph(),
                    *root,
                ) {
                    let eqsat_result_2 = eqsat::eqsat(
                        conf.clone(),
                        (&extracted_guide).into(),
                        &<$type as RewriteSystem>::full_rules(),
                        None,
                        egg::SimpleScheduler,
                    );
                    let second_report_str = serde_json::to_string(&eqsat_result_2).unwrap();
                    if let Some((_, extracted_target)) = $crate::meta_lang::sketch::eclass_extract(
                        &target.0,
                        egg::AstSize,
                        &eqsat_result.egraph(),
                        *root,
                    ) {
                        return (
                            first_report_str,
                            Some((second_report_str, extracted_guide.to_string())),
                            Some(extracted_target.to_string()),
                        );
                    } else {
                        return (
                            first_report_str,
                            Some((second_report_str, extracted_guide.to_string())),
                            None,
                        );
                    }
                }
            }
            (first_report_str, None, None)
        }

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<RecExpr>()?;
            module.add_class::<Guide>()?;

            module.add_function(pyo3::wrap_pyfunction!(eqsat_check, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(eqsat_guide_check, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(eqsat_two_guide_check, m)?)?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub(crate) use monomorphize;
