/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::RecExpr;
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::*;
        use rayon::prelude::*;

        use $crate::eqsat::conf::EqsatConf;
        use $crate::eqsat::{Eqsat, StartMaterial};
        use $crate::error::EggshellError;
        use $crate::features::{AsFeatures, Feature, FeatureError, Featurizer};
        use $crate::python::raw_ast::RawAst;
        use $crate::trs::TermRewriteSystem;
        use $crate::utils::Tree;

        type L = <$type as TermRewriteSystem>::Language;
        // type N = <$type as TermRewriteSystem>::Analysis;

        #[gen_stub_pyclass]
        #[pyclass(module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyAst {
            raw_ast: RawAst<L>,
            rec_expr: RecExpr<L>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyAst {
            /// Parse from string
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(s_expr_str: &str, featurizer: &PyFeaturizer) -> pyo3::PyResult<Self> {
                let rec_expr = s_expr_str
                    .parse::<egg::RecExpr<L>>()
                    .map_err(|e| EggshellError::from(e))?;
                let raw_ast = RawAst::new(&rec_expr, &featurizer.0)
                    .map_err(|e| EggshellError::<L>::from(e))?;
                Ok(Self { raw_ast, rec_expr })
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.rec_expr.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[getter(name)]
            #[must_use]
            pub fn name(&self) -> String {
                self.raw_ast.node().to_string()
            }

            #[getter(is_leaf)]
            #[must_use]
            pub fn is_leaf(&self) -> bool {
                self.raw_ast.is_leaf()
            }

            #[getter(arity)]
            #[must_use]
            pub fn arity(&self) -> usize {
                self.raw_ast.arity()
            }

            #[must_use]
            pub fn size(&self) -> usize {
                self.raw_ast.size()
            }

            #[must_use]
            pub fn depth(&self) -> usize {
                self.raw_ast.depth()
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn count_symbols(&self, featurizer: &PyFeaturizer) -> pyo3::PyResult<Vec<usize>> {
                let x = self
                    .raw_ast
                    .count_symbols(&featurizer.0)
                    .map_err(|e| EggshellError::<L>::from(e))?;
                Ok(x)
            }

            #[expect(clippy::cast_precision_loss, clippy::missing_errors_doc)]
            pub fn feature_vec_simple(
                &self,
                featurizer: &PyFeaturizer,
            ) -> pyo3::PyResult<Vec<f64>> {
                let mut features = self
                    .raw_ast
                    .count_symbols(&featurizer.0)
                    .map_err(|e| EggshellError::<L>::from(e))?;
                features.push(self.size());
                features.push(self.depth());
                Ok(features.into_iter().map(|v| v as f64).collect())
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn feature_vec_ml(&self) -> pyo3::PyResult<Vec<f64>> {
                match self.raw_ast.features() {
                    Feature::Leaf(f) => Ok(f.to_owned()),
                    Feature::NonLeaf(_) => {
                        Err(EggshellError::<L>::from(FeatureError::NonLeaf(self.name())))?
                    }
                    Feature::IgnoredSymbol => Err(EggshellError::<L>::from(
                        FeatureError::IgnoredSymbol(self.name()),
                    ))?,
                }
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn node_id(&self) -> pyo3::PyResult<usize> {
                match self.raw_ast.features() {
                    Feature::NonLeaf(id) => Ok(*id),
                    Feature::Leaf(_) => {
                        Err(EggshellError::<L>::from(FeatureError::Leaf(self.name())))?
                    }
                    Feature::IgnoredSymbol => Err(EggshellError::<L>::from(
                        FeatureError::IgnoredSymbol(self.name()),
                    ))?,
                }
            }
        }

        #[gen_stub_pyclass]
        #[pyclass(module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyFeaturizer(Featurizer<L>);

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyFeaturizer {
            /// Parse from string
            #[new]
            #[pyo3(signature = (variable_names, ignore_unknown=false))]
            #[must_use]
            pub fn new(variable_names: Vec<String>, ignore_unknown: bool) -> Self {
                let mut featurizer = L::featurizer(variable_names);
                if ignore_unknown {
                    featurizer.set_ignore_unknown(true);
                }
                PyFeaturizer(featurizer)
            }

            #[must_use]
            pub fn feature_names_simple(&self) -> Vec<String> {
                let mut symbol_names = self.0.symbol_names();
                symbol_names.push(String::from("SIZE"));
                symbol_names.push(String::from("DEPTH"));
                symbol_names
            }
        }

        #[expect(clippy::cast_precision_loss, clippy::missing_errors_doc)]
        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        pub fn many_featurize_simple<'py>(
            py: pyo3::Python<'py>,
            expression_strings: Vec<String>,
            featurizer: &PyFeaturizer,
        ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray2<f64>>> {
            let rust_vec = expression_strings
                .par_iter()
                .map(|s| {
                    let raw_sketch = s
                        .parse::<egg::RecExpr<L>>()
                        .map_err(|e| EggshellError::from(e))?;
                    let raw_ast = RawAst::new(&raw_sketch, &featurizer.0)?;
                    let mut features = raw_ast.count_symbols(&featurizer.0)?;
                    features.push(raw_ast.size());
                    features.push(raw_ast.depth());
                    Ok(features.into_iter().map(|v| v as f64).collect())
                })
                .collect::<Result<Vec<_>, EggshellError<_>>>()?;

            Ok(numpy::PyArray::from_vec2(py, &rust_vec).unwrap())
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn eqsat_check(start: PyAst, goals: Vec<PyAst>) -> String {
            let conf = EqsatConf::builder().root_check(true).build();
            let start_material = StartMaterial::RecExprs(vec![start.rec_expr]);
            let goals = goals.into_iter().map(|x| x.rec_expr).collect();
            let rules = <$type as TermRewriteSystem>::full_rules();
            let eqsat_result = Eqsat::new(start_material)
                .with_conf(conf)
                .with_goals(goals)
                .run(&rules);
            serde_json::to_string(&eqsat_result).unwrap()
        }

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<PyAst>()?;
            module.add_class::<PyFeaturizer>()?;

            module.add_function(pyo3::wrap_pyfunction!(eqsat_check, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(many_featurize_simple, m)?)?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub mod simple {
    monomorphize!(crate::trs::Simple, "eggshell.simple");
}

pub mod arithmetic {
    monomorphize!(crate::trs::Arithmetic, "eggshell.arithmetic");
}

pub mod halide {
    monomorphize!(crate::trs::Halide, "eggshell.halide");
}

pub mod rise {
    monomorphize!(crate::trs::Rise, "eggshell.rise");
}

use pyo3_stub_gen::define_stub_info_gatherer;
define_stub_info_gatherer!(stub_info);
