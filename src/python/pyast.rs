/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        type Lang = <$type as $crate::trs::TermRewriteSystem>::Language;

        #[pyo3::pyclass]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyAst($crate::python::raw_ast::RawAst<Lang>);

        #[pyo3::pymethods]
        impl PyAst {
            /// Parse from string
            #[new]
            pub fn new(s_expr_str: &str, featurizer: &PyFeaturizer) -> pyo3::PyResult<Self> {
                let raw_sketch = s_expr_str.parse::<egg::RecExpr<Lang>>().map_err(|e| {
                    $crate::EggshellError::from($crate::python::EggError::RecExprParse(e))
                })?;
                let raw_ast = $crate::python::raw_ast::RawAst::new(&raw_sketch, &featurizer.0)
                    .map_err(|e| $crate::EggshellError::<Lang>::from(e))?;
                Ok(Self(raw_ast))
            }

            fn __str__(&self) -> String {
                self.0.to_string()
            }

            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[getter(name)]
            pub fn name(&self) -> String {
                self.0.node().to_string()
            }

            #[getter(is_leaf)]
            pub fn is_leaf(&self) -> bool {
                self.0.is_leaf()
            }

            #[getter(arity)]
            pub fn arity(&self) -> usize {
                self.0.arity()
            }

            pub fn size(&self) -> usize {
                <$crate::python::raw_ast::RawAst<Lang> as $crate::utils::Tree>::size(&self.0)
            }

            pub fn depth(&self) -> usize {
                <$crate::python::raw_ast::RawAst<Lang> as $crate::utils::Tree>::depth(&self.0)
            }

            pub fn count_symbols(&self, variable_names: Vec<String>) -> Vec<usize> {
                self.0.count_symbols(variable_names)
            }

            #[expect(clippy::cast_precision_loss)]
            pub fn feature_vec_simple(&self, variable_names: Vec<String>) -> Vec<f64> {
                let mut features = self.0.count_symbols(variable_names.clone());
                features.push(self.size());
                features.push(self.depth());
                features.into_iter().map(|v| v as f64).collect()
            }

            pub fn feature_vec_ml(&self) -> pyo3::PyResult<Vec<f64>> {
                match self.0.features() {
                    $crate::features::Feature::Leaf(f) => Ok(f.to_owned()),
                    $crate::features::Feature::NonLeaf(_) => {
                        Err($crate::EggshellError::<Lang>::from(
                            $crate::features::FeatureError::NonLeaf(self.name()),
                        ))?
                    }
                }
            }

            pub fn node_id(&self) -> pyo3::PyResult<usize> {
                match self.0.features() {
                    $crate::features::Feature::NonLeaf(id) => Ok(*id),
                    $crate::features::Feature::Leaf(_) => {
                        Err($crate::EggshellError::<Lang>::from(
                            $crate::features::FeatureError::Leaf(self.name()),
                        ))?
                    }
                }
            }

            pub fn invert_flatten(&self) -> Vec<Self> {
                self.0
                    .flatten()
                    .into_iter()
                    .map(|c| Self(c.to_owned()))
                    .rev()
                    .collect()
            }
        }

        #[pyo3::pyclass]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyFeaturizer($crate::features::Featurizer<Lang>);

        #[pyo3::pymethods]
        impl PyFeaturizer {
            /// Parse from string
            #[new]
            pub fn new(variable_names: Vec<String>) -> Self {
                PyFeaturizer(<Lang as $crate::features::AsFeatures>::featurizer(
                    variable_names,
                ))
            }

            pub fn feature_names_simple(&self) -> Vec<String> {
                let mut symbol_names = self.0.symbol_names();
                symbol_names.push(String::from("SIZE"));
                symbol_names.push(String::from("DEPTH"));
                symbol_names
            }

            pub fn __setstate__(&mut self, state: Vec<u8>) -> pyo3::PyResult<()> {
                self.0 = postcard::from_bytes(&state).unwrap();
                Ok(())
            }

            pub fn __getstate__(&self) -> pyo3::PyResult<Vec<u8>> {
                Ok(postcard::to_allocvec(&self.0).unwrap())
            }

            pub fn __getnewargs__(&self) -> pyo3::PyResult<Vec<String>> {
                Ok(self
                    .0
                    .symbols()
                    .iter()
                    .filter(|s| {
                        matches!(
                            <Lang as $crate::features::AsFeatures>::symbol_type(s),
                            $crate::features::SymbolType::Variable(_)
                        )
                    })
                    .map(|s| s.to_string())
                    .collect())
            }
        }

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<PyAst>()?;
            // module.add_function(pyo3::wrap_pyfunction!(str_to_simple_feature_vec, m)?)?;
            module.add_class::<PyFeaturizer>()?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub mod simple {
    monomorphize!(crate::trs::Simple);
}

pub mod arithmetic {
    monomorphize!(crate::trs::Arithmetic);
}

pub mod halide {
    monomorphize!(crate::trs::Halide);
}

pub mod rise {
    monomorphize!(crate::trs::Rise);
}
