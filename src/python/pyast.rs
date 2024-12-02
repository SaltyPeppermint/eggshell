/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        type Lang = <$type as $crate::trs::TermRewriteSystem>::Language;

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let bound = pyo3::prelude::PyModule::new_bound(m.py(), module_name)?;

            m.add_submodule(&bound)?;
            Ok(())
        }

        #[pyo3::pyclass]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyAst(pub(crate) $crate::python::raw_ast::RawAst);

        #[pyo3::pymethods]
        impl PyAst {
            /// Parse from string
            #[staticmethod]
            pub fn new(s_expr_str: &str, symbol_list: Vec<String>) -> pyo3::PyResult<Self> {
                let raw_sketch = s_expr_str
                    .parse::<egg::RecExpr<Lang>>()
                    .map_err(|e| $crate::python::EggError::RecExprParse(e))?;
                Ok(PyAst($crate::python::raw_ast::RawAst::new(
                    &raw_sketch,
                    symbol_list,
                )))
            }

            fn __str__(&self) -> String {
                self.0.to_string()
            }

            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            pub fn name(&self) -> String {
                self.0.name().to_owned()
            }

            pub fn size(&self) -> usize {
                <$crate::python::raw_ast::RawAst as $crate::utils::Tree>::size(&self.0)
            }

            pub fn depth(&self) -> usize {
                <$crate::python::raw_ast::RawAst as $crate::utils::Tree>::depth(&self.0)
            }

            pub fn is_leaf(&self) -> bool {
                self.0.is_leaf()
            }

            pub fn arity(&self) -> usize {
                self.0.arity()
            }

            pub fn feature_vec(&self) -> Option<Vec<f64>> {
                match self.0.features() {
                    $crate::features::Feature::Leaf(f) => Some(f.clone()),
                    $crate::features::Feature::NonLeaf(_) => None,
                }
            }

            pub fn node_id(&self) -> Option<usize> {
                match self.0.features() {
                    $crate::features::Feature::NonLeaf(id) => Some(*id),
                    $crate::features::Feature::Leaf(_) => None,
                }
            }
        }

        impl From<$crate::python::raw_ast::RawAst> for PyAst {
            fn from(value: $crate::python::raw_ast::RawAst) -> Self {
                PyAst(value)
            }
        }
    };
}

pub mod simple {
    monomorphize!(crate::trs::Simple);
}

pub mod arithmatic {
    monomorphize!(crate::trs::Arithmetic);
}

pub mod halide {
    monomorphize!(crate::trs::Halide);
}

pub mod rise {
    monomorphize!(crate::trs::Rise);
}
