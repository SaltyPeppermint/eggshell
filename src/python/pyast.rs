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

            let bound = pyo3::prelude::PyModule::new(m.py(), module_name)?;

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
            #[new]
            pub fn new(s_expr_str: &str, symbol_list: Vec<String>) -> pyo3::PyResult<Self> {
                let raw_sketch = s_expr_str
                    .parse::<egg::RecExpr<Lang>>()
                    .map_err(|e| $crate::python::EggError::RecExprParse(e))?;
                let raw_ast = $crate::python::raw_ast::RawAst::new(&raw_sketch, symbol_list);
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
                self.0.name().to_owned()
            }

            #[getter(is_leaf)]
            pub fn is_leaf(&self) -> bool {
                self.0.is_leaf()
            }

            #[getter(arity)]
            pub fn arity(&self) -> usize {
                self.0.arity()
            }

            pub fn children(&self) -> Vec<Self> {
                <$crate::python::raw_ast::RawAst as $crate::utils::Tree>::children(&self.0)
                    .iter()
                    .map(|c| Self(c.to_owned()))
                    .collect()
            }

            pub fn size(&self) -> usize {
                <$crate::python::raw_ast::RawAst as $crate::utils::Tree>::size(&self.0)
            }

            pub fn depth(&self) -> usize {
                <$crate::python::raw_ast::RawAst as $crate::utils::Tree>::depth(&self.0)
            }

            pub fn feature_vec<'py>(
                &self,
                py: pyo3::Python<'py>,
            ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f64>>> {
                match self.0.features() {
                    $crate::features::Feature::Leaf(f) => Ok(numpy::PyArray::from_slice(py, f)),
                    $crate::features::Feature::NonLeaf(_) => {
                        Err(pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Tried calling feature_vec on a non-leaf node",
                        ))
                    }
                }
            }

            pub fn node_id(&self) -> pyo3::PyResult<usize> {
                match self.0.features() {
                    $crate::features::Feature::NonLeaf(id) => Ok(*id),
                    $crate::features::Feature::Leaf(_) => {
                        Err(pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Tried calling node_id on a leaf node",
                        ))
                    }
                }
            }
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
