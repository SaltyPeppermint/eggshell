use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};

use super::raw_ast::RawAstError;

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
            /// This always generates a new node that has [open] as its children
            #[new]
            fn new(node: &str, arity: usize) -> pyo3::PyResult<Self> {
                let new_children = vec![$crate::python::raw_ast::RawAst::Open; arity];
                let raw_sketch = $crate::python::raw_ast::RawAst::new(node, new_children)?;
                Ok(PyAst(raw_sketch))
            }

            /// Generate a new root with an [active] node
            #[staticmethod]
            pub fn new_root() -> Self {
                PyAst($crate::python::raw_ast::RawAst::Active)
            }

            /// Parse from string
            #[staticmethod]
            pub fn from_str(s_expr_str: &str) -> pyo3::PyResult<Self> {
                let raw_sketch = s_expr_str
                    .parse()
                    .map_err($crate::python::raw_ast::RawAstError::BadSexp)?;
                Ok(PyAst(raw_sketch))
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
                <$crate::python::raw_ast::RawAst as crate::utils::Tree>::size(&self.0)
            }

            pub fn depth(&self) -> usize {
                <$crate::python::raw_ast::RawAst as crate::utils::Tree>::depth(&self.0)
            }

            /// Appends at the current [active] node and turns an open [open]
            /// into a new [active]
            /// Returns if the sketch is finished
            pub fn append(&mut self, new_child: Self) -> bool {
                self.0.append(new_child.0)
            }

            /// Checks if sketch has open [active]
            pub fn finished(&self) -> bool {
                self.0.finished()
            }

            /// Checks if sketch has open [active]
            pub fn sketch_symbols(&self) -> usize {
                self.0.sketch_symbols()
            }

            /// Checks if it is a sketch
            pub fn is_sketch(&self) -> bool {
                self.0.is_sketch()
            }

            /// Checks if it is a sketch
            pub fn is_partial_sketch(&self) -> bool {
                self.0.is_partial_sketch()
            }

            pub fn features(&self) -> Option<Vec<f64>> {
                self.0.features()
            }
        }

        impl From<$crate::python::raw_ast::RawAst> for PyAst {
            fn from(value: $crate::python::raw_ast::RawAst) -> Self {
                PyAst(value)
            }
        }

        impl From<&egg::RecExpr<Lang>> for PyAst {
            fn from(expr: &egg::RecExpr<Lang>) -> Self {
                let raw_ast = expr.into();
                PyAst(raw_ast)
            }
        }

        impl From<&crate::sketch::PartialSketch<Lang>> for PyAst {
            fn from(sketch: &$crate::sketch::PartialSketch<Lang>) -> Self {
                let raw_partial_sketch = sketch.into();
                PyAst(raw_partial_sketch)
            }
        }

        impl From<&$crate::sketch::Sketch<Lang>> for PyAst {
            fn from(sketch: &$crate::sketch::Sketch<Lang>) -> Self {
                let raw_sketch = sketch.into();
                PyAst(raw_sketch)
            }
        }

        impl std::str::FromStr for PyAst {
            type Err = $crate::python::raw_ast::RawAstParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let raw_sketch = s.parse()?;
                Ok(PyAst(raw_sketch))
            }
        }
    };
}

// /// Macro to generate the necessary implementations so pyo3 doesnt freak out about
// /// self-referential types using boxes
// // macro_rules! pyboxable {
// //     ($type: ty) => {
// //         impl pyo3::FromPyObject<'_> for std::boxed::Box<$type> {
// //             fn extract_bound(ob: &Bound<'_, PyAny>) -> pyo3::PyResult<Self> {
// //                 ob.extract::<$type>().map(Box::new)
// //             }
// //         }
// //         impl pyo3::IntoPy<pyo3::PyObject> for std::boxed::Box<$type> {
// //             fn into_py(self, py: pyo3::Python<'_>) -> pyo3::PyObject {
// //                 (*self).into_py(py)
// //             }
// //         }
// //     };
// // }

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

impl From<RawAstError> for PyErr {
    fn from(err: RawAstError) -> PyErr {
        PySketchException::new_err(err.to_string())
    }
}

create_exception!(
    eggshell,
    PySketchException,
    PyException,
    "Error dealing with a PySketch."
);
