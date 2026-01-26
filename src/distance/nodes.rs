use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use super::ids::{EClassId, NatId, NatOrDTId, TypeId};

/// Trait for node labels
pub trait Label:
    Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>
{
    fn type_of() -> Self;
}

impl Label for String {
    fn type_of() -> Self {
        "typeOf".to_owned()
    }
}

impl Label for Cow<'_, str> {
    fn type_of() -> Self {
        Cow::from("typeOf")
    }
}

/// Macro to generate node types with configurable ID types
///
/// # Example
/// ```
/// define_node!(
///     /// A node for natural numbers
///     NatNode, NatId
/// );
/// define_node!(
///     /// A node for types
///     TyNode, TyId
/// );
/// ```
macro_rules! define_node {
    ($(#[$meta:meta])* $name:ident, $id_type:ty) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name<L> {
            #[serde(rename = "node")]
            label: L,
            children: Vec<$id_type>,
        }

        impl<L: Label> $name<L> {
            pub fn leaf(label: L) -> Self {
                Self {
                    label,
                    children: Vec::new(),
                }
            }

            pub fn new(label: L, children: Vec<$id_type>) -> Self {
                Self { label, children }
            }

            pub fn label(&self) -> &L {
                &self.label
            }

            pub fn children(&self) -> &[$id_type] {
                &self.children
            }
        }
    };
}

define_node!(
    /// `ENode` must take all children
    /// Children are indices into the `EGraph` array `EClasses`
    ENode, EClassId
);

define_node!(
    /// Node for natural number expressions
    /// Children are either indices into the nat array or the
    NatNode, NatId
);

define_node!(
    /// Node for type expressions
    FunTyNode, TypeId
);

define_node!(
    /// Node for datatype expressions
    DataTyNode, NatOrDTId
);
