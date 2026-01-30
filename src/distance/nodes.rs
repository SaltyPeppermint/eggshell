use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use super::ids::{DataChildId, ExprChildId, NatId, TypeChildId};

/// Trait for node labels in e-graphs and trees.
pub trait Label:
    Clone + Eq + std::hash::Hash + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + Send + Sync
{
    /// Returns the label used for type annotations (e.g., "typeOf").
    fn type_of() -> Self;

    /// Returns true if this label is the type annotation label.
    fn is_type_of(&self) -> bool {
        &Self::type_of() == self
    }
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
    /// Expression node in an e-graph. Children reference `EClass` entries.
    ENode, ExprChildId
);

define_node!(
    /// Node for natural number expressions. Children are nats.
    NatNode, NatId
);

define_node!(
    /// Node for function type expressions. Children can be function types, datatypes, or nats.
    FunTyNode, TypeChildId
);

define_node!(
    /// Node for datatype expressions. Children can be datatypes or nats.
    DataTyNode, DataChildId
);
