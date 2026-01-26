/// Macro to generate newtype ID wrappers around `usize`
///
/// # Example
/// ```
/// define_id!(EClassId);
/// define_id!(NatId);
/// ```
macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(usize);

        impl $name {
            #[must_use]
            pub fn new(id: usize) -> Self {
                Self(id)
            }
        }

        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

define_id!(EClassId);
define_id!(NatId);
define_id!(FunTyId);
define_id!(DataTyId);

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum TypeId {
    Nat(NatId),
    Type(FunTyId),
    DataType(DataTyId),
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum NatOrDTId {
    Nat(NatId),
    DataType(DataTyId),
}
