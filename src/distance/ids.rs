use std::str::FromStr;

// use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Macro to generate newtype ID wrappers around `usize`
///
/// # Example
/// ```
/// define_id!(EClassId, "EClassId");
/// define_id!(NatId, "n");
/// ```
macro_rules! define_id {
    ($name:ident, $prefix:literal) => {
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

        impl FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                s.trim()
                    .strip_prefix($prefix)
                    .and_then(|r| r.trim().strip_prefix('('))
                    .and_then(|r| r.strip_suffix(')'))
                    .ok_or_else(|| format!("expected '{}(id)' format", $prefix))?
                    .trim()
                    .parse::<usize>()
                    .map(Self::new)
                    .map_err(|e| format!("failed to parse id: {}", e))
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&format!("{}({})", $prefix, self.0))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let s = String::deserialize(deserializer)?;
                s.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}

define_id!(EClassId, "EClassId");
define_id!(NatId, "NatId");
define_id!(FunTyId, "NotDataTypeId");
define_id!(DataTyId, "DataTypeId");

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeId {
    Nat(NatId),
    Type(FunTyId),
    DataType(DataTyId),
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NatOrDTId {
    Nat(NatId),
    DataType(DataTyId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_eclass_id() {
        assert_eq!("EClassId(42)".parse(), Ok(EClassId::new(42)));
        assert_eq!("EClassId(0)".parse(), Ok(EClassId::new(0)));
        assert_eq!("  EClassId( 123 )  ".parse(), Ok(EClassId::new(123)));
    }

    #[test]
    fn parse_nat_id() {
        assert_eq!("NatId(2342)".parse(), Ok(NatId::new(2342)));
    }

    #[test]
    fn parse_funty_id() {
        assert_eq!("NotDataTypeId(141)".parse(), Ok(FunTyId::new(141)));
    }

    #[test]
    fn parse_dataty_id() {
        assert_eq!("DataTypeId(242)".parse(), Ok(DataTyId::new(242)));
    }

    #[test]
    fn parse_invalid() {
        assert!("EClassId(abc)".parse::<EClassId>().is_err());
        assert!("WrongPrefix(42)".parse::<EClassId>().is_err());
        assert!("EClassId42".parse::<EClassId>().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let id = EClassId::new(99);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"EClassId(99)\"");
        let parsed: EClassId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }
}
