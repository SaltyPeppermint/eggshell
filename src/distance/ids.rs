use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Trait for ID types that wrap a numeric index.
pub trait NumericId: Sized + Copy + Eq + std::hash::Hash {
    /// Create an ID from a numeric index.
    fn from_index(index: usize) -> Self;
    /// Convert the ID back to its numeric index.
    fn to_index(self) -> usize;
}

/// Macro to generate newtype ID wrappers around `usize`
///
/// # Example
/// ```
/// define_id!(EClassId, "EClassId");
/// define_id!(NatId, "n");
/// ```
macro_rules! define_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
        pub struct $name(usize);

        impl $name {
            #[must_use]
            pub fn new(id: usize) -> Self {
                Self(id)
            }
        }

        impl NumericId for $name {
            fn from_index(index: usize) -> Self {
                Self::new(index)
            }

            fn to_index(self) -> usize {
                self.0
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

/// Type identifier: can be a nat, function type, or datatype.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypeId {
    Nat(NatId),
    Type(FunTyId),
    DataType(DataTyId),
}

/// Identifier for nat or datatype nodes.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NatOrDTId {
    Nat(NatId),
    DataType(DataTyId),
}

/// Serde helpers for `Vec<EClassId>` (e.g., `[0, 1, 2]` -> `Vec<EClassId>`)
pub mod eclass_id_vec {
    use super::EClassId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(vec: &[EClassId], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let indices: Vec<usize> = vec.iter().map(|k| k.0).collect();
        indices.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<EClassId>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<usize> = Vec::deserialize(deserializer)?;
        Ok(vec.into_iter().map(EClassId::new).collect())
    }
}

/// Serde helpers for `HashMaps` with numeric string keys (e.g., "0", "1", "2")
pub mod numeric_key_map {
    use super::NumericId;

    use hashbrown::HashMap;
    use serde::de::MapAccess;
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<K, V, S>(map: &HashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: NumericId,
        V: Serialize,
        S: Serializer,
    {
        let mut ser_map = serializer.serialize_map(Some(map.len()))?;
        for (k, v) in map {
            ser_map.serialize_entry(&k.to_index().to_string(), v)?;
        }
        ser_map.end()
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: NumericId,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        struct NumericKeyMapVisitor<K, V>(std::marker::PhantomData<(K, V)>);

        impl<'de, K, V> serde::de::Visitor<'de> for NumericKeyMapVisitor<K, V>
        where
            K: NumericId,
            V: Deserialize<'de>,
        {
            type Value = HashMap<K, V>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a map with numeric string keys")
            }

            fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut map = HashMap::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((key, value)) = access.next_entry::<String, V>()? {
                    let index: usize = key.parse().map_err(serde::de::Error::custom)?;
                    map.insert(K::from_index(index), value);
                }
                Ok(map)
            }
        }

        deserializer.deserialize_map(NumericKeyMapVisitor(std::marker::PhantomData))
    }
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
