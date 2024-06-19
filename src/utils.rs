use std::collections::HashMap;
use std::hash::Hash;

use hashbrown::HashMap as HashBrownMap;
// use indexmap::IndexMap;

pub trait MapGet<K, V> {
    fn get(&self, key: &K) -> Option<&V>;
}

#[allow(clippy::implicit_hasher)]
impl<K, V> MapGet<K, V> for HashMap<K, V>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }
}

#[allow(clippy::implicit_hasher)]
impl<K, V> MapGet<K, V> for &HashMap<K, V>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }
}

#[allow(clippy::implicit_hasher)]
impl<K, V> MapGet<K, V> for HashBrownMap<K, V>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashBrownMap::get(self, key)
    }
}

#[allow(clippy::implicit_hasher)]
impl<K, V> MapGet<K, V> for &HashBrownMap<K, V>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashBrownMap::get(self, key)
    }
}

// impl<K, V> MapGet<K, V> for IndexMap<K, V>
// where
//     K: Eq + std::hash::Hash,
// {
//     fn get(&self, key: &K) -> Option<&V> {
//         IndexMap::get(self, key)
//     }
// }

// impl<K, V> MapGet<K, V> for &IndexMap<K, V>
// where
//     K: Eq + std::hash::Hash,
// {
//     fn get(&self, key: &K) -> Option<&V> {
//         IndexMap::get(self, key)
//     }
// }
