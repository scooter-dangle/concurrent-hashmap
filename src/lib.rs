//! generic, thread safe, in-memory, key-value data store
//!
//! As indicated by the name, the `Library` struct is the main export of
//! this module. Some supporting types and traits are also exported.
//!
//! The `Library` structure supports two operations: `get` and `insert`, which fill similar roles to
//! their counter parts in the `HashMap` collection.
//! `get` accepts a key and returns a value.
//! `insert` updates provided key in the data store with the provided value.
//!
//! The thread safety property of the `Library` is the result of wrapping multiple concurrency
//! primitives; Arc, ArcCell, and Mutex.
//!
//! Mutex and Arc are part of the standard library. We use mutex to prevent multiple writers from
//! adding a new key simultaneously. Note: this is subtly, but significantly, different from
//! preventing multiple insertions of the same key.
//!
//! ## Insertion
//!
//! There are two different scenarios to consider: inserting a new value under a new key and
//! inserting a new value for an existing key and
//!
//! ### New value under new key
//!
//! Inserting a value with a new key requires allocating additional space in the `HashMap` and
//! potentially rearranging the underlying data. To prevent consistency errors the `Library` has an
//! internal `Mutex` (`Library.insert_mutex`) which must be obtained before inserting a key.
//!
//! ```
//! use concurrent_hashmap::Library;
//!
//! let lib: Library<String, i64> = Library::new();
//! lib.insert("qwerty".into(), 12345);
//! let val0 = lib.get("qwerty").unwrap();
//! lib.insert("asdfgh".into(), 67890);
//! let val1 = lib.get("asdfgh").unwrap();
//! assert_eq!(val0, 12345.into());
//! assert_eq!(val1, 67890.into());
//! ```
//!
//! ### New value under existing key
//!
//! Since the key already exists the `HashMap` does not need to allocate any additional storage (we
//! are just swapping the contents of an ArcCell). So we can short-circuit the insertion process,
//! and thus skipping lock acquisition, by providing a reference to the ArcCell and swapping directly.
//!
//! This tradeoff for performance is what introduces the "Last Writer Wins" behavior for multiple
//! insertions to the same key.
//!
//! ```
//! use concurrent_hashmap::Library;
//! use std::sync::Arc;
//!
//! let lib0: Arc<Library<String, i64>> = Library::new().into();
//! let val0 = lib0.get("abc");
//! assert_eq!(val0, None.into());
//! let lib1 = lib0.clone();
//! lib0.insert("abc".into(), 123);
//! let val123 = lib1.get("abc");
//! assert_eq!(val123, lib0.get("abc"));
//! assert_eq!(val0, None.into());
//! lib1.insert("abc".into(), 456);
//! let val456 = lib1.get("abc");
//! assert_eq!(val456, Some(456.into()));
//! assert_eq!(val123, Some(123.into()));
//! ```
//!
//! ## `ArcCell`
//!
//! [`ArcCell`](https://github.com/aturon/crossbeam/blob/master/src/sync/arc_cell.rs) is provided by
//! the crossbeam crate. The naming and documentation are atrocious, so we attempt to provide an
//! explaination here.
//!
//! As the figure below attempts to depict, The defining feature of this type is the ability to swap
//! out the contents of the heap allocated value (i.e. `Cell`) atomically. So a more accurate name
//! would be `AtomicCell`.
//!
//! ```text
//!          A ----> N
//!   A -\
//!   A ---> A ----> O
//!   A --/      /
//!            A-
//!
//!          A -\ /- N
//!   A -\       X
//!   A ---> A -/ \- > O
//!   A --/       /
//!             A-
//!
//! see: https://internals.rust-lang.org/t/atomic-arc-swap/3588
//! ```
//!
//! ## Caveats
//!
//! It is up to the user of `Library` to ensure that only a single update to an individual key happens concurrently.
//! Otherwise the `Library` will default to the "Last Writer Wins" conflict resolution strategy (hardly ever the
//! desired behavior from an end user perspective).

extern crate crossbeam;

use std::collections::hash_map::HashMap;
use std::sync::{Arc, Mutex};
use std::borrow::Borrow;
use std::clone::Clone;
use crossbeam::sync::ArcCell;

pub type LibraryStore<K, V> = HashMap<K, Arc<ArcCell<V>>>;

pub trait LibraryKey: ::std::cmp::Eq + ::std::hash::Hash + ::std::clone::Clone {}
impl<              K: ::std::cmp::Eq + ::std::hash::Hash + ::std::clone::Clone> LibraryKey for K {}

pub struct Library<K, V> where K: LibraryKey {
    internal_data: ArcCell<LibraryStore<K, V>>,
    insert_mutex: Mutex<()>,
}

impl<K, V> ::std::default::Default for Library<K, V> where K: LibraryKey {
    fn default() -> Library<K, V> {
        Library {
            internal_data: ArcCell::new(HashMap::new().into()),
            insert_mutex: Mutex::new(()),
        }
    }
}

impl<K, V> Library<K, V> where K: LibraryKey {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Library {
            internal_data: ArcCell::new(HashMap::with_capacity(capacity).into()),
            insert_mutex: Mutex::new(()),
        }
    }

    #[inline]
    fn internal_data(&self) -> Arc<LibraryStore<K, V>> {
        self.internal_data.get()
    }

    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<Arc<V>>
    where K: Borrow<Q>, Q: ::std::hash::Hash + Eq {
        self.internal_data().get(key).map(|el| el.get())
    }

    pub fn insert(&self, key: K, value: V) {
        let store = self.internal_data();

        // short circuit if domain already exists
        if let Some(arccell) = store.get(&key) {
                arccell.set(value.into());
                return;
        }

        // obtain lock (released at end of function scope)
        let _guard = self.insert_mutex.lock().unwrap();

        let store = self.internal_data();

        // exact copy of first `if let`
        if let Some(arccell) = store.get(&key) {
                arccell.set(value.into());
                return;
        }

        let new_hash: LibraryStore<K, V> = {
            // Multiple bindings because rust is incapable of inferring types
            let new_hash: &LibraryStore<K, V> = store.borrow();

            // Note: Potential room for future optimization if HashMap
            // ever provides a custom implementation of `clone_from`
            // that re-uses the allocated memory of `self`.
            //
            // If it did, we could create `new_hash` prior to the write-lock:
            //
            //     // (Before the lock)
            //     let mut new_hash: LibraryStore<K, V> = LibraryStore::with_capacity(store.capacity());
            //     // Ensure new_hash can have an additional element without resizing
            //     new_hash.reserve(1);
            //
            // And then here, avoid having to allocate memory (unless a
            // bajillion elements were added between when we grabbed the
            // hash map and when we write-locked Library).
            //
            //     // (After the lock)
            //     new_hash.clone_from(&store);
            //     new_hash.insert(...);
            //
            let mut new_hash = new_hash.clone();
            new_hash.insert(key,
                            ArcCell::new(value.into()).into());

            new_hash
        };

        self.internal_data.set(new_hash.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::std::sync::Arc;

    #[test]
    fn basic() {
        let lib1: Arc<Library<String, i64>> = Library::new().into();
        lib1.insert("abc".into(), 123);

        assert_eq!(lib1.get(&String::from("abc")), Some(Arc::new(123)));
    }

    #[test]
    fn old_value_replace_same_key() {
        let lib0: Arc<Library<String, i64>> = Library::new().into();
        let val0 = lib0.get("abc");
        assert_eq!(val0, None.into());
        let lib1 = lib0.clone();
        lib0.insert("abc".into(), 123);
        let val123 = lib1.get("abc");
        assert_eq!(val123, lib0.get("abc"));
        assert_eq!(val0, None.into());
        lib1.insert("abc".into(), 456);
        let val456 = lib1.get("abc");
        assert_eq!(val456, Some(456.into()));
        assert_eq!(val123, Some(123.into()));
    }

    #[test]
    fn old_value_insert_new_key() {
        let lib: Library<String, i64> = Library::new();
        lib.insert("qwerty".into(), 12345);
        let val0 = lib.get("qwerty").unwrap();
        lib.insert("asdfgh".into(), 67890);
        let val1 = lib.get("asdfgh").unwrap();
        assert_eq!(val0, 12345.into());
        assert_eq!(val1, 67890.into());
    }
}
