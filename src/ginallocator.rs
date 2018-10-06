use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::mem::replace;

struct GinAllocatorItem<T> {
    item: T,
    generation: u32,
}

impl<T> GinAllocatorItem<T> {
    pub(crate) fn is_active(&self) -> bool {
        self.generation % 2 == 0
    }
}

pub struct GinAllocator<T, M> {
    items: Vec<GinAllocatorItem<T>>,
    freelist: VecDeque<u32>,
    refcounts: HashMap<u32, u32>,
    phantom: PhantomData<M>,
}

#[must_use]
pub enum DestroyResult {
    StillAlive,
    ShouldDrop,
}

pub struct GinAllocatorChild<T, M>
where
    T: Default,
{
    items: Vec<T>,
    // Can't refer to the parent yet because it creates a self-referential
    // struct as whatever owns this and the parent
    // This means that the caller must check that any handle used to access
    // items is actually valid or they'll get a panic. Still safe, obviously,
    // but not ideal.
    // parent: &'a GinAllocator<U, M>,
    phantom: PhantomData<M>,
}

pub trait Marker {
    const ID: &'static str;
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Handle<M: Marker> {
    pub(crate) index: u32,
    pub(crate) generation: u32,
    pub(crate) phantom: PhantomData<M>,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct CheckedHandle<M: Marker> {
    index: u32,
    phantom: PhantomData<M>,
}

impl<T> fmt::Display for GinAllocatorItem<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GinAllocatorItem {{ generation: {} }}", self.generation)
    }
}

impl<M: Marker> fmt::Display for Handle<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}Handle {{ index: {}, generation: {} }}",
            M::ID,
            self.index,
            self.generation
        )
    }
}

impl<T, M: Marker> GinAllocator<T, M> {
    pub fn new() -> GinAllocator<T, M> {
        GinAllocator::<T, M> {
            items: Vec::new(),
            freelist: VecDeque::new(),
            refcounts: HashMap::new(),
            phantom: PhantomData,
        }
    }

    /// Create a child allocator. The child allocator shares `Handle`s with its
    /// parent, and the parent allocator is responsible for generation tracking
    /// and reference counting.
    pub fn create_child<U: Default>(&self) -> GinAllocatorChild<U, M> {
        GinAllocatorChild::<U, M> {
            items: Vec::new(),
            phantom: PhantomData,
        }
    }

    /// Inserts `data` into a new item in the allocator store
    pub fn insert(&mut self, data: T) -> Handle<M> {
        // check the freelist for a free slot first
        if self.freelist.len() != 0 {
            // safe to unwrap here as we've checked we're not empty
            let index = self.freelist.pop_front().unwrap();
            let item = &mut self.items[index as usize];
            assert!(item.generation % 2 == 1);
            item.generation = item.generation.wrapping_add(1);
            // reset the refcount to 1 for the userspace key we're about to
            // return
            self.refcounts.insert(index, 1);
            Handle {
                index,
                generation: item.generation,
                phantom: PhantomData,
            }
        } else {
            // push a new item onto the vec
            let index = self.items.len() as u32;
            self.items.push(GinAllocatorItem::<T> {
                item: data,
                generation: 0,
            });
            // Set the refcount to 1 for the userspace key we're about to
            // return
            self.refcounts.insert(index, 1);
            Handle {
                index,
                generation: 0,
                phantom: PhantomData,
            }
        }
    }

    /// Check that the given `handle` is valid. Specifically this checks that
    /// the item it refers to is in range, active and of the correct generation.
    pub fn is_valid(&self, handle: &Handle<M>) -> bool {
        if let Some(item) = self.items.get(handle.index as usize) {
            if item.is_active() && handle.generation == item.generation {
                return true;
            }
        }
        false
    }

    pub fn check_handle(&self, handle: Handle<M>) -> Option<CheckedHandle<M>> {
        if self.is_valid(&handle) {
            Some(
                CheckedHandle::<M> {
                    index: handle.index,
                    phantom: PhantomData,
                }
            )
        } else {
            None
        }
    }

    /// Gets the item if it is in range, is active and of the correct generation
    /// or `None` otherwise
    fn get_item(&self, handle: Handle<M>) -> Option<&GinAllocatorItem<T>> {
        if let Some(item) = self.items.get(handle.index as usize) {
            if item.is_active() && handle.generation == item.generation {
                return Some(item);
            }
        }

        None
    }

    /// Gets the item if it is in range, is active and of the correct generation
    /// or `None` otherwise
    fn get_item_mut(&mut self, handle: Handle<M>) -> Option<&mut GinAllocatorItem<T>> {
        if let Some(item) = self.items.get_mut(handle.index as usize) {
            if item.is_active() && handle.generation == item.generation {
                return Some(item);
            }
        }

        None
    }

    pub fn incref(&mut self, handle: Handle<M>) {
        let refcount = self.refcounts.get_mut(&handle.index).unwrap();
        println!("Incrementing {} to {}", handle, *refcount + 1);
        *refcount += 1;
    }

    // fn decref(&mut self, handle: Handle<M>) {
    //     let refcount = self.refcounts.get_mut(&handle.index).unwrap();
    //     *refcount -= 1;
    // }

    /// "Destroys" the item referenced by `handle`. All this does is return the
    /// item to the allocator to be recycled.
    pub fn destroy(&mut self, handle: Handle<M>) -> DestroyResult {
        // first, check that the handle is valid
        let item = &mut self.items[handle.index as usize];
        assert!(item.generation % 2 == 0);
        if item.generation != handle.generation {
            panic!(
                "Item and Handle generations do not match: {} {}",
                item, handle
            );
        }
        // decrement the ref count, panic if it's not there
        let refcount = self.refcounts.get_mut(&handle.index).unwrap();
        *refcount = refcount.saturating_sub(1);
        println!("Decremented {} - refcount is {}", handle, refcount);
        if *refcount == 0 {
            println!("Refcount is {} on {} - Dropping", refcount, handle);
            // Bump the generation on the item to indicate that it's empty, and
            // push the index to the free list so the allocator can reuse it
            item.generation = item.generation.wrapping_add(1);
            self.freelist.push_back(handle.index);
            DestroyResult::ShouldDrop
        } else {
            DestroyResult::StillAlive
        }
    }

    /// Get a reference to the item referred to by `handle`. Returns None if the
    /// item is not found, is inactive or of the wrong generation
    pub fn get(&self, handle: Handle<M>) -> Option<&T> {
        if let Some(item) = self.get_item(handle) {
            Some(&item.item)
        } else {
            None
        }
    }

    /// Get a mutable reference to the item referred to by `handle`. Returns
    /// `None` if the item is not found, is inactive or of the wrong generation
    pub fn get_mut(&mut self, handle: Handle<M>) -> Option<&mut T> {
        if let Some(item) = self.get_item_mut(handle) {
            Some(&mut item.item)
        } else {
            None
        }
    }
}

impl<T: Default, M: Marker> GinAllocatorChild<T, M> {
    /// Insert the given item into the array at `handle`. This should only be
    /// used after the handle has been validated by the parent allocator
    pub fn insert(&mut self, handle: &CheckedHandle<M>, item: T) {
        assert!((handle.index as usize) <= self.items.len());

        if (handle.index as usize) < self.items.len() {
            self.items[handle.index as usize] = item;
        } else {
            self.items.push(item);
        }
    }

    /// Get a reference to the item referred to by `handle`.  
    /// # Panics
    /// If the handle is out of bounds
    pub fn get(&self, handle: CheckedHandle<M>) -> &T {
        assert!((handle.index as usize) < self.items.len());

        &self.items[handle.index as usize]
    }

    /// Get a mutable reference to the item referred to by `handle`.
    /// # Panics
    /// If the handle is out of bounds
    pub fn get_mut(&mut self, handle: CheckedHandle<M>) -> &mut T {
        assert!((handle.index as usize) < self.items.len());
        &mut self.items[handle.index as usize]
    }

    /// Remove the given item from the storage, returning it and replacing it
    /// with a default value.
    pub fn remove(&mut self, handle: CheckedHandle<M>) -> T {
        assert!((handle.index as usize) < self.items.len());
        std::mem::replace(&mut self.items[handle.index as usize], T::default())
    }
}

