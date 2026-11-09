//! Ring buffer with O(1) front/back operations
//!
//! Optimized for sliding window buffers where old messages
//! are frequently removed from the front.

use std::collections::VecDeque;

/// Efficient FIFO buffer with bounded capacity
///
/// # Example
/// ```
/// use caliche::memory::container::RingBuffer;
///
/// let mut buffer = RingBuffer::new();
/// buffer.push(1);
/// buffer.push(2);
/// buffer.push(3);
///
/// assert_eq!(buffer.len(), 3);
/// assert_eq!(buffer.pop_front(), Some(1));
/// ```
pub struct RingBuffer<T> {
    items: VecDeque<T>,
    capacity: Option<usize>,
}

impl<T> RingBuffer<T> {
    /// Create unbounded ring buffer
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items: VecDeque::new(),
            capacity: None,
        }
    }

    /// Create ring buffer with maximum capacity
    ///
    /// When capacity is exceeded, oldest items are automatically removed.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity: Some(capacity),
        }
    }

    /// Push item to back, potentially removing oldest
    pub fn push(&mut self, item: T) -> Option<T> {
        let removed = if let Some(cap) = self.capacity {
            if self.items.len() >= cap {
                self.items.pop_front()
            } else {
                None
            }
        } else {
            None
        };

        self.items.push_back(item);
        removed
    }

    /// Remove and return front item (O(1))
    pub fn pop_front(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    /// Remove and return back item (O(1))
    pub fn pop_back(&mut self) -> Option<T> {
        self.items.pop_back()
    }

    /// Get iterator over items
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Extend from iterator
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Get as contiguous slices
    #[must_use]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.items.as_slices()
    }

    /// Remove the item at the given index (O(n) operation)
    /// Returns the removed item if the index is valid
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index < self.items.len() {
            self.items.remove(index)
        } else {
            None
        }
    }

    /// Remove the first item matching the predicate (O(n) operation)
    /// Returns the removed item if found
    pub fn remove_first_matching<F>(&mut self, predicate: F) -> Option<T>
    where
        F: Fn(&T) -> bool,
    {
        if let Some(pos) = self.items.iter().position(predicate) {
            self.items.remove(pos)
        } else {
            None
        }
    }
}

impl<T> Default for RingBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FromIterator<T> for RingBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::new();
        buffer.extend(iter);
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buffer = RingBuffer::new();
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop_front(), Some(1));
        assert_eq!(buffer.pop_front(), Some(2));
        assert_eq!(buffer.pop_front(), Some(3));
        assert_eq!(buffer.pop_front(), None);
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut buffer = RingBuffer::with_capacity(3);
        assert_eq!(buffer.push(1), None);
        assert_eq!(buffer.push(2), None);
        assert_eq!(buffer.push(3), None);
        assert_eq!(buffer.push(4), Some(1)); // Oldest removed

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop_front(), Some(2));
    }

    #[test]
    fn test_ring_buffer_pop_back() {
        let mut buffer = RingBuffer::new();
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.pop_back(), Some(3));
        assert_eq!(buffer.pop_back(), Some(2));
        assert_eq!(buffer.pop_back(), Some(1));
    }

    #[test]
    fn test_ring_buffer_extend() {
        let mut buffer = RingBuffer::new();
        buffer.extend(vec![1, 2, 3]);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop_front(), Some(1));
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut buffer = RingBuffer::new();
        buffer.push(1);
        buffer.push(2);
        buffer.clear();

        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.is_empty(), true);
    }

    #[test]
    fn test_ring_buffer_from_iter() {
        let buffer: RingBuffer<i32> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(buffer.len(), 3);
    }
}
