//! Block-aligned buffer allocation for O_DIRECT.
//!
//! `O_DIRECT` requires the user buffer address, the file offset and the transfer
//! length to be aligned to the logical block size (4096 on this hardware).
//! The std allocator is not guaranteed to return such addresses, so we allocate
//! through `std::alloc` with an explicit `Layout`.

use std::alloc::{alloc, dealloc, Layout};

#[derive(Debug)]
pub struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

// The buffer is exclusively owned; moving it between threads is sound.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    /// Allocate `len` bytes aligned to `align` (rounded up to a power of two).
    pub fn new(len: usize, align: usize) -> Self {
        let align = align.max(64).next_power_of_two();
        // Layout requires size to be a multiple of align for this use case;
        // keep the requested length and align the *base address* instead.
        let layout = Layout::from_size_align(len.max(1), align)
            .expect("invalid aligned layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "aligned allocation of {len} bytes failed");
        Self { ptr, layout }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for layout.size() bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    /// True when the base address satisfies the alignment requirement.
    pub fn is_aligned(&self, align: usize) -> bool {
        (self.ptr as usize).is_multiple_of(align)
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated with this exact layout in `new`.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}
