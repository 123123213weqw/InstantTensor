//! Read planning: turn tensor byte ranges into block-aligned read requests.
//!
//! safetensors packs tensors back-to-back, but the header length is only
//! guaranteed to be 8-byte aligned, so tensor start offsets are almost never
//! 4096-aligned. `O_DIRECT` needs both offset and length aligned, which forces
//! every tensor to be rounded outward. Merging the rounded ranges is what keeps
//! read amplification at 1.0x instead of issuing one request per tensor.
//!
//! # Arithmetic safety
//!
//! Everything here is overflow-checked. A malformed header cannot produce a
//! negative length (which would wrap into a huge allocation in release builds)
//! or a wrapped merge decision:
//!
//! * `ceil_to` saturates instead of wrapping;
//! * every candidate range is clamped to `ceil(file_size, block)`;
//! * ranges with `end <= start` are dropped rather than emitted;
//! * merge decisions use `saturating_add` for the gap.

use crate::header::ShardHeader;

/// A block-aligned read request within one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub offset: u64,
    pub len: u64,
}

impl Range {
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.len)
    }
}

pub fn floor_to(x: u64, b: u64) -> u64 {
    let b = b.max(1);
    x / b * b
}

/// Round `x` up to a multiple of `b`, saturating at `u64::MAX` rather than
/// wrapping.
pub fn ceil_to(x: u64, b: u64) -> u64 {
    let b = b.max(1);
    x.div_ceil(b).saturating_mul(b)
}

/// Plan the reads for one shard.
///
/// Tensor ranges are rounded outward to `block`, then merged while the gap to
/// the next rounded range is at most `max_gap`. `max_gap == 0` yields one
/// request per contiguous run; a large `max_gap` collapses the file into a
/// single request.
pub fn plan_shard(h: &ShardHeader, block: u64, max_gap: u64) -> Vec<Range> {
    let block = block.max(1);
    let file_ceil = ceil_to(h.file_size, block);
    let mut out: Vec<Range> = Vec::new();

    for t in &h.tensors {
        // A zero-byte tensor needs no read. Without this skip, a zero-element
        // tensor sitting at `data_start` would still emit the alignment padding
        // in front of it as a request.
        if t.nbytes() == 0 {
            continue;
        }

        // Invariant from `read_header`: data_start + end <= file_size.
        let (abs_s, abs_e) = t.abs_range(h.data_start);

        let a0 = floor_to(abs_s, block).min(file_ceil);
        // Rounded up, possibly past EOF. Clamping this to `file_size` instead
        // would break block alignment and O_DIRECT would return EINVAL; a read
        // that runs past EOF is legal and reports a short count, which
        // `reader` accounts for in `short_reads`.
        let a1 = ceil_to(abs_e, block).min(file_ceil);

        // Zero-length or inverted candidates are skipped, never emitted.
        if a1 <= a0 {
            continue;
        }

        match out.last_mut() {
            Some(last) if a0 <= last.end().saturating_add(max_gap) => {
                let new_end = last.end().max(a1);
                last.len = new_end - last.offset;
            }
            _ => out.push(Range { offset: a0, len: a1 - a0 }),
        }
    }

    out
}

/// Plan one request per tensor, each rounded outward, without merging.
///
/// This is the control case: it shows what `O_DIRECT` alignment padding costs
/// when adjacent tensors are not coalesced.
pub fn plan_shard_unmerged(h: &ShardHeader, block: u64) -> Vec<Range> {
    let block = block.max(1);
    let file_ceil = ceil_to(h.file_size, block);
    h.tensors
        .iter()
        .filter_map(|t| {
            if t.nbytes() == 0 {
                return None;
            }
            let (abs_s, abs_e) = t.abs_range(h.data_start);
            let a0 = floor_to(abs_s, block).min(file_ceil);
            let a1 = ceil_to(abs_e, block).min(file_ceil);
            if a1 <= a0 {
                None
            } else {
                Some(Range { offset: a0, len: a1 - a0 })
            }
        })
        .collect()
}

/// Plan reads across a whole model. Returns `(shard index, range)` pairs.
///
/// Each shard index is guaranteed to be a valid index into `headers`.
pub fn plan_model(headers: &[ShardHeader], block: u64, max_gap: u64) -> Vec<(usize, Range)> {
    let mut out = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        for r in plan_shard(h, block, max_gap) {
            out.push((i, r));
        }
    }
    out
}

/// Plan with one request per tensor, across a whole model.
pub fn plan_model_unmerged(headers: &[ShardHeader], block: u64) -> Vec<(usize, Range)> {
    let mut out = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        for r in plan_shard_unmerged(h, block) {
            out.push((i, r));
        }
    }
    out
}
