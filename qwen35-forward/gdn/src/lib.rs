//! Gated delta net (`Qwen3_5GatedDeltaNet`) — the linear-attention block of qwen35.
//!
//! This is the shell around the delta rule. The rule itself lives in the
//! `deltarule` crate; everything here is the plumbing that produces its operands
//! and consumes its output.
//!
//! # The chain
//!
//! ```text
//! x = input_layernorm(h)                       (per-row RMSNorm, x*(1+w))
//! mixed = in_proj_qkv(x)                       [B, T, 2*key_dim + value_dim]
//! mixed = conv1d_silu(mixed)                   depthwise, causal, kernel 4
//! q, k, v = split(mixed, [key_dim, key_dim, value_dim])
//! q,k -> [B,T,num_k_heads,head_k_dim];  v -> [B,T,num_v_heads,head_v_dim]
//! z = in_proj_z(x) -> [B,T,num_v_heads,head_v_dim]
//! b = in_proj_b(x);  a = in_proj_a(x)          [B, T, num_v_heads]
//! beta = sigmoid(b)
//! g = -exp(A_log) * softplus(a + dt_bias)
//! q,k = repeat_interleave(q,k, num_v_heads/num_k_heads)   <- GQA, before the rule
//! out, state = delta_rule(q, k, v, g, beta)    [B,T,H,head_v_dim], [B,H,K,V]
//! out = norm(out, z)                           gated RMSNorm over head_v_dim
//! y = out_proj(out)
//! ```
//!
//! # Two conventions that are easy to get backwards
//!
//! * `input_layernorm` is `Qwen3_5RMSNorm`: **`x * (1 + w)`** with `w`
//!   zero-initialised, so the default is a no-op scale.
//! * `linear_attn.norm` is `Qwen3_5RMSNormGated`: **`w * x_hat * silu(z)`**, i.e.
//!   plain `w`, not `(1 + w)`. Using the wrong one of these zeroes every
//!   activation or doubles them, without raising an error.
//!
//! # Where the GQA expansion goes
//!
//! `repeat_interleave` on the head axis happens **before** the delta rule, so the
//! rule sees `num_v_heads`, not `num_k_heads`. The captured operands confirm it:
//! `q`/`k` are `[1, 6, 4, 16]` while `num_k_heads` is 2.

use deltarule::{forward_prepared, forward_prepared_into, Shape};

pub mod attention;
pub mod layer;
pub mod loader;
pub mod model;
pub mod real;
pub mod safetensors;
pub mod tokenizer;
pub mod unicode_gc;
pub mod unicode_tables;

/// Default RMSNorm epsilon. The reference reads it from `config.rms_norm_eps`,
/// which is `1e-6` for every qwen35 configuration checked; `GdnConfig::eps`
/// carries the per-model value for the mixer, and `layer` uses this default for
/// the two block-level norms.
pub const EPS: f32 = 1e-6;

/// Sizes for one gated-delta-net block.
#[derive(Debug, Clone, Copy)]
pub struct GdnConfig {
    pub hidden: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

impl GdnConfig {
    pub fn key_dim(&self) -> usize {
        self.head_k_dim * self.num_k_heads
    }
    pub fn value_dim(&self) -> usize {
        self.head_v_dim * self.num_v_heads
    }
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    /// `num_v_heads / num_k_heads`; 1 means no GQA expansion is applied.
    pub fn kv_ratio(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }
}

/// All weights of one block, in the layout PyTorch stores them.
#[derive(Debug, Clone)]
pub struct GdnWeights {
    /// `[conv_dim, hidden]`
    pub in_proj_qkv: Vec<f32>,
    /// `[value_dim, hidden]`
    pub in_proj_z: Vec<f32>,
    /// `[num_v_heads, hidden]`
    pub in_proj_b: Vec<f32>,
    /// `[num_v_heads, hidden]`
    pub in_proj_a: Vec<f32>,
    /// `[conv_dim, 1, conv_kernel]` as stored; only `[conv_dim, conv_kernel]` is used.
    pub conv1d: Vec<f32>,
    /// `[num_v_heads]`
    pub a_log: Vec<f32>,
    /// `[num_v_heads]`
    pub dt_bias: Vec<f32>,
    /// `[head_v_dim]` — `Qwen3_5RMSNormGated`, applied as `w * x_hat * silu(z)`.
    pub norm: Vec<f32>,
    /// `[hidden, value_dim]`
    pub out_proj: Vec<f32>,
}

// --------------------------------------------------------------------------- //
// primitives
// --------------------------------------------------------------------------- //

/// Dot product of two equal-length `f32` slices, accumulated in `f64`.
///
/// # Why this is not just cosmetic
///
/// A naive `f32` accumulation of `n` products carries a worst-case error of about
/// `n * eps`, because every partial sum rounds. For the reductions in this model --
/// `n = 1024` for `hidden`, `3584` for `intermediate_size` -- that is a relative error
/// around `2e-4` and `8e-4`, and it is applied once per matmul across 24 layers.
///
/// BLAS does not accumulate this way: it uses blocked or pairwise reductions, whose
/// error grows like `log n * eps`, roughly two orders of magnitude smaller here. So a
/// sequential loop is not "the same computation in different order" -- it is a
/// measurably worse one, and it shows up as the engine disagreeing with the reference
/// by several times the reference's own disagreement with itself.
///
/// Accumulating in `f64` removes the question: the products are exact (an `f32`
/// times an `f32` is representable in `f64`), and the sum of `n` of them in `f64`
/// carries an error of `n * eps_f64`, which is below `f32` resolution for any `n`
/// this model uses. The result is a dot product that is correctly rounded to `f32`,
/// so the remaining disagreement with a reference is the reference's own error.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f64;
    for i in 0..a.len() {
        acc += a[i] as f64 * b[i] as f64;
    }
    acc as f32
}

/// Multiply-accumulate work above which `linear` spreads rows over threads.
///
/// Small matrices are left alone: a real model does thousands of these calls and
/// the spawn cost dominates for anything the size of a test fixture.
const PARALLEL_MIN_WORK: usize = 1 << 18;

/// Upper bound on threads used per call. The box has 88 cores, but a single
/// matmul is memory-bound well before that and oversubscribing only adds
/// contention with whatever else is running.
const MAX_THREADS: usize = 32;

/// `y[r, o] = sum_i w[o, i] * x[r, i] (+ bias[o])`
///
/// `w` is `[out_dim, in_dim]`, matching `nn.Linear`'s storage.
///
/// Rows are independent, so the parallel path splits the output by row. Each output
/// element is still accumulated in the same order over `i`, so the result is
/// bit-identical to the serial path -- threading changes throughput, not numerics.
pub fn linear(
    w: &[f32],
    bias: Option<&[f32]>,
    x: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim, "linear weight size");
    assert_eq!(x.len(), rows * in_dim, "linear input size");
    let mut y = vec![0f32; rows * out_dim];
    if rows == 0 || out_dim == 0 {
        return y;
    }

    let row = |r: usize, yr: &mut [f32]| {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for (o, yo) in yr.iter_mut().enumerate() {
            let wo = &w[o * in_dim..(o + 1) * in_dim];
            // f64 accumulation: see `dot` for why a plain f32 loop is measurably
            // worse rather than merely differently ordered.
            let mut acc = 0f64;
            for i in 0..in_dim {
                acc += wo[i] as f64 * xr[i] as f64;
            }
            if let Some(b) = bias {
                acc += b[o] as f64;
            }
            *yo = acc as f32;
        }
    };

    let work = rows.saturating_mul(in_dim).saturating_mul(out_dim);
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(MAX_THREADS);
    if nthreads <= 1 || work < PARALLEL_MIN_WORK {
        for (r, yr) in y.chunks_mut(out_dim).enumerate() {
            row(r, yr);
        }
        return y;
    }

    let chunk = rows.div_ceil(nthreads).max(1);
    let xref = x;
    let wref = w;
    let biasref = bias;
    std::thread::scope(|s| {
        for (ci, ychunk) in y.chunks_mut(chunk * out_dim).enumerate() {
            let r0 = ci * chunk;
            s.spawn(move || {
                for (rr, yr) in ychunk.chunks_mut(out_dim).enumerate() {
                    let r = r0 + rr;
                    let xr = &xref[r * in_dim..(r + 1) * in_dim];
                    for (o, yo) in yr.iter_mut().enumerate() {
                        let wo = &wref[o * in_dim..(o + 1) * in_dim];
                        let mut acc = 0f64;
                        for i in 0..in_dim {
                            acc += wo[i] as f64 * xr[i] as f64;
                        }
                        if let Some(b) = biasref {
                            acc += b[o] as f64;
                        }
                        *yo = acc as f32;
                    }
                }
            });
        }
    });
    y
}

/// `Qwen3_5RMSNorm`: `x * rsqrt(mean(x^2) + eps) * (1 + w)`, over the last dim.
///
/// Note the `1 + w`. The reference stores the weight zero-initialised and adds
/// one, so a zero weight means identity.
pub fn rmsnorm_1plus(w: &[f32], x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(w.len(), dim);
    assert_eq!(x.len(), rows * dim);
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let yr = &mut y[r * dim..(r + 1) * dim];
        // f64 sum of squares, for the same reason as `dot`.
        let mut ss = 0f64;
        for v in xr {
            ss += *v as f64 * *v as f64;
        }
        let inv = 1.0f32 / ((ss / dim as f64) as f32 + eps).sqrt();
        for i in 0..dim {
            yr[i] = xr[i] * inv * (1.0 + w[i]);
        }
    }
    y
}

/// `Qwen3_5RMSNormGated`: `w * x_hat * silu(z)`, over the last dim.
///
/// Note plain `w`, not `(1 + w)` — the opposite convention from `rmsnorm_1plus`.
/// `silu(z)` is computed in f32 and the product is returned in f32.
pub fn rmsnorm_gated(w: &[f32], x: &[f32], z: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(w.len(), dim);
    assert_eq!(x.len(), rows * dim);
    assert_eq!(z.len(), rows * dim);
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let zr = &z[r * dim..(r + 1) * dim];
        let yr = &mut y[r * dim..(r + 1) * dim];
        let mut ss = 0f64;
        for v in xr {
            ss += *v as f64 * *v as f64;
        }
        let inv = 1.0f32 / ((ss / dim as f64) as f32 + eps).sqrt();
        for i in 0..dim {
            let xhat = xr[i] * inv;
            let g = zr[i];
            let silu = g / (1.0 + (-g).exp()); // silu(x) = x * sigmoid(x)
            yr[i] = w[i] * xhat * silu;
        }
    }
    y
}

/// Softplus, matching `torch.nn.functional.softplus` with `beta = 1` and the
/// default `threshold = 20`: above the threshold the result is the identity, which
/// avoids `exp` overflow.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Depthwise causal convolution followed by SiLU, over `[B, C, T]`, optionally
/// continuing from a carried state.
///
/// Returns `(out, new_state)`: `out` is `[B, C, T]`, `new_state` is `[B, C, K-1]`.
///
/// The reference is
///
/// ```python
/// out = F.conv1d(x, weight.unsqueeze(1), bias, padding=K-1, groups=C)[:, :, :T]
/// out = silu(out)
/// ```
///
/// so the whole `(K-1)` padding is on the **left** and the trailing positions are
/// dropped. That is what makes it causal: position `p` reads `x[p-(K-1) ..= p]` and
/// never a future step.
///
/// Both call modes evaluate the same expression, and they differ only in where the
/// left context comes from. Position `p` needs the previous `K-1` inputs:
///
/// * no state: they must be inside the chunk, so a short chunk is zero-padded. This is
///   the uncached path, and re-processing the whole sequence at every step is what it
///   costs.
/// * a state: they are prepended. This is what makes a **single-token** step possible,
///   because a chunk of one has no left context of its own.
///
/// Assembling `full = [left context (K-1), chunk (t)]` first makes the inner loop
/// `out[ti] = sum_k w[k] * full[ti + k]` for both modes with no branch inside, and makes
/// the new state simply the last `K-1` entries of `full`. Zero padding is not a special
/// case here, it is the same window with a zero prefix.
///
/// `K-1` is the minimum the next step needs. The reference keeps `K` and discards one
/// extra output; a test asserts the two agree exactly.
///
/// # No bias in practice
///
/// `bias` is an `Option` because the checkpoints have no `conv1d.bias` at all: Qwen3.5
/// constructs the module with `bias=False`, and none of the 18 linear layers in
/// `Qwen3.5-0.8B` has one. It is kept in the signature so the reference's
/// `F.conv1d(..., bias, ...)` is visibly accounted for rather than silently dropped.
#[allow(clippy::too_many_arguments)]
pub fn conv_forward(
    w: &[f32],
    bias: Option<&[f32]>,
    x_bct: &[f32],
    b: usize,
    channels: usize,
    t: usize,
    k: usize,
    state_in: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), channels * k, "conv weight size");
    assert_eq!(x_bct.len(), b * channels * t, "conv input size");
    let keep = k.saturating_sub(1);
    if let Some(st) = state_in {
        assert_eq!(st.len(), b * channels * keep, "conv state size");
    }

    let mut out = vec![0f32; b * channels * t];
    let mut new_state = vec![0f32; b * channels * keep];
    // Reused across (batch, channel); holds [left context, chunk].
    let mut full = vec![0f32; keep + t];

    for bi in 0..b {
        for c in 0..channels {
            let wc = &w[c * k..(c + 1) * k];
            let off = (bi * channels + c) * t;
            let so = (bi * channels + c) * keep;

            match state_in {
                Some(st) => full[..keep].copy_from_slice(&st[so..so + keep]),
                None => full[..keep].fill(0.0),
            }
            full[keep..].copy_from_slice(&x_bct[off..off + t]);

            for ti in 0..t {
                let mut acc = 0f32;
                for (ki, wk) in wc.iter().enumerate() {
                    acc += wk * full[ti + ki];
                }
                if let Some(bb) = bias {
                    acc += bb[c];
                }
                out[off + ti] = silu(acc);
            }

            // The last `keep` entries of `full` are the context the next chunk needs.
            new_state[so..so + keep].copy_from_slice(&full[t..t + keep]);
        }
    }
    (out, new_state)
}

/// `repeat_interleave(x, ratio)` on the head axis of a `[B, T, H, D]` tensor.
///
/// Each head is duplicated `ratio` times consecutively, so head `h` maps to
/// output heads `h*ratio .. h*ratio + ratio - 1`.
pub fn repeat_interleave_heads(x: &[f32], b: usize, t: usize, h: usize, d: usize, ratio: usize) -> Vec<f32> {
    if ratio == 1 {
        return x.to_vec();
    }
    let oh = h * ratio;
    let mut y = vec![0f32; b * t * oh * d];
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let src = ((bi * t + ti) * h + hi) * d;
                for rep in 0..ratio {
                    let dst = ((bi * t + ti) * oh + hi * ratio + rep) * d;
                    y[dst..dst + d].copy_from_slice(&x[src..src + d]);
                }
            }
        }
    }
    y
}

/// `[B, T, C]` -> `[B, C, T]`. The convolution works channels-first.
pub fn btc_to_bct(x: &[f32], b: usize, t: usize, c: usize) -> Vec<f32> {
    assert_eq!(x.len(), b * t * c);
    let mut y = vec![0f32; b * c * t];
    for bi in 0..b {
        for ti in 0..t {
            for ci in 0..c {
                y[(bi * c + ci) * t + ti] = x[(bi * t + ti) * c + ci];
            }
        }
    }
    y
}

/// `[B, C, T]` -> `[B, T, C]`, the inverse of [`btc_to_bct`].
pub fn bct_to_btc(x: &[f32], b: usize, c: usize, t: usize) -> Vec<f32> {
    assert_eq!(x.len(), b * c * t);
    let mut y = vec![0f32; b * t * c];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..t {
                y[(bi * t + ti) * c + ci] = x[(bi * c + ci) * t + ti];
            }
        }
    }
    y
}

// --------------------------------------------------------------------------- //
// the block
// --------------------------------------------------------------------------- //

/// Every intermediate of one block, so a caller can compare against a golden
/// bundle at whichever step diverges first.
#[derive(Debug, Clone)]
pub struct GdnTrace {
    pub in_proj_qkv: Vec<f32>,
    pub conv_in: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub in_proj_z: Vec<f32>,
    pub in_proj_b: Vec<f32>,
    pub in_proj_a: Vec<f32>,
    /// post-GQA-expansion, `[B, T, num_v_heads, head_k_dim]`
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
    pub delta_out: Vec<f32>,
    pub delta_state: Vec<f32>,
    pub norm: Vec<f32>,
    pub out_proj: Vec<f32>,
}

/// Per-layer state the linear-attention mixer carries between calls.
///
/// The mixer has two kinds of memory, and they are not the same thing:
///
/// * [`conv`](GdnState::conv) is the previous `conv_kernel - 1` inputs. The depthwise
///   causal convolution needs exactly that much left context, so this is tiny and
///   bounded.
/// * [`ssm`](GdnState::ssm) is the delta rule's recurrent state, `[B, H, K, V]`. This is
///   the reason a linear-attention layer exists at all: it is a **constant** amount of
///   state, so unlike a KV cache it does not grow with context. For `Qwen3.5-0.8B` one
///   layer is `16*128*128` = 262144 `f32` = 1 MiB, and 18 of them is 19.3 MiB no matter
///   how long the conversation runs. The 6 full-attention layers are the ones whose
///   cache grows, and they grow linearly.
#[derive(Debug, Clone)]
pub struct GdnState {
    /// `[B, conv_dim, K-1]`, channels-first to match the convolution's layout.
    pub conv: Vec<f32>,
    /// `[B, num_v_heads, head_k_dim, head_v_dim]`
    pub ssm: Vec<f32>,
}

impl GdnState {
    /// A zeroed state, which is what the first chunk of a sequence starts from.
    pub fn new(cfg: &GdnConfig, b: usize) -> Self {
        Self {
            conv: vec![0f32; b * cfg.conv_dim() * cfg.conv_kernel.saturating_sub(1)],
            ssm: vec![0f32; b * cfg.num_v_heads * cfg.head_k_dim * cfg.head_v_dim],
        }
    }

    /// Bytes of state, so a caller can report what the cache costs.
    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.ssm.len()) * std::mem::size_of::<f32>()
    }
}

/// Run the mixer on an **already-normalised** input, with no prior state.
///
/// `x` is the output of the decoder layer's `input_layernorm`, which the layer
/// owns: `Qwen3_5GatedDeltaNet` has no `input_layernorm` of its own, and neither
/// does `Qwen3_5Attention`. Applying it here instead would look correct on a
/// single layer while double-normalising in a chain, so the layer does it.
///
/// `x` is `[B, T, hidden]`. The returned `out_proj` is the mixer's output.
///
/// This is the whole-sequence path: it re-runs from zero state every time, so a caller
/// that decodes one token at a time pays for the entire prefix at each step. Use
/// [`forward_with_state`] to avoid that.
pub fn forward(cfg: &GdnConfig, w: &GdnWeights, x: &[f32], b: usize, t: usize) -> GdnTrace {
    forward_impl(cfg, w, x, b, t, None)
}

/// Run the mixer continuing from `state`, which is advanced in place.
///
/// `x` is the new chunk, `[B, T, hidden]`. Decoding passes `T = 1` and lets `state`
/// carry everything the mixer remembers.
///
/// Because the state is *continued* rather than restarted, this is the same arithmetic
/// as [`forward`] over the concatenated sequence, not an approximation of it. A test
/// asserts the two agree bit for bit, which is a stronger statement than the reference
/// can make: its uncached and cached delta-rule paths are different algorithms.
pub fn forward_with_state(
    cfg: &GdnConfig,
    w: &GdnWeights,
    x: &[f32],
    b: usize,
    t: usize,
    state: &mut GdnState,
) -> GdnTrace {
    forward_impl(cfg, w, x, b, t, Some(state))
}

fn forward_impl(
    cfg: &GdnConfig,
    w: &GdnWeights,
    x: &[f32],
    b: usize,
    t: usize,
    state: Option<&mut GdnState>,
) -> GdnTrace {
    let h = cfg.hidden;
    let rows = b * t;
    assert_eq!(x.len(), rows * h, "gdn input size vs config");

    // 1. projections
    let mixed = linear(&w.in_proj_qkv, None, x, rows, h, cfg.conv_dim()); // [rows, conv_dim]
    let z_full = linear(&w.in_proj_z, None, x, rows, h, cfg.value_dim()); // [rows, value_dim]
    let bb = linear(&w.in_proj_b, None, x, rows, h, cfg.num_v_heads);
    let aa = linear(&w.in_proj_a, None, x, rows, h, cfg.num_v_heads);

    // 2. conv over [B, C, T]. The reference transposes to channels-first, so this
    //    does too. `w.conv1d` is stored [C, 1, K]; a contiguous [C, 1, K] tensor is
    //    exactly [C*K] in memory, so it is used directly and `conv_forward` asserts
    //    the length it needs.
    let mixed_bct = btc_to_bct(&mixed, b, t, cfg.conv_dim()); // [B, C, T]
    let (conv_out, new_conv_state) = match &state {
        Some(st) => conv_forward(
            &w.conv1d,
            None,
            &mixed_bct,
            b,
            cfg.conv_dim(),
            t,
            cfg.conv_kernel,
            Some(&st.conv),
        ),
        None => conv_forward(
            &w.conv1d,
            None,
            &mixed_bct,
            b,
            cfg.conv_dim(),
            t,
            cfg.conv_kernel,
            None,
        ),
    };
    let conv_btc = bct_to_btc(&conv_out, b, cfg.conv_dim(), t); // [B, T, C]

    // 3. split into q, k, v along the last axis
    let kd = cfg.key_dim();
    let vd = cfg.value_dim();
    let mut q_raw = vec![0f32; rows * kd];
    let mut k_raw = vec![0f32; rows * kd];
    let mut v_raw = vec![0f32; rows * vd];
    for r in 0..rows {
        let src = &conv_btc[r * cfg.conv_dim()..(r + 1) * cfg.conv_dim()];
        q_raw[r * kd..(r + 1) * kd].copy_from_slice(&src[0..kd]);
        k_raw[r * kd..(r + 1) * kd].copy_from_slice(&src[kd..2 * kd]);
        v_raw[r * vd..(r + 1) * vd].copy_from_slice(&src[2 * kd..2 * kd + vd]);
    }

    // 4. g and beta
    let mut beta = vec![0f32; rows * cfg.num_v_heads];
    let mut g = vec![0f32; rows * cfg.num_v_heads];
    for r in 0..rows {
        for hh in 0..cfg.num_v_heads {
            let i = r * cfg.num_v_heads + hh;
            beta[i] = sigmoid(bb[i]);
            // `-exp(A_log) * softplus(a + dt_bias)`, with the f32 casts the
            // reference applies explicitly (a comment there notes that in fp16
            // `A` can otherwise become -inf).
            g[i] = -w.a_log[hh].exp() * softplus(aa[i] + w.dt_bias[hh]);
        }
    }

    // 5. GQA expansion, before the rule
    let ratio = cfg.kv_ratio();
    let q = repeat_interleave_heads(&q_raw, b, t, cfg.num_k_heads, cfg.head_k_dim, ratio);
    let k = repeat_interleave_heads(&k_raw, b, t, cfg.num_k_heads, cfg.head_k_dim, ratio);

    // 6. the rule, continued from the state when there is one
    let shape = Shape { b, t, h: cfg.num_v_heads, k: cfg.head_k_dim, v: cfg.head_v_dim };
    let (delta_out, delta_state) = match state {
        Some(st) => {
            let out = forward_prepared_into(&shape, &q, &k, &v_raw, &g, &beta, &mut st.ssm);
            // Snapshot for the trace, then advance the conv state. The clone is
            // diagnostic only -- `gdncheck` compares this tensor -- and could be
            // skipped when nobody is looking.
            let snap = st.ssm.clone();
            st.conv = new_conv_state;
            (out, snap)
        }
        None => forward_prepared(&shape, &q, &k, &v_raw, &g, &beta),
    };

    // 7. gated norm over head_v_dim, with z as the gate.
    //
    //    The reference reshapes both to `(-1, head_v_dim)`:
    //        core_attn_out.reshape(-1, head_v_dim)   from [B, T, H, V]
    //        z.reshape(-1, head_v_dim)               from [B, T, H, V]
    //    so the norm sees `B*T*H` rows, not `B*T`. Using `rows` here would
    //    normalise across the wrong axis and produce plausible-looking garbage.
    let norm_rows = rows * cfg.num_v_heads;
    let norm = rmsnorm_gated(&w.norm, &delta_out, &z_full, norm_rows, cfg.head_v_dim, cfg.eps);

    // 8. out_proj
    let out_proj = linear(&w.out_proj, None, &norm, rows, vd, h);

    GdnTrace {
        in_proj_qkv: mixed,
        conv_in: mixed_bct,
        conv_out,
        in_proj_z: z_full,
        in_proj_b: bb,
        in_proj_a: aa,
        q,
        k,
        v: v_raw,
        g,
        beta,
        delta_out,
        delta_state,
        norm,
        out_proj,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_and_sigmoid_at_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-7);
        // above the threshold softplus is the identity
        assert_eq!(softplus(30.0), 30.0);
    }

    #[test]
    fn rmsnorm_1plus_is_identity_at_zero_weight() {
        // w = 0 -> scale 1, so the output is x normalised, not zero.
        let x = vec![3.0f32, 4.0, 0.0, 0.0];
        let w = vec![0.0f32, 0.0];
        let y = rmsnorm_1plus(&w, &x, 2, 2, 1e-6);
        // mean(x^2) = 12.5 -> scale = 1/sqrt(12.5)
        let inv = 1.0f32 / 12.5f32.sqrt();
        assert!((y[0] - 3.0 * inv).abs() < 1e-6, "{y:?}");
        assert!((y[1] - 4.0 * inv).abs() < 1e-6, "{y:?}");
    }

    #[test]
    fn rmsnorm_gated_uses_plain_w_not_1_plus_w() {
        // With w = 1 and gate = 0 (silu(0) = 0) the output must be 0 either way,
        // so use a large gate to separate the two conventions.
        let x = vec![2.0f32, 2.0];
        let z = vec![10.0f32, 10.0];
        let w1 = vec![1.0f32, 1.0];
        let y = rmsnorm_gated(&w1, &x, &z, 1, 2, 1e-6);
        // x_hat = 1 each (x equal), silu(10) ~ 10
        let silu10 = 10.0f32 / (1.0 + (-10.0f32).exp());
        assert!((y[0] - 1.0 * silu10).abs() < 1e-4, "{y:?}");
        // If the implementation had used (1+w) this would be ~2x larger.
        assert!((y[0] - 2.0 * silu10).abs() > 1.0, "looks like (1+w) was used");
    }

    #[test]
    fn conv_is_causal_and_shapes_hold() {
        let (b, c, t, k) = (1usize, 1usize, 5usize, 4usize);
        let w = vec![1.0f32, 1.0, 1.0, 1.0];
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (y, _) = conv_forward(&w, None, &x, b, c, t, k, None);
        assert_eq!(y.len(), b * c * t);
        // out[t] = silu(sum of x[t-3..=t] present)
        // t=0 -> x[0] = 1
        assert!((y[0] - silu(1.0)).abs() < 1e-6);
        // t=1 -> x[0]+x[1] = 3
        assert!((y[1] - silu(3.0)).abs() < 1e-6);
        // t=3 -> x[0..=3] = 10
        assert!((y[3] - silu(10.0)).abs() < 1e-6);
        // causality: changing x[4] must not alter y[0..=3]
        let mut x2 = x.clone();
        x2[4] = 100.0;
        let (y2, _) = conv_forward(&w, None, &x2, b, c, t, k, None);
        for i in 0..4 {
            assert!((y[i] - y2[i]).abs() < 1e-6, "position {i} saw the future");
        }
        assert!((y[4] - y2[4]).abs() > 1e-3, "position 4 should change");
    }

    /// Two batches must not mix, and a batch's channel block must not read the
    /// neighbouring batch's samples.
    #[test]
    fn conv_does_not_mix_batches() {
        let (b, c, t, k) = (2usize, 1usize, 3usize, 2usize);
        let w = vec![1.0f32, 0.0];
        let x = vec![1.0f32, 2.0, 3.0, /* batch 1 */ 10.0, 20.0, 30.0];
        let (y, _) = conv_forward(&w, None, &x, b, c, t, k, None);
        // out[b, 0, t] = silu(x[b, 0, t-1]) with the left pad dropped
        assert!((y[0] - silu(0.0)).abs() < 1e-6);
        assert!((y[1] - silu(1.0)).abs() < 1e-6);
        assert!((y[2] - silu(2.0)).abs() < 1e-6);
        assert!((y[3] - silu(0.0)).abs() < 1e-6, "batch 1 leaked batch 0");
        assert!((y[4] - silu(10.0)).abs() < 1e-6, "batch 1 leaked batch 0");
        assert!((y[5] - silu(20.0)).abs() < 1e-6);
    }

    #[test]
    fn repeat_interleave_duplicates_consecutively() {
        // [1, 1, 2, 2] heads of dim 1
        let x = vec![10.0f32, 20.0];
        let y = repeat_interleave_heads(&x, 1, 1, 2, 1, 2);
        assert_eq!(y, vec![10.0, 10.0, 20.0, 20.0]);
        // ratio 1 is a copy
        let z = repeat_interleave_heads(&x, 1, 1, 2, 1, 1);
        assert_eq!(z, x);
    }

    #[test]
    fn transpose_round_trips_and_places_elements() {
        let (b, t, c) = (2usize, 4usize, 3usize);
        // x is [B, T, C]; element (bi, ti, ci) carries a unique tag.
        let mut x = vec![0f32; b * t * c];
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..c {
                    x[(bi * t + ti) * c + ci] = (bi * 100 + ti * 10 + ci) as f32;
                }
            }
        }
        let y = btc_to_bct(&x, b, t, c);
        // y must be [B, C, T]: (bi, ci, ti) reads the same tag
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..c {
                    let want = (bi * 100 + ti * 10 + ci) as f32;
                    assert_eq!(y[(bi * c + ci) * t + ti], want,
                               "bct layout wrong at bi={bi} ci={ci} ti={ti}");
                }
            }
        }
        let z = bct_to_btc(&y, b, c, t);
        assert_eq!(x, z, "round trip");
    }

    /// Threading splits by row and leaves each output element's accumulation order
    /// untouched, so the parallel path must agree with the serial one bit for bit.
    /// A test that only compared within a tolerance would not notice a reduction
    /// order change, which is exactly what would make results irreproducible.
    #[test]
    fn parallel_linear_is_bit_identical_to_serial() {
        let (rows, in_dim, out_dim) = (200usize, 64usize, 40usize);
        // Enough work to clear the threshold, so the parallel branch is taken.
        assert!(rows * in_dim * out_dim >= super::PARALLEL_MIN_WORK);
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i * 40503) % 997) as f32 / 500.0 - 1.0)
            .collect();

        let par = linear(&w, None, &x, rows, in_dim, out_dim);

        // Reproduce the serial reduction exactly, with the same f64 accumulator the
        // implementation uses. The property under test is that threading does not
        // change the result, so the two paths must differ in *scheduling* only.
        let mut ser = vec![0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] as f64 * x[r * in_dim + i] as f64;
                }
                ser[r * out_dim + o] = acc as f32;
            }
        }
        assert_eq!(par.len(), ser.len());
        for i in 0..par.len() {
            assert_eq!(
                par[i].to_bits(),
                ser[i].to_bits(),
                "index {i}: parallel {} vs serial {}",
                par[i],
                ser[i]
            );
        }
    }

    #[test]
    fn linear_matches_explicit_matmul() {
        // w [2, 3], x [1, 3]
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 1.0, 1.0];
        let y = linear(&w, None, &x, 1, 3, 2);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    /// A test fixture: a small but structurally complete gated delta net.
    fn gdn_fixture() -> (GdnConfig, GdnWeights) {
        let cfg = GdnConfig {
            hidden: 8,
            num_k_heads: 2,
            num_v_heads: 2,
            head_k_dim: 3,
            head_v_dim: 4,
            conv_kernel: 4,
            eps: 1e-6,
        };
        let det = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                    ((x >> 8) % 2000) as f32 / 1000.0 - 1.0
                })
                .collect()
        };
        let w = GdnWeights {
            in_proj_qkv: det(cfg.conv_dim() * cfg.hidden, 1),
            in_proj_z: det(cfg.value_dim() * cfg.hidden, 2),
            in_proj_b: det(cfg.num_v_heads * cfg.hidden, 3),
            in_proj_a: det(cfg.num_v_heads * cfg.hidden, 4),
            conv1d: det(cfg.conv_dim() * cfg.conv_kernel, 5),
            a_log: vec![0.1; cfg.num_v_heads],
            dt_bias: vec![0.0; cfg.num_v_heads],
            norm: vec![0.5; cfg.head_v_dim],
            out_proj: det(cfg.hidden * cfg.value_dim(), 6),
        };
        (cfg, w)
    }

    /// The convolution with a carried state must equal the convolution of the whole
    /// signal at once. This is the conv half of what a cache depends on.
    #[test]
    fn conv_with_state_equals_the_whole_signal() {
        let (b, c, k) = (2usize, 3usize, 4usize);
        let t = 9usize;
        let w: Vec<f32> = (0..c * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let x: Vec<f32> = (0..b * c * t).map(|i| ((i * 37 % 23) as f32) - 11.0).collect();

        let (whole, _) = conv_forward(&w, None, &x, b, c, t, k, None);

        // Feed it in chunks of 1, 2, 3 and 3, carrying the state.
        let mut state = vec![0f32; b * c * (k - 1)];
        let mut got = vec![0f32; b * c * t];
        let mut done = 0usize;
        for len in [1usize, 2, 3, 3] {
            // Gather the chunk for every (batch, channel).
            let mut chunk = vec![0f32; b * c * len];
            for bi in 0..b {
                for ci in 0..c {
                    let src = (bi * c + ci) * t + done;
                    let dst = (bi * c + ci) * len;
                    chunk[dst..dst + len].copy_from_slice(&x[src..src + len]);
                }
            }
            let (out, ns) = conv_forward(&w, None, &chunk, b, c, len, k, Some(&state));
            for bi in 0..b {
                for ci in 0..c {
                    let dst = (bi * c + ci) * t + done;
                    let src = (bi * c + ci) * len;
                    got[dst..dst + len].copy_from_slice(&out[src..src + len]);
                }
            }
            state = ns;
            done += len;
        }
        assert_eq!(done, t);
        for (i, (a, b)) in got.iter().zip(whole.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "index {i}: chunked {a} vs whole {b}");
        }
    }

    /// The full mixer, one token at a time with a state, must equal the whole-sequence
    /// call. This is the invariant the KV/recurrent cache rests on.
    #[test]
    fn incremental_mixer_equals_whole_sequence() {
        let (cfg, w) = gdn_fixture();
        let (b, t) = (1usize, 6usize);
        let x: Vec<f32> = (0..b * t * cfg.hidden)
            .map(|i| ((i * 17 % 31) as f32) * 0.1 - 1.5)
            .collect();

        let whole = forward(&cfg, &w, &x, b, t);

        let mut st = GdnState::new(&cfg, b);
        let mut got = vec![0f32; b * t * cfg.hidden];
        for ti in 0..t {
            let chunk: Vec<f32> = x[ti * cfg.hidden..(ti + 1) * cfg.hidden].to_vec();
            let tr = forward_with_state(&cfg, &w, &chunk, b, 1, &mut st);
            got[ti * cfg.hidden..(ti + 1) * cfg.hidden].copy_from_slice(&tr.out_proj);
        }
        for (i, (a, b)) in got.iter().zip(whole.out_proj.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "index {i}: incremental {a} vs whole {b}"
            );
        }
    }

    /// The state must actually be used and advanced. If the conv state were ignored, a
    /// single-token step would lose its left context; if it were not written back, the
    /// next step would too. Both would still produce finite numbers.
    #[test]
    fn the_state_is_used_and_advanced() {
        let (cfg, w) = gdn_fixture();
        let b = 1usize;
        let x: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.7];

        // A fresh state gives one answer.
        let mut st_fresh = GdnState::new(&cfg, b);
        let a = forward_with_state(&cfg, &w, &x, b, 1, &mut st_fresh).out_proj;

        // A pre-used state must give a different one, or the state is being ignored.
        let mut st_used = GdnState::new(&cfg, b);
        let warm: Vec<f32> = vec![0.9; cfg.hidden];
        forward_with_state(&cfg, &w, &warm, b, 1, &mut st_used);
        let b_out = forward_with_state(&cfg, &w, &x, b, 1, &mut st_used).out_proj;

        let differs = a.iter().zip(b_out.iter()).any(|(p, q)| (p - q).abs() > 1e-6);
        assert!(differs, "the initial state had no effect on the output");

        // And the state must move after the call.
        let mut st2 = GdnState::new(&cfg, b);
        let before = st2.conv.clone();
        let ssm_before = st2.ssm.clone();
        forward_with_state(&cfg, &w, &x, b, 1, &mut st2);
        assert_ne!(st2.conv, before, "conv state was not advanced");
        assert_ne!(st2.ssm, ssm_before, "ssm state was not advanced");
    }
}
