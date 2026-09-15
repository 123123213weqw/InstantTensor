//! `Qwen3_5Attention` -- the full-attention mixer, used by 1 layer in 4.
//!
//! # The chain
//!
//! ```text
//! q_full = q_proj(x)                    [B, T, num_heads * head_dim * 2]
//! q_full viewed as [B, T, num_heads, head_dim * 2]
//! q, gate = chunk(q_full, 2, dim=-1)    each [B, T, num_heads, head_dim]
//! gate = gate.reshape(B, T, num_heads * head_dim)
//! q = q_norm(q)                         RMSNorm on the head dim, x*(1+w)
//! k = k_norm(k_proj(x))                 [B, T, num_kv_heads, head_dim]
//! v = v_proj(x)                         [B, T, num_kv_heads, head_dim]
//! q, k = rope(q), rope(k)               first rotary_dim dims only
//! q,k,v -> [B, H, T, D]
//! attn = softmax(q @ k^T * head_dim**-0.5 + causal_mask)   in fp32
//! out  = attn @ v -> [B, H, T, D] -> [B, T, H*D]
//! out  = out * sigmoid(gate)            <- the output gate
//! y    = o_proj(out)
//! ```
//!
//! # Three things that are easy to get wrong
//!
//! **The query/gate split is per head, not front-and-back.** `q_proj` emits
//! `num_heads * head_dim * 2` values which are *viewed* as
//! `[B, T, num_heads, head_dim * 2]` before being chunked along the last axis. So
//! head `h` takes columns `h*2D .. h*2D + D` as its query and
//! `h*2D + D .. (h+1)*2D` as its gate. Splitting the flat output in half instead --
//! "the first half is the query" -- produces plausible output and is wrong.
//!
//! **The gate uses `sigmoid`, not the `swish` the config claims.** The config key is
//! `output_gate_type` and the reference ignores it: the code is
//! `attn_output * torch.sigmoid(gate)`. Read the source, not the config.
//!
//! **Only part of the head is rotated.** `partial_rotary_factor` is 0.25, so with
//! `head_dim = 32` only the first 8 dimensions are rotated and the remaining 24
//! pass through untouched. `rotate_half` splits that 8 into 4+4 and returns
//! `cat(-x2, x1)` -- the half convention, not the interleaved one.
//!
//! # Text-only MRoPE
//!
//! `Qwen3_5TextRotaryEmbedding` builds three interleaved frequency rows (temporal,
//! height, width) and overwrites row 0 with rows 1 and 2 at interleaved indices.
//! For text-only input all three rows come from the same `arange`, so the
//! interleave is a no-op: the result is bit-identical to plain RoPE. That is
//! checked rather than assumed -- see `rope_matches_plain_for_text_only`.

use crate::{linear, rmsnorm_1plus};

/// Sizes for one full-attention layer.
#[derive(Debug, Clone, Copy)]
pub struct AttnConfig {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `int(head_dim * partial_rotary_factor)` -- 8 of 32 in the tiny model.
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl AttnConfig {
    pub fn q_out_dim(&self) -> usize {
        self.num_heads * self.head_dim * 2
    }
    pub fn kv_out_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// `num_attention_heads // num_key_value_heads`
    pub fn kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
    pub fn scaling(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}

/// Attention weights. `q_norm`/`k_norm` are `Qwen3_5RMSNorm`, i.e. `x * (1 + w)`.
#[derive(Debug, Clone)]
pub struct AttnWeights {
    /// `[num_heads * head_dim * 2, hidden]`
    pub q_proj: Vec<f32>,
    /// `[num_kv_heads * head_dim, hidden]`
    pub k_proj: Vec<f32>,
    /// `[num_kv_heads * head_dim, hidden]`
    pub v_proj: Vec<f32>,
    /// `[hidden, num_heads * head_dim]`
    pub o_proj: Vec<f32>,
    /// `[head_dim]`
    pub q_norm: Vec<f32>,
    /// `[head_dim]`
    pub k_norm: Vec<f32>,
}

/// Every intermediate, in chain order.
#[derive(Debug, Clone)]
pub struct AttnTrace {
    /// `[B, T, num_heads * head_dim * 2]`
    pub q_proj: Vec<f32>,
    /// `[B, T, num_heads, head_dim]`, post-norm, pre-rope
    pub q_norm: Vec<f32>,
    /// `[B, T, num_kv_heads * head_dim]`
    pub k_proj: Vec<f32>,
    /// `[B, T, num_kv_heads, head_dim]`, post-norm, pre-rope
    pub k_norm: Vec<f32>,
    /// `[B, T, num_kv_heads, head_dim]`
    pub v_proj: Vec<f32>,
    /// `[B, T, num_heads * head_dim]` -- after the sigmoid gate, before `o_proj`.
    /// This is the tensor the bundle records as `self_attn__out0`.
    pub gated: Vec<f32>,
    pub o_proj: Vec<f32>,
}

/// Build plain RoPE tables for positions `start .. start + t`.
///
/// Returns `(cos, sin)`, each `[t, rotary_dim]`, laid out so index `i` of the
/// rotary slice pairs with index `i` of the rotated vector.
pub fn build_rope(cfg: &AttnConfig, t: usize, start: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.rotary_dim / 2;
    // inv_freq[i] = 1 / theta^(2i / rotary_dim), matching
    // `1.0 / (base ** (arange(0, dim, 2) / dim))`.
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            let e = (2 * i) as f32 / cfg.rotary_dim as f32;
            1.0f32 / cfg.rope_theta.powf(e)
        })
        .collect();

    let mut cos = vec![0f32; t * cfg.rotary_dim];
    let mut sin = vec![0f32; t * cfg.rotary_dim];
    for pos in 0..t {
        let p = (start + pos) as f32;
        for i in 0..half {
            let f = p * inv_freq[i];
            // `emb = torch.cat((freqs, freqs), dim=-1)`: the frequency vector is
            // duplicated, so slot i and slot i+half carry the same angle.
            cos[pos * cfg.rotary_dim + i] = f.cos();
            cos[pos * cfg.rotary_dim + i + half] = f.cos();
            sin[pos * cfg.rotary_dim + i] = f.sin();
            sin[pos * cfg.rotary_dim + i + half] = f.sin();
        }
    }
    (cos, sin)
}

/// `rotate_half(x) = cat(-x[half:], x[:half])` over the last axis of length
/// `rotary_dim`.
fn rotate_half(x: &[f32], rotary_dim: usize) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = vec![0f32; rotary_dim];
    for i in 0..half {
        out[i] = -x[i + half];
        out[i + half] = x[i];
    }
    out
}

/// Rotate the first `rotary_dim` dims of each head vector; the rest pass through.
///
/// `x` is `[rows, head_dim]` flattened in the `[B, T, H, D]` layout, so row `r`
/// belongs to position `(r / heads) % t`. Getting this wrong -- `r % t` looks
/// plausible and is correct when there is a single head -- scrambles which position
/// each head is rotated by, while leaving every tensor *shape* unchanged.
///
/// A single-head unit test cannot distinguish the two, so `rope_uses_the_position_
/// not_the_row` exercises two heads over three positions.
#[allow(clippy::too_many_arguments)]
fn apply_rope(
    x: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
    t: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    let mut rot = vec![0f32; rotary_dim];
    for r in 0..rows {
        let base = r * head_dim;
        // Row `r` is (batch, time, head) flattened as (b*T + t)*heads + h, so the
        // position is `(r / heads) % T`.
        let pos = (r / heads) % t;
        rot.copy_from_slice(&x[base..base + rotary_dim]);
        let rh = rotate_half(&rot, rotary_dim);
        for i in 0..rotary_dim {
            let c = cos[pos * rotary_dim + i];
            let s = sin[pos * rotary_dim + i];
            out[base + i] = rot[i] * c + rh[i] * s;
        }
        // dims beyond rotary_dim are already copied by `x.to_vec()`
    }
    out
}

/// Softmax over the last axis, in f32, matching
/// `nn.functional.softmax(x, dim=-1, dtype=torch.float32)`.
fn softmax_in_place(v: &mut [f32], len: usize) {
    for row in v.chunks_mut(len) {
        let mut m = f32::NEG_INFINITY;
        for x in row.iter() {
            if *x > m {
                m = *x;
            }
        }
        let mut sum = 0f32;
        for x in row.iter_mut() {
            // A fully-masked row would produce NaN; causality guarantees at least
            // one unmasked entry (position t always attends to itself).
            let e = (*x - m).exp();
            *x = e;
            sum += e;
        }
        let inv = 1.0f32 / sum;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

/// Run one full-attention layer on an already-normalised input.
///
/// `x` is `[B, T, hidden]` (the output of the layer's `input_layernorm`).
pub fn forward(
    cfg: &AttnConfig,
    w: &AttnWeights,
    x: &[f32],
    b: usize,
    t: usize,
    cos: &[f32],
    sin: &[f32],
) -> AttnTrace {
    let rows = b * t;
    let h = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let d = cfg.head_dim;

    // 1. projections
    let q_flat = linear(&w.q_proj, None, x, rows, cfg.hidden, cfg.q_out_dim()); // [rows, H*2D]
    let k_flat = linear(&w.k_proj, None, x, rows, cfg.hidden, cfg.kv_out_dim());
    let v_flat = linear(&w.v_proj, None, x, rows, cfg.hidden, cfg.kv_out_dim());

    // 2. per-head query/gate split. The reference views the projection output as
    //    [B, T, num_heads, head_dim*2] and chunks the *last* axis, so head h owns
    //    columns h*2D..h*2D+D (query) and h*2D+D..(h+1)*2D (gate).
    let two_d = 2 * d;
    let mut q = vec![0f32; rows * h * d];
    let mut gate = vec![0f32; rows * h * d];
    for r in 0..rows {
        for hh in 0..h {
            let src = r * (h * two_d) + hh * two_d;
            let dst = r * (h * d) + hh * d;
            q[dst..dst + d].copy_from_slice(&q_flat[src..src + d]);
            gate[dst..dst + d].copy_from_slice(&q_flat[src + d..src + two_d]);
        }
    }

    // 3. q_norm / k_norm, both Qwen3_5RMSNorm (x*(1+w)) on the head dim.
    let q_normed = rmsnorm_1plus(&w.q_norm, &q, rows * h, d, cfg.eps);
    let k_normed = rmsnorm_1plus(&w.k_norm, &k_flat, rows * kvh, d, cfg.eps);

    // 4. rope on the first rotary_dim dims. q_normed/k_normed are [rows*heads, d]
    //    with row index r*heads + head, so `row % t` gives the position.
    let q_rot = apply_rope(&q_normed, rows * h, h, d, cos, sin, t, cfg.rotary_dim);
    let k_rot = apply_rope(&k_normed, rows * kvh, kvh, d, cos, sin, t, cfg.rotary_dim);

    // 5. to [B, H, T, D] for the contraction.
    let to_bhtd = |src: &[f32], heads: usize| -> Vec<f32> {
        let mut out = vec![0f32; b * heads * t * d];
        for bi in 0..b {
            for ti in 0..t {
                for hh in 0..heads {
                    let s = ((bi * t + ti) * heads + hh) * d;
                    let dst = ((bi * heads + hh) * t + ti) * d;
                    out[dst..dst + d].copy_from_slice(&src[s..s + d]);
                }
            }
        }
        out
    };
    let q_bhtd = to_bhtd(&q_rot, h);
    let k_bhtd = to_bhtd(&k_rot, kvh);
    let v_bhtd = to_bhtd(&v_flat, kvh);

    // 6. GQA: repeat each kv head `kv_groups` times, consecutively. This is
    //    `repeat_kv`, which expands+reshapes and so matches repeat_interleave.
    let groups = cfg.kv_groups();
    let mut k_full = vec![0f32; b * h * t * d];
    let mut v_full = vec![0f32; b * h * t * d];
    for bi in 0..b {
        for g in 0..groups {
            for hh in 0..kvh {
                let dst_head = hh * groups + g;
                let src = ((bi * kvh + hh) * t) * d;
                let dst = ((bi * h + dst_head) * t) * d;
                k_full[dst..dst + t * d].copy_from_slice(&k_bhtd[src..src + t * d]);
                v_full[dst..dst + t * d].copy_from_slice(&v_bhtd[src..src + t * d]);
            }
        }
    }

    // 7. scores, causal mask, softmax, then the value contraction.
    let scaling = cfg.scaling();
    let mut out_bhtd = vec![0f32; b * h * t * d];
    for bi in 0..b {
        for hh in 0..h {
            let qb = ((bi * h + hh) * t) * d;
            let kb = ((bi * h + hh) * t) * d;
            for ti in 0..t {
                // scores over all keys, masked to keys <= ti
                let mut scores = vec![f32::NEG_INFINITY; t];
                let qr = qb + ti * d;
                for (tj, sc) in scores.iter_mut().enumerate().take(ti + 1) {
                    let kr = kb + tj * d;
                    let mut acc = 0f32;
                    for i in 0..d {
                        acc += q_bhtd[qr + i] * k_full[kr + i];
                    }
                    *sc = acc * scaling;
                }
                softmax_in_place(&mut scores, t);
                // a masked key has weight exactly 0, so it contributes nothing
                let orow = qb + ti * d;
                for i in 0..d {
                    let mut acc = 0f32;
                    for tj in 0..=ti {
                        acc += scores[tj] * v_full[kb + tj * d + i];
                    }
                    out_bhtd[orow + i] = acc;
                }
            }
        }
    }

    // 8. back to [B, T, H*D]
    let mut attn_out = vec![0f32; rows * h * d];
    for bi in 0..b {
        for hh in 0..h {
            for ti in 0..t {
                let s = ((bi * h + hh) * t + ti) * d;
                let dst = ((bi * t + ti) * h + hh) * d;
                attn_out[dst..dst + d].copy_from_slice(&out_bhtd[s..s + d]);
            }
        }
    }

    // 9. the output gate: sigmoid, per the source (the config says "swish").
    let mut gated = vec![0f32; rows * h * d];
    for i in 0..gated.len() {
        let g = gate[i];
        gated[i] = attn_out[i] * (1.0 / (1.0 + (-g).exp()));
    }

    // 10. o_proj
    let o = linear(&w.o_proj, None, &gated, rows, h * d, cfg.hidden);

    AttnTrace {
        q_proj: q_flat,
        q_norm: q_normed,
        k_proj: k_flat,
        k_norm: k_normed,
        v_proj: v_flat,
        gated,
        o_proj: o,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AttnConfig {
        AttnConfig {
            hidden: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            rope_theta: 10000.0,
            eps: 1e-6,
        }
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let c = cfg();
        let (cos, sin) = build_rope(&c, 3, 0);
        // pos 0 -> angle 0 -> cos 1, sin 0
        for i in 0..c.rotary_dim {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[{i}]={}", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[{i}]={}", sin[i]);
        }
    }

    /// The reference duplicates the frequency vector (`cat((freqs, freqs))`), so
    /// slot `i` and slot `i + rotary_dim/2` carry the same angle.
    #[test]
    fn rope_duplicates_the_frequency_vector() {
        let c = cfg();
        let (cos, sin) = build_rope(&c, 5, 1);
        let half = c.rotary_dim / 2;
        for pos in 0..5 {
            for i in 0..half {
                let a = pos * c.rotary_dim + i;
                let b = a + half;
                assert!((cos[a] - cos[b]).abs() < 1e-6, "cos duplicated mismatch");
                assert!((sin[a] - sin[b]).abs() < 1e-6, "sin duplicated mismatch");
            }
        }
    }

    #[test]
    fn rope_preserves_norm() {
        // A rotation must not change the length of the rotated segment.
        let c = cfg();
        let (cos, sin) = build_rope(&c, 4, 0);
        let head_dim = c.head_dim;
        let rows = 4;
        let mut x = vec![0f32; rows * head_dim];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 37 % 11) as f32) - 5.0;
        }
        let y = apply_rope(&x, rows, 1, head_dim, &cos, &sin, 4, c.rotary_dim);
        for r in 0..rows {
            let (mut n0, mut n1) = (0f32, 0f32);
            for i in 0..c.rotary_dim {
                n0 += x[r * head_dim + i] * x[r * head_dim + i];
                n1 += y[r * head_dim + i] * y[r * head_dim + i];
            }
            assert!((n0 - n1).abs() < 1e-4, "norm changed: {n0} vs {n1}");
            // dims beyond rotary_dim must be untouched
            for i in c.rotary_dim..head_dim {
                assert_eq!(x[r * head_dim + i], y[r * head_dim + i]);
            }
        }
    }

    /// The bug `r % t` would introduce: with more than one head, the row index no
    /// longer identifies the position. Two heads over three positions means the row
    /// order is (t0,h0) (t0,h1) (t1,h0) (t1,h1) (t2,h0) (t2,h1), so rows 0 and 1 are
    /// both position 0 -- `r % t` would put row 1 at position 1.
    #[test]
    fn rope_uses_the_position_not_the_row() {
        let c = AttnConfig {
            hidden: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            rotary_dim: 4,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let t = 3usize;
        let heads = 2usize;
        let d = c.head_dim;
        let (cos, sin) = build_rope(&c, t, 0);

        // One distinct vector per (position, head): the value encodes both.
        let rows = t * heads;
        let mut x = vec![0f32; rows * d];
        for ti in 0..t {
            for hh in 0..heads {
                let base = (ti * heads + hh) * d;
                for i in 0..d {
                    x[base + i] = (ti * 10 + hh) as f32 + i as f32 * 0.1;
                }
            }
        }
        let y = apply_rope(&x, rows, heads, d, &cos, &sin, t, c.rotary_dim);
        // Row for (ti, hh) must equal the rotation of x at position ti.
        for ti in 0..t {
            for hh in 0..heads {
                let src = (ti * heads + hh) * d;
                let row = &x[src..src + d];
                let want = apply_rope(row, 1, 1, d, &cos, &sin, t, c.rotary_dim);
                // `apply_rope` with one row reads position 0, so compare against the
                // reference rotation built for this position explicitly.
                let _ = want;
                let mut rot = vec![0f32; c.rotary_dim];
                rot.copy_from_slice(&row[..c.rotary_dim]);
                let rh = rotate_half(&rot, c.rotary_dim);
                for i in 0..c.rotary_dim {
                    let ang = ti * c.rotary_dim + i;
                    let expect = rot[i] * cos[ang] + rh[i] * sin[ang];
                    let got = y[src + i];
                    assert!(
                        (got - expect).abs() < 1e-5,
                        "row for (t={ti}, h={hh}) dim {i}: got {got}, want {expect}\
                         -- rotation used the wrong position"
                    );
                }
            }
        }
    }

    #[test]
    fn rotate_half_is_the_half_convention_not_interleaved() {
        // [1, 2, 3, 4] with rotary_dim 4 -> cat(-[3,4], [1,2]) = [-3,-4,1,2]
        let out = rotate_half(&[1.0, 2.0, 3.0, 4.0], 4);
        assert_eq!(out, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    /// The trap: the query/gate split is per head, not front-half / back-half.
    #[test]
    fn query_gate_split_is_per_head() {
        let c = AttnConfig { hidden: 1, num_heads: 2, num_kv_heads: 2, head_dim: 2, rotary_dim: 2, rope_theta: 1e4, eps: 1e-6 };
        // q_proj = identity-ish so the projection output is just x repeated.
        // Use q_proj that maps x=[1] to [q_h0(2), gate_h0(2), q_h1(2), gate_h1(2)]
        let q_proj = vec![
            10.0, // -> q_h0[0]
            11.0, // -> q_h0[1]
            20.0, // -> gate_h0[0]
            21.0, // -> gate_h0[1]
            30.0, // -> q_h1[0]
            31.0, // -> q_h1[1]
            40.0, // -> gate_h1[0]
            41.0, // -> gate_h1[1]
        ];
        let w = AttnWeights {
            q_proj,
            k_proj: vec![0.0; 4],
            v_proj: vec![0.0; 4],
            o_proj: vec![0.0; 4],
            q_norm: vec![0.0; 2],
            k_norm: vec![0.0; 2],
        };
        let (cos, sin) = build_rope(&c, 1, 0);
        let tr = forward(&c, &w, &[1.0], 1, 1, &cos, &sin);
        assert_eq!(tr.q_proj, vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]);
        // q = [10,11,30,31], and the gate [20,21,40,41] must NOT be the last four
        // in a front/back split ([30,31,40,41] would be that wrong reading).
        // q_norm with zero weight scales each head to unit RMS, so compare ratios.
        let head0 = &tr.q_norm[0..2];
        let head1 = &tr.q_norm[2..4];
        assert!((head0[0] / head0[1] - 10.0 / 11.0).abs() < 1e-4, "head0 {head0:?}");
        assert!((head1[0] / head1[1] - 30.0 / 31.0).abs() < 1e-4, "head1 {head1:?}");
    }

    /// The gate is `sigmoid`, not `swish`. With gate = 0 swish and sigmoid agree,
    /// so use a large positive gate where sigmoid saturates but swish does not.
    #[test]
    fn output_gate_is_sigmoid_not_swish() {
        let c = AttnConfig { hidden: 1, num_heads: 1, num_kv_heads: 1, head_dim: 1, rotary_dim: 1, rope_theta: 1e4, eps: 1e-6 };
        let w = AttnWeights {
            q_proj: vec![0.0, 10.0], // q = 0, gate = 10
            k_proj: vec![0.0],
            v_proj: vec![0.0],
            o_proj: vec![1.0],
            q_norm: vec![0.0],
            k_norm: vec![0.0],
        };
        let (cos, sin) = build_rope(&c, 1, 0);
        // With v = 0 the attention output is zero and the gate cannot be observed,
        // so the first pass only checks that the call works; the real assertion is
        // below with v != 0.
        let zero_v = forward(&c, &w, &[1.0], 1, 1, &cos, &sin);
        assert!(zero_v.gated[0].abs() < 1e-7, "v = 0 should give 0 output");
        let w = AttnWeights {
            q_proj: vec![0.0, 10.0],
            k_proj: vec![0.0],
            v_proj: vec![1.0],
            o_proj: vec![1.0],
            q_norm: vec![0.0],
            k_norm: vec![0.0],
        };
        let tr = forward(&c, &w, &[1.0], 1, 1, &cos, &sin);
        let sig = 1.0 / (1.0 + (-10.0f32).exp());
        let swish = 10.0 / (1.0 + (-10.0f32).exp());
        assert!((tr.gated[0] - sig).abs() < 1e-5, "gated={} sigmoid={sig}", tr.gated[0]);
        assert!((tr.gated[0] - swish).abs() > 1.0, "looks like swish was used");
    }

    /// Which kv head each query head reads.
    ///
    /// `repeat_kv` expands as `hidden[:, :, None].expand(b, kvh, n_rep, ...)` and
    /// then reshapes, so output head `j` reads kv head `j / n_rep` -- kv head 0 feeds
    /// query heads 0..n_rep-1, contiguously.
    ///
    /// This needs its own test because the tiny golden model has
    /// `num_key_value_heads = 1`, where the wrong ordering
    /// (`g * kvh + hh`) is *identical* to the right one. Injecting that bug into the
    /// implementation is not caught by any golden check; the real 27B model has a
    /// ratio of 3, so the convention has to be pinned down here or nowhere.
    #[test]
    fn gqa_head_mapping_is_contiguous_per_kv_head() {
        let c = AttnConfig {
            hidden: 1,
            num_heads: 4,
            num_kv_heads: 2, // n_rep = 2, so head j reads kv head j / 2
            head_dim: 2,
            rotary_dim: 2,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        // q_proj emits num_heads*head_dim*2 = 16 values; all zero, so every query is
        // zero and the softmax is uniform over the single key.
        let w = AttnWeights {
            q_proj: vec![0.0; 16],
            k_proj: vec![0.0; 4],
            // kv head 0 -> 10, kv head 1 -> 20, both dims
            v_proj: vec![10.0, 10.0, 20.0, 20.0],
            o_proj: vec![1.0; 8],
            q_norm: vec![0.0; 2],
            k_norm: vec![0.0; 2],
        };
        let (cos, sin) = build_rope(&c, 1, 0);
        let tr = forward(&c, &w, &[1.0], 1, 1, &cos, &sin);
        assert_eq!(tr.gated.len(), 4 * 2);
        // gate is 0 -> sigmoid(0) = 0.5, so each head's output is v * 0.5
        let want = [
            5.0, 5.0, // query head 0 <- kv head 0
            5.0, 5.0, // query head 1 <- kv head 0
            10.0, 10.0, // query head 2 <- kv head 1
            10.0, 10.0, // query head 3 <- kv head 1
        ];
        for (i, wv) in want.iter().enumerate() {
            assert!(
                (tr.gated[i] - wv).abs() < 1e-5,
                "head {} dim {}: got {}, want {wv} -- kv heads interleaved instead of \
                 contiguous (got {:?})",
                i / 2,
                i % 2,
                tr.gated[i],
                tr.gated
            );
        }
    }

    /// Causality: position 0 attends only to key 0, so changing a later position
    /// must leave position 0's output bit-identical. Config: hidden=2, one head,
    /// head_dim=2, one kv head.
    #[test]
    fn causal_mask_hides_the_future() {
        let c = AttnConfig {
            hidden: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let w = AttnWeights {
            // q_out_dim = num_heads * head_dim * 2 = 4, hidden = 2
            q_proj: vec![
                1.0, 0.0, // q[0]
                0.0, 1.0, // q[1]
                0.0, 0.0, // gate[0]
                0.0, 0.0, // gate[1]
            ],
            k_proj: vec![1.0, 0.0, 0.0, 1.0],
            v_proj: vec![1.0, 0.0, 0.0, 1.0],
            o_proj: vec![1.0, 0.0, 0.0, 1.0],
            q_norm: vec![0.0, 0.0],
            k_norm: vec![0.0, 0.0],
        };
        let (cos, sin) = build_rope(&c, 3, 0);
        // positions 0,1 identical; position 2 differs
        let a = forward(&c, &w, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1, 3, &cos, &sin);
        let b = forward(&c, &w, &[1.0, 0.0, 1.0, 0.0, 9.0, 7.0], 1, 3, &cos, &sin);
        for i in 0..c.head_dim {
            assert!(
                (a.gated[i] - b.gated[i]).abs() < 1e-6,
                "position 0 dim {i} saw the future: {} vs {}",
                a.gated[i],
                b.gated[i]
            );
        }
        // And position 2 must change, so the test cannot pass by the mixer being a
        // no-op.
        let tail = 2 * c.head_dim;
        let changed = (0..c.head_dim).any(|i| (a.gated[tail + i] - b.gated[tail + i]).abs() > 1e-6);
        assert!(changed, "position 2 did not change when its own input changed");
    }
}
