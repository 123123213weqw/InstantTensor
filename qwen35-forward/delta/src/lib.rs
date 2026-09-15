//! The gated delta rule — Qwen3.5's linear-attention recurrence.
//!
//! This is the core of the 48-of-64 layers that are `linear_attention` in
//! qwen35. It is a fixed-size recurrence, not a GEMM, so it is the part of the
//! model a hand-written implementation must get right by construction.
//!
//! # Layout
//!
//! The real model passes `[B, T, H, K]`, because the rule's first statement is a
//! `transpose(1, 2)` into `[B, H, T, K]`. The golden bundle uses the same
//! convention, so `q/k/v` are `[B, T, H, *]` and `g/beta` are `[B, T, H]`.
//! Outputs are `out [B, T, H, V]` and `state [B, H, K, V]`.
//!
//! # The rule
//!
//! For each time step `t`, per `(batch, head)`:
//!
//! ```text
//! state  = state * exp(g_t)                    // decay
//! kv_mem = (state * k_t).sum(over K)           // read out
//! delta  = (v_t - kv_mem) * beta_t             // prediction error
//! state  = state + outer(k_t, delta)           // rank-1 correction
//! out_t  = (state * q_t).sum(over K)           // read out
//! ```
//!
//! `scale = 1/sqrt(K)` is folded into `q` once, before the loop.

/// L2-normalise every `[H, D]` slice across `D`, in place.
fn l2norm_last_dim(buf: &mut [f32], rows: usize, d: usize, eps: f32) {
    for r in 0..rows {
        let base = r * d;
        let mut ss = 0f32;
        for i in 0..d {
            let v = buf[base + i];
            ss += v * v;
        }
        // rsqrt(ss + eps), matching torch.rsqrt((x*x).sum(dim=-1, keepdim=True) + eps)
        let inv = 1.0f32 / (ss + eps).sqrt();
        for i in 0..d {
            buf[base + i] *= inv;
        }
    }
}

/// All pre-loop normalisation and scaling the reference applies.
///
/// `use_qk_l2norm_in_kernel=True` in the reference, so `q` and `k` are
/// L2-normalised across the head dimension first, and `q` is then scaled by
/// `1/sqrt(K)`.
pub fn prepare_qk(q: &mut [f32], k: &mut [f32], b: usize, t: usize, h: usize, d: usize) {
    let eps = 1e-6f32;
    let rows = b * t * h;
    l2norm_last_dim(q, rows, d, eps);
    l2norm_last_dim(k, rows, d, eps);
    let scale = 1.0f32 / (d as f32).sqrt();
    for v in q.iter_mut() {
        *v *= scale;
    }
}

/// Dense layout description of one delta-rule problem.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    /// key head dim
    pub k: usize,
    /// value head dim
    pub v: usize,
}

/// Index helper for a `[B, T, H, D]` tensor.
#[inline]
fn idx4(s: &Shape, bi: usize, ti: usize, hi: usize, d: usize, dsz: usize) -> usize {
    ((bi * s.t + ti) * s.h + hi) * dsz + d
}

/// Index helper for a `[B, H, K, V]` tensor.
#[inline]
fn idx_state(s: &Shape, bi: usize, hi: usize, ki: usize, vi: usize) -> usize {
    ((bi * s.h + hi) * s.k + ki) * s.v + vi
}

/// Run the recurrence.
///
/// * `q`, `k`: `[B, T, H, K]`, already prepared (L2-normalised and scaled).
/// * `v`: `[B, T, H, V]`.
/// * `g`, `beta`: `[B, T, H]`, `g` being a *log* decay (this function applies
///   `exp`, matching the reference's `g_t = g[:, :, i].exp()`).
/// * `state`: `[B, H, K, V]`, the initial state on entry; overwritten in place
///   with the final state.
///
/// Returns `out [B, T, H, V]`.
pub fn forward(
    s: &Shape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
) -> Vec<f32> {
    let mut out = vec![0f32; s.b * s.t * s.h * s.v];

    for bi in 0..s.b {
        for ti in 0..s.t {
            for hi in 0..s.h {
                // exp(g) is a per-(batch, head, time) scalar decay.
                let g_t = g[(bi * s.t + ti) * s.h + hi].exp();
                let beta_t = beta[(bi * s.t + ti) * s.h + hi];

                // state = state * g_t
                for ki in 0..s.k {
                    for vi in 0..s.v {
                        let si = idx_state(s, bi, hi, ki, vi);
                        state[si] *= g_t;
                    }
                }

                // kv_mem[vi] = sum_ki state[ki, vi] * k_t[ki]
                let mut kv_mem = vec![0f32; s.v];
                for ki in 0..s.k {
                    let kkv = k[idx4(s, bi, ti, hi, ki, s.k)];
                    for vi in 0..s.v {
                        kv_mem[vi] += state[idx_state(s, bi, hi, ki, vi)] * kkv;
                    }
                }

                // delta[vi] = (v_t[vi] - kv_mem[vi]) * beta_t
                let mut delta = vec![0f32; s.v];
                for (vi, d) in delta.iter_mut().enumerate() {
                    let vv = v[idx4(s, bi, ti, hi, vi, s.v)];
                    *d = (vv - kv_mem[vi]) * beta_t;
                }

                // state[ki, vi] += k_t[ki] * delta[vi]   (rank-1 update)
                for ki in 0..s.k {
                    let kkv = k[idx4(s, bi, ti, hi, ki, s.k)];
                    let sbase = idx_state(s, bi, hi, ki, 0);
                    for (vi, d) in delta.iter().enumerate() {
                        state[sbase + vi] += kkv * d;
                    }
                }

                // out_t[vi] = sum_ki state[ki, vi] * q_t[ki]
                for ki in 0..s.k {
                    let qq = q[idx4(s, bi, ti, hi, ki, s.k)];
                    for vi in 0..s.v {
                        let oi = idx4(s, bi, ti, hi, vi, s.v);
                        out[oi] += state[idx_state(s, bi, hi, ki, vi)] * qq;
                    }
                }
            }
        }
    }
    out
}

/// Convenience: prepare and run in one call, returning `(out, final_state)`.
pub fn forward_prepared(
    s: &Shape,
    q_in: &[f32],
    k_in: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut q = q_in.to_vec();
    let mut k = k_in.to_vec();
    prepare_qk(&mut q, &mut k, s.b, s.t, s.h, s.k);
    let mut state = vec![0f32; s.b * s.h * s.k * s.v];
    let out = forward(s, &q, &k, v, g, beta, &mut state);
    (out, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape { b: 1, t: 3, h: 2, k: 4, v: 5 }
    }

    fn ramp(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37 + seed).sin()).collect()
    }

    /// With `T = 1` and a zero initial state the recurrence collapses to a
    /// closed form, which gives a target that is **independent of the reference
    /// implementation**:
    ///
    /// ```text
    /// state  = 0, so  state = 0 * g = 0
    /// kv_mem = 0
    /// delta  = (v - 0) * beta = v * beta
    /// state  = 0 + outer(k, v*beta)
    /// out    = sum_ki state[ki,vi] * q[ki] = (k . q) * beta * v[vi]
    /// ```
    ///
    /// This is a stronger check than "agrees with golden" because it cannot
    /// inherit a mistake from the reference.
    #[test]
    fn t1_matches_closed_form() {
        let s = Shape { b: 2, t: 1, h: 3, k: 6, v: 4 };
        let q_in = ramp(s.b * s.t * s.h * s.k, 0.11);
        let k_in = ramp(s.b * s.t * s.h * s.k, 1.7);
        let v = ramp(s.b * s.t * s.h * s.v, 2.3);
        let g = vec![-0.4f32; s.b * s.t * s.h];
        let beta = ramp(s.b * s.t * s.h, 3.1).iter().map(|x| x.abs()).collect::<Vec<_>>();

        let (out, state) = forward_prepared(&s, &q_in, &k_in, &v, &g, &beta);

        // Prepare q/k the same way to get the scaled/normalised values.
        let mut q = q_in.clone();
        let mut k = k_in.clone();
        prepare_qk(&mut q, &mut k, s.b, s.t, s.h, s.k);

        let mut worst_out = 0f32;
        let mut worst_state = 0f32;
        for bi in 0..s.b {
            for hi in 0..s.h {
                let bt = bi * s.h + hi;
                let b_t = beta[bt];
                // k . q over the head dim
                let mut kq = 0f32;
                for ki in 0..s.k {
                    kq += k[idx4(&s, bi, 0, hi, ki, s.k)] * q[idx4(&s, bi, 0, hi, ki, s.k)];
                }
                for vi in 0..s.v {
                    let vv = v[idx4(&s, bi, 0, hi, vi, s.v)];
                    let expect = kq * b_t * vv;
                    let got = out[idx4(&s, bi, 0, hi, vi, s.v)];
                    worst_out = worst_out.max((got - expect).abs());
                }
                for ki in 0..s.k {
                    let kk = k[idx4(&s, bi, 0, hi, ki, s.k)];
                    for vi in 0..s.v {
                        let vv = v[idx4(&s, bi, 0, hi, vi, s.v)];
                        let expect = kk * vv * b_t;
                        let got = state[idx_state(&s, bi, hi, ki, vi)];
                        worst_state = worst_state.max((got - expect).abs());
                    }
                }
            }
        }
        assert!(worst_out < 1e-6, "T=1 out vs closed form: {worst_out:e}");
        assert!(worst_state < 1e-6, "T=1 state vs closed form: {worst_state:e}");
    }

    #[test]
    fn shapes_and_sizes() {
        let s = shape();
        let q = ramp(s.b * s.t * s.h * s.k, 0.0);
        let k = ramp(s.b * s.t * s.h * s.k, 0.5);
        let v = ramp(s.b * s.t * s.h * s.v, 1.0);
        let g = vec![-0.2f32; s.b * s.t * s.h];
        let beta = vec![0.3f32; s.b * s.t * s.h];
        let (out, state) = forward_prepared(&s, &q, &k, &v, &g, &beta);
        assert_eq!(out.len(), s.b * s.t * s.h * s.v);
        assert_eq!(state.len(), s.b * s.h * s.k * s.v);
    }

    #[test]
    fn deterministic() {
        let s = shape();
        let q = ramp(s.b * s.t * s.h * s.k, 0.0);
        let k = ramp(s.b * s.t * s.h * s.k, 0.5);
        let v = ramp(s.b * s.t * s.h * s.v, 1.0);
        let g = vec![-0.2f32; s.b * s.t * s.h];
        let beta = vec![0.3f32; s.b * s.t * s.h];
        let (o1, s1) = forward_prepared(&s, &q, &k, &v, &g, &beta);
        let (o2, s2) = forward_prepared(&s, &q, &k, &v, &g, &beta);
        assert_eq!(o1, o2);
        assert_eq!(s1, s2);
    }

    /// A strongly negative `g` decays the state to nothing, so the output of
    /// every step after the first must lose its dependence on earlier steps.
    /// With initial state zero and `g -> -inf`, step `t` sees only its own
    /// contribution.
    #[test]
    fn strong_decay_forgets_history() {
        let s = Shape { b: 1, t: 3, h: 1, k: 3, v: 2 };
        let q = ramp(s.b * s.t * s.h * s.k, 0.9);
        let k = ramp(s.b * s.t * s.h * s.k, 0.2);
        let v = ramp(s.b * s.t * s.h * s.v, 1.1);
        let beta = vec![1.0f32; s.b * s.t * s.h];

        let g_decay = vec![-60.0f32; s.b * s.t * s.h];
        let (out, _) = forward_prepared(&s, &q, &k, &v, &g_decay, &beta);

        // With no history, each step reduces to (k_t . q_t) * beta * v_t.
        let mut qq = q.clone();
        let mut kk = k.clone();
        prepare_qk(&mut qq, &mut kk, s.b, s.t, s.h, s.k);
        let mut worst = 0f32;
        for ti in 0..s.t {
            let mut kq = 0f32;
            for ki in 0..s.k {
                kq += kk[idx4(&s, 0, ti, 0, ki, s.k)] * qq[idx4(&s, 0, ti, 0, ki, s.k)];
            }
            for vi in 0..s.v {
                let expect = kq * v[idx4(&s, 0, ti, 0, vi, s.v)];
                let got = out[idx4(&s, 0, ti, 0, vi, s.v)];
                worst = worst.max((got - expect).abs());
            }
        }
        assert!(worst < 1e-4, "strong-decay limit mismatch: {worst:e}");
    }

    /// `l2norm` must divide by the norm of the head vector, not of the whole
    /// tensor: with `K` elements each of magnitude `a`, the result is `1/sqrt(K)`
    /// per element regardless of `a`.
    #[test]
    fn l2norm_is_per_head_vector() {
        let (b, t, h, d) = (2, 3, 2, 8);
        let rows = b * t * h;
        for a in [0.5f32, 1.0, 7.0] {
            let mut x = vec![a; rows * d];
            l2norm_last_dim(&mut x, rows, d, 1e-6);
            let expect = 1.0f32 / (d as f32).sqrt();
            for v in &x {
                assert!((v - expect).abs() < 1e-6, "a={a}: got {v}, want {expect}");
            }
        }
    }
}
