//! The decoder layer: a token mixer (gated delta net, or attention) plus an MLP,
//! each wrapped in a residual connection.
//!
//! # Shape of the layer
//!
//! `Qwen3_5DecoderLayer.forward` is, in full:
//!
//! ```python
//! residual = hidden_states
//! hidden_states = self.input_layernorm(hidden_states)
//! hidden_states = self.linear_attn(..) or self.self_attn(..)
//! hidden_states = residual + hidden_states      # first residual
//!
//! residual = hidden_states                      # note: the *updated* value
//! hidden_states = self.post_attention_layernorm(hidden_states)
//! hidden_states = self.mlp(hidden_states)
//! hidden_states = residual + hidden_states      # second residual
//! ```
//!
//! Three details that are easy to get wrong and produce plausible-looking output:
//!
//! * The first residual adds the value from **before** `input_layernorm` — this is
//!   pre-norm, not post-norm.
//! * The second residual adds the value from **after** the first addition, not the
//!   original layer input.
//! * There is no scale, no dropout and no gate on either residual: plain `x + f(x)`.
//!
//! # The MLP
//!
//! `Qwen3_5MLP.forward` is a single SwiGLU expression with no biases:
//!
//! ```python
//! down_proj(act_fn(gate_proj(x)) * up_proj(x))
// ```
//!
//! with `hidden_act = "silu"`. The elementwise product `silu(gate) * up` is where
//! an implementation most plausibly goes wrong (wrong branch activated, or the two
//! swapped), which is why the golden bundle records it as its own tensor.

use crate::{linear, rmsnorm_1plus, GdnTrace, EPS};

/// MLP weights, in PyTorch's `[out, in]` storage order. No biases.
#[derive(Debug, Clone)]
pub struct MlpWeights {
    /// `[intermediate, hidden]`
    pub gate_proj: Vec<f32>,
    /// `[intermediate, hidden]`
    pub up_proj: Vec<f32>,
    /// `[hidden, intermediate]`
    pub down_proj: Vec<f32>,
}

/// Everything an MLP produces, so a caller can locate the first divergence.
#[derive(Debug, Clone)]
pub struct MlpTrace {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    /// `silu(gate_proj(x)) * up_proj(x)`
    pub swiglu_product: Vec<f32>,
    pub down_proj: Vec<f32>,
}

/// `silu`, matching `ACT2FN["silu"]`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
///
/// `x` is `[rows, hidden]`; the result is `[rows, hidden]`.
pub fn mlp_forward(w: &MlpWeights, x: &[f32], rows: usize, hidden: usize, intermediate: usize) -> MlpTrace {
    let gate = linear(&w.gate_proj, None, x, rows, hidden, intermediate);
    let up = linear(&w.up_proj, None, x, rows, hidden, intermediate);

    // The elementwise product. `silu` is applied to the gate branch only; applying
    // it to `up` instead, or to the product, are both silent errors.
    let mut product = vec![0f32; rows * intermediate];
    for i in 0..product.len() {
        product[i] = silu(gate[i]) * up[i];
    }

    let down = linear(&w.down_proj, None, &product, rows, intermediate, hidden);
    MlpTrace {
        gate_proj: gate,
        up_proj: up,
        swiglu_product: product,
        down_proj: down,
    }
}

/// Weights of a full decoder layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// `[hidden]`, applied as `x * (1 + w)`.
    pub input_layernorm: Vec<f32>,
    /// `[hidden]`, likewise.
    pub post_attention_layernorm: Vec<f32>,
    pub mlp: MlpWeights,
}

/// Trace of a full layer.
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub input_layernorm: Vec<f32>,
    /// Output of the token mixer (the gated delta net today).
    pub mixer: GdnTrace,
    /// `input + mixer_out_proj`
    pub after_first_residual: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub mlp: MlpTrace,
    /// `after_first_residual + mlp_out` — the layer's output.
    pub out: Vec<f32>,
}

/// Run the MLP half of a layer given the already-computed mixer output.
///
/// Kept separate from [`layer_forward`] so that the parts can be checked
/// independently: the mixer half is already validated on its own.
pub fn layer_forward_from_mixer(
    lw: &LayerWeights,
    mixer_out: &[f32],
    layer_input: &[f32],
    layernorm_out: &[f32],
    mixer_trace: GdnTrace,
    rows: usize,
) -> LayerTrace {
    let hidden = lw.input_layernorm.len();
    assert_eq!(layer_input.len(), rows * hidden, "layer input size");
    assert_eq!(mixer_out.len(), rows * hidden, "mixer output size");
    assert_eq!(layernorm_out.len(), rows * hidden, "layernorm output size");

    // First residual: the value from *before* input_layernorm.
    let after_first: Vec<f32> =
        (0..rows * hidden).map(|i| layer_input[i] + mixer_out[i]).collect();

    // post_attention_layernorm feeds the MLP.
    let post_ln = rmsnorm_1plus(
        &lw.post_attention_layernorm,
        &after_first,
        rows,
        hidden,
        EPS,
    );

    let intermediate = lw.mlp.gate_proj.len() / hidden;
    let mlp = mlp_forward(&lw.mlp, &post_ln, rows, hidden, intermediate);

    // Second residual: the value from *after* the first addition.
    let out: Vec<f32> = (0..rows * hidden).map(|i| after_first[i] + mlp.down_proj[i]).collect();

    LayerTrace {
        input_layernorm: layernorm_out.to_vec(),
        mixer: mixer_trace,
        after_first_residual: after_first,
        post_attention_layernorm: post_ln,
        mlp,
        out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_definition() {
        for x in [-8.0f32, -1.0, 0.0, 0.5, 3.0, 20.0] {
            let want = x * (1.0 / (1.0 + (-x).exp()));
            assert!((silu(x) - want).abs() < 1e-6, "x={x}");
        }
        assert!((silu(0.0)).abs() < 1e-7);
    }

    #[test]
    fn swiglu_applies_activation_to_the_gate_branch_only() {
        // hidden=1, intermediate=1: gate_proj = [2], up_proj = [3], down_proj = [1].
        // x = [1]  ->  gate = 2, up = 3, product = silu(2)*3, out = that.
        let w = MlpWeights {
            gate_proj: vec![2.0],
            up_proj: vec![3.0],
            down_proj: vec![1.0],
        };
        let t = mlp_forward(&w, &[1.0], 1, 1, 1);
        let want = silu(2.0) * 3.0;
        assert!((t.swiglu_product[0] - want).abs() < 1e-6, "{:?}", t.swiglu_product);
        assert!((t.down_proj[0] - want).abs() < 1e-6);
        // If the activation had been applied to `up` instead, the value would be
        // silu(3)*2, which differs enough to catch.
        let wrong = silu(3.0) * 2.0;
        assert!((t.swiglu_product[0] - wrong).abs() > 0.1, "looks like up was activated");
    }

    #[test]
    fn residuals_are_plain_adds_and_second_uses_updated_value() {
        // hidden=2, rows=1, intermediate=2; make the MLP an identity-ish map so the
        // expected arithmetic is easy to state.
        let lw = LayerWeights {
            input_layernorm: vec![0.0, 0.0],       // x*(1+0) -> scale 1
            post_attention_layernorm: vec![0.0, 0.0],
            mlp: MlpWeights {
                // produce constant zeros so `out == after_first`
                gate_proj: vec![0.0, 0.0, 0.0, 0.0],
                up_proj: vec![0.0, 0.0, 0.0, 0.0],
                down_proj: vec![0.0, 0.0, 0.0, 0.0],
            },
        };
        let layer_in = vec![1.0f32, 2.0];
        let mixer_out = vec![10.0f32, 20.0];
        let ln_out = vec![0.5f32, 0.5];
        // Build a minimal GdnTrace standing in for the mixer.
        let mixer = GdnTrace {
            input_layernorm: ln_out.clone(),
            in_proj_qkv: vec![],
            conv_in: vec![],
            conv_out: vec![],
            in_proj_z: vec![],
            in_proj_b: vec![],
            in_proj_a: vec![],
            q: vec![],
            k: vec![],
            v: vec![],
            g: vec![],
            beta: vec![],
            delta_out: vec![],
            delta_state: vec![],
            norm: vec![],
            out_proj: mixer_out.clone(),
        };
        let t = layer_forward_from_mixer(&lw, &mixer_out, &layer_in, &ln_out, mixer, 1);
        // first residual adds the pre-norm input
        assert_eq!(t.after_first_residual, vec![11.0, 22.0]);
        // second residual adds to the *updated* value, not the original
        assert_eq!(t.out, vec![11.0, 22.0]);
        assert_ne!(t.out, vec![1.0, 2.0]);
    }
}
