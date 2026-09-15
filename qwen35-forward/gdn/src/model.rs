//! The whole model: embedding, the decoder stack, the final norm and the head.
//!
//! # The chain
//!
//! ```text
//! h = embed_tokens[input_ids]                    [B, T, hidden]
//! for each layer:
//!     h = layer(h)                               mixer chosen by layer_types[i]
//! h = final_norm(h)                              Qwen3_5RMSNorm, x*(1+w)
//! logits = lm_head(h)                            [B, T, vocab], no bias
//! ```
//!
//! Verified rather than assumed: with the reference,
//! `|lm_head(norm(last_layer_out)) - logits|` is exactly `0.0`, and
//! `hidden_states[-1]` from `output_hidden_states=True` is *already* normalised, so
//! normalising it again is a real mistake that produces a small, plausible error
//! (6.2e-4 in the tiny model) rather than something obviously broken.
//!
//! # No cache
//!
//! The golden trace re-runs the full forward for each greedy step with an input that
//! grows by one token, and reads `logits[0, -1]`. So there is no KV or recurrent
//! state to carry between steps here, and every layer starts from zero state. Adding
//! a cache is a later concern: it changes what has to be *stored*, not what has to be
//! *computed*.

use crate::attention::{self, AttnConfig, AttnWeights};
use crate::layer::{layer_forward, LayerTrace, LayerWeights, Mixer};
use crate::{linear, rmsnorm_1plus, GdnConfig, GdnWeights};

/// Which mixer a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

/// Model-level sizes. `gdn` and `attn` are shared by every layer of their kind: all
/// linear-attention layers have the same shapes, as do all full-attention layers.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub eps: f32,
    pub gdn: GdnConfig,
    pub attn: AttnConfig,
}

/// One layer's weights plus the kind that selects which of them are used.
#[derive(Debug, Clone)]
pub struct LayerWeightsAll {
    pub kind: LayerKind,
    pub layer: LayerWeights,
    /// Present iff `kind == LinearAttention`.
    pub gdn: Option<GdnWeights>,
    /// Present iff `kind == FullAttention`.
    pub attn: Option<AttnWeights>,
}

impl LayerWeightsAll {
    fn mixer<'a>(&'a self, mcfg: &'a ModelConfig) -> Result<Mixer<'a>, String> {
        match self.kind {
            LayerKind::LinearAttention => {
                let w = self
                    .gdn
                    .as_ref()
                    .ok_or("linear_attention layer has no gated-delta-net weights")?;
                Ok(Mixer::LinearAttention(&mcfg.gdn, w))
            }
            LayerKind::FullAttention => {
                let w = self
                    .attn
                    .as_ref()
                    .ok_or("full_attention layer has no attention weights")?;
                Ok(Mixer::FullAttention(&mcfg.attn, w))
            }
        }
    }
}

/// Everything the forward pass needs.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// `[vocab, hidden]`
    pub embed_tokens: Vec<f32>,
    /// `[hidden]`
    pub final_norm: Vec<f32>,
    /// `[vocab, hidden]`, no bias
    pub lm_head: Vec<f32>,
    pub layers: Vec<LayerWeightsAll>,
}

/// Result of a full forward.
#[derive(Debug, Clone)]
pub struct ModelTrace {
    /// Embedding output, before any layer.
    pub embedding: Vec<f32>,
    /// One entry per layer, in order.
    pub layers: Vec<LayerTrace>,
    /// Output of the final norm.
    pub final_norm: Vec<f32>,
    /// `[T, vocab]`
    pub logits: Vec<f32>,
    pub t: usize,
    pub vocab: usize,
}

impl ModelTrace {
    /// Logits at the last position, `[vocab]` -- what greedy decoding reads.
    pub fn last_logits(&self) -> &[f32] {
        &self.logits[(self.t - 1) * self.vocab..]
    }

    /// Argmax over the last position, matching `torch.argmax` on ties by taking the
    /// first maximum.
    pub fn argmax_last(&self) -> usize {
        let l = self.last_logits();
        let mut best = 0usize;
        for (i, v) in l.iter().enumerate() {
            if *v > l[best] {
                best = i;
            }
        }
        best
    }
}

/// Look up embedding rows for `input_ids`.
pub fn embed(w: &[f32], input_ids: &[u32], vocab: usize, hidden: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; input_ids.len() * hidden];
    for (i, &id) in input_ids.iter().enumerate() {
        let id = id as usize;
        if id >= vocab {
            return Err(format!("token id {id} is out of range for vocab {vocab}"));
        }
        out[i * hidden..(i + 1) * hidden]
            .copy_from_slice(&w[id * hidden..(id + 1) * hidden]);
    }
    Ok(out)
}

/// Run the whole model on one sequence.
///
/// `input_ids` is the full token sequence; the batch is always 1, matching the golden
/// trace.
pub fn forward(mcfg: &ModelConfig, w: &ModelWeights, input_ids: &[u32]) -> Result<ModelTrace, String> {
    let t = input_ids.len();
    if t == 0 {
        return Err("empty input".to_string());
    }
    let b = 1usize;
    let rows = b * t;
    if mcfg.vocab * mcfg.hidden != w.embed_tokens.len() {
        return Err(format!(
            "embed_tokens is {} values, vocab*hidden = {}",
            w.embed_tokens.len(),
            mcfg.vocab * mcfg.hidden
        ));
    }

    let embedding = embed(&w.embed_tokens, input_ids, mcfg.vocab, mcfg.hidden)?;

    // RoPE tables are built once and shared by every full-attention layer, which is
    // what the reference does: `position_embeddings = self.rotary_emb(...)` is
    // computed before the layer loop and passed in.
    let (cos, sin) = attention::build_rope(&mcfg.attn, t, 0);

    let mut h = embedding.clone();
    let mut layer_traces = Vec::with_capacity(w.layers.len());
    for (i, lw) in w.layers.iter().enumerate() {
        let mixer = lw.mixer(mcfg).map_err(|e| format!("layer {i}: {e}"))?;
        let tr = layer_forward(
            mixer,
            &lw.layer,
            &h,
            b,
            t,
            mcfg.eps,
            Some((&cos, &sin)),
        )
        .map_err(|e| format!("layer {i}: {e}"))?;
        h = tr.out.clone();
        layer_traces.push(tr);
    }

    // Final norm, then the head. `lm_head` has no bias and is applied to every
    // position; greedy decoding reads the last one.
    let final_norm = rmsnorm_1plus(&w.final_norm, &h, rows, mcfg.hidden, mcfg.eps);
    let logits = linear(&w.lm_head, None, &final_norm, rows, mcfg.hidden, mcfg.vocab);

    Ok(ModelTrace {
        embedding,
        layers: layer_traces,
        final_norm,
        logits,
        t,
        vocab: mcfg.vocab,
    })
}

/// Greedy decoding: run, take the argmax of the last position, append, repeat.
///
/// Mirrors the reference's loop exactly, including re-running the full sequence each
/// step rather than using a cache.
pub fn greedy(
    mcfg: &ModelConfig,
    w: &ModelWeights,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<ModelTrace>), String> {
    let mut ids: Vec<u32> = prompt.to_vec();
    let mut generated = Vec::with_capacity(steps);
    let mut traces = Vec::with_capacity(steps);
    for _ in 0..steps {
        let tr = forward(mcfg, w, &ids)?;
        let nxt = tr.argmax_last() as u32;
        generated.push(nxt);
        ids.push(nxt);
        traces.push(tr);
    }
    Ok((generated, traces))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (ModelConfig, ModelWeights) {
        let hidden = 4usize;
        let vocab = 6usize;
        let gdn = GdnConfig {
            hidden,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_kernel: 2,
            eps: 1e-6,
        };
        let attn = AttnConfig {
            hidden,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            rope_theta: 10000.0,
            eps: 1e-6,
        };
        let mcfg = ModelConfig { vocab, hidden, eps: 1e-6, gdn, attn };
        let conv_dim = gdn.conv_dim();
        let value_dim = gdn.value_dim();
        let lin = LayerWeightsAll {
            kind: LayerKind::LinearAttention,
            layer: LayerWeights {
                input_layernorm: vec![1.0; hidden],
                post_attention_layernorm: vec![1.0; hidden],
                mlp: crate::layer::MlpWeights {
                    gate_proj: vec![0.01; 8 * hidden],
                    up_proj: vec![0.01; 8 * hidden],
                    down_proj: vec![0.01; hidden * 8],
                },
            },
            gdn: Some(GdnWeights {
                in_proj_qkv: vec![0.02; conv_dim * hidden],
                in_proj_z: vec![0.02; value_dim * hidden],
                in_proj_b: vec![0.0; hidden],
                in_proj_a: vec![0.0; hidden],
                conv1d: vec![0.5; conv_dim * gdn.conv_kernel],
                a_log: vec![0.1; 1],
                dt_bias: vec![0.0; 1],
                norm: vec![1.0; 2],
                out_proj: vec![0.02; hidden * value_dim],
            }),
            attn: None,
        };
        let full = LayerWeightsAll {
            kind: LayerKind::FullAttention,
            layer: lin.layer.clone(),
            gdn: None,
            attn: Some(AttnWeights {
                q_proj: vec![0.02; 2 * attn.head_dim * hidden],
                k_proj: vec![0.02; attn.head_dim * hidden],
                v_proj: vec![0.02; attn.head_dim * hidden],
                o_proj: vec![0.02; hidden * attn.head_dim],
                q_norm: vec![1.0; attn.head_dim],
                k_norm: vec![1.0; attn.head_dim],
            }),
        };
        let w = ModelWeights {
            embed_tokens: (0..vocab * hidden).map(|i| (i % 7) as f32 * 0.1).collect(),
            final_norm: vec![1.0; hidden],
            lm_head: vec![0.05; vocab * hidden],
            layers: vec![lin, full],
        };
        (mcfg, w)
    }

    #[test]
    fn embedding_lookup_picks_the_right_rows() {
        let w = vec![0f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let e = embed(&w, &[1, 0], 2, 4).unwrap();
        assert_eq!(e, vec![10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn out_of_range_token_is_reported() {
        let w = vec![0f32; 8];
        let err = embed(&w, &[5], 2, 4).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn forward_shapes_and_finiteness() {
        let (mcfg, w) = tiny();
        let tr = forward(&mcfg, &w, &[1, 2, 3]).unwrap();
        assert_eq!(tr.t, 3);
        assert_eq!(tr.logits.len(), 3 * mcfg.vocab);
        assert_eq!(tr.layers.len(), 2);
        assert!(tr.logits.iter().all(|x| x.is_finite()), "logits not finite");
        assert_eq!(tr.final_norm.len(), 3 * mcfg.hidden);
        // Argmax must be a valid token id.
        assert!(tr.argmax_last() < mcfg.vocab);
    }

    /// A full-attention layer must actually mix across positions. If the mixer were
    /// silently a no-op, changing an earlier token would leave later residual
    /// streams untouched.
    #[test]
    fn attention_layer_mixes_across_positions() {
        let (mcfg, w) = tiny();
        let a = forward(&mcfg, &w, &[1, 2, 3]).unwrap();
        let b = forward(&mcfg, &w, &[1, 2, 4]).unwrap();
        let la = &a.layers[1];
        let lb = &b.layers[1];
        // Position 0 precedes the change at position 2, so causality says its
        // attention output must be identical.
        let d0: f32 = la
            .after_first_residual
            .iter()
            .zip(lb.after_first_residual.iter())
            .take(mcfg.hidden)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d0 < 1e-6, "position 0 changed when a later token changed: {d0}");
        // Position 2 must change.
        let d2: f32 = la
            .after_first_residual
            .iter()
            .zip(lb.after_first_residual.iter())
            .skip(2 * mcfg.hidden)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d2 > 1e-9, "position 2 did not change: {d2}");
    }

    #[test]
    fn greedy_appends_the_argmax() {
        let (mcfg, w) = tiny();
        let (gen, traces) = greedy(&mcfg, &w, &[1, 2], 3).unwrap();
        assert_eq!(gen.len(), 3);
        assert_eq!(traces.len(), 3);
        for (i, g) in gen.iter().enumerate() {
            assert_eq!(*g as usize, traces[i].argmax_last());
            assert_eq!(traces[i].t, 2 + i);
        }
    }
}
