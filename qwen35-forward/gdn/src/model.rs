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

use crate::attention::{self, AttnConfig, AttnState, AttnWeights};
use crate::layer::{
    layer_forward, layer_forward_with_state, LayerTrace, LayerWeights, Mixer, MixerState,
};
use crate::{linear, rmsnorm_1plus, GdnConfig, GdnState, GdnWeights};

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

/// Everything the stack carries between chunks: one mixer state per layer, plus how many
/// positions have been processed.
///
/// The states in here could hardly be more different from each other -- see
/// [`crate::GdnState`] and [`AttnState`] -- and that is the design, not an accident:
/// linear-attention layers keep a fixed-size summary, full-attention layers keep
/// everything.
///
/// [`Cache::len`] is the part that makes decoding *correct* rather than merely fast. It is
/// the absolute position of the next token, and it is what the rotary tables must be built
/// from. Building them from 0 instead leaves every tensor shape intact and rotates every
/// decoded token as though it were the first one.
#[derive(Debug, Clone)]
pub struct Cache {
    /// One entry per layer, in order, each matching its layer's kind.
    pub states: Vec<MixerState>,
    /// Positions processed so far.
    pub len: usize,
}

impl Cache {
    /// A zeroed cache, which is what the first chunk of a sequence starts from.
    pub fn new(mcfg: &ModelConfig, w: &ModelWeights, b: usize) -> Result<Self, String> {
        let mut states = Vec::with_capacity(w.layers.len());
        for lw in w.layers.iter() {
            states.push(match lw.kind {
                LayerKind::LinearAttention => {
                    MixerState::LinearAttention(GdnState::new(&mcfg.gdn, b))
                }
                LayerKind::FullAttention => MixerState::FullAttention(AttnState::new()),
            });
        }
        Ok(Cache { states, len: 0 })
    }

    pub fn bytes(&self) -> usize {
        self.states.iter().map(|s| s.bytes()).sum()
    }

    /// Bytes split into `(linear, full)`, so the two growth behaviours can be reported
    /// side by side.
    pub fn bytes_by_kind(&self) -> (usize, usize) {
        let (mut lin, mut full) = (0usize, 0usize);
        for s in &self.states {
            match s {
                MixerState::LinearAttention(g) => lin += g.bytes(),
                MixerState::FullAttention(a) => full += a.bytes(),
            }
        }
        (lin, full)
    }
}

/// Run the model on a **new chunk** of tokens, continuing from `cache`.
///
/// The first call is prefill: pass the whole prompt and the cache grows to cover it. Every
/// later call passes a single token. The result equals what [`forward`] would give on the
/// concatenation, because the state is continued rather than restarted.
pub fn forward_cached(
    mcfg: &ModelConfig,
    w: &ModelWeights,
    cache: &mut Cache,
    tokens: &[u32],
) -> Result<ModelTrace, String> {
    let t = tokens.len();
    if t == 0 {
        return Err("empty chunk".to_string());
    }
    let b = 1usize;
    let rows = b * t;
    let start = cache.len;
    if cache.states.len() != w.layers.len() {
        return Err(format!(
            "cache holds {} layer states but the model has {} layers",
            cache.states.len(),
            w.layers.len()
        ));
    }

    let embedding = embed(&w.embed_tokens, tokens, mcfg.vocab, mcfg.hidden)?;

    // The tables start at the chunk's first absolute position, not at zero.
    let (cos, sin) = attention::build_rope(&mcfg.attn, t, start);

    let mut h = embedding.clone();
    let mut layer_traces = Vec::with_capacity(w.layers.len());
    for (i, lw) in w.layers.iter().enumerate() {
        let mixer = lw.mixer(mcfg).map_err(|e| format!("layer {i}: {e}"))?;
        let tr = layer_forward_with_state(
            mixer,
            &lw.layer,
            &h,
            b,
            t,
            mcfg.eps,
            Some((&cos, &sin)),
            &mut cache.states[i],
        )
        .map_err(|e| format!("layer {i}: {e}"))?;
        h = tr.out.clone();
        layer_traces.push(tr);
    }
    cache.len = start + t;

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

/// Greedy decoding with a cache: prefill once, then one token per step.
///
/// The generated ids must equal [`greedy`]'s. That is checked, not assumed, because a
/// cache that is subtly wrong still produces fluent text -- the failure mode is a
/// different continuation, not an error.
///
/// Each returned trace covers only the tokens of the call that produced the token, so
/// `traces[0]` has `t == prompt.len()` (the prefill) and every later one has `t == 1`.
/// `last_logits` is the last position either way, which is what greedy reads.
pub fn greedy_cached(
    mcfg: &ModelConfig,
    w: &ModelWeights,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<ModelTrace>), String> {
    if prompt.is_empty() {
        return Err("empty prompt".to_string());
    }
    let mut cache = Cache::new(mcfg, w, 1)?;
    // Prefill once.
    let mut tr = forward_cached(mcfg, w, &mut cache, prompt)?;
    let mut generated = Vec::with_capacity(steps);
    let mut traces = Vec::with_capacity(steps);
    for _ in 0..steps {
        let nxt = tr.argmax_last() as u32;
        generated.push(nxt);
        traces.push(tr);
        // Decode one token; the cache already holds everything before it.
        tr = forward_cached(mcfg, w, &mut cache, &[nxt])?;
    }
    Ok((generated, traces))
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

    /// Prefilling with a fresh cache must equal the uncached whole-sequence path, bit for
    /// bit. This is the strongest statement available: not "close enough", but the same
    /// arithmetic reached two ways.
    #[test]
    fn prefill_equals_the_uncached_path() {
        let (mcfg, w) = tiny();
        let prompt = [1u32, 2, 3, 4];
        let a = forward(&mcfg, &w, &prompt).unwrap();
        let mut cache = Cache::new(&mcfg, &w, 1).unwrap();
        let b = forward_cached(&mcfg, &w, &mut cache, &prompt).unwrap();
        assert_eq!(a.logits.len(), b.logits.len());
        for i in 0..a.logits.len() {
            assert_eq!(
                a.logits[i].to_bits(),
                b.logits[i].to_bits(),
                "logit {i}: uncached {} vs prefill {}",
                a.logits[i],
                b.logits[i]
            );
        }
        assert_eq!(cache.len, prompt.len());
    }

    /// Decoding one token at a time with a cache must equal processing the whole growing
    /// sequence at once. If the rotary tables were built from position 0 instead of the
    /// absolute position, this is the test that would fail.
    #[test]
    fn token_at_a_time_equals_the_whole_sequence() {
        let (mcfg, w) = tiny();
        // Ids must stay inside the fixture's vocabulary (6), which is also why this
        // asserts bit-level closeness rather than exactness: the two paths call `linear`
        // with different row counts.
        let seq = [1u32, 2, 3, 4, 5, 0];
        let whole = forward(&mcfg, &w, &seq).unwrap();

        let mut cache = Cache::new(&mcfg, &w, 1).unwrap();
        let split = 3usize;
        let _ = forward_cached(&mcfg, &w, &mut cache, &seq[..split]).unwrap();
        let mut last = None;
        for &tok in &seq[split..] {
            last = Some(forward_cached(&mcfg, &w, &mut cache, &[tok]).unwrap());
        }
        let step = last.unwrap();
        assert_eq!(cache.len, seq.len());
        // Bit-exact, not "close". Both paths evaluate the same recurrence in the same
        // order -- `linear` computes each row independently, the convolution and the
        // delta rule are continued rather than restarted, and `attend` sums over the
        // same keys in the same order. A tolerance here would hide a cache that drops
        // part of its contents, because the fixture's attention output is small in
        // absolute terms; an earlier version of this test used 1e-4 and missed exactly
        // that bug.
        let a = whole.last_logits();
        let b = step.last_logits();
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "logit {i}: whole {} vs incremental {}",
                a[i],
                b[i]
            );
        }
    }

    /// Cached greedy must generate the same tokens as uncached greedy.
    #[test]
    fn cached_greedy_matches_uncached_greedy() {
        let (mcfg, w) = tiny();
        let prompt = [1u32, 2, 3];
        let (want, want_tr) = greedy(&mcfg, &w, &prompt, 5).unwrap();
        let (got, got_tr) = greedy_cached(&mcfg, &w, &prompt, 5).unwrap();
        assert_eq!(got, want, "cached greedy diverged from the uncached path");
        assert_eq!(got_tr.len(), want_tr.len());
    }

    /// The cache must exist per layer and in the shape the model needs; a mismatch would
    /// otherwise be caught only by an index panic somewhere deep.
    #[test]
    fn cache_is_allocated_per_layer_and_the_two_kinds_differ() {
        let (mcfg, w) = tiny();
        let cache = Cache::new(&mcfg, &w, 1).unwrap();
        assert_eq!(cache.states.len(), w.layers.len());
        assert_eq!(cache.len, 0);
        let (lin, full) = cache.bytes_by_kind();
        // The fixture has one linear layer and one full layer, and at zero sequence length
        // the linear state dominates: it is fixed-size, while the KV part starts empty.
        assert!(lin > 0, "linear state should be allocated up front");
        assert_eq!(full, 0, "attention state should start empty, not preallocated");
        assert_eq!(cache.bytes(), lin + full);
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
