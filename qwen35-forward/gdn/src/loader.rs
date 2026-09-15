//! Reading a golden bundle: model-level shapes, per-layer weights, and the tensor
//! names a layer's intermediates live under.
//!
//! # Why the names need a module of their own
//!
//! The convolution and delta-rule captures are named by **position among the
//! linear-attention layers**, not by layer index:
//!
//! ```text
//! causal_conv1d_fn_call<K>_in / _out
//! delta_torch_chunk_gated_delta_rule_<K>__{q,k,v,g,beta,out,state}
//! ```
//!
//! `K` counts only the layers that run the delta rule. In the tiny model the
//! linear layers are 0, 1, 2, 4, 5, 6, so layer 4's convolution is `call3`, not
//! `call4`. A checker that indexes by layer number silently reads a *different
//! layer's* tensor from layer 4 onward: it was correct for layers 0-2 by
//! coincidence, which is exactly the kind of bug that survives a single-layer
//! test. [`LayerCapture`] is the one place the mapping lives.

use crate::attention::{AttnConfig, AttnWeights};
use crate::layer::{LayerWeights, MlpWeights};
use crate::model::{LayerKind, LayerWeightsAll, ModelWeights};
use crate::{GdnConfig, GdnWeights};
use goldenbundle::Bundle;

/// Shapes and settings shared by every layer, derived from the bundle.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub b: usize,
    pub t: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    /// `"linear_attention"` or `"full_attention"`, one entry per layer.
    pub layer_types: Vec<String>,
    pub eps: f32,
    pub ssm_gain: f64,
    pub vocab: usize,
    /// Config for the linear-attention mixer, shared by every such layer.
    pub gdn: GdnConfig,
    /// Config for the full-attention mixer, shared by every such layer.
    pub attn: AttnConfig,
}

impl ModelInfo {
    pub fn layer_type(&self, layer: usize) -> &str {
        self.layer_types
            .get(layer)
            .map(|s| s.as_str())
            .unwrap_or("unknown")
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.layer_type(layer) == "linear_attention"
    }

    /// Position of `layer` among the linear-attention layers — the `K` used in the
    /// convolution and delta-rule capture names. `None` for a full-attention layer.
    pub fn ssm_ordinal(&self, layer: usize) -> Option<usize> {
        if !self.is_linear(layer) {
            return None;
        }
        Some(
            (0..layer)
                .filter(|&l| self.is_linear(l))
                .count(),
        )
    }

    /// The tensor holding this layer's **input**. For layer 0 that is the embedding
    /// output; for every other layer it is the previous layer's output, which the
    /// bundle captures because decoder layers are hooked.
    pub fn input_name(&self, layer: usize) -> String {
        if layer == 0 {
            "model__embed_tokens".to_string()
        } else {
            format!("model__layers__{}", layer - 1)
        }
    }

    /// A chain from layer 0 to the last layer.
    ///
    /// Both mixer kinds are implemented, so the stack is one unbroken run. An
    /// earlier version split this at every `full_attention` layer, because those had
    /// no implementation and their output could not be computed; that is no longer
    /// true and the splitting is gone with it.
    pub fn chain_layers(&self) -> Vec<usize> {
        (0..self.num_layers).collect()
    }

    /// Collapse to the config the forward pass takes.
    pub fn model_config(&self) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            vocab: self.vocab,
            hidden: self.hidden,
            eps: self.eps,
            gdn: self.gdn,
            attn: self.attn,
        }
    }

    pub fn counts(&self) -> (usize, usize) {
        let lin = (0..self.num_layers).filter(|&l| self.is_linear(l)).count();
        (lin, self.num_layers - lin)
    }
}

/// Derive model-level shapes from the bundle.
pub fn load_model(b: &Bundle) -> Result<ModelInfo, String> {
    let cfg = &b.manifest.config;

    let emb = b
        .entry("model__embed_tokens")
        .ok_or("bundle has no model__embed_tokens; cannot determine B and T")?;
    if emb.shape.len() != 3 {
        return Err(format!("model__embed_tokens has shape {:?}, expected 3 axes", emb.shape));
    }
    let (b_sz, t_sz) = (emb.shape[0], emb.shape[1]);

    let hidden = b
        .entry("model__layers__0__input_layernorm__weight")
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .filter(|&h| h > 0)
        .ok_or("cannot determine hidden_size from layer 0's input_layernorm weight")?;

    let intermediate = b
        .entry("model__layers__0__mlp__gate_proj__weight")
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .filter(|&i| i > 0)
        .ok_or("cannot determine intermediate_size from layer 0's gate_proj weight")?;

    // Number of layers: the largest N for which a `model__layers__<N>` output
    // tensor exists.
    let mut num_layers = 0usize;
    for name in b.names() {
        if let Some(rest) = name.strip_prefix("model__layers__") {
            if !rest.contains("__") {
                if let Ok(n) = rest.parse::<usize>() {
                    num_layers = num_layers.max(n + 1);
                }
            }
        }
    }
    if num_layers == 0 {
        return Err("bundle captures no decoder-layer outputs (model__layers__<N>)".into());
    }

    // Layer types: prefer the config, fall back to which submodules exist.
    let from_config: Vec<String> = cfg
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let layer_types: Vec<String> = if from_config.len() >= num_layers {
        from_config[..num_layers].to_vec()
    } else {
        (0..num_layers)
            .map(|l| {
                if b.entry(&format!("model__layers__{l}__linear_attn")).is_some() {
                    "linear_attention".to_string()
                } else {
                    "full_attention".to_string()
                }
            })
            .collect()
    };

    let eps = cfg
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::EPS);

    // Linear-attention dims, from the first linear layer.
    let first = (0..num_layers)
        .find(|&l| layer_types.get(l).map(|s| s == "linear_attention").unwrap_or(false))
        .ok_or("no linear_attention layers in this bundle")?;
    let p = format!("model__layers__{first}__linear_attn__");

    let conv_dim = b
        .entry(&format!("{p}in_proj_qkv__weight"))
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .ok_or("missing in_proj_qkv weight")?;
    let value_dim = b
        .entry(&format!("{p}in_proj_z__weight"))
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .ok_or("missing in_proj_z weight")?;
    let num_v_heads = b
        .entry(&format!("{p}in_proj_b__weight"))
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .ok_or("missing in_proj_b weight")?;
    let head_v_dim = b
        .entry(&format!("{p}norm__weight"))
        .map(|e| e.shape.first().copied().unwrap_or(0))
        .ok_or("missing linear_attn norm weight")?;
    let conv_kernel = b
        .entry(&format!("{p}conv1d__weight"))
        .and_then(|e| e.shape.last().copied())
        .ok_or("missing conv1d weight")?;

    // Post-GQA q is [B, T, num_v_heads, head_k_dim]; repeat_interleave only
    // duplicates heads, so the trailing axis is still head_k_dim.
    let head_k_dim = b
        .entry("delta_torch_chunk_gated_delta_rule_0__q")
        .and_then(|e| e.shape.last().copied())
        .ok_or(
            "bundle has no delta-rule operand capture for the first linear layer; \
             regenerating with the current generator is required",
        )?;

    // conv_dim = 2*key_dim + value_dim, key_dim = num_k_heads * head_k_dim.
    let key_dim = conv_dim.saturating_sub(value_dim) / 2;
    let num_k_heads = key_dim.checked_div(head_k_dim).unwrap_or(0);

    if head_k_dim == 0
        || num_k_heads == 0
        || key_dim != num_k_heads * head_k_dim
        || conv_dim != 2 * key_dim + value_dim
        || value_dim != num_v_heads * head_v_dim
        || num_v_heads % num_k_heads != 0
    {
        return Err(format!(
            "derived sizes are inconsistent:\n  conv_dim={conv_dim} value_dim={value_dim}\n  \
             num_k_heads={num_k_heads} num_v_heads={num_v_heads}\n  \
             head_k_dim={head_k_dim} head_v_dim={head_v_dim}"
        ));
    }

    let gdn = GdnConfig {
        hidden,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel,
        eps,
    };

    // ---- full attention --------------------------------------------------
    let vocab = cfg
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            b.entry("model__embed_tokens__weight")
                .and_then(|e| e.shape.first().copied())
        })
        .ok_or("cannot determine vocab_size")?;

    let num_attention_heads = cfg
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or("config has no num_attention_heads")?;
    let num_key_value_heads = cfg
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or("config has no num_key_value_heads")?;
    let head_dim = cfg
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or("config has no head_dim")?;
    let rope_params = cfg.get("rope_parameters");
    let rope_theta = rope_params
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;
    let partial = rope_params
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    // `rotary_dim` is measured from the captured cos/sin rather than derived from
    // the config, so a config that disagrees with the run cannot silently pass.
    let rotary_dim = b
        .entry("model__rotary_emb__out0")
        .and_then(|e| e.shape.last().copied())
        .unwrap_or((head_dim as f32 * partial) as usize);
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary_dim {rotary_dim} is not a positive even number <= head_dim {head_dim}"
        ));
    }
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(format!(
            "num_attention_heads {num_attention_heads} is not a multiple of num_key_value_heads {num_key_value_heads}"
        ));
    }

    let attn = AttnConfig {
        hidden,
        num_heads: num_attention_heads,
        num_kv_heads: num_key_value_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        eps,
    };

    Ok(ModelInfo {
        b: b_sz,
        t: t_sz,
        hidden,
        intermediate,
        num_layers,
        layer_types,
        eps,
        ssm_gain: b.manifest.source.ssm_gain,
        vocab,
        gdn,
        attn,
    })
}

/// The bundle tensor names for one layer's intermediates.
///
/// Every accessor returns an owned `String`; the ones that only exist for
/// linear-attention layers return `None` for a full-attention layer rather than
/// producing a name that happens to be absent.
#[derive(Debug, Clone)]
pub struct LayerCapture {
    pub layer: usize,
    /// Position among linear-attention layers — the capture index. `None` for a
    /// full-attention layer.
    pub ssm: Option<usize>,
    prefix: String,
    attn: String,
}

impl LayerCapture {
    pub fn new(layer: usize, ssm: Option<usize>) -> Self {
        let prefix = format!("model__layers__{layer}__");
        let attn = format!("{prefix}linear_attn__");
        Self { layer, ssm, prefix, attn }
    }

    fn conv(&self, part: &str) -> Option<String> {
        self.ssm.map(|k| format!("causal_conv1d_fn_call{k}_{part}"))
    }

    fn delta(&self, part: &str) -> Option<String> {
        self.ssm
            .map(|k| format!("delta_torch_chunk_gated_delta_rule_{k}__{part}"))
    }

    // ---- intermediates -----------------------------------------------------
    pub fn layer_out(&self) -> String {
        format!("model__layers__{}", self.layer)
    }
    pub fn input_layernorm(&self) -> String {
        format!("{}input_layernorm", self.prefix)
    }
    pub fn in_proj_qkv(&self) -> String {
        format!("{}in_proj_qkv", self.attn)
    }
    pub fn in_proj_z(&self) -> String {
        format!("{}in_proj_z", self.attn)
    }
    pub fn in_proj_b(&self) -> String {
        format!("{}in_proj_b", self.attn)
    }
    pub fn in_proj_a(&self) -> String {
        format!("{}in_proj_a", self.attn)
    }
    pub fn conv_in(&self) -> Option<String> {
        self.conv("in")
    }
    pub fn conv_out(&self) -> Option<String> {
        self.conv("out")
    }
    pub fn delta_operand(&self, operand: &str) -> Option<String> {
        self.delta(operand)
    }
    pub fn delta_out(&self) -> Option<String> {
        self.delta("out")
    }
    pub fn delta_state(&self) -> Option<String> {
        self.delta("state")
    }
    pub fn mixer_norm(&self) -> String {
        format!("{}norm", self.attn)
    }
    pub fn out_proj(&self) -> String {
        format!("{}out_proj", self.attn)
    }
    pub fn linear_attn(&self) -> String {
        format!("{}linear_attn", self.prefix)
    }
    pub fn post_attention_layernorm(&self) -> String {
        format!("{}post_attention_layernorm", self.prefix)
    }
    pub fn mlp_gate_proj(&self) -> String {
        format!("{}mlp__gate_proj", self.prefix)
    }
    pub fn mlp_up_proj(&self) -> String {
        format!("{}mlp__up_proj", self.prefix)
    }
    pub fn mlp_swiglu(&self) -> String {
        format!("{}mlp__swiglu_product", self.prefix)
    }
    pub fn mlp_down_proj(&self) -> String {
        format!("{}mlp__down_proj", self.prefix)
    }
    pub fn mlp(&self) -> String {
        format!("{}mlp", self.prefix)
    }

    // ---- weights -----------------------------------------------------------
    pub fn w_input_layernorm(&self) -> String {
        format!("{}input_layernorm__weight", self.prefix)
    }
    pub fn w_post_attention_layernorm(&self) -> String {
        format!("{}post_attention_layernorm__weight", self.prefix)
    }
    pub fn w_in_proj_qkv(&self) -> String {
        format!("{}in_proj_qkv__weight", self.attn)
    }
    pub fn w_in_proj_z(&self) -> String {
        format!("{}in_proj_z__weight", self.attn)
    }
    pub fn w_in_proj_b(&self) -> String {
        format!("{}in_proj_b__weight", self.attn)
    }
    pub fn w_in_proj_a(&self) -> String {
        format!("{}in_proj_a__weight", self.attn)
    }
    pub fn w_conv1d(&self) -> String {
        format!("{}conv1d__weight", self.attn)
    }
    pub fn w_a_log(&self) -> String {
        format!("{}A_log", self.attn)
    }
    pub fn w_dt_bias(&self) -> String {
        format!("{}dt_bias", self.attn)
    }
    pub fn w_norm(&self) -> String {
        format!("{}norm__weight", self.attn)
    }
    pub fn w_out_proj(&self) -> String {
        format!("{}out_proj__weight", self.attn)
    }
    pub fn w_gate_proj(&self) -> String {
        format!("{}mlp__gate_proj__weight", self.prefix)
    }
    pub fn w_up_proj(&self) -> String {
        format!("{}mlp__up_proj__weight", self.prefix)
    }
    pub fn w_down_proj(&self) -> String {
        format!("{}mlp__down_proj__weight", self.prefix)
    }

    fn sa(&self, part: &str) -> String {
        format!("{}self_attn__{part}", self.prefix)
    }
    pub fn w_q_proj(&self) -> String {
        self.sa("q_proj__weight")
    }
    pub fn w_k_proj(&self) -> String {
        self.sa("k_proj__weight")
    }
    pub fn w_v_proj(&self) -> String {
        self.sa("v_proj__weight")
    }
    pub fn w_o_proj(&self) -> String {
        self.sa("o_proj__weight")
    }
    pub fn w_q_norm(&self) -> String {
        self.sa("q_norm__weight")
    }
    pub fn w_k_norm(&self) -> String {
        self.sa("k_norm__weight")
    }
    /// `Qwen3_5Attention` returns `(attn_output, attn_weights)` where `attn_output`
    /// has already been through `o_proj`, so this is the module's *output*, not the
    /// post-gate tensor.
    pub fn self_attn_out0(&self) -> String {
        self.sa("out0")
    }
    /// The post-gate, pre-`o_proj` tensor, recovered from `o_proj`'s input.
    pub fn o_proj_in(&self) -> String {
        self.sa("o_proj__in")
    }
    pub fn q_proj(&self) -> String {
        self.sa("q_proj")
    }
    pub fn q_norm(&self) -> String {
        self.sa("q_norm")
    }
    pub fn k_proj(&self) -> String {
        self.sa("k_proj")
    }
    pub fn k_norm(&self) -> String {
        self.sa("k_norm")
    }
    pub fn v_proj(&self) -> String {
        self.sa("v_proj")
    }
    pub fn o_proj(&self) -> String {
        self.sa("o_proj")
    }
    pub fn self_attn(&self) -> String {
        format!("{}self_attn", self.prefix)
    }
}

/// Read a tensor and require an exact length.
pub fn read_exact(b: &Bundle, name: &str, expect: usize) -> Result<Vec<f32>, String> {
    let v = b.read(name).map_err(|e| format!("{name}: {e}"))?;
    if v.len() != expect {
        return Err(format!("{name}: got {} values, expected {expect}", v.len()));
    }
    Ok(v)
}

/// Load the linear-attention mixer weights for one layer.
pub fn load_gdn_weights(b: &Bundle, m: &ModelInfo, layer: usize) -> Result<GdnWeights, String> {
    let c = LayerCapture::new(layer, m.ssm_ordinal(layer));
    if c.ssm.is_none() {
        return Err(format!("layer {layer} is {}; it has no gated delta net", m.layer_type(layer)));
    }
    let (h, vd, nh) = (
        m.gdn.hidden,
        m.gdn.value_dim(),
        m.gdn.num_v_heads,
    );
    Ok(GdnWeights {
        in_proj_qkv: read_exact(b, &c.w_in_proj_qkv(), m.gdn.conv_dim() * h)?,
        in_proj_z: read_exact(b, &c.w_in_proj_z(), vd * h)?,
        in_proj_b: read_exact(b, &c.w_in_proj_b(), nh * h)?,
        in_proj_a: read_exact(b, &c.w_in_proj_a(), nh * h)?,
        conv1d: read_exact(b, &c.w_conv1d(), m.gdn.conv_dim() * m.gdn.conv_kernel)?,
        a_log: read_exact(b, &c.w_a_log(), nh)?,
        dt_bias: read_exact(b, &c.w_dt_bias(), nh)?,
        norm: read_exact(b, &c.w_norm(), m.gdn.head_v_dim)?,
        out_proj: read_exact(b, &c.w_out_proj(), h * vd)?,
    })
}

/// Load the parts of a layer that are not the mixer.
pub fn load_layer_weights(b: &Bundle, m: &ModelInfo, layer: usize) -> Result<LayerWeights, String> {
    let c = LayerCapture::new(layer, m.ssm_ordinal(layer));
    let (h, i) = (m.hidden, m.intermediate);
    Ok(LayerWeights {
        input_layernorm: read_exact(b, &c.w_input_layernorm(), h)?,
        post_attention_layernorm: read_exact(b, &c.w_post_attention_layernorm(), h)?,
        mlp: MlpWeights {
            gate_proj: read_exact(b, &c.w_gate_proj(), i * h)?,
            up_proj: read_exact(b, &c.w_up_proj(), i * h)?,
            down_proj: read_exact(b, &c.w_down_proj(), h * i)?,
        },
    })
}

/// The layer kind, read from the same source as `ModelInfo::layer_types`.
pub fn layer_kind(m: &ModelInfo, layer: usize) -> LayerKind {
    if m.is_linear(layer) {
        LayerKind::LinearAttention
    } else {
        LayerKind::FullAttention
    }
}

/// Load the full-attention mixer weights for one layer.
pub fn load_attn_weights(b: &Bundle, m: &ModelInfo, layer: usize) -> Result<AttnWeights, String> {
    let c = LayerCapture::new(layer, m.ssm_ordinal(layer));
    let a = &m.attn;
    Ok(AttnWeights {
        q_proj: read_exact(b, &c.w_q_proj(), a.q_out_dim() * a.hidden)?,
        k_proj: read_exact(b, &c.w_k_proj(), a.kv_out_dim() * a.hidden)?,
        v_proj: read_exact(b, &c.w_v_proj(), a.kv_out_dim() * a.hidden)?,
        o_proj: read_exact(b, &c.w_o_proj(), a.hidden * a.num_heads * a.head_dim)?,
        q_norm: read_exact(b, &c.w_q_norm(), a.head_dim)?,
        k_norm: read_exact(b, &c.w_k_norm(), a.head_dim)?,
    })
}

/// Load every layer, whichever kind it is.
pub fn load_all_layers(b: &Bundle, m: &ModelInfo) -> Result<Vec<LayerWeightsAll>, String> {
    let mut out = Vec::with_capacity(m.num_layers);
    for layer in 0..m.num_layers {
        let kind = layer_kind(m, layer);
        let lw = load_layer_weights(b, m, layer)?;
        let (gdn, attn) = match kind {
            LayerKind::LinearAttention => (Some(load_gdn_weights(b, m, layer)?), None),
            LayerKind::FullAttention => (None, Some(load_attn_weights(b, m, layer)?)),
        };
        out.push(LayerWeightsAll { kind, layer: lw, gdn, attn });
    }
    Ok(out)
}

/// Model-level weights: the embedding table, the final norm, and the head.
pub fn load_model_weights(b: &Bundle, m: &ModelInfo) -> Result<ModelWeights, String> {
    Ok(ModelWeights {
        embed_tokens: read_exact(
            b,
            "model__embed_tokens__weight",
            m.vocab * m.hidden,
        )?,
        final_norm: read_exact(b, "model__norm__weight", m.hidden)?,
        lm_head: read_exact(b, "lm_head__weight", m.vocab * m.hidden)?,
        // Layers are attached by the caller, which already needs the list.
        layers: Vec::new(),
    })
}

/// The captured RoPE tables, `(cos, sin)`, each flattened `[T, rotary_dim]`.
///
/// Read rather than rebuilt so the check can compare against the reference's own
/// tables; `attention::build_rope` is separately verified against these.
pub fn load_rope(b: &Bundle, m: &ModelInfo) -> Result<(Vec<f32>, Vec<f32>), String> {
    let want = m.b * m.t * m.attn.rotary_dim;
    let cos = read_exact(b, "model__rotary_emb__out0", want)?;
    let sin = read_exact(b, "model__rotary_emb__out1", want)?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(types: &[&str]) -> ModelInfo {
        ModelInfo {
            b: 1,
            t: 1,
            hidden: 4,
            intermediate: 8,
            num_layers: types.len(),
            layer_types: types.iter().map(|s| s.to_string()).collect(),
            eps: 1e-6,
            ssm_gain: 1.0,
            vocab: 16,
            attn: AttnConfig {
                hidden: 4,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                rotary_dim: 2,
                rope_theta: 10000.0,
                eps: 1e-6,
            },
            gdn: GdnConfig {
                hidden: 4,
                num_k_heads: 1,
                num_v_heads: 1,
                head_k_dim: 2,
                head_v_dim: 2,
                conv_kernel: 2,
                eps: 1e-6,
            },
        }
    }

    /// The bug this module exists to prevent: `K` counts linear layers, not layers.
    #[test]
    fn ssm_ordinal_skips_full_attention_layers() {
        let m = info(&[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]);
        assert_eq!(m.ssm_ordinal(0), Some(0));
        assert_eq!(m.ssm_ordinal(1), Some(1));
        assert_eq!(m.ssm_ordinal(2), Some(2));
        assert_eq!(m.ssm_ordinal(3), None);
        // Layer 4 is the fourth linear layer, so its capture index is 3, not 4.
        assert_eq!(m.ssm_ordinal(4), Some(3));
        assert_eq!(m.ssm_ordinal(5), Some(4));
        assert_eq!(m.ssm_ordinal(6), Some(5));
        assert_eq!(m.ssm_ordinal(7), None);
    }

    #[test]
    fn capture_names_use_the_ordinal_not_the_layer() {
        let c = LayerCapture::new(4, Some(3));
        assert_eq!(c.conv_in().unwrap(), "causal_conv1d_fn_call3_in");
        assert_eq!(c.delta_out().unwrap(), "delta_torch_chunk_gated_delta_rule_3__out");
        assert_eq!(c.delta_state().unwrap(), "delta_torch_chunk_gated_delta_rule_3__state");
        // But the per-module names do use the layer index.
        assert_eq!(c.in_proj_qkv(), "model__layers__4__linear_attn__in_proj_qkv");
        assert_eq!(c.layer_out(), "model__layers__4");
    }

    #[test]
    fn full_attention_layer_has_no_delta_names() {
        let c = LayerCapture::new(3, None);
        assert_eq!(c.conv_in(), None);
        assert_eq!(c.delta_out(), None);
        // The shared parts still have names.
        assert_eq!(c.layer_out(), "model__layers__3");
        assert_eq!(c.mlp_swiglu(), "model__layers__3__mlp__swiglu_product");
    }

    #[test]
    fn chain_covers_every_layer_once_both_mixers_exist() {
        let m = info(&[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]);
        assert_eq!(m.chain_layers(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(m.counts(), (6, 2));
    }

    #[test]
    fn input_name_chains_from_the_previous_layer() {
        let m = info(&["linear_attention", "linear_attention"]);
        assert_eq!(m.input_name(0), "model__embed_tokens");
        assert_eq!(m.input_name(1), "model__layers__0");
    }
}
