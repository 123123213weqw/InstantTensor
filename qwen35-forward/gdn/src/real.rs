//! Assembling a real Qwen3.5 checkpoint into the structures the forward pass wants.
//!
//! # What a real checkpoint looks like, and how it differs from the golden bundle
//!
//! The golden bundles are a *test artifact*: a tiny synthetic model whose tensor
//! names are the Python module path with `.` rewritten to `__`. A real checkpoint
//! has different names, a different dtype and a different layout:
//!
//! | | golden bundle | real checkpoint |
//! |---|---|---|
//! | names | `model__layers__0__linear_attn__in_proj_qkv__weight` | `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` |
//! | dtype | `f32` | `bf16` |
//! | layout | one file per tensor | one flat shard, 488 tensors |
//! | extras | none | a vision tower and an MTP head |
//! | head | a separate `lm_head` | tied to `embed_tokens` |
//!
//! So this module maps *structure*, not strings, and it verifies that mapping
//! against the shapes it actually finds.
//!
//! # Sizes come from the tensors, not the config
//!
//! Every dimension is read off a tensor's shape, then **cross-checked** against the
//! config. A config that disagrees is an error rather than something to average
//! over: the shapes are what the forward pass will actually index with, so a config
//! that says `head_dim = 256` while `q_proj` implies 128 must not be papered over.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::attention::{AttnConfig, AttnWeights};
use crate::layer::{LayerWeights, MlpWeights};
use crate::model::{LayerKind, LayerWeightsAll, ModelConfig, ModelWeights};
use crate::safetensors::Model;
use crate::{GdnConfig, GdnWeights};

/// Where the language model's tensors live inside the checkpoint.
///
/// A `Qwen3_5ForConditionalGeneration` checkpoint nests the text model under
/// `model.language_model.` and puts the vision tower in `model.visual.`. A
/// text-only checkpoint would use `model.`. This is detected rather than assumed,
/// by locating the tensor that must exist either way.
pub fn detect_prefix(m: &Model) -> Result<String, String> {
    let mut found: Vec<String> = m
        .names()
        .filter(|n| n.ends_with("embed_tokens.weight"))
        .cloned()
        .collect();
    found.sort();
    match found.len() {
        0 => Err("no tensor ending in `embed_tokens.weight`".to_string()),
        1 => {
            let n = &found[0];
            Ok(n[..n.len() - "embed_tokens.weight".len()].to_string())
        }
        _ => Err(format!(
            "ambiguous: {} tensors end in `embed_tokens.weight` ({found:?})",
            found.len()
        )),
    }
}

/// The sub-trees a checkpoint may carry that inference does not use.
fn is_unused(name: &str) -> Option<&'static str> {
    if name.starts_with("model.visual.") || name.starts_with("visual.") {
        Some("vision tower")
    } else if name.starts_with("mtp.") {
        Some("multi-token-prediction head")
    } else {
        None
    }
}

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> Option<usize> {
    cfg.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// Pick the object holding the language-model settings.
///
/// A `Qwen3_5ForConditionalGeneration` config puts them under `text_config`; a
/// text-only config has them at the top level. Both are accepted, and the choice is
/// reported so a wrong pick is visible.
fn text_config(root: &serde_json::Value) -> (&serde_json::Value, &'static str) {
    match root.get("text_config") {
        Some(t) if t.is_object() => (t, "text_config"),
        _ => (root, "top level"),
    }
}

/// Reporting from a load.
#[derive(Debug, Clone)]
pub struct LoadInfo {
    pub dir: PathBuf,
    pub prefix: String,
    pub config_source: &'static str,
    pub shards: usize,
    pub tensors_total: usize,
    pub tensor_bytes: u64,
    pub dtype_counts: BTreeMap<String, usize>,
    pub skipped: Vec<(String, usize)>,
    pub tied_embeddings: bool,
    /// Sizes as *derived from tensor shapes*, before any config cross-check.
    pub derived: BTreeMap<String, usize>,
}

/// One loaded model.
pub struct RealModel {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub info: LoadInfo,
}

/// Deterministically pick the layer kind.
fn kind_of(layer_types: &[String], layer: usize) -> Result<LayerKind, String> {
    match layer_types.get(layer).map(|s| s.as_str()) {
        Some("linear_attention") => Ok(LayerKind::LinearAttention),
        Some("full_attention") => Ok(LayerKind::FullAttention),
        Some(other) => Err(format!("layer {layer}: unknown layer type {other}")),
        None => Err(format!("layer {layer}: layer_types is too short")),
    }
}

/// Read a tensor and require an exact element count.
fn want(m: &Model, name: &str, expect: usize) -> Result<Vec<f32>, String> {
    let v = m.tensor(name)?;
    if v.len() != expect {
        return Err(format!("{name}: got {} values, expected {expect}", v.len()));
    }
    Ok(v)
}

fn shape_of(m: &Model, name: &str) -> Result<Vec<usize>, String> {
    m.entry(name)
        .map(|e| e.shape.clone())
        .ok_or_else(|| format!("missing tensor {name}"))
}

/// Load a checkpoint directory.
pub fn load(dir: impl AsRef<Path>) -> Result<RealModel, String> {
    let dir = dir.as_ref().to_path_buf();

    // ---- config ----------------------------------------------------------
    let cfg_path = dir.join("config.json");
    if !cfg_path.exists() {
        return Err(format!("{}: no config.json", dir.display()));
    }
    let root: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
    let (tc, cfg_source) = text_config(&root);

    let model_type = tc
        .get("model_type")
        .or_else(|| root.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !model_type.starts_with("qwen3_5") {
        return Err(format!(
            "model_type is `{model_type}`; this forward pass implements qwen3_5"
        ));
    }

    // ---- tensors ---------------------------------------------------------
    let m = Model::open(&dir)?;
    let prefix = detect_prefix(&m)?;

    let mut skipped: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();
    for name in m.names() {
        let e = m.entry(name).unwrap();
        *dtype_counts.entry(e.dtype.as_str().to_string()).or_insert(0) += 1;
        if let Some(why) = is_unused(name) {
            *skipped.entry(why).or_insert(0) += 1;
        }
    }

    // ---- layer kinds -----------------------------------------------------
    let layer_types: Vec<String> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if layer_types.is_empty() {
        return Err("config has no layer_types; cannot tell linear from full layers".into());
    }
    let num_layers = cfg_usize(tc, "num_hidden_layers").ok_or("config has no num_hidden_layers")?;
    if layer_types.len() < num_layers {
        return Err(format!(
            "layer_types has {} entries but num_hidden_layers is {num_layers}",
            layer_types.len()
        ));
    }

    // ---- derived sizes ---------------------------------------------------
    let emb_name = format!("{prefix}embed_tokens.weight");
    let emb_shape = shape_of(&m, &emb_name)?;
    if emb_shape.len() != 2 {
        return Err(format!("{emb_name} has shape {emb_shape:?}, expected 2 axes"));
    }
    let (vocab, hidden) = (emb_shape[0], emb_shape[1]);

    // The first layer's MLP gives intermediate_size; every layer has one.
    let first_mlp = format!("{prefix}layers.0.mlp.gate_proj.weight");
    let intermediate = shape_of(&m, &first_mlp)?[0];

    // Find the first linear and the first full layer, since the two kinds have
    // different submodules.
    let first_linear = (0..num_layers).find(|&l| kind_of(&layer_types, l).ok() == Some(LayerKind::LinearAttention));
    let first_full = (0..num_layers).find(|&l| kind_of(&layer_types, l).ok() == Some(LayerKind::FullAttention));
    let (Some(first_linear), Some(first_full)) = (first_linear, first_full) else {
        return Err("the checkpoint has only one kind of layer; nothing to cross-check against".into());
    };

    let lin = |s: &str| format!("{prefix}layers.{first_linear}.linear_attn.{s}");
    let att = |s: &str| format!("{prefix}layers.{first_full}.self_attn.{s}");

    let conv_dim = shape_of(&m, &lin("in_proj_qkv.weight"))?[0];
    let value_dim = shape_of(&m, &lin("in_proj_z.weight"))?[0];
    let num_v_heads = shape_of(&m, &lin("in_proj_b.weight"))?[0];
    let head_v_dim = shape_of(&m, &lin("norm.weight"))?[0];
    let conv_shape = shape_of(&m, &lin("conv1d.weight"))?;
    let conv_kernel = *conv_shape.last().ok_or("conv1d weight has no axes")?;

    let head_dim = cfg_usize(tc, "head_dim").ok_or("config has no head_dim")?;
    let num_heads = cfg_usize(tc, "num_attention_heads").ok_or("config has no num_attention_heads")?;
    let num_kv_heads = cfg_usize(tc, "num_key_value_heads").ok_or("config has no num_key_value_heads")?;

    let q_out = shape_of(&m, &att("q_proj.weight"))?[0];
    let kv_out = shape_of(&m, &att("k_proj.weight"))?[0];

    // Linear-attention head counts are not directly in the shapes: key_dim is
    // implied by conv_dim and value_dim, and the head split comes from the config,
    // which is then cross-checked by reconstruction.
    let key_dim = conv_dim
        .checked_sub(value_dim)
        .map(|d| d / 2)
        .ok_or("conv_dim < value_dim")?;
    let num_k_heads_cfg = cfg_usize(tc, "linear_num_key_heads").ok_or("config has no linear_num_key_heads")?;
    let num_v_heads_cfg = cfg_usize(tc, "linear_num_value_heads").ok_or("config has no linear_num_value_heads")?;
    let head_k_dim_cfg = cfg_usize(tc, "linear_key_head_dim").ok_or("config has no linear_key_head_dim")?;
    let head_v_dim_cfg = cfg_usize(tc, "linear_value_head_dim").ok_or("config has no linear_value_head_dim")?;

    let mut derived = BTreeMap::new();
    derived.insert("vocab".into(), vocab);
    derived.insert("hidden".into(), hidden);
    derived.insert("intermediate".into(), intermediate);
    derived.insert("num_layers".into(), num_layers);
    derived.insert("conv_dim".into(), conv_dim);
    derived.insert("key_dim".into(), key_dim);
    derived.insert("value_dim".into(), value_dim);
    derived.insert("num_v_heads".into(), num_v_heads);
    derived.insert("head_v_dim".into(), head_v_dim);
    derived.insert("conv_kernel".into(), conv_kernel);
    derived.insert("head_dim".into(), head_dim);
    derived.insert("num_heads".into(), num_heads);
    derived.insert("num_kv_heads".into(), num_kv_heads);
    derived.insert("per_layer_tensors".into(), 0);

    // ---- cross-checks ----------------------------------------------------
    let mut problems: Vec<String> = Vec::new();
    // A macro rather than a closure: the closure would hold a mutable borrow of
    // `problems` for its whole lifetime, blocking the direct pushes below.
    macro_rules! check {
        ($what:expr, $from_config:expr, $from_shapes:expr) => {
            if $from_config != $from_shapes {
                problems.push(format!(
                    "{}: config says {}, the tensors imply {}",
                    $what, $from_config, $from_shapes
                ));
            }
        };
    }
    check!("num_v_heads", num_v_heads_cfg, num_v_heads);
    check!("head_v_dim", head_v_dim_cfg, head_v_dim);
    check!("key_dim = num_k_heads * head_k_dim", num_k_heads_cfg * head_k_dim_cfg, key_dim);
    check!("value_dim = num_v_heads * head_v_dim", num_v_heads_cfg * head_v_dim_cfg, value_dim);
    check!("conv_dim = 2*key_dim + value_dim", 2 * key_dim + value_dim, conv_dim);
    check!("q_proj out = num_heads * head_dim * 2", num_heads * head_dim * 2, q_out);
    check!("k_proj out = num_kv_heads * head_dim", num_kv_heads * head_dim, kv_out);
    if num_heads % num_kv_heads != 0 {
        problems.push(format!(
            "num_attention_heads {num_heads} is not a multiple of num_key_value_heads {num_kv_heads}"
        ));
    }
    if num_v_heads % num_k_heads_cfg != 0 {
        problems.push(format!(
            "num_v_heads {num_v_heads} is not a multiple of num_k_heads {num_k_heads_cfg}"
        ));
    }
    // The o_proj shape pins the attention head count independently of the config.
    let o_shape = shape_of(&m, &att("o_proj.weight"))?;
    check!("o_proj in = num_heads * head_dim", o_shape[1], num_heads * head_dim);
    check!("o_proj out = hidden", o_shape[0], hidden);
    if !problems.is_empty() {
        return Err(format!(
            "the config and the tensor shapes disagree, so one of them is not what \
             the forward pass should use:\n  {}",
            problems.join("\n  ")
        ));
    }

    // ---- rope ------------------------------------------------------------
    let rope = tc.get("rope_parameters");
    let rope_theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;
    let partial = rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let rotary_dim = (head_dim as f32 * partial) as usize;
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary_dim {rotary_dim} (head_dim {head_dim} * partial {partial}) is not a \
             positive even number <= head_dim"
        ));
    }

    let eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or("config has no rms_norm_eps")?;

    let gdn = GdnConfig {
        hidden,
        num_k_heads: num_k_heads_cfg,
        num_v_heads,
        head_k_dim: head_k_dim_cfg,
        head_v_dim,
        conv_kernel,
        eps,
    };
    let attn = AttnConfig {
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        eps,
    };

    // ---- head ------------------------------------------------------------
    let tied = tc
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let lm_name = if m.contains(&format!("{prefix}lm_head.weight")) {
        format!("{prefix}lm_head.weight")
    } else if m.contains("lm_head.weight") {
        "lm_head.weight".to_string()
    } else if tied {
        // Tied: the head *is* the embedding table. The reference does the same, so
        // this is a real parameter sharing and not an approximation.
        emb_name.clone()
    } else {
        return Err(
            "tie_word_embeddings is false and the checkpoint has no lm_head.weight".into()
        );
    };

    let embed = want(&m, &emb_name, vocab * hidden)?;
    let lm_head = want(&m, &lm_name, vocab * hidden)?;

    // ---- layers ----------------------------------------------------------
    let mut layers = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let kind = kind_of(&layer_types, layer)?;
        let p = format!("{prefix}layers.{layer}.");
        let layer_weights = LayerWeights {
            input_layernorm: want(&m, &format!("{p}input_layernorm.weight"), hidden)?,
            post_attention_layernorm: want(
                &m,
                &format!("{p}post_attention_layernorm.weight"),
                hidden,
            )?,
            mlp: MlpWeights {
                gate_proj: want(&m, &format!("{p}mlp.gate_proj.weight"), intermediate * hidden)?,
                up_proj: want(&m, &format!("{p}mlp.up_proj.weight"), intermediate * hidden)?,
                down_proj: want(&m, &format!("{p}mlp.down_proj.weight"), hidden * intermediate)?,
            },
        };
        let (g, a) = match kind {
            LayerKind::LinearAttention => {
                let lp = format!("{p}linear_attn.");
                (
                    Some(GdnWeights {
                        in_proj_qkv: want(&m, &format!("{lp}in_proj_qkv.weight"), conv_dim * hidden)?,
                        in_proj_z: want(&m, &format!("{lp}in_proj_z.weight"), value_dim * hidden)?,
                        in_proj_b: want(&m, &format!("{lp}in_proj_b.weight"), num_v_heads * hidden)?,
                        in_proj_a: want(&m, &format!("{lp}in_proj_a.weight"), num_v_heads * hidden)?,
                        conv1d: want(&m, &format!("{lp}conv1d.weight"), conv_dim * conv_kernel)?,
                        a_log: want(&m, &format!("{lp}A_log"), num_v_heads)?,
                        dt_bias: want(&m, &format!("{lp}dt_bias"), num_v_heads)?,
                        norm: want(&m, &format!("{lp}norm.weight"), head_v_dim)?,
                        out_proj: want(&m, &format!("{lp}out_proj.weight"), hidden * value_dim)?,
                    }),
                    None,
                )
            }
            LayerKind::FullAttention => {
                let ap = format!("{p}self_attn.");
                (
                    None,
                    Some(AttnWeights {
                        q_proj: want(&m, &format!("{ap}q_proj.weight"), q_out * hidden)?,
                        k_proj: want(&m, &format!("{ap}k_proj.weight"), kv_out * hidden)?,
                        v_proj: want(&m, &format!("{ap}v_proj.weight"), kv_out * hidden)?,
                        o_proj: want(&m, &format!("{ap}o_proj.weight"), hidden * num_heads * head_dim)?,
                        q_norm: want(&m, &format!("{ap}q_norm.weight"), head_dim)?,
                        k_norm: want(&m, &format!("{ap}k_norm.weight"), head_dim)?,
                    }),
                )
            }
        };
        layers.push(LayerWeightsAll { kind, layer: layer_weights, gdn: g, attn: a });
    }

    let config = ModelConfig { vocab, hidden, eps, gdn, attn };
    let weights = ModelWeights {
        embed_tokens: embed,
        final_norm: want(&m, &format!("{prefix}norm.weight"), hidden)?,
        lm_head,
        layers,
    };

    let info = LoadInfo {
        dir: dir.clone(),
        prefix,
        config_source: cfg_source,
        shards: m.shards.len(),
        tensors_total: m.names().count(),
        tensor_bytes: m.tensor_bytes(),
        dtype_counts,
        skipped: skipped.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        tied_embeddings: lm_name == emb_name,
        derived,
    };

    Ok(RealModel { config, weights, info })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unused_subtrees_are_classified() {
        assert_eq!(is_unused("model.visual.blocks.0.attn.qkv.weight"), Some("vision tower"));
        assert_eq!(is_unused("visual.patch_embed.proj.weight"), Some("vision tower"));
        assert_eq!(is_unused("mtp.layers.0.mlp.gate_proj.weight"), Some("multi-token-prediction head"));
        assert_eq!(is_unused("model.language_model.layers.0.mlp.up_proj.weight"), None);
    }

    #[test]
    fn text_config_prefers_the_nested_object() {
        let v = serde_json::json!({ "model_type": "qwen3_5", "text_config": { "hidden_size": 1024 } });
        let (tc, src) = text_config(&v);
        assert_eq!(src, "text_config");
        assert_eq!(tc["hidden_size"], 1024);

        let v = serde_json::json!({ "model_type": "qwen3_5_text", "hidden_size": 2048 });
        let (tc, src) = text_config(&v);
        assert_eq!(src, "top level");
        assert_eq!(tc["hidden_size"], 2048);
    }

    #[test]
    fn layer_kind_reports_unknown_and_short_lists() {
        let types = vec!["linear_attention".to_string(), "full_attention".to_string()];
        assert_eq!(kind_of(&types, 0).unwrap(), LayerKind::LinearAttention);
        assert_eq!(kind_of(&types, 1).unwrap(), LayerKind::FullAttention);
        let e = kind_of(&types, 2).unwrap_err();
        assert!(e.contains("too short"), "{e}");
        let bad = vec!["sliding_attention".to_string()];
        let e = kind_of(&bad, 0).unwrap_err();
        assert!(e.contains("unknown layer type"), "{e}");
    }
}
