//! `gdncheck` — run the gated delta net against a golden bundle, step by step.
//!
//! ```text
//! gdncheck <bundle> [--layer N] [--top N]
//! ```
//!
//! Compares every intermediate the bundle captured for one block, in chain order,
//! so the first divergence names the operator that is wrong rather than just
//! reporting that the output differs.

use std::process::ExitCode;

use gdn::layer::{layer_forward_from_mixer, LayerWeights, MlpWeights};
use gdn::{forward, GdnConfig, GdnWeights};
use goldenbundle::Bundle;

const TOL: f32 = 1e-5;

struct Report {
    checks: usize,
    failed: usize,
}

impl Report {
    fn step(&mut self, label: &str, mine: &[f32], golden: &[f32]) {
        self.checks += 1;
        if mine.len() != golden.len() {
            println!(
                "   {:<44} SHAPE  mine={} golden={}",
                label,
                mine.len(),
                golden.len()
            );
            self.failed += 1;
            return;
        }
        let (d, at) = max_abs_diff(mine, golden);
        let scale = golden.iter().fold(0f32, |m, x| m.max(x.abs()));
        let rel = if scale > 0.0 { d / scale } else { d };
        let ok = d <= TOL;
        if !ok {
            self.failed += 1;
        }
        println!(
            "   {:<44} abs={:<11.3e} rel={:<11.3e} {}",
            label,
            d,
            rel,
            if ok { "ok" } else { "FAIL" }
        );
        if !ok {
            println!("        worst at index {at}  mine={} golden={}", mine[at], golden[at]);
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0f32;
    let mut at = 0usize;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// Load a golden weight tensor, checking its length against what the config needs.
fn w(b: &Bundle, name: &str, expect_len: usize) -> Result<Vec<f32>, String> {
    let v = b.read(name).map_err(|e| format!("{name}: {e}"))?;
    if v.len() != expect_len {
        return Err(format!("{name}: got {} values, expected {expect_len}", v.len()));
    }
    Ok(v)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("usage: gdncheck <bundle> [--layer N]");
        return ExitCode::from(2);
    };
    let layer: usize = args
        .iter()
        .position(|a| a == "--layer")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let b = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let p = format!("model__layers__{layer}__");
    let attn = format!("{p}linear_attn__");

    // Shapes come from the bundle rather than being hardcoded, so the same
    // binary works for any layer.
    // The block's *input*, not the layernorm's output. For layer 0 that is the
    // embedding table's output. For layers > 0 it would be the previous layer's
    // residual stream, which the bundle does not capture (only submodule outputs
    // are hooked, and the decoder layer's own output is not among them), so an
    // explicit override is required.
    let input_name = args
        .iter()
        .position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            if layer == 0 {
                "model__embed_tokens".to_string()
            } else {
                String::new()
            }
        });
    if input_name.is_empty() {
        eprintln!(
            "layer {layer} needs an explicit --input <tensor>; the bundle does not \
             capture a decoder layer's residual-stream output"
        );
        return ExitCode::FAILURE;
    }
    let g_in = match b.read(&input_name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {input_name}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let shape_of = |name: &str| -> Vec<usize> {
        b.entry(name).map(|e| e.shape.clone()).unwrap_or_default()
    };
    let inorm_shape = shape_of(&format!("{p}input_layernorm"));
    let qkv_shape = shape_of(&format!("{attn}in_proj_qkv"));
    let z_shape = shape_of(&format!("{attn}in_proj_z"));
    let b_shape = shape_of(&format!("{attn}in_proj_b"));
    let conv_shape = shape_of(&format!("{attn}conv1d__weight"));
    let q_shape = shape_of(&format!("delta_torch_chunk_gated_delta_rule_{layer}__q"));

    if inorm_shape.len() != 3 || qkv_shape.len() != 3 {
        eprintln!("unexpected shapes: input_layernorm {inorm_shape:?}, in_proj_qkv {qkv_shape:?}");
        return ExitCode::FAILURE;
    }
    let (b_sz, t_sz, hidden) = (inorm_shape[0], inorm_shape[1], inorm_shape[2]);
    let conv_dim = qkv_shape[2];
    let value_dim = z_shape[2];
    let num_v_heads = b_shape[2];
    let head_v_dim = shape_of(&format!("{attn}norm__weight")).first().copied().unwrap_or(0);
    // Post-GQA `q` is `[B, T, num_v_heads, head_k_dim]`: repeat_interleave only
    // duplicates heads, so the trailing dim is still head_k_dim.
    let head_k_dim = *q_shape.last().unwrap_or(&0);
    // conv_dim = 2*key_dim + value_dim, and key_dim = num_k_heads * head_k_dim.
    let key_dim = conv_dim.saturating_sub(value_dim) / 2;
    let num_k_heads = key_dim.checked_div(head_k_dim).unwrap_or(0);

    if head_v_dim == 0
        || head_k_dim == 0
        || num_k_heads == 0
        || num_v_heads == 0
        || value_dim != num_v_heads * head_v_dim
        || key_dim != num_k_heads * head_k_dim
        || conv_dim != 2 * key_dim + value_dim
    {
        eprintln!(
            "derived sizes are inconsistent:\n  conv_dim={conv_dim} value_dim={value_dim}\n  \
             num_k_heads={num_k_heads} num_v_heads={num_v_heads}\n  \
             head_k_dim={head_k_dim} head_v_dim={head_v_dim}\n  q_shape={q_shape:?} b_shape={b_shape:?}"
        );
        return ExitCode::FAILURE;
    }
    if num_v_heads % num_k_heads != 0 {
        eprintln!("num_v_heads ({num_v_heads}) is not a multiple of num_k_heads ({num_k_heads})");
        return ExitCode::FAILURE;
    }

    let cfg = GdnConfig {
        hidden,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel: conv_shape.last().copied().unwrap_or(4),
        eps: 1e-6,
    };

    println!("== gated delta net vs {}", b.root.display());
    println!("   layer {layer}   B={b_sz} T={t_sz} hidden={hidden}");
    println!("   block input: {input_name}");
    println!(
        "   k_heads={} v_heads={} head_k={} head_v={} conv_k={} ratio={}",
        cfg.num_k_heads,
        cfg.num_v_heads,
        cfg.head_k_dim,
        cfg.head_v_dim,
        cfg.conv_kernel,
        cfg.kv_ratio()
    );
    println!("   conv_dim={conv_dim} value_dim={value_dim}");
    println!("   tolerance {TOL:.1e}");
    println!();

    if !cfg.num_v_heads.is_multiple_of(cfg.num_k_heads.max(1)) {
        eprintln!("   derived head counts are inconsistent; aborting");
        return ExitCode::FAILURE;
    }

    let conv_raw = match b.read(&format!("{attn}conv1d__weight")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("conv1d weight: {e}");
            return ExitCode::FAILURE;
        }
    };

    let weights = GdnWeights {
        input_layernorm: match w(&b, &format!("{p}input_layernorm__weight"), hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        in_proj_qkv: match w(&b, &format!("{attn}in_proj_qkv__weight"), conv_dim * hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        in_proj_z: match w(&b, &format!("{attn}in_proj_z__weight"), value_dim * hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        in_proj_b: match w(&b, &format!("{attn}in_proj_b__weight"), num_v_heads * hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        in_proj_a: match w(&b, &format!("{attn}in_proj_a__weight"), num_v_heads * hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        conv1d: conv_raw,
        a_log: match w(&b, &format!("{attn}A_log"), num_v_heads) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        dt_bias: match w(&b, &format!("{attn}dt_bias"), num_v_heads) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        norm: match w(&b, &format!("{attn}norm__weight"), head_v_dim) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        out_proj: match w(&b, &format!("{attn}out_proj__weight"), hidden * value_dim) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
    };

    let trace = forward(&cfg, &weights, &g_in, b_sz, t_sz);

    // ---- MLP + the two residuals -------------------------------------------
    let intermediate = shape_of(&format!("{p}mlp__gate_proj"))
        .last()
        .copied()
        .unwrap_or(0);
    if intermediate == 0 {
        eprintln!("cannot derive intermediate_size from {p}mlp__gate_proj");
        return ExitCode::FAILURE;
    }
    let layer_weights = LayerWeights {
        input_layernorm: weights.input_layernorm.clone(),
        post_attention_layernorm: match w(&b, &format!("{p}post_attention_layernorm__weight"), hidden) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        },
        mlp: MlpWeights {
            gate_proj: match w(&b, &format!("{p}mlp__gate_proj__weight"), intermediate * hidden) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            },
            up_proj: match w(&b, &format!("{p}mlp__up_proj__weight"), intermediate * hidden) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            },
            down_proj: match w(&b, &format!("{p}mlp__down_proj__weight"), hidden * intermediate) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            },
        },
    };
    let ltrace = layer_forward_from_mixer(
        &layer_weights,
        &trace.out_proj,
        &g_in,
        &trace.input_layernorm,
        trace.clone(),
        b_sz * t_sz,
    );

    let mut r = Report { checks: 0, failed: 0 };
    let get = |n: &str| b.read(n).ok();

    macro_rules! cmp {
        ($label:expr, $mine:expr, $golden_name:expr) => {
            match get($golden_name) {
                Some(g) => r.step($label, &$mine, &g),
                None => println!("   {:<44} (golden missing)", $label),
            }
        };
    }

    cmp!("1. input_layernorm", trace.input_layernorm, &format!("{p}input_layernorm"));
    cmp!("2. in_proj_qkv", trace.in_proj_qkv, &format!("{attn}in_proj_qkv"));
    cmp!(
        "3. conv input (channels-first)",
        trace.conv_in,
        &format!("causal_conv1d_fn_call{layer}_in")
    );
    cmp!(
        "4. conv + silu",
        trace.conv_out,
        &format!("causal_conv1d_fn_call{layer}_out")
    );
    cmp!("5. in_proj_z", trace.in_proj_z, &format!("{attn}in_proj_z"));
    cmp!("6. in_proj_b", trace.in_proj_b, &format!("{attn}in_proj_b"));
    cmp!("7. in_proj_a", trace.in_proj_a, &format!("{attn}in_proj_a"));
    cmp!(
        "8. q (post-GQA)",
        trace.q,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__q")
    );
    cmp!(
        "9. k (post-GQA)",
        trace.k,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__k")
    );
    cmp!(
        "10. v",
        trace.v,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__v")
    );
    cmp!(
        "11. g (decay)",
        trace.g,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__g")
    );
    cmp!(
        "12. beta (gate)",
        trace.beta,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__beta")
    );
    cmp!(
        "13. delta rule out",
        trace.delta_out,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__out")
    );
    cmp!(
        "14. delta rule state",
        trace.delta_state,
        &format!("delta_torch_chunk_gated_delta_rule_{layer}__state")
    );
    cmp!("15. gated norm", trace.norm, &format!("{attn}norm"));
    cmp!("16. out_proj", trace.out_proj, &format!("{attn}out_proj"));
    cmp!("17. block output", trace.out_proj, &format!("{p}linear_attn"));
    cmp!(
        "18. post_attention_layernorm",
        ltrace.post_attention_layernorm,
        &format!("{p}post_attention_layernorm")
    );
    cmp!(
        "19. mlp gate_proj",
        ltrace.mlp.gate_proj,
        &format!("{p}mlp__gate_proj")
    );
    cmp!("20. mlp up_proj", ltrace.mlp.up_proj, &format!("{p}mlp__up_proj"));
    cmp!(
        "21. mlp swiglu product",
        ltrace.mlp.swiglu_product,
        &format!("{p}mlp__swiglu_product")
    );
    cmp!(
        "22. mlp down_proj",
        ltrace.mlp.down_proj,
        &format!("{p}mlp__down_proj")
    );
    cmp!("23. mlp output", ltrace.mlp.down_proj, &format!("{p}mlp"));
    // The decoder layer's own name has no trailing separator: `model__layers__0`.
    cmp!(
        "24. layer output (both residuals)",
        ltrace.out,
        format!("model__layers__{layer}").as_str()
    );

    println!();
    if r.failed == 0 {
        println!("   RESULT: PASS ({} checks)", r.checks);
        ExitCode::SUCCESS
    } else {
        println!("   RESULT: FAIL ({} of {} checks)", r.failed, r.checks);
        ExitCode::FAILURE
    }
}
