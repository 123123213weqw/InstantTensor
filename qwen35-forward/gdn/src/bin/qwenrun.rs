//! `qwenrun` — run a real Qwen3.5 checkpoint.
//!
//! ```text
//! qwenrun <model-dir> [--prompt 1,2,3] [--tokens N] [--dump-logits FILE]
//!                     [--compare FILE] [--topk K]
//! ```
//!
//! Prints what it loaded, runs the prompt through the stack, reports the next-token
//! distribution, and optionally decodes greedily.
//!
//! # Comparing against the reference
//!
//! `--dump-logits` writes the last position's logits as raw little-endian `f32`, and
//! `--compare` reads such a file and diffs it. The expected workflow is: run the
//! reference (transformers) once, dump the same tensor, then compare. A tolerance of
//! `1e-5` is used by default; a `bf16` checkpoint cannot be expected to agree much
//! more closely than its own storage precision, so the interesting number is whether
//! the **argmax** and the top-k ordering match, which the tool reports separately.

use std::process::ExitCode;

use gdn::model;
use gdn::real;

/// Decode a raw little-endian `f32` blob.
///
/// `chunks(4)` rather than `chunks_exact(4)`: clippy flags the latter with a constant
/// chunk size, and the trailing-partial case cannot arise here anyway because the
/// caller rejects a length that is not a multiple of 4.
fn read_f32_blob(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .filter(|c| c.len() == 4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_list(s: &str) -> Result<Vec<u32>, String> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<u32>().map_err(|e| format!("bad token id `{p}`: {e}")))
        .collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let val = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!(
            "usage: qwenrun <model-dir> [--prompt 1,2,3] [--tokens N] [--topk K]\n\
             \x20                    [--dump-logits FILE] [--compare FILE] [--tol F]"
        );
        return ExitCode::from(2);
    };
    let topk: usize = val("--topk").and_then(|s| s.parse().ok()).unwrap_or(10);
    let steps: usize = val("--tokens").and_then(|s| s.parse().ok()).unwrap_or(0);
    // Default tolerance, and why it is not 1e-5.
    //
    // A `bf16` checkpoint loaded into an `f32` reference is still an `f32`
    // computation, and `f32` does not reproduce itself across implementations. On
    // this model the reference disagrees with *itself* by 2.46e-5 between CPU and
    // GPU (same weights, same dtype, same eager attention), and sits 1.6e-5 to
    // 2.6e-5 from a float64 ground truth.
    //
    // So a bound tighter than that is not a test of correctness; it is a demand that
    // this engine agree with the reference more closely than the reference agrees
    // with itself. 3e-5 is above the reference's own spread and below what a real
    // operator mistake produces (a wrong gate or a missing rotation moves logits by
    // ~1e-1, not 1e-5).
    let tol: f32 = val("--tol").and_then(|s| s.parse().ok()).unwrap_or(3e-5);
    // A tiny default prompt: one token is enough to exercise every layer and makes
    // the reference comparison cheap.
    let prompt: Vec<u32> = match val("--prompt") {
        Some(s) => match parse_list(&s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        None => vec![9419],
    };
    if prompt.is_empty() {
        eprintln!("error: empty prompt");
        return ExitCode::from(2);
    }

    // ---- load -------------------------------------------------------------
    println!("== loading {dir}");
    let t0 = std::time::Instant::now();
    let rm = match real::load(dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let load_s = t0.elapsed().as_secs_f64();
    let i = &rm.info;
    println!("   loaded in {load_s:.1}s");
    println!(
        "   prefix `{}`  config from {}  {} shard(s)  {} tensors  {:.2} GiB of tensor data",
        i.prefix,
        i.config_source,
        i.shards,
        i.tensors_total,
        i.tensor_bytes as f64 / (1u64 << 30) as f64
    );
    println!(
        "   dtypes: {}",
        i.dtype_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !i.skipped.is_empty() {
        println!(
            "   unused sub-trees: {}",
            i.skipped
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!(
        "   head: {}",
        if i.tied_embeddings {
            "tied to embed_tokens".to_string()
        } else {
            "separate lm_head".to_string()
        }
    );
    print!("   derived:");
    for (k, v) in &i.derived {
        print!(" {k}={v}");
    }
    println!();
    let c = &rm.config;
    println!(
        "   gdn: k_heads={} v_heads={} head_k={} head_v={} conv_k={}",
        c.gdn.num_k_heads, c.gdn.num_v_heads, c.gdn.head_k_dim, c.gdn.head_v_dim, c.gdn.conv_kernel
    );
    println!(
        "   attn: heads={} kv_heads={} head_dim={} rotary_dim={} theta={}",
        c.attn.num_heads, c.attn.num_kv_heads, c.attn.head_dim, c.attn.rotary_dim, c.attn.rope_theta
    );
    let lin = rm
        .weights
        .layers
        .iter()
        .filter(|l| l.kind == model::LayerKind::LinearAttention)
        .count();
    println!(
        "   {} layers ({} linear_attention, {} full_attention), hidden={} vocab={} eps={:e}",
        rm.weights.layers.len(),
        lin,
        rm.weights.layers.len() - lin,
        c.hidden,
        c.vocab,
        c.eps
    );

    // ---- forward ----------------------------------------------------------
    println!();
    println!("   prompt: {prompt:?}  ({} tokens)", prompt.len());
    let t1 = std::time::Instant::now();
    let tr = match model::forward(c, &rm.weights, &prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("forward failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let fwd_s = t1.elapsed().as_secs_f64();
    println!(
        "   forward: {fwd_s:.2}s for {} token(s)  ({:.3}s/token)",
        prompt.len(),
        fwd_s / prompt.len() as f64
    );

    let last = tr.last_logits();
    if !last.iter().all(|x| x.is_finite()) {
        let bad = last.iter().filter(|x| !x.is_finite()).count();
        eprintln!("   !! {bad} non-finite logits");
        return ExitCode::FAILURE;
    }
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("   next-token argmax = {}", idx[0]);
    println!("   top-{topk}:");
    for &t in idx.iter().take(topk) {
        println!("      {:>7}  {:+.6}", t, last[t]);
    }
    // A distribution that is nearly uniform would mean the stack is not doing
    // anything; a healthy model puts most of its mass on a few tokens.
    let max = last[idx[0]];
    let sum: f64 = last.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let p0 = 1.0 / sum;
    println!("   p(argmax) = {p0:.4}   (uniform would be {:.2e})", 1.0 / c.vocab as f64);

    let mut failed = false;

    // ---- per-layer comparison ---------------------------------------------
    // The point of this: an end-to-end logit diff of 1e-4 tells you nothing about
    // whether it is 24 layers of slow accumulation or one layer that is wrong. The
    // reference dumps every layer output, so each layer can be judged on its own.
    if let Some(dir) = val("--compare-layers") {
        let dir = std::path::PathBuf::from(dir);
        let shapes_path = dir.join("layer_shapes.json");
        let blob_path = dir.join("layers.f32");
        let shapes: Vec<Vec<usize>> = match std::fs::read(&shapes_path)
            .map_err(|e| format!("{}: {e}", shapes_path.display()))
            .and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", shapes_path.display()))
            }) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let blob = match std::fs::read(&blob_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: {}: {e}", blob_path.display());
                return ExitCode::FAILURE;
            }
        };
        let refv = read_f32_blob(&blob);
        let expect: usize = shapes.iter().map(|s| s.iter().product::<usize>()).sum();
        if refv.len() != expect {
            eprintln!(
                "{}: {} values but the shapes need {expect}",
                blob_path.display(),
                refv.len()
            );
            return ExitCode::FAILURE;
        }
        println!();
        println!("   per-layer agreement (each layer's output, mine vs reference)");
        println!("     {:<6} {:>12} {:>12} {:>12}", "layer", "max abs", "rel", "growth");
        let mut off = 0usize;
        let mut prev = 0f32;
        for (k, sh) in shapes.iter().enumerate() {
            let n: usize = sh.iter().product();
            let gold = &refv[off..off + n];
            off += n;
            if k >= tr.layers.len() {
                println!("     {k:<6} (mine has only {} layers)", tr.layers.len());
                break;
            }
            let mine = &tr.layers[k].out;
            if mine.len() != n {
                println!("     {k:<6} SHAPE mine={} ref={n}", mine.len());
                failed = true;
                continue;
            }
            let mut worst = 0f32;
            for j in 0..n {
                let d = (mine[j] - gold[j]).abs();
                if d > worst {
                    worst = d;
                }
            }
            let scale = gold.iter().fold(0f32, |m, x| m.max(x.abs()));
            let rel = if scale > 0.0 { worst / scale } else { worst };
            println!(
                "     {k:<6} {worst:>12.3e} {rel:>12.3e} {:>12}",
                if prev > 0.0 {
                    format!("{:.1}x", worst / prev)
                } else {
                    "-".to_string()
                }
            );
            prev = worst;
            if worst > tol {
                failed = true;
            }
        }
        println!(
            "     note: growth is the ratio to the previous layer, so a gradual rise is \
             accumulation and a jump is a layer that is wrong"
        );
    }

    // ---- dump / compare ----------------------------------------------------
    if let Some(path) = val("--dump-logits") {
        let mut bytes = Vec::with_capacity(last.len() * 4);
        for v in last {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        match std::fs::write(&path, &bytes) {
            Ok(()) => println!("   dumped {} f32 logits to {path}", last.len()),
            Err(e) => {
                eprintln!("cannot write {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(path) = val("--compare") {
        match std::fs::read(&path) {
            Ok(bytes) => {
                if bytes.len() % 4 != 0 {
                    eprintln!("{path}: {} bytes is not a multiple of 4", bytes.len());
                    return ExitCode::FAILURE;
                }
                let refv = read_f32_blob(&bytes);
                if refv.len() != last.len() {
                    eprintln!(
                        "{path}: {} values but the model has {} logits",
                        refv.len(),
                        last.len()
                    );
                    return ExitCode::FAILURE;
                }
                let (mut worst, mut at) = (0f32, 0usize);
                for k in 0..refv.len() {
                    let d = (last[k] - refv[k]).abs();
                    if d > worst {
                        worst = d;
                        at = k;
                    }
                }
                let mut ridx: Vec<usize> = (0..refv.len()).collect();
                ridx.sort_by(|&a, &b| refv[b].partial_cmp(&refv[a]).unwrap());
                let my_top: Vec<usize> = idx.iter().take(topk).copied().collect();
                let ref_top: Vec<usize> = ridx.iter().take(topk).copied().collect();
                println!();
                println!("   compare vs {path}");
                println!("     max abs diff      {worst:.3e}  (tolerance {tol:.0e})");
                println!("     worst at index    {at}  mine={} ref={}", last[at], refv[at]);
                println!("     argmax            mine={} ref={}", idx[0], ridx[0]);
                println!(
                    "     top-{topk} ordering  {}",
                    if my_top == ref_top { "identical" } else { "DIFFER" }
                );
                if my_top != ref_top {
                    println!("       mine {my_top:?}");
                    println!("       ref  {ref_top:?}");
                }
                // The tokens are what matters; the absolute tolerance is a proxy.
                // Two independent gates, reported separately: the functional one
                // (do the tokens agree) and the numerical one (are the values within
                // the tolerance). They fail for different reasons and the distinction
                // matters -- identical tokens with a large value error means a
                // precision problem, whereas differing tokens means a wrong operator.
                let ok = idx[0] == ridx[0] && my_top == ref_top;
                if !ok || worst > tol {
                    failed = true;
                }
                println!(
                    "     => {}",
                    if !ok {
                        "FAIL: the tokens differ, which is an operator error rather than precision"
                    } else if worst <= tol {
                        "PASS"
                    } else {
                        "FAIL: tokens agree but the values exceed the tolerance"
                    }
                );
            }
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    // ---- greedy -----------------------------------------------------------
    if steps > 0 {
        println!();
        println!("   greedy decoding {steps} token(s)");
        let mut ids = prompt.clone();
        let t2 = std::time::Instant::now();
        for k in 0..steps {
            let t = match model::forward(c, &rm.weights, &ids) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("forward failed at step {k}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let nxt = t.argmax_last() as u32;
            ids.push(nxt);
            println!(
                "     step {k:>2}  len={:<4} -> {nxt}   ({:.2}s elapsed)",
                ids.len() - 1,
                t2.elapsed().as_secs_f64()
            );
        }
        println!("   generated ids: {}", ids[prompt.len()..].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }

    println!();
    if failed {
        println!("   RESULT: FAIL");
        ExitCode::FAILURE
    } else {
        println!("   RESULT: OK");
        ExitCode::SUCCESS
    }
}
