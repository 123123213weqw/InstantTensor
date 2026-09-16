//! `gdncheck` -- run the qwen35 stack against a golden bundle.
//!
//! ```text
//! gdncheck <bundle>            every layer, each fed the golden input for that layer
//! gdncheck <bundle> --chain    each layer's own output feeds the next
//! gdncheck <bundle> --layer N  one layer, every check printed
//! gdncheck <bundle> --model    the whole model: logits and the greedy token trace
//! gdncheck <bundle> --verbose  all checks for every layer
//! ```
//!
//! # Three levels, each answering a different question
//!
//! **Isolated** per-layer checks feed each layer the input the *reference* produced
//! for it. A faulty layer cannot contaminate the verdict on a later one, so a
//! failure names the layer that caused it.
//!
//! **Chained** feeds each layer's own output onward. It localises worse, but it is
//! the only way to show the stack works end to end and that rounding does not
//! accumulate.
//!
//! **Model** runs embedding, all layers, the final norm and the head, then decodes
//! greedily and compares the generated tokens. The per-layer checks can all pass
//! while the assembled model is wrong -- a wrong final norm, a head applied at the
//! wrong position, or layers wired in the wrong order are all invisible to them.

use std::process::ExitCode;

use gdn::attention;
use gdn::layer::{layer_forward, LayerTrace, Mixer, MixerTrace};
use gdn::loader::{self, LayerCapture, ModelInfo};
use gdn::model::{self, LayerKind};
use gdn::GdnConfig;
use goldenbundle::Bundle;

const TOL: f32 = 1e-5;

/// Absolute tolerance for a bundle.
///
/// `--ssm-gain` multiplies `linear_attn.out_proj`, so the recurrent path's
/// contribution to the residual is scaled by the same factor -- and so is the
/// absolute rounding error it carries. Holding the tolerance fixed would make the
/// amplified bundle fail on arithmetic that is proportionally identical, so the
/// tolerance is scaled with the gain. The relative error, which is what actually
/// matters, is unchanged.
fn tolerance_for(ssm_gain: f64) -> f32 {
    TOL * (ssm_gain.max(1.0) as f32)
}

struct Check {
    label: String,
    abs: f32,
    rel: f32,
    ok: bool,
    missing: bool,
}

struct Checker<'a> {
    b: &'a Bundle,
    checks: Vec<Check>,
    n: usize,
    /// Absolute tolerance, already scaled for the bundle's `ssm_gain`.
    tol: f32,
}

impl<'a> Checker<'a> {
    fn new(b: &'a Bundle, tol: f32) -> Checker<'a> {
        Checker { b, checks: Vec::new(), n: 0, tol }
    }

    /// Compare against a named golden tensor. `None` means "not applicable to this
    /// layer kind" (a full-attention layer has no convolution), which is not a
    /// failure.
    fn step_opt(&mut self, label: &str, mine: &[f32], name: Option<String>) {
        self.n += 1;
        let label = format!("{:>2}. {label}", self.n);
        let Some(name) = name else {
            self.checks.push(Check { label, abs: f32::NAN, rel: f32::NAN, ok: true, missing: true });
            return;
        };
        match self.b.read(&name) {
            Ok(golden) => self.compare(label, mine, &golden),
            // Applicable but absent. For a layer that should have this tensor, its
            // absence is a real gap: treating it as a pass once let a capture-index
            // bug hide nine checks behind a cheerful "ok".
            Err(e) => self.checks.push(Check {
                label: format!("{label} MISSING {name} ({e})"),
                abs: f32::INFINITY,
                rel: f32::INFINITY,
                ok: false,
                missing: false,
            }),
        }
    }

    fn step(&mut self, label: &str, mine: &[f32], name: &str) {
        self.step_opt(label, mine, Some(name.to_string()));
    }

    fn compare(&mut self, label: String, mine: &[f32], golden: &[f32]) {
        if mine.len() != golden.len() {
            self.checks.push(Check {
                label: format!("{label} SHAPE mine={} golden={}", mine.len(), golden.len()),
                abs: f32::INFINITY,
                rel: f32::INFINITY,
                ok: false,
                missing: false,
            });
            return;
        }
        let (abs, _) = max_abs_diff(mine, golden);
        let scale = golden.iter().fold(0f32, |m, x| m.max(x.abs()));
        let rel = if scale > 0.0 { abs / scale } else { abs };
        self.checks.push(Check { label, abs, rel, ok: abs <= self.tol, missing: false });
    }

    fn failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.ok && !c.missing).count()
    }

    fn first_failure(&self) -> Option<&Check> {
        self.checks.iter().find(|c| !c.ok && !c.missing)
    }

    fn worst(&self) -> Option<&Check> {
        self.checks
            .iter()
            .filter(|c| !c.missing)
            .max_by(|x, y| x.abs.partial_cmp(&y.abs).unwrap())
    }

    /// The layer-output error, which is the number that would accumulate along a
    /// chain.
    fn out_abs(&self) -> f32 {
        self.checks.last().map(|c| c.abs).unwrap_or(f32::NAN)
    }

    fn describe(&self) -> (String, f32) {
        match self.first_failure() {
            Some(f) => (format!("FAIL {}", f.label), f.abs),
            None => match self.worst() {
                Some(w) => (format!("worst: {}", w.label), w.abs),
                None => (String::from("(nothing compared)"), f32::NAN),
            },
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

/// Run one layer and compare every intermediate the bundle holds for it.
fn verify_layer<'a>(
    b: &'a Bundle,
    m: &ModelInfo,
    layer: usize,
    input: &[f32],
    rope: Option<(&[f32], &[f32])>,
    tol: f32,
) -> Result<(Checker<'a>, LayerTrace), String> {
    let c = LayerCapture::new(layer, m.ssm_ordinal(layer));
    let lw = loader::load_layer_weights(b, m, layer)?;
    let kind = loader::layer_kind(m, layer);

    let cfg_gdn: GdnConfig = m.gdn;
    let attn_cfg = m.attn;
    // The mixer borrows the weights, so each arm must keep them alive across the
    // call rather than storing them in an outer `Option`.
    let tr = match kind {
        LayerKind::LinearAttention => {
            let w = loader::load_gdn_weights(b, m, layer)?;
            layer_forward(
                Mixer::LinearAttention(&cfg_gdn, &w),
                &lw, input, m.b, m.t, m.eps, rope,
            )?
        }
        LayerKind::FullAttention => {
            let w = loader::load_attn_weights(b, m, layer)?;
            layer_forward(
                Mixer::FullAttention(&attn_cfg, &w),
                &lw, input, m.b, m.t, m.eps, rope,
            )?
        }
    };

    let mut ck = Checker::new(b, tol);
    ck.step("input_layernorm", &tr.input_layernorm, &c.input_layernorm());

    match &tr.mixer {
        MixerTrace::LinearAttention(g) => {
            ck.step("in_proj_qkv", &g.in_proj_qkv, &c.in_proj_qkv());
            ck.step_opt("conv input (channels-first)", &g.conv_in, c.conv_in());
            ck.step_opt("conv + silu", &g.conv_out, c.conv_out());
            ck.step("in_proj_z", &g.in_proj_z, &c.in_proj_z());
            ck.step("in_proj_b", &g.in_proj_b, &c.in_proj_b());
            ck.step("in_proj_a", &g.in_proj_a, &c.in_proj_a());
            ck.step_opt("q (post-GQA)", &g.q, c.delta_operand("q"));
            ck.step_opt("k (post-GQA)", &g.k, c.delta_operand("k"));
            ck.step_opt("v", &g.v, c.delta_operand("v"));
            ck.step_opt("g (decay)", &g.g, c.delta_operand("g"));
            ck.step_opt("beta (gate)", &g.beta, c.delta_operand("beta"));
            ck.step_opt("delta rule out", &g.delta_out, c.delta_out());
            ck.step_opt("delta rule state", &g.delta_state, c.delta_state());
            ck.step("gated norm", &g.norm, &c.mixer_norm());
            ck.step("out_proj", &g.out_proj, &c.out_proj());
            ck.step("linear_attn output", &g.out_proj, &c.linear_attn());
        }
        MixerTrace::FullAttention(a) => {
            ck.step("q_proj", &a.q_proj, &c.q_proj());
            ck.step("q_norm (pre-rope)", &a.q_norm, &c.q_norm());
            ck.step("k_proj", &a.k_proj, &c.k_proj());
            ck.step("k_norm (pre-rope)", &a.k_norm, &c.k_norm());
            ck.step("v_proj", &a.v_proj, &c.v_proj());
            // The gate multiply happens on a tensor that is never bound to a name,
            // so it is recovered from `o_proj`'s *input*. This is the check that
            // pins down `sigmoid` vs the `swish` the config claims.
            ck.step("attn out (post-gate, pre-o_proj)", &a.gated, &c.o_proj_in());
            ck.step("o_proj", &a.o_proj, &c.o_proj());
            // `Qwen3_5Attention` returns `(attn_output, attn_weights)` with
            // `attn_output` already through `o_proj`, so `out0` is the module output.
            ck.step("self_attn output", &a.o_proj, &c.self_attn_out0());
        }
    }

    ck.step("post_attention_layernorm", &tr.post_attention_layernorm, &c.post_attention_layernorm());
    ck.step("mlp gate_proj", &tr.mlp.gate_proj, &c.mlp_gate_proj());
    ck.step("mlp up_proj", &tr.mlp.up_proj, &c.mlp_up_proj());
    ck.step("mlp swiglu product", &tr.mlp.swiglu_product, &c.mlp_swiglu());
    ck.step("mlp down_proj", &tr.mlp.down_proj, &c.mlp_down_proj());
    ck.step("mlp output", &tr.mlp.down_proj, &c.mlp());
    ck.step("layer output (both residuals)", &tr.out, &c.layer_out());

    Ok((ck, tr))
}

fn print_checks(ck: &Checker<'_>) {
    for c in &ck.checks {
        if c.missing {
            println!("   {:<52} (n/a for this layer kind)", c.label);
        } else {
            println!(
                "   {:<52} abs={:<11.3e} rel={:<11.3e} {}",
                c.label,
                c.abs,
                c.rel,
                if c.ok { "ok" } else { "FAIL" }
            );
        }
    }
}

struct Row {
    layer: usize,
    ltype: String,
    note: String,
    label: String,
    abs: f32,
    out_abs: f32,
    failed: usize,
    ok: bool,
    skipped: bool,
}

fn print_table(rows: &[Row]) {
    println!(
        "   {:<5} {:<17} {:<34} {:>10} {:>10}",
        "layer", "type", "first failing / worst check", "abs", "layer-out"
    );
    for r in rows {
        if r.skipped {
            println!("   {:<5} {:<17} {}", r.layer, r.ltype, r.note);
            continue;
        }
        println!(
            "   {:<5} {:<17} {:<34} {:>10.3e} {:>10.3e}  {}",
            r.layer,
            r.ltype,
            r.label,
            r.abs,
            r.out_abs,
            if r.ok { "ok".to_string() } else { format!("FAIL ({} checks)", r.failed) }
        );
    }
}

/// Every layer's golden input is `model__layers__<N-1>`; layer 0 comes from the
/// embedding table.
fn layer_input_name(m: &ModelInfo, layer: usize) -> String {
    m.input_name(layer)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let val = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };

    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("usage: gdncheck <bundle> [--layer N] [--chain] [--model] [--verbose]");
        return ExitCode::from(2);
    };
    let verbose = flag("--verbose");
    let chain = flag("--chain");
    let do_model = flag("--model") || flag("--greedy");
    let only: Option<usize> = val("--layer").and_then(|s| s.parse().ok());

    let b = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let m = match loader::load_model(&b) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let (lin, full) = m.counts();
    println!("== qwen35 vs {}", b.root.display());
    println!(
        "   B={} T={} hidden={} layers={} ({} linear_attention, {} full_attention) vocab={}",
        m.b, m.t, m.hidden, m.num_layers, lin, full, m.vocab
    );
    println!(
        "   gdn: k_heads={} v_heads={} head_k={} head_v={} conv_k={}",
        m.gdn.num_k_heads, m.gdn.num_v_heads, m.gdn.head_k_dim, m.gdn.head_v_dim, m.gdn.conv_kernel
    );
    println!(
        "   attn: heads={} kv_heads={} head_dim={} rotary_dim={} theta={}",
        m.attn.num_heads, m.attn.num_kv_heads, m.attn.head_dim, m.attn.rotary_dim, m.attn.rope_theta
    );
    let tol = tolerance_for(m.ssm_gain);
    println!("   eps={:e} ssm_gain={} tolerance {tol:.1e}", m.eps, m.ssm_gain);
    println!();

    let mut any_failed = false;

    // ---- RoPE tables ------------------------------------------------------
    // The builder is checked against the reference's own tables here, once, so a
    // failure inside an attention layer can be attributed elsewhere.
    match loader::load_rope(&b, &m) {
        Ok((cos_ref, sin_ref)) => {
            let (cos, sin) = attention::build_rope(&m.attn, m.t, 0);
            let dc = max_abs_diff(&cos, &cos_ref).0;
            let ds = max_abs_diff(&sin, &sin_ref).0;
            let ok = dc <= tol && ds <= tol;
            any_failed |= !ok;
            println!(
                "   rope tables vs reference:   cos abs={dc:.3e}  sin abs={ds:.3e}  {}",
                if ok { "ok" } else { "FAIL" }
            );
        }
        Err(e) => println!("   rope tables: cannot compare ({e})"),
    }

    // ---- single layer -----------------------------------------------------
    if let Some(layer) = only {
        if layer >= m.num_layers {
            eprintln!("layer {layer} is out of range (0..{})", m.num_layers);
            return ExitCode::from(2);
        }
        let input_name = layer_input_name(&m, layer);
        let input = match b.read(&input_name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cannot read {input_name}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let (cos, sin) = match loader::load_rope(&b, &m) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("cannot read rope tables: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("   layer {layer} ({})  input: {input_name}", m.layer_type(layer));
        match verify_layer(&b, &m, layer, &input, Some((&cos, &sin)), tol) {
            Ok((ck, _)) => {
                print_checks(&ck);
                let failed = ck.failed();
                println!();
                if failed == 0 {
                    println!("   RESULT: PASS ({} checks)", ck.checks.len());
                } else {
                    println!("   RESULT: FAIL ({failed} of {} checks)", ck.checks.len());
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                eprintln!("   layer {layer}: {e}");
                return ExitCode::FAILURE;
            }
        }
        if !do_model {
            return ExitCode::SUCCESS;
        }
    }

    let (cos, sin) = match loader::load_rope(&b, &m) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot read rope tables: {e}");
            return ExitCode::FAILURE;
        }
    };

    // ---- isolated, every layer --------------------------------------------
    if only.is_none() {
        println!("   mode: isolated (each layer fed the golden input recorded for it)");
        let mut rows: Vec<Row> = Vec::new();
        for layer in 0..m.num_layers {
            let input_name = layer_input_name(&m, layer);
            let input = match b.read(&input_name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("   layer {layer}: cannot read {input_name}: {e}");
                    any_failed = true;
                    continue;
                }
            };
            match verify_layer(&b, &m, layer, &input, Some((&cos, &sin)), tol) {
                Ok((ck, _)) => {
                    let failed = ck.failed();
                    any_failed |= failed > 0;
                    let (label, abs) = ck.describe();
                    if verbose {
                        println!();
                        println!("   ---- layer {layer} ({}) ----", m.layer_type(layer));
                        print_checks(&ck);
                    }
                    rows.push(Row {
                        layer,
                        ltype: m.layer_type(layer).to_string(),
                        note: String::new(),
                        label,
                        abs,
                        out_abs: ck.out_abs(),
                        failed,
                        ok: failed == 0,
                        skipped: false,
                    });
                }
                Err(e) => {
                    eprintln!("   layer {layer}: {e}");
                    any_failed = true;
                }
            }
        }
        println!();
        print_table(&rows);
        let isolated_out: Vec<(usize, f32)> =
            rows.iter().map(|r| (r.layer, r.out_abs)).collect();

        // ---- chained ------------------------------------------------------
        if chain {
            println!();
            println!("   mode: chained (each layer's own output feeds the next)");
            println!();
            let mut chain_rows: Vec<Row> = Vec::new();
            let run = m.chain_layers();
            let first = layer_input_name(&m, run[0]);
            match b.read(&first) {
                Ok(mut input) => {
                for &layer in &run {
                    match verify_layer(&b, &m, layer, &input, Some((&cos, &sin)), tol) {
                        Ok((ck, tr)) => {
                            let failed = ck.failed();
                            any_failed |= failed > 0;
                            let (label, abs) = ck.describe();
                            chain_rows.push(Row {
                                layer,
                                ltype: m.layer_type(layer).to_string(),
                                note: String::new(),
                                label,
                                abs,
                                out_abs: ck.out_abs(),
                                failed,
                                ok: failed == 0,
                                skipped: false,
                            });
                            input = tr.out;
                        }
                        Err(e) => {
                            eprintln!("   layer {layer}: {e}");
                            any_failed = true;
                            break;
                        }
                    }
                }
                }
                Err(e) => {
                    eprintln!("   chain: cannot read {first}: {e}");
                    any_failed = true;
                }
            }
            print_table(&chain_rows);
            println!();
            println!("   layer-output error, chained vs isolated (the drift check)");
            for r in &chain_rows {
                let base = isolated_out
                    .iter()
                    .find(|(l, _)| *l == r.layer)
                    .map(|(_, a)| *a)
                    .unwrap_or(f32::NAN);
                let ratio = if base > 0.0 { r.out_abs / base } else { f32::NAN };
                println!(
                    "     layer {:<3} chained={:<11.3e} isolated={:<11.3e} ratio={:.2}",
                    r.layer, r.out_abs, base, ratio
                );
            }
        }
    }

    // ---- the whole model --------------------------------------------------
    if do_model {
        println!();
        println!("   mode: whole model (embedding -> all layers -> final norm -> head)");
        let mw = match loader::load_model_weights(&b, &m) {
            Ok(mut mw) => {
                mw.layers = match loader::load_all_layers(&b, &m) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("cannot load layers: {e}");
                        return ExitCode::FAILURE;
                    }
                };
                mw
            }
            Err(e) => {
                eprintln!("cannot load model weights: {e}");
                return ExitCode::FAILURE;
            }
        };

        let prompt: Vec<u32> = b.manifest.prompt_ids.clone();
        let mcfg = m.model_config();
        let tr = match model::forward(&mcfg, &mw, &prompt) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("forward failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut ck = Checker::new(&b, tol);
        ck.step("embedding", &tr.embedding, "model__embed_tokens");
        for (i, lt) in tr.layers.iter().enumerate() {
            ck.step(
                &format!("layer {i} output"),
                &lt.out,
                &format!("model__layers__{i}"),
            );
        }
        ck.step("final norm", &tr.final_norm, "model__norm");
        ck.step("logits (all positions)", &tr.logits, "lm_head");
        print_checks(&ck);
        any_failed |= ck.failed() > 0;

        // ---- greedy -------------------------------------------------------
        let steps = b.manifest.greedy.steps.len();
        let use_cache = args.iter().any(|a| a == "--cached");
        if use_cache {
            if let Ok(cz) = model::Cache::new(&mcfg, &mw, 1) {
                let (lin, full) = cz.bytes_by_kind();
                println!(
                    "   cache: linear {:.2} MiB (constant) + full {:.2} MiB at len 0",
                    lin as f64 / 1048576.0,
                    full as f64 / 1048576.0
                );
            }
        }
        let (gen, traces) = match if use_cache {
            model::greedy_cached(&mcfg, &mw, &prompt, steps)
        } else {
            model::greedy(&mcfg, &mw, &prompt, steps)
        } {
            Ok(g) => g,
            Err(e) => {
                eprintln!("greedy failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!();
        println!(
            "   greedy trace ({steps} steps, {})",
            if use_cache { "cached" } else { "uncached" }
        );
        let mut token_ok = 0usize;
        let mut worst_logit = 0f32;
        for (k, st) in b.manifest.greedy.steps.iter().enumerate() {
            let want_name = format!("greedy_step{k:02}__logits");
            let (logit_abs, topk_ok) = match b.read(&want_name) {
                Ok(golden) => {
                    let mine = traces[k].last_logits();
                    let (a, _) = max_abs_diff(mine, &golden);
                    worst_logit = worst_logit.max(a);
                    let gtok: Vec<usize> = st.topk_tokens.iter().map(|x| *x as usize).collect();
                    let mut mine_top: Vec<(usize, f32)> =
                        mine.iter().copied().enumerate().collect();
                    mine_top.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                    let mine_top: Vec<usize> = mine_top.iter().take(gtok.len()).map(|(i, _)| *i).collect();
                    (a, gtok == mine_top)
                }
                Err(_) => (f32::NAN, false),
            };
            let got = gen[k] as usize;
            let want = st.argmax_token as usize;
            let ok = got == want;
            if ok {
                token_ok += 1;
            }
            println!(
                "     step {k:>2}  argmax want={want:<4} got={got:<4} {status:<9} logits abs={logit_abs:.3e}  top{n_top} {topk_status:<6} input_len={ilen}",
                status = if ok { "ok" } else { "MISMATCH" },
                n_top = st.topk_tokens.len(),
                topk_status = if topk_ok { "same" } else { "DIFFER" },
                ilen = traces[k].t
            );
        }
        println!();
        println!(
            "     tokens: {token_ok}/{steps} match   worst logit abs={worst_logit:.3e}"
        );
        any_failed |= token_ok != steps || worst_logit > tol;

        let want_ids: Vec<u32> = b.manifest.greedy.final_ids.clone();
        let mut mine_ids = prompt.clone();
        mine_ids.extend(gen.iter().copied());
        let ids_ok = mine_ids == want_ids;
        any_failed |= !ids_ok;
        if ids_ok {
            println!("     full token sequence: identical");
        } else {
            println!("     full token sequence: DIFFERS");
            println!("       want {want_ids:?}");
            println!("       got  {mine_ids:?}");
        }
    }

    println!();
    if any_failed {
        println!("   RESULT: FAIL");
        ExitCode::FAILURE
    } else {
        println!("   RESULT: PASS");
        ExitCode::SUCCESS
    }
}
