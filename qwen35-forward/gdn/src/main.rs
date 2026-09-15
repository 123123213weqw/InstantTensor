//! `gdncheck` — run the qwen35 decoder stack against a golden bundle.
//!
//! ```text
//! gdncheck <bundle>                 every verifiable layer, each fed the golden
//!                                   input recorded for that layer
//! gdncheck <bundle> --layer N       one layer, all checks printed
//! gdncheck <bundle> --chain         feed each layer's own output forward
//! gdncheck <bundle> --verbose       all checks for every layer
//! ```
//!
//! # Two ways to check a layer, and why both
//!
//! **Isolated** (the default) feeds each layer the input the *reference* produced
//! for it, so an error in layer 2 cannot contaminate the verdict on layer 4. Errors
//! point at the layer that caused them.
//!
//! **Chained** (`--chain`) feeds each layer's own output into the next. An error in
//! layer 2 shows up in layer 4 too, so it localises worse, but it is the only way to
//! show that the stack works end to end and that rounding does not accumulate.
//!
//! A full-attention layer breaks a chain: its output cannot be computed yet, so the
//! layer after it has no input this implementation can produce. `--chain` therefore
//! runs over maximal *runs* of linear layers, starting each run from the golden
//! input of its first layer.

use std::process::ExitCode;

use gdn::layer::{layer_forward, LayerTrace, Mixer};
use gdn::loader::{self, LayerCapture, ModelInfo};
use gdn::GdnConfig;
use goldenbundle::Bundle;

const TOL: f32 = 1e-5;

struct Check {
    label: String,
    abs: f32,
    rel: f32,
    ok: bool,
    /// Golden tensor absent from the bundle — reported, not counted as a failure.
    missing: bool,
}

struct Checker<'a> {
    b: &'a Bundle,
    checks: Vec<Check>,
}

impl Checker<'_> {
    fn new(b: &Bundle) -> Checker<'_> {
        Checker { b, checks: Vec::new() }
    }

    fn cmp_opt(&mut self, label: &str, mine: &[f32], name: Option<String>) {
        let Some(name) = name else {
            // Not applicable to this layer kind: a full-attention layer has no
            // convolution and no delta rule, so there is nothing to compare.
            self.checks.push(Check {
                label: label.to_string(),
                abs: f32::NAN,
                rel: f32::NAN,
                ok: true,
                missing: true,
            });
            return;
        };
        match self.b.read(&name) {
            Ok(golden) => self.compare(label, mine, &golden),
            // Applicable but absent. The layer is a linear-attention layer, so the
            // bundle is *supposed* to hold this tensor, and treating its absence as
            // a pass silently drops the check: under a capture-index bug, layer 6
            // looked up names that do not exist and nine checks vanished without a
            // word. Absence is a failure.
            Err(e) => self.checks.push(Check {
                label: format!("{label} MISSING {name} ({e})"),
                abs: f32::INFINITY,
                rel: f32::INFINITY,
                ok: false,
                missing: false,
            }),
        }
    }

    fn cmp(&mut self, label: &str, mine: &[f32], name: &str) {
        self.cmp_opt(label, mine, Some(name.to_string()));
    }

    fn compare(&mut self, label: &str, mine: &[f32], golden: &[f32]) {
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
        let (abs, _at) = max_abs_diff(mine, golden);
        let scale = golden.iter().fold(0f32, |m, x| m.max(x.abs()));
        let rel = if scale > 0.0 { abs / scale } else { abs };
        self.checks.push(Check {
            label: label.to_string(),
            abs,
            rel,
            ok: abs <= TOL,
            missing: false,
        });
    }

    fn failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.ok && !c.missing).count()
    }

    /// The first failing check: the earliest point in the chain that diverged.
    fn first_failure(&self) -> Option<&Check> {
        self.checks.iter().find(|c| !c.ok && !c.missing)
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

/// Run one layer and compare all of its captured intermediates.
fn verify_layer<'a>(
    b: &'a Bundle,
    m: &ModelInfo,
    layer: usize,
    input: &[f32],
) -> Result<(Checker<'a>, LayerTrace), String> {
    let c = LayerCapture::new(layer, m.ssm_ordinal(layer));
    let lw = loader::load_layer_weights(b, m, layer)?;
    let gw = loader::load_gdn_weights(b, m, layer)?;
    let cfg: GdnConfig = m.gdn;
    let tr = layer_forward(
        Mixer::LinearAttention(&cfg, &gw),
        &lw,
        input,
        m.b,
        m.t,
        m.eps,
    )?;

    let mut ck = Checker::new(b);
    let g = &tr.linear_attn;

    ck.cmp("1. input_layernorm", &tr.input_layernorm, &c.input_layernorm());
    ck.cmp("2. in_proj_qkv", &g.in_proj_qkv, &c.in_proj_qkv());
    ck.cmp_opt("3. conv input (channels-first)", &g.conv_in, c.conv_in());
    ck.cmp_opt("4. conv + silu", &g.conv_out, c.conv_out());
    ck.cmp("5. in_proj_z", &g.in_proj_z, &c.in_proj_z());
    ck.cmp("6. in_proj_b", &g.in_proj_b, &c.in_proj_b());
    ck.cmp("7. in_proj_a", &g.in_proj_a, &c.in_proj_a());
    ck.cmp_opt("8. q (post-GQA)", &g.q, c.delta_operand("q"));
    ck.cmp_opt("9. k (post-GQA)", &g.k, c.delta_operand("k"));
    ck.cmp_opt("10. v", &g.v, c.delta_operand("v"));
    ck.cmp_opt("11. g (decay)", &g.g, c.delta_operand("g"));
    ck.cmp_opt("12. beta (gate)", &g.beta, c.delta_operand("beta"));
    ck.cmp_opt("13. delta rule out", &g.delta_out, c.delta_out());
    ck.cmp_opt("14. delta rule state", &g.delta_state, c.delta_state());
    ck.cmp("15. gated norm", &g.norm, &c.mixer_norm());
    ck.cmp("16. out_proj", &g.out_proj, &c.out_proj());
    ck.cmp("17. block output", &g.out_proj, &c.linear_attn());
    ck.cmp(
        "18. post_attention_layernorm",
        &tr.post_attention_layernorm,
        &c.post_attention_layernorm(),
    );
    ck.cmp("19. mlp gate_proj", &tr.mlp.gate_proj, &c.mlp_gate_proj());
    ck.cmp("20. mlp up_proj", &tr.mlp.up_proj, &c.mlp_up_proj());
    ck.cmp("21. mlp swiglu product", &tr.mlp.swiglu_product, &c.mlp_swiglu());
    ck.cmp("22. mlp down_proj", &tr.mlp.down_proj, &c.mlp_down_proj());
    ck.cmp("23. mlp output", &tr.mlp.down_proj, &c.mlp());
    ck.cmp("24. layer output (both residuals)", &tr.out, &c.layer_out());

    Ok((ck, tr))
}

fn print_checks(ck: &Checker<'_>) {
    for c in &ck.checks {
        if c.missing {
            println!("   {:<44} (golden missing)", c.label);
        } else {
            println!(
                "   {:<44} abs={:<11.3e} rel={:<11.3e} {}",
                c.label,
                c.abs,
                c.rel,
                if c.ok { "ok" } else { "FAIL" }
            );
        }
    }
}

/// One row of the multi-layer table.
struct Row {
    layer: usize,
    ltype: String,
    note: String,
    worst_label: String,
    worst_abs: f32,
    /// Error on the layer's own output. This is the number that would accumulate
    /// along a chain, so the isolated and chained tables can be compared directly.
    out_abs: f32,
    failed: usize,
    ok: bool,
    skipped: bool,
}

/// Pull the label and error of a check by its numeric prefix.
fn pick(ck: &Checker<'_>, prefix: &str) -> (String, f32) {
    ck.checks
        .iter()
        .find(|c| c.label.starts_with(prefix))
        .map(|c| (c.label.clone(), c.abs))
        .unwrap_or_default()
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
            r.worst_label,
            r.worst_abs,
            r.out_abs,
            if r.ok { "ok".to_string() } else { format!("FAIL ({} checks)", r.failed) }
        );
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let val = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };

    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("usage: gdncheck <bundle> [--layer N] [--chain] [--verbose]");
        return ExitCode::from(2);
    };
    let verbose = flag("--verbose");
    let chain = flag("--chain");
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
    println!("== qwen35 decoder stack vs {}", b.root.display());
    println!(
        "   B={} T={} hidden={} intermediate={} layers={} ({} linear_attention, {} full_attention)",
        m.b, m.t, m.hidden, m.intermediate, m.num_layers, lin, full
    );
    println!(
        "   eps={:e} ssm_gain={} k_heads={} v_heads={} head_k={} head_v={} conv_k={}",
        m.eps,
        m.ssm_gain,
        m.gdn.num_k_heads,
        m.gdn.num_v_heads,
        m.gdn.head_k_dim,
        m.gdn.head_v_dim,
        m.gdn.conv_kernel
    );
    println!("   tolerance {TOL:.1e}");
    println!();

    let mut rows: Vec<Row> = Vec::new();
    let mut any_failed = false;

    // ---- single layer, full detail -----------------------------------------
    if let Some(layer) = only {
        if layer >= m.num_layers {
            eprintln!("layer {layer} is out of range (0..{})", m.num_layers);
            return ExitCode::from(2);
        }
        if !m.is_linear(layer) {
            println!("   layer {layer} is {} — not implemented", m.layer_type(layer));
            println!();
            println!("   RESULT: SKIPPED");
            return ExitCode::from(3);
        }
        let input_name = m.input_name(layer);
        let input = match b.read(&input_name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cannot read {input_name}: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("   layer {layer}   input: {input_name}");
        match verify_layer(&b, &m, layer, &input) {
            Ok((ck, _)) => {
                print_checks(&ck);
                let failed = ck.failed();
                any_failed |= failed > 0;
                println!();
                if failed == 0 {
                    println!("   RESULT: PASS ({} checks)", ck.checks.len());
                } else {
                    println!("   RESULT: FAIL ({failed} of {} checks)", ck.checks.len());
                }
            }
            Err(e) => {
                eprintln!("   layer {layer}: {e}");
                return ExitCode::FAILURE;
            }
        }
        return if any_failed { ExitCode::FAILURE } else { ExitCode::SUCCESS };
    }

    // ---- isolated: every layer against its own recorded input ---------------
    println!("   mode: isolated (each layer fed the golden input recorded for it)");
    for layer in 0..m.num_layers {
        if !m.is_linear(layer) {
            rows.push(Row {
                layer,
                ltype: m.layer_type(layer).to_string(),
                note: "— not implemented (full_attention) —".to_string(),
                worst_label: String::new(),
                worst_abs: 0.0,
                out_abs: 0.0,
                failed: 0,
                ok: true,
                skipped: true,
            });
            continue;
        }
        let input_name = m.input_name(layer);
        let input = match b.read(&input_name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("   layer {layer}: cannot read {input_name}: {e}");
                any_failed = true;
                continue;
            }
        };
        match verify_layer(&b, &m, layer, &input) {
            Ok((ck, _)) => {
                let failed = ck.failed();
                any_failed |= failed > 0;
                // Report the first divergence: the earliest operator that is wrong.
                let (label, abs) = match ck.first_failure() {
                    Some(f) => (format!("FAIL {}", f.label), f.abs),
                    None => {
                        let worst = ck
                            .checks
                            .iter()
                            .filter(|c| !c.missing)
                            .max_by(|x, y| x.abs.partial_cmp(&y.abs).unwrap());
                        (
                            format!(
                                "worst: {}",
                                worst.map(|c| c.label.as_str()).unwrap_or("-")
                            ),
                            worst.map(|c| c.abs).unwrap_or(0.0),
                        )
                    }
                };
                let (_, out_abs) = pick(&ck, "24.");
                if verbose {
                    println!();
                    println!("   ---- layer {layer} ({}) ----", m.layer_type(layer));
                    print_checks(&ck);
                }
                rows.push(Row {
                    layer,
                    ltype: m.layer_type(layer).to_string(),
                    note: String::new(),
                    worst_label: label,
                    worst_abs: abs,
                    out_abs,
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
        rows.iter().filter(|r| !r.skipped).map(|r| (r.layer, r.out_abs)).collect();

    // ---- chained: each layer's own output feeds the next -------------------
    if chain {
        println!();
        println!("   mode: chained (each layer's own output feeds the next)");
        println!("   runs: {:?}", m.runs());
        println!();
        let mut chain_rows: Vec<Row> = Vec::new();
        for run in m.runs() {
            let start = run[0];
            let first_input = m.input_name(start);
            let mut input = match b.read(&first_input) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("   run {run:?}: cannot read {first_input}: {e}");
                    any_failed = true;
                    continue;
                }
            };
            for &layer in &run {
                match verify_layer(&b, &m, layer, &input) {
                    Ok((ck, tr)) => {
                        let failed = ck.failed();
                        any_failed |= failed > 0;
                        let (label, abs) = match ck.first_failure() {
                            Some(f) => (format!("FAIL {}", f.label), f.abs),
                            None => {
                                let worst = ck
                                    .checks
                                    .iter()
                                    .filter(|c| !c.missing)
                                    .max_by(|x, y| x.abs.partial_cmp(&y.abs).unwrap());
                                (
                                    format!(
                                        "worst: {}",
                                        worst.map(|c| c.label.as_str()).unwrap_or("-")
                                    ),
                                    worst.map(|c| c.abs).unwrap_or(0.0),
                                )
                            }
                        };
                        let (_, out_abs) = pick(&ck, "24.");
                        chain_rows.push(Row {
                            layer,
                            ltype: m.layer_type(layer).to_string(),
                            note: String::new(),
                            worst_label: label,
                            worst_abs: abs,
                            out_abs,
                            failed,
                            ok: failed == 0,
                            skipped: false,
                        });
                        // Feed our own output forward, not the golden one.
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
        print_table(&chain_rows);

        // The question the chained mode exists to answer: does feeding our own
        // output forward make the error grow? Compare the layer-output error
        // against the isolated run, which is fed the reference's own input.
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

    println!();
    if any_failed {
        println!("   RESULT: FAIL");
        ExitCode::FAILURE
    } else {
        println!("   RESULT: PASS ({} layers)", rows.iter().filter(|r| !r.skipped).count());
        ExitCode::SUCCESS
    }
}
