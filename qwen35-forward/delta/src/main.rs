//! `deltacheck` — run the delta rule against the golden bundle's unit cases.
//!
//! ```text
//! deltacheck <bundle>            check every delta_* case
//! deltacheck <bundle> --list     list the cases and their shapes
//! ```
//!
//! For each case the bundle holds `q k v g beta` and **two** expected results:
//! the recurrent form and the chunked (prefill) form. Both are compared, so the
//! implementation has to satisfy two independently computed targets; the
//! generator already asserts those two agree with each other to ~1e-8.

use std::collections::BTreeSet;
use std::process::ExitCode;

use deltarule::{forward_prepared, Shape};
use goldenbundle::Bundle;

/// Tolerance: the reference is fp32 accumulated in fp32, and the bundle's own
/// recurrent-vs-chunked agreement is ~1e-7. 1e-5 leaves room for a different
/// summation order without admitting a real error.
const TOL: f32 = 1e-5;

struct Case {
    tag: String,
    shape: Shape,
}

fn discover(b: &Bundle) -> Vec<Case> {
    let mut tags = BTreeSet::new();
    for name in b.names() {
        if let Some(rest) = name.strip_prefix("delta_") {
            if let Some((tag, _)) = rest.split_once("__") {
                tags.insert(tag.to_string());
            }
        }
    }

    let mut out = Vec::new();
    for tag in tags {
        // Tag format: B{b}_H{h}_T{t}_K{k}_V{v}
        let mut b_ = 0usize;
        let mut h = 0usize;
        let mut t = 0usize;
        let mut k = 0usize;
        let mut v = 0usize;
        for part in tag.split('_') {
            let (key, val) = part.split_at(1);
            let n: usize = match val.parse() {
                Ok(n) => n,
                Err(_) => continue,
            };
            match key {
                "B" => b_ = n,
                "H" => h = n,
                "T" => t = n,
                "K" => k = n,
                "V" => v = n,
                _ => {}
            }
        }
        out.push(Case { tag, shape: Shape { b: b_, t, h, k, v } });
    }
    out
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

fn max_abs(a: &[f32]) -> f32 {
    a.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.iter().skip(1).find(|a| !a.starts_with("--")) else {
        eprintln!("usage: deltacheck <bundle> [--list]");
        return ExitCode::from(2);
    };
    let list_only = args.iter().any(|a| a == "--list");

    let b = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let cases = discover(&b);
    if cases.is_empty() {
        eprintln!("no delta_* tensors in bundle");
        return ExitCode::FAILURE;
    }

    println!("== delta rule vs {}", b.root.display());
    println!("   tolerance {TOL:.1e}   {} case(s)", cases.len());

    if list_only {
        for c in &cases {
            println!(
                "   {:<22} B={} T={} H={} K={} V={}",
                c.tag, c.shape.b, c.shape.t, c.shape.h, c.shape.k, c.shape.v
            );
        }
        return ExitCode::SUCCESS;
    }

    let mut failures = 0usize;

    for c in &cases {
        let s = &c.shape;
        let g = |nm: &str| -> Result<Vec<f32>, String> {
            b.read(&format!("delta_{}__{nm}", c.tag))
                .map_err(|e| format!("{nm}: {e}"))
        };

        let (q, k, v, gg, beta) = match (g("q"), g("k"), g("v"), g("g"), g("beta")) {
            (Ok(a), Ok(b2), Ok(c2), Ok(d), Ok(e)) => (a, b2, c2, d, e),
            (qa, kb, vc, gd, be) => {
                for r in [qa.err(), kb.err(), vc.err(), gd.err(), be.err()].into_iter().flatten() {
                    eprintln!("   {} {r}", c.tag);
                }
                failures += 1;
                continue;
            }
        };

        // Dimension check up front: a wrong shape would otherwise show up as a
        // mysterious numeric error.
        let want_q = s.b * s.t * s.h * s.k;
        let want_v = s.b * s.t * s.h * s.v;
        let want_g = s.b * s.t * s.h;
        if q.len() != want_q || k.len() != want_q || v.len() != want_v
            || gg.len() != want_g || beta.len() != want_g
        {
            eprintln!(
                "   {} SHAPE MISMATCH: q={} k={} v={} g={} beta={}  (expected {} {} {} {} {})",
                c.tag, q.len(), k.len(), v.len(), gg.len(), beta.len(),
                want_q, want_q, want_v, want_g, want_g
            );
            failures += 1;
            continue;
        }

        let (out, state) = forward_prepared(s, &q, &k, &v, &gg, &beta);

        let mut worst_out = 0f32;
        let mut worst_state = 0f32;
        let mut ok = true;

        for form in ["recurrent", "chunked"] {
            match b.read(&format!("delta_{}__{form}_out", c.tag)) {
                Ok(expect) => {
                    let (d, at) = max_abs_diff(&out, &expect);
                    let scale = max_abs(&expect);
                    let rel = if scale > 0.0 { d / scale } else { d };
                    let pass = d <= TOL;
                    ok &= pass;
                    println!(
                        "   {:<22} {:>9} out   abs={:<11.3e} rel={:<11.3e} {}",
                        c.tag, form, d, rel, if pass { "ok" } else { "FAIL" }
                    );
                    if !pass {
                        println!("        worst at index {at}  mine={} golden={}", out[at], expect[at]);
                    }
                    worst_out = worst_out.max(d);
                }
                Err(e) => {
                    eprintln!("   {} {form}_out: {e}", c.tag);
                    ok = false;
                }
            }
            match b.read(&format!("delta_{}__{form}_state", c.tag)) {
                Ok(expect) => {
                    let (d, at) = max_abs_diff(&state, &expect);
                    let scale = max_abs(&expect);
                    let rel = if scale > 0.0 { d / scale } else { d };
                    let pass = d <= TOL;
                    ok &= pass;
                    println!(
                        "   {:<22} {:>9} state abs={:<11.3e} rel={:<11.3e} {}",
                        c.tag, form, d, rel, if pass { "ok" } else { "FAIL" }
                    );
                    if !pass {
                        println!("        worst at index {at}  mine={} golden={}", state[at], expect[at]);
                    }
                    worst_state = worst_state.max(d);
                }
                Err(e) => {
                    eprintln!("   {} {form}_state: {e}", c.tag);
                    ok = false;
                }
            }
        }

        if !ok {
            failures += 1;
        }
        let _ = (worst_out, worst_state);
    }

    println!();
    if failures == 0 {
        println!("   RESULT: PASS ({} cases, both recurrent and chunked forms)", cases.len());
        ExitCode::SUCCESS
    } else {
        println!("   RESULT: FAIL ({failures} of {} cases)", cases.len());
        ExitCode::FAILURE
    }
}
