//! `bundlecmp` — compare a candidate implementation against a golden bundle.
//!
//! ```text
//! bundlecmp summary  <bundle>                       list tensors, tokens, config
//! bundlecmp show     <bundle> <tensor>              print a tensor's stats
//! bundlecmp compare  <golden> <candidate> [opts]    per-tensor + token diff
//! bundlecmp selftest <golden>                       golden vs itself (must be exact)
//! ```
//!
//! `compare` options:
//!   --group <name>    only compare this group (weights / intermediates / units)
//!   --max-rel <f>     relative tolerance for PASS (default 1e-5)
//!   --top <n>         show the N worst tensors (default 12)
//!
//! The candidate must use the same layout and the same tensor names. Writing it
//! is a matter of dumping your intermediates with the bundle writer; the naming
//! rule is the module path with `.` replaced by `__`.

use std::process::ExitCode;

use goldenbundle::{check_token_selfconsistency, diff, diff_tokens, Bundle};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         bundlecmp summary <bundle>\n  \
         bundlecmp show    <bundle> <tensor>\n  \
         bundlecmp compare <golden> <candidate> [--group G] [--max-rel F] [--top N]\n  \
         bundlecmp selftest <golden>"
    );
    ExitCode::from(2)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn positional(args: &[String], n: usize) -> Option<&String> {
    // Skip the subcommand and any flag/value pairs.
    let value_flags = ["--group", "--max-rel", "--top"];
    let mut out = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i].starts_with("--") {
            if value_flags.contains(&args[i].as_str()) {
                i += 1;
            }
        } else {
            out.push(&args[i]);
        }
        i += 1;
    }
    out.get(n).copied()
}

fn cmd_summary(args: &[String]) -> ExitCode {
    let Some(dir) = positional(args, 0) else { return usage() };
    let b = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("== {}", b.root.display());
    println!("   schema {} | dtype {} | byte_order {}", b.manifest.schema,
             b.manifest.dtype, b.manifest.byte_order);
    println!("   params {} | ssm_gain {} | seed {}",
             b.manifest.source.params, b.manifest.source.ssm_gain, b.manifest.source.seed);
    if let Some(t) = &b.manifest.source.transformers {
        println!("   transformers {} | torch {}",
                 t, b.manifest.source.torch.as_deref().unwrap_or("?"));
    }

    let mut groups: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for t in &b.manifest.tensors {
        let e = groups.entry(t.group.as_str()).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.nbytes;
    }
    println!("\n   groups:");
    for (g, (n, bytes)) in &groups {
        println!("     {:<16} {:>5} tensors  {:>10.1} KiB", g, n, *bytes as f64 / 1024.0);
    }

    println!("\n   prompt_ids: {:?}", b.manifest.prompt_ids);
    println!("   greedy steps: {}", b.manifest.greedy.steps.len());
    for s in b.manifest.greedy.steps.iter().take(6) {
        println!("     step {:>2}: argmax={:<4} topk={:?}", s.step, s.argmax_token, s.topk_tokens);
    }
    if b.manifest.greedy.steps.len() > 6 {
        println!("     ... {} more", b.manifest.greedy.steps.len() - 6);
    }
    println!("   final_ids: {:?}", b.manifest.greedy.final_ids);
    ExitCode::SUCCESS
}

fn cmd_show(args: &[String]) -> ExitCode {
    let (Some(dir), Some(name)) = (positional(args, 0), positional(args, 1)) else {
        return usage();
    };
    let b = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(entry) = b.entry(name) else {
        eprintln!("no tensor {name} in bundle");
        let prefix: Vec<&String> = b.names().filter(|n| n.contains(name.as_str())).take(10).collect();
        if !prefix.is_empty() {
            eprintln!("did you mean:");
            for p in prefix {
                eprintln!("   {p}");
            }
        }
        return ExitCode::FAILURE;
    };
    let v = match b.read_entry(entry) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let absmax = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
    println!("== {name}");
    println!("   group {}  shape {:?}  dtype {}", entry.group, entry.shape, entry.dtype);
    println!("   numel {}  nbytes {}", entry.numel, entry.nbytes);
    println!("   absmax {absmax:.6e}  mean|x| {mean:.6e}");
    println!("   first 8: {:?}", &v[..v.len().min(8)]);
    ExitCode::SUCCESS
}

fn cmd_compare(args: &[String]) -> ExitCode {
    let (Some(gdir), Some(cdir)) = (positional(args, 0), positional(args, 1)) else {
        return usage();
    };
    let group = flag(args, "--group").map(str::to_string);
    let max_rel: f32 = flag(args, "--max-rel").and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let top: usize = flag(args, "--top").and_then(|s| s.parse().ok()).unwrap_or(12);

    let g = match Bundle::open(gdir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("golden: {e}");
            return ExitCode::FAILURE;
        }
    };
    let c = match Bundle::open(cdir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("candidate: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("== {} vs {}", g.root.display(), c.root.display());
    println!("   tolerance: max_rel {max_rel:.1e}   group {}", group.as_deref().unwrap_or("<all>"));

    let mut diffs = Vec::new();
    let mut compared = 0usize;
    let mut identical = 0usize;
    let mut missing = Vec::new();

    for entry in &g.manifest.tensors {
        if let Some(gr) = &group {
            if &entry.group != gr {
                continue;
            }
        }
        let Some(centry) = c.entry(&entry.name) else {
            missing.push(entry.name.clone());
            continue;
        };
        let Ok(gv) = g.read_entry(entry) else {
            eprintln!("   golden read failed for {}", entry.name);
            continue;
        };
        let Ok(cv) = c.read_entry(centry) else {
            eprintln!("   candidate read failed for {}", entry.name);
            continue;
        };
        compared += 1;
        match diff(&entry.name, &entry.shape, &gv, &cv) {
            None => identical += 1,
            Some(d) => diffs.push(d),
        }
    }

    diffs.sort_by(|a, b| b.max_rel.partial_cmp(&a.max_rel).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n   compared {compared} tensors: {identical} bit-identical, {} differing",
             diffs.len());
    if !missing.is_empty() {
        println!("   missing in candidate: {} (first: {:?})",
                 missing.len(), &missing[..missing.len().min(3)]);
    }

    if !diffs.is_empty() {
        println!("\n   worst {} by relative error:", top.min(diffs.len()));
        for d in diffs.iter().take(top) {
            println!("     {}", d.describe());
        }
    }

    let td = diff_tokens(&g.manifest, &c.manifest);
    println!("\n   token trace:");
    match td.first_divergence {
        None => println!("     identical over {} tokens", td.golden_ids.len()),
        Some(i) => {
            println!("     FIRST DIVERGENCE at token {i}");
            println!("       golden    {:?}", &td.golden_ids[..(i + 4).min(td.golden_ids.len())]);
            println!("       candidate {:?}", &td.candidate_ids[..(i + 4).min(td.candidate_ids.len())]);
        }
    }
    println!("     max top-k logit delta {:.3e}", td.max_logit_delta);

    // A candidate could record correct token ids while its logits are wrong.
    // Deriving argmax from the candidate's own logits closes that hole.
    let g_bad = check_token_selfconsistency(&g);
    let c_bad = check_token_selfconsistency(&c);
    if !g_bad.is_empty() {
        println!("\n   !! GOLDEN is internally inconsistent at {} steps: {:?}",
                 g_bad.len(), &g_bad[..g_bad.len().min(3)]);
    }
    if !c_bad.is_empty() {
        println!("\n   !! CANDIDATE token ids disagree with its own logits at {} of {} steps:",
                 c_bad.len(), c.manifest.greedy.steps.len());
        for (i, recorded, derived) in c_bad.iter().take(5) {
            println!("        step {i}: manifest says {recorded}, its logits say {derived}");
        }
        println!("      (recorded ids are not trustworthy; the logits are the ground truth)");
    }

    let worst_rel = diffs.first().map(|d| d.max_rel).unwrap_or(0.0);
    let ok = worst_rel <= max_rel
        && td.first_divergence.is_none()
        && missing.is_empty()
        && g_bad.is_empty()
        && c_bad.is_empty();
    println!(
        "\n   RESULT: {} (worst rel {:.3e} vs tol {:.1e}, tokens {}, missing {}, inconsistent {})",
        if ok { "PASS" } else { "FAIL" },
        worst_rel,
        max_rel,
        if td.first_divergence.is_none() { "match" } else { "diverge" },
        missing.len(),
        c_bad.len()
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_selftest(args: &[String]) -> ExitCode {
    let Some(dir) = positional(args, 0) else { return usage() };
    let g = match Bundle::open(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut checked = 0usize;
    let mut nonzero = 0usize;
    for entry in &g.manifest.tensors {
        let v = match g.read_entry(entry) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("   read failed {}: {e}", entry.name);
                return ExitCode::FAILURE;
            }
        };
        checked += 1;
        if let Some(d) = diff(&entry.name, &entry.shape, &v, &v) {
            nonzero += 1;
            eprintln!("   self-compare of {} is non-zero: {}", entry.name, d.max_rel);
        }
    }
    let td = diff_tokens(&g.manifest, &g.manifest);
    println!("== selftest on {}", g.root.display());
    println!("   {checked} tensors self-compared, {nonzero} non-zero");
    println!("   token trace first divergence: {:?}", td.first_divergence);
    if nonzero == 0 && td.first_divergence.is_none() {
        println!("   RESULT: PASS (the comparator reports agreement on identical input)");
        ExitCode::SUCCESS
    } else {
        println!("   RESULT: FAIL (comparator is not reflexive)");
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1) else { return usage() };
    match cmd.as_str() {
        "summary" => cmd_summary(&args),
        "show" => cmd_show(&args),
        "compare" => cmd_compare(&args),
        "selftest" => cmd_selftest(&args),
        _ => usage(),
    }
}
