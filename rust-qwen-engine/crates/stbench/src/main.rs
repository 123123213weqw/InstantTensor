//! `stbench` — cold-cache load benchmark for the Rust safetensors reader.
//!
//! ```
//! stbench list   <model_dir>
//! stbench read   <model_dir> [--depth N] [--chunk-mb N] [--buffered] [--cold]
//!                            [--repeat N] [--block N] [--gap N] [--no-direct]
//! stbench verify <model_dir> [--tensors N]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use stloader::cache;
use stloader::header::{self, cross_check_index, Dtype};
use stloader::plan;
use stloader::reader::{self, ReadConfig};
use stloader::{summarize, summarize_plan, ModelSummary};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         stbench list     <model_dir> [--block N] [--gap N]\n  \
         stbench read     <model_dir> [--depth N] [--chunk-mb N] [--buffered] [--cold]\n  \
         \x20                          [--repeat N] [--block N] [--gap N] [--per-tensor]\n  \
         stbench verify   <model_dir> [--tensors N] [--large N]\n  \
         stbench selftest [scratch_dir]\n  \
         stbench fuzz     [scratch_dir] [--iters N] [--seed N] [--read] [--verbose]"
    );
    ExitCode::from(2)
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    flag_val(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Flags that consume the following token as their value.
const VALUE_FLAGS: &[&str] = &[
    "--iters", "--seed", "--tensors", "--large", "--depth", "--chunk-mb", "--repeat",
    "--block", "--gap",
];

/// Positional arguments, skipping subcommands and flag values.
///
/// Without this, `stbench fuzz --iters 20000` would treat `20000` as the
/// scratch directory, because a naive scan for the first non-`--` token cannot
/// tell a positional from a flag's value.
fn positionals(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 2; // args[0] = argv[0], args[1] = subcommand
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 1; // skip the value
            }
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

/// First positional argument after the subcommand.
fn model_dir(args: &[String]) -> Option<PathBuf> {
    positionals(args).first().map(PathBuf::from)
}

fn print_summary(dir: &Path, s: &ModelSummary, block: u64) {
    println!("== {}", dir.display());
    println!(
        "   shards={} tensors={}  file={:.3} GiB  tensor_bytes={:.3} GiB  header={:.1} MiB",
        s.shards,
        s.tensors,
        s.file_bytes as f64 / 1024f64.powi(3),
        s.tensor_bytes as f64 / 1024f64.powi(3),
        s.header_bytes as f64 / 1024f64.powi(2),
    );
    println!(
        "   unaligned to {block}: offset {}/{} ({:.1}%)  size {}/{} ({:.1}%)",
        s.unaligned_offsets,
        s.tensors,
        100.0 * s.unaligned_offsets as f64 / s.tensors.max(1) as f64,
        s.unaligned_sizes,
        s.tensors,
        100.0 * s.unaligned_sizes as f64 / s.tensors.max(1) as f64,
    );
    println!(
        "   planned: {} requests  {:.3} GiB  amplification {:.4}x",
        s.planned_requests,
        s.planned_bytes as f64 / 1024f64.powi(3),
        s.amplification(),
    );
}

fn cmd_list(args: &[String]) -> ExitCode {
    let Some(dir) = model_dir(args) else { return usage() };
    let block: u64 = num(args, "--block", 4096);
    let gap: u64 = num(args, "--gap", 0);

    let t0 = Instant::now();
    let headers = match header::read_model(&dir) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let s = summarize(&headers, block, gap);
    print_summary(&dir, &s, block);
    println!("   header parse: {parse_ms:.1} ms");

    let dtypes = {
        let mut m = std::collections::BTreeMap::new();
        for h in &headers {
            for t in &h.tensors {
                *m.entry(t.dtype.as_str()).or_insert(0usize) += 1;
            }
        }
        m
    };
    println!("   dtypes: {dtypes:?}");

    match header::read_index(&dir) {
        Ok(Some(idx)) => {
            let (listed, present, missing, extra) = cross_check_index(&headers, &idx);
            println!("   index.json: {listed} listed, {present} present in shards");
            if !missing.is_empty() {
                println!("   index entries MISSING from shards: {} (first: {:?})", missing.len(), &missing[..missing.len().min(3)]);
            }
            if !extra.is_empty() {
                println!("   shard tensors NOT in index: {} (first: {:?})", extra.len(), &extra[..extra.len().min(3)]);
            }
            if missing.is_empty() && extra.is_empty() {
                println!("   index/shard tensor sets agree exactly");
            }
        }
        Ok(None) => println!("   index.json: absent"),
        Err(e) => println!("   index.json: error {e}"),
    }
    ExitCode::SUCCESS
}

fn cmd_selftest(args: &[String]) -> ExitCode {
    let dir = positionals(args)
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("stloader_selftest"));

    let checks = match stloader::selftest::run(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("selftest could not run: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("== synthetic robustness cases ({})", dir.display());
    let mut failed = 0usize;
    for c in &checks {
        let tag = if c.pass { "PASS" } else { "FAIL" };
        println!("   [{tag}] {:<46} {}", c.name, c.detail);
        if !c.pass {
            failed += 1;
        }
    }
    println!(
        "\n   {} cases, {} passed, {} failed",
        checks.len(),
        checks.len() - failed,
        failed
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_read(args: &[String]) -> ExitCode {
    let Some(dir) = model_dir(args) else { return usage() };
    let block = num(args, "--block", 4096usize);
    let gap = num(args, "--gap", 0u64);
    let depth = num(args, "--depth", 32usize);
    let chunk_mb = num(args, "--chunk-mb", 1usize);
    let repeat = num(args, "--repeat", 1usize);
    let buffered = has(args, "--buffered") || has(args, "--no-direct") || has(args, "--no-mmap");
    let cold = has(args, "--cold");
    let per_tensor = has(args, "--per-tensor");

    let files = match header::discover_shards(&dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let t0 = Instant::now();
    let headers = match header::read_model(&dir) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parse_s = t0.elapsed().as_secs_f64();

    // --per-tensor issues one aligned request per tensor without merging: the
    // control case that exposes what alignment padding costs.
    let plan = if per_tensor {
        plan::plan_model_unmerged(&headers, block as u64)
    } else {
        plan::plan_model(&headers, block as u64, gap)
    };
    let s = summarize_plan(&headers, block as u64, &plan);
    print_summary(&dir, &s, block as u64);
    println!("   header parse: {:.1} ms", parse_s * 1000.0);

    let cfg = ReadConfig {
        depth,
        chunk_bytes: chunk_mb * 1024 * 1024,
        direct: !buffered,
        block,
    };
    println!(
        "   config: depth={depth} chunk={chunk_mb}MiB direct={} block={block} mode={}",
        cfg.direct,
        if per_tensor { "per-tensor" } else { "coalesced" }
    );

    let mut best = f64::INFINITY;
    let mut best_stats = None;
    for i in 0..repeat {
        if cold || repeat > 1 {
            if let Err(e) = cache::drop_page_cache(&files) {
                eprintln!("   warn: drop_page_cache failed: {e}");
            }
            let r = cache::min_resident_ratio(&files);
            if i == 0 || cold {
                println!("   page-cache residency after drop: {r:.3}");
            }
        }

        let t1 = Instant::now();
        let stats = match reader::read_files(&files, &plan, &cfg) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("   error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let wall = t1.elapsed().as_secs_f64();

        println!(
            "   run {}: {:.3} GiB in {:.3}s  {:.2} GB/s  (io {:.3}s, {} requests, {} short)",
            i + 1,
            stats.gib(),
            wall,
            stats.gbps(),
            stats.seconds,
            stats.requests,
            stats.short_reads,
        );
        if wall < best {
            best = wall;
            best_stats = Some(stats);
        }
    }

    if let Some(st) = best_stats {
        println!(
            "   BEST: {:.3} GiB / {:.3}s = {:.2} GB/s (decimal)   peak RSS {:.2} GiB",
            st.gib(),
            best,
            st.bytes as f64 / 1e9 / best,
            cache::peak_rss_bytes() as f64 / 1024f64.powi(3),
        );
    }
    ExitCode::SUCCESS
}

fn cmd_verify(args: &[String]) -> ExitCode {
    let Some(dir) = model_dir(args) else { return usage() };
    let n = num(args, "--tensors", 32usize);
    let block = num(args, "--block", 4096usize);

    let headers = match header::read_model(&dir) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Small tensors are the interesting case: they are the ones that cannot be
    // block-aligned, so they exercise the padding logic.
    let mut cands: Vec<(&PathBuf, &stloader::TensorInfo, u64)> = Vec::new();
    for h in &headers {
        for t in &h.tensors {
            if t.nbytes() <= 64 * 1024 && t.nbytes() > 0 {
                cands.push((&h.path, t, h.data_start));
            }
        }
    }
    cands.sort_by_key(|(_, t, _)| t.nbytes());

    let mut checked = 0usize;
    let mut failed = 0usize;
    for (path, t, data_start) in cands.iter().take(n) {
        let (off, end) = t.abs_range(*data_start);
        let len = (end - off) as usize;
        let a = reader::read_small_range(path, off, len, true, block);
        let b = reader::read_small_range(path, off, len, false, block);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                if a != b {
                    failed += 1;
                    let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                    println!(
                        "   MISMATCH {} {} ({}) off={} len={} first_diff={:?}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        t.name,
                        t.dtype.as_str(),
                        off,
                        len,
                        first
                    );
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                failed += 1;
                println!("   ERROR {} {}: {e}", path.display(), t.name);
            }
        }
    }

    // Also spot-check a few F32 scalars, since those are the tensors whose byte
    // count is least likely to be a block multiple.
    let mut f32_small = 0usize;
    for h in &headers {
        for t in &h.tensors {
            if t.dtype == Dtype::F32 && t.nbytes() <= 8192 && f32_small < 8 {
                let (off, end) = t.abs_range(h.data_start);
                let len = (end - off) as usize;
                let a = reader::read_small_range(&h.path, off, len, true, block);
                let b = reader::read_small_range(&h.path, off, len, false, block);
                if let (Ok(a), Ok(b)) = (a, b) {
                    checked += 1;
                    f32_small += 1;
                    if a != b {
                        failed += 1;
                        println!("   MISMATCH (F32 small) {} len={len}", t.name);
                    }
                }
            }
        }
    }

    // Large tensors carry almost all the bytes; small-tensor checks alone would
    // miss a bug in the bulk path. Compare the head of each of the N largest.
    let large_n = num(args, "--large", 4usize);
    let mut big: Vec<(&PathBuf, &stloader::TensorInfo, u64)> = Vec::new();
    for h in &headers {
        for t in &h.tensors {
            if t.nbytes() > 64 * 1024 {
                big.push((&h.path, t, h.data_start));
            }
        }
    }
    big.sort_by_key(|(_, t, _)| std::cmp::Reverse(t.nbytes()));
    for (path, t, data_start) in big.iter().take(large_n) {
        let (off, end) = t.abs_range(*data_start);
        let len = ((end - off) as usize).min(4 << 20);
        let a = reader::read_small_range(path, off, len, true, block);
        let b = reader::read_small_range(path, off, len, false, block);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                if a != b {
                    failed += 1;
                    let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                    println!(
                        "   MISMATCH (large) {} {} nbytes={} first {len}B diff={:?}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        t.name,
                        t.nbytes(),
                        first
                    );
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                failed += 1;
                println!("   ERROR (large) {} {}: {e}", path.display(), t.name);
            }
        }
    }

    println!("   verified {checked} tensor reads, {failed} mismatches");
    println!("   coverage: {} smallest (<=64 KiB) + {} largest (first 4 MiB each)", n.min(checked), large_n.min(big.len()));
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_fuzz(args: &[String]) -> ExitCode {
    let iters = num(args, "--iters", 20_000usize);
    let seed = num(args, "--seed", 0x5EED_1234u64);
    let verbose = has(args, "--verbose");
    // Reading is slower, so it is opt-in; parsing is where most bugs live.
    let do_reads = has(args, "--read") || has(args, "--with-reads");
    let dir = positionals(args)
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("stloader_fuzz"));

    println!(
        "== fuzzing ({iters} iterations, seed {seed}, reads {}, scratch {})",
        if do_reads { "on" } else { "off" },
        dir.display()
    );

    let t0 = Instant::now();
    let report = match stloader::fuzz::run(iters, seed, &dir, do_reads, verbose) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("fuzzer could not run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dt = t0.elapsed().as_secs_f64();

    println!(
        "   {} cases in {dt:.2}s ({:.0}/s): {} parsed, {} correctly rejected, {} reads executed",
        report.iterations,
        report.iterations as f64 / dt.max(1e-9),
        report.parsed_ok,
        report.rejected,
        report.reads_executed,
    );
    if !report.panics.is_empty() {
        println!("\n   PANICS: {}", report.panics.len());
        for p in report.panics.iter().take(10) {
            println!("     {p}");
        }
    }
    if !report.invariant_failures.is_empty() {
        println!("\n   INVARIANT FAILURES: {}", report.invariant_failures.len());
        for p in report.invariant_failures.iter().take(10) {
            println!("     {p}");
        }
    }
    println!(
        "\n   dtype table consistent: {}",
        stloader::fuzz::dtype_table_is_consistent()
    );

    if report.ok() {
        println!("   RESULT: clean (no panics, no invariant violations)");
        ExitCode::SUCCESS
    } else {
        println!("   RESULT: FAILURES FOUND");
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(cmd) = args.get(1) else { return usage() };
    match cmd.as_str() {
        "list" => cmd_list(&args),
        "read" => cmd_read(&args),
        "verify" => cmd_verify(&args),
        "selftest" => cmd_selftest(&args),
        "fuzz" => cmd_fuzz(&args),
        _ => usage(),
    }
}
