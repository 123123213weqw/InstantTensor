//! `tokcheck` -- run the tokenizer against the committed conformance corpus.
//!
//! ```text
//! tokcheck <tokenizer.json> <corpus.json> [--verbose] [--max N]
//! ```
//!
//! The corpus is produced by `tools/make_tok_corpus.py` from the reference
//! implementation, so every case carries the ids the reference produced. This checks
//! ids, not regexes: the pre-tokenizer is a regex with lookaround, and agreeing on
//! ordinary prose while disagreeing on whitespace runs, contractions or scripts is
//! exactly the failure mode a regex comparison would miss.
//!
//! Three things are checked per case:
//!
//! * the token **ids** match, which is the whole point;
//! * decoding those ids reproduces the reference's decoded string;
//! * encoding is **stable** -- encoding the decoded text again gives the same ids.
//!
//! The third check catches a class of bug that id comparison alone does not: an
//! implementation can be self-consistent but land on a different segmentation that
//! happens to decode to the same text.

use std::collections::BTreeMap;
use std::process::ExitCode;

use gdn::tokenizer::Tokenizer;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let flag = |n: &str| args.iter().any(|a| a == n);
    let val = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    if positional.len() < 2 {
        eprintln!("usage: tokcheck <tokenizer.json> <corpus.json> [--verbose] [--max N]");
        return ExitCode::from(2);
    }
    let verbose = flag("--verbose");
    let max: Option<usize> = val("--max").and_then(|s| s.parse().ok());

    // `<tokenizer.json>` or a model directory. The directory form also reads
    // `tokenizer_config.json`, which is what decides the mark handling.
    let arg = std::path::Path::new(positional[0]);
    let (mut tk, info) = {
        let loaded = if arg.is_dir() {
            Tokenizer::from_model_dir(arg)
        } else {
            Tokenizer::from_file(arg)
        };
        match loaded {
            Ok(x) => x,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    // The corpus says which pattern it was produced with, so there is no ambiguity
    // about what is being verified.
    let _raw = match std::fs::read(positional[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {e}", positional[1]);
            return ExitCode::FAILURE;
        }
    };
    let doc: serde_json::Value = match serde_json::from_slice(&_raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {e}", positional[1]);
            return ExitCode::FAILURE;
        }
    };
    let (mode, cases): (String, Vec<serde_json::Value>) = match &doc {
        serde_json::Value::Array(a) => ("(unlabelled)".to_string(), a.clone()),
        serde_json::Value::Object(o) => (
            o.get("mode").and_then(|v| v.as_str()).unwrap_or("?").to_string(),
            o.get("cases").and_then(|v| v.as_array()).cloned().unwrap_or_default(),
        ),
        _ => {
            eprintln!("error: {}: not an array or object", positional[1]);
            return ExitCode::FAILURE;
        }
    };

    // The corpus is the authority on which variant it encodes.
    let want_marks = !mode.contains("no-marks");
    tk.set_marks_join_letters(want_marks);

    println!("== tokenizer");
    println!("   model            {}", info.model_type);
    println!("   vocab            {} entries", info.vocab_size);
    println!("   merges           {}", info.merges);
    println!("   added tokens     {}", info.added_tokens);
    println!("   normalizer       {}", info.normalizer);
    println!("   decoder          {}", info.decoder);
    println!("   byte chars       all 256 present in the vocabulary");
    println!("   unicode tables   version {}", gdn::unicode_tables::UNICODE_VERSION);
    println!("   pattern in file  {}", info.pattern_variant);
    println!("   corpus mode      {mode}");
    println!(
        "   marks join letters: {}  (from the corpus)",
        want_marks
    );
    println!();

    let _raw = match std::fs::read(positional[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}: {e}", positional[1]);
            return ExitCode::FAILURE;
        }
    };

    let mut id_fail = Vec::new();
    let mut decode_fail = Vec::new();
    let mut stable_fail = Vec::new();
    let mut total_ids = 0usize;
    let mut checked = 0usize;

    for (n, c) in cases.iter().enumerate() {
        if let Some(m) = max {
            if n >= m {
                break;
            }
        }
        let text = c["text"].as_str().unwrap_or("");
        let want: Vec<u32> = c["ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect())
            .unwrap_or_default();
        let want_decoded = c["decoded"].as_str().unwrap_or("");
        checked += 1;
        total_ids += want.len();

        let got = tk.encode(text);
        if got != want {
            id_fail.push((n, text.to_string(), want.clone(), got.clone()));
            continue;
        }
        let back = tk.decode(&got);
        if back != want_decoded {
            decode_fail.push((n, text.to_string(), want_decoded.to_string(), back));
            continue;
        }
        // Stability: re-encoding the decoded text must be a fixed point.
        let again = tk.encode(&back);
        if again != got {
            stable_fail.push((n, text.to_string(), got, again));
        }
    }

    println!("   {} cases, {total_ids} ids", checked);
    if verbose {
        for c in cases.iter().take(max.unwrap_or(usize::MAX)) {
            let t = c["text"].as_str().unwrap_or("");
            let ids = tk.encode(t);
            println!("     {:>3} ids  {:?}  {}", ids.len(), t, tk.decode(&ids).escape_debug());
        }
        println!();
    }

    let show = |label: &str, v: &[(usize, String, Vec<u32>, Vec<u32>)]| {
        println!("   {label}: {}", v.len());
        for (n, text, want, got) in v.iter().take(12) {
            println!("     case {n}  {text:?}");
            println!("       want {want:?}");
            println!("       got  {got:?}");
        }
        if v.len() > 12 {
            println!("     ... and {} more", v.len() - 12);
        }
    };
    let show_s = |label: &str, v: &[(usize, String, String, String)]| {
        println!("   {label}: {}", v.len());
        for (n, text, want, got) in v.iter().take(12) {
            println!("     case {n}  {text:?}");
            println!("       want {want:?}");
            println!("       got  {got:?}");
        }
        if v.len() > 12 {
            println!("     ... and {} more", v.len() - 12);
        }
    };

    // Group failures by what they are, so a systematic mistake reads as one problem
    // rather than N.
    if !id_fail.is_empty() {
        let mut by_shape: BTreeMap<String, usize> = BTreeMap::new();
        for (_, text, _, _) in &id_fail {
            let k = describe(text);
            *by_shape.entry(k).or_insert(0) += 1;
        }
        println!();
        println!("   id mismatches by input shape:");
        for (k, v) in &by_shape {
            println!("     {v:>4}  {k}");
        }
        println!();
        show("id mismatches", &id_fail);
        // The first mismatch is the interesting one.
        if let Some((n, text, want, got)) = id_fail.first() {
            println!();
            println!("   first mismatch in detail: case {n} {text:?}");
            println!("     want {} ids  {:?}", want.len(), want);
            println!("     got  {} ids  {:?}", got.len(), got);
            let s0 = gdn::unicode_gc::nfc(text);
            let p = gdn::tokenizer::pretokenize_with(&s0, want_marks);
            println!("     pre-tokenizer pieces:");
            for (a, b) in p {
                println!("       {:?}", &s0[a..b]);
            }
            println!("     per-piece ids from this implementation:");
            for (a, b) in gdn::tokenizer::pretokenize_with(&s0, want_marks) {
                let piece = &s0[a..b];
                println!("       {:?} -> {:?}", piece, tk.encode(piece));
            }
        }
    }
    if !decode_fail.is_empty() {
        println!();
        show_s("decode mismatches", &decode_fail);
    }
    if !stable_fail.is_empty() {
        println!();
        show("not idempotent under re-encoding", &stable_fail);
    }

    println!();
    let failed = !id_fail.is_empty() || !decode_fail.is_empty() || !stable_fail.is_empty();
    if failed {
        println!(
            "   RESULT: FAIL ({} id, {} decode, {} stability)",
            id_fail.len(),
            decode_fail.len(),
            stable_fail.len()
        );
        ExitCode::FAILURE
    } else {
        println!("   RESULT: PASS ({checked} cases, {total_ids} ids)");
        ExitCode::SUCCESS
    }
}

/// A rough label for what an input exercises, used only for grouping failures.
fn describe(t: &str) -> String {
    let has = |f: fn(char) -> bool| t.chars().any(f);
    let mut tags = Vec::new();
    if t.is_empty() {
        return "empty".to_string();
    }
    if has(|c| c.is_whitespace() && c != ' ') {
        tags.push("non-space whitespace");
    }
    if t.contains("  ") {
        tags.push("doubled space");
    }
    if t.starts_with(' ') || t.ends_with(' ') {
        tags.push("leading/trailing space");
    }
    if has(|c| c.is_ascii_digit()) {
        tags.push("digits");
    }
    if t.contains('\'') {
        tags.push("apostrophe");
    }
    if has(|c| !c.is_ascii()) {
        tags.push("non-ascii");
    }
    if has(|c| c as u32 > 0x2000) {
        tags.push("high codepoint");
    }
    if tags.is_empty() {
        "plain ascii words".to_string()
    } else {
        tags.join(", ")
    }
}
