//! Built-in mutational fuzzer.
//!
//! A hand-written test suite only covers the cases its author thought of. This
//! module generates malformed inputs automatically so that assumptions baked
//! into `header` and `plan` get falsified without depending on imagination.
//!
//! Two things are checked on every generated case:
//!
//! 1. **No panic.** The whole parse/plan/read pipeline runs inside
//!    `catch_unwind`, so an index-out-of-bounds or an unwrap surfaces as a
//!    recorded failure instead of killing the run.
//! 2. **A real safety oracle.** If a header parses, then the plan derived from
//!    it must satisfy:
//!      * every range is block-aligned on both ends and non-empty;
//!      * every range lies within `[0, ceil(file_size, block)]`, so no read can
//!        touch bytes outside the file;
//!      * the ranges are sorted and non-overlapping;
//!      * **every tensor's absolute byte span is fully contained in a range**,
//!        so no tensor byte can be silently skipped.
//!
//! Point 2 is what makes this more than a crash hunt: a loader that returns
//! success while planning a read that misses part of a tensor is broken even
//! though it does not panic.
//!
//! # Running it
//!
//! `panic = "abort"` in the release profile would turn a caught panic into a
//! process abort, which defeats `catch_unwind`. Fuzz runs therefore use the
//! dedicated `fuzz` profile (`inherits = "release"`, `panic = "unwind"`):
//!
//! ```text
//! cargo run --profile fuzz -p stbench -- fuzz --iters 200000
//! ```

use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use crate::header::{read_header, Dtype, ShardHeader};
use crate::plan::{ceil_to, floor_to, plan_shard, Range};
use crate::reader::{read_files, ReadConfig};

/// xorshift64*, so the fuzzer is self-contained and reproducible from a seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform-ish value in `[0, n)`.
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            0
        } else {
            self.next_u64() % n
        }
    }

    #[inline]
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

#[derive(Debug, Default)]
pub struct FuzzReport {
    pub iterations: usize,
    pub parsed_ok: usize,
    pub rejected: usize,
    pub reads_executed: usize,
    /// Cases where the pipeline panicked (name -> payload byte length).
    pub panics: Vec<String>,
    /// Cases that parsed but violated a plan/safety invariant.
    pub invariant_failures: Vec<String>,
}

impl FuzzReport {
    pub fn ok(&self) -> bool {
        self.panics.is_empty() && self.invariant_failures.is_empty()
    }
}

const DTYPES: &[&str] = &[
    "F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL",
    // Not in the dtype table: must be rejected, never panic.
    "F8_E4M3", "F8_E5M2", "F7", "", "f32", "F32 ", "FLOAT32", "U16", "I4",
];

const WEIRD_NUMBERS: &[u64] = &[
    0, 1, 2, 7, 8, 15, 16, 4095, 4096, 4097, 8192, 65535, 65536,
    u32::MAX as u64, u32::MAX as u64 + 1, 1 << 40, 1 << 48, u64::MAX - 1, u64::MAX,
];

fn rand_shape(rng: &mut Rng, out: &mut String) {
    let n = rng.below(4); // 0..3 dims
    out.push('[');
    for i in 0..n {
        if i > 0 {
            out.push(',');
        }
        match rng.below(8) {
            0 => out.push_str(&rng.pick(WEIRD_NUMBERS).to_string()),
            1 => out.push_str("-1"),
            2 => out.push_str("1.5"),
            3 => out.push_str("null"),
            4 => out.push_str("\"3\""),
            5 => out.push_str("18446744073709551615"), // u64::MAX
            6 => out.push_str(&rng.below(1 << 20).to_string()),
            _ => out.push_str(&rng.below(64).to_string()),
        }
    }
    out.push(']');
}

/// Generate one case: `(header_json, payload, forced_header_len)`.
fn gen_case(rng: &mut Rng) -> (String, Vec<u8>, Option<u64>) {
    let n_entries = rng.below(5); // 0..4 entries
    let mut entries: Vec<String> = Vec::new();
    let mut cursor: u64 = 0;

    for i in 0..n_entries {
        let name = match rng.below(6) {
            0 => format!("t{i}"),
            1 => "__metadata__".to_string(), // reserved: must be an object, not a tensor
            2 => String::new(),
            3 => format!("layer.{i}.weight"),
            4 => "\u{1F600}".to_string(),
            _ => "dup".to_string(),
        };
        let dtype = *rng.pick(DTYPES);
        let mut shape = String::new();
        rand_shape(rng, &mut shape);

        // Offsets: usually coherent, sometimes deliberately not.
        let (start, end) = match rng.below(10) {
            0 => (rng.pick(WEIRD_NUMBERS).to_owned(), rng.pick(WEIRD_NUMBERS).to_owned()),
            1 => {
                let a = rng.below(4096);
                (a, a.saturating_sub(1)) // start > end
            }
            2 => {
                let a = cursor;
                let n = rng.below(4096);
                (a, a + n)
            }
            3 => (0, u64::MAX),
            _ => {
                let n = rng.below(1024);
                let s = cursor;
                (s, s + n)
            }
        };
        cursor = cursor.max(end).min(1 << 22);

        match rng.below(12) {
            // Deliberately malformed entry shapes.
            0 => entries.push(format!(r#""{name}":42"#)),
            1 => entries.push(format!(r#""{name}":{{"dtype":"{dtype}","shape":{shape}}}"#)),
            2 => entries.push(format!(r#""{name}":{{"dtype":"{dtype}","data_offsets":[{start},{end}]}}"#)),
            3 => entries.push(format!(r#""{name}":{{"shape":{shape},"data_offsets":[{start},{end}]}}"#)),
            4 => entries.push(format!(r#""{name}":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{start}]}}"#)),
            5 => entries.push(format!(r#""{name}":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{start},{end},3]}}"#)),
            _ => entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{start},{end}]}}"#
            )),
        }
    }

    let json = if rng.below(16) == 0 {
        // Sometimes emit structurally broken JSON.
        match rng.below(4) {
            0 => "{".to_string(),
            1 => format!("{{{}}}", entries.join(",")),
            2 => format!("[{}]", entries.join(",")),
            _ => "{not json at all".to_string(),
        }
    } else {
        format!("{{{}}}", entries.join(","))
    };

    let payload_len = rng.below(1 << 16) as usize;
    let mut payload = vec![0u8; payload_len];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(rng.below(256) as u8);
    }

    // Rarely: force a specific header_len instead of using the real JSON length,
    // which decouples "declared header size" from "actual bytes".
    let forced = if rng.below(8) == 0 {
        Some(rng.pick(WEIRD_NUMBERS).to_owned())
    } else {
        None
    };

    (json, payload, forced)
}

fn assemble(json: &str, payload: &[u8], forced_len: Option<u64>) -> Vec<u8> {
    let declared = forced_len.unwrap_or(json.len() as u64);
    let mut v = Vec::with_capacity(8 + json.len() + payload.len());
    v.extend_from_slice(&declared.to_le_bytes());
    v.extend_from_slice(json.as_bytes());
    v.extend_from_slice(payload);
    v
}

/// Byte-level mutation of an otherwise valid file.
fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut v = base.to_vec();
    if v.is_empty() {
        return v;
    }
    let ops = 1 + rng.below(4);
    for _ in 0..ops {
        // A previous op may have truncated the buffer to nothing; indexing ops
        // below would then panic. The fuzzer's own generator must be as robust
        // as the code it tests, otherwise it crashes before finding anything.
        if v.is_empty() {
            break;
        }
        match rng.below(6) {
            0 => {
                let i = rng.below(v.len() as u64) as usize;
                v[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(v.len() as u64) as usize;
                v[i] = rng.below(256) as u8;
            }
            2 => {
                // Corrupt the declared header length.
                if v.len() >= 8 {
                    let x = rng.pick(WEIRD_NUMBERS).to_owned().to_le_bytes();
                    v[..8].copy_from_slice(&x);
                }
            }
            3 => {
                if v.len() > 1 {
                    let n = rng.below(v.len() as u64) as usize;
                    v.truncate(n);
                }
            }
            4 => {
                let i = rng.below(v.len() as u64) as usize;
                v.insert(i, rng.below(256) as u8);
            }
            _ => {
                if v.len() > 1 {
                    let i = rng.below(v.len() as u64) as usize;
                    v.remove(i);
                }
            }
        }
    }
    v
}

/// The safety oracle: everything the rest of the crate assumes after a parse.
fn check_plan(h: &ShardHeader, block: u64) -> Option<String> {
    let file_ceil = ceil_to(h.file_size, block);
    let plan = plan_shard(h, block, 0);

    for r in &plan {
        if r.len == 0 {
            return Some(format!("zero-length range at {}", r.offset));
        }
        if !r.offset.is_multiple_of(block) {
            return Some(format!("range offset {} not block-aligned", r.offset));
        }
        if !r.len.is_multiple_of(block) {
            return Some(format!("range len {} not block-aligned", r.len));
        }
        if r.end() > file_ceil {
            return Some(format!(
                "range [{}, {}) extends past ceil(file_size)={file_ceil}",
                r.offset,
                r.end()
            ));
        }
    }
    for w in plan.windows(2) {
        if w[0].end() > w[1].offset {
            return Some(format!(
                "ranges overlap: [{},{}) then [{},{})",
                w[0].offset,
                w[0].end(),
                w[1].offset,
                w[1].end()
            ));
        }
    }

    // Every tensor's absolute span must sit inside a single range.
    let mut ri = 0usize;
    for t in &h.tensors {
        if t.nbytes() == 0 {
            continue;
        }
        let (s, e) = t.abs_range(h.data_start);
        if e > h.file_size {
            return Some(format!("tensor {} ends at {e} past file_size {}", t.name, h.file_size));
        }
        while ri < plan.len() && plan[ri].end() <= s {
            ri += 1;
        }
        let Some(r) = plan.get(ri) else {
            return Some(format!("tensor {} [{s},{e}) has no covering range", t.name));
        };
        if r.offset > s || r.end() < e {
            return Some(format!(
                "tensor {} [{s},{e}) not contained in range [{},{})",
                t.name,
                r.offset,
                r.end()
            ));
        }
    }

    // The converse: every range must cover at least some tensor byte. A range
    // that only spans alignment padding (or the header tail) is not unsafe, but
    // it is wasted I/O — the class of bug where a zero-element tensor still
    // emits a request. Checking safety alone would miss it.
    let mut ti = 0usize;
    for r in &plan {
        let mut covers_something = false;
        while ti < h.tensors.len() {
            let t = &h.tensors[ti];
            if t.nbytes() == 0 {
                ti += 1;
                continue;
            }
            let (s, e) = t.abs_range(h.data_start);
            let ts = floor_to(s, block);
            let te = ceil_to(e, block);
            if te <= r.offset {
                ti += 1;
                continue;
            }
            covers_something = ts < r.end();
            break;
        }
        if !covers_something {
            return Some(format!(
                "range [{}, {}) covers no tensor bytes (wasted read)",
                r.offset,
                r.end()
            ));
        }
    }
    None
}

/// One case end to end. Returns `(parsed, read_ok)`.
///
/// Deliberately does not catch panics itself: the caller wraps this in
/// `catch_unwind` so the original panic message survives.
fn run_case(path: &Path, bytes: &[u8], block: u64, do_read: bool) -> (bool, bool) {
    std::fs::write(path, bytes).expect("write fuzz case");

    let h = match read_header(path) {
        Ok(h) => h,
        Err(_) => return (false, false),
    };

    if let Some(msg) = check_plan(&h, block) {
        panic!("plan invariant violated: {msg}");
    }

    if do_read && h.file_size > 0 && h.file_size <= (1 << 20) {
        let plan = plan_shard(&h, block, 0);
        let cfg = ReadConfig { depth: 4, chunk_bytes: 4096, direct: true, block: block as usize };
        let files = [path.to_path_buf()];
        match read_files(&files, &plan_as_pairs(&plan), &cfg) {
            Ok(_) => return (true, true),
            Err(_) => return (true, false),
        }
    }
    (true, false)
}

fn plan_as_pairs(plan: &[Range]) -> Vec<(usize, Range)> {
    plan.iter().map(|r| (0usize, *r)).collect()
}

/// Syntactically valid files used as mutation seeds.
fn seed_corpus() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let cases: &[(&str, usize)] = &[
        (r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#, 16),
        (
            r#"{"a":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]},"b":{"dtype":"F16","shape":[8],"data_offsets":[4096,4112]}}"#,
            4112,
        ),
        (
            r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#,
            8,
        ),
        (r#"{"z":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#, 0),
        (r#"{"s":{"dtype":"F64","shape":[],"data_offsets":[0,8]}}"#, 8),
    ];
    for (json, payload) in cases {
        out.push(assemble(json, &vec![0x5Au8; *payload], None));
    }
    out
}

/// Run `iterations` generated cases.
///
/// `scratch` is where cases are written; `seed` makes the run reproducible.
pub fn run(
    iterations: usize,
    seed: u64,
    scratch: &Path,
    do_reads: bool,
    verbose: bool,
) -> io::Result<FuzzReport> {
    std::fs::create_dir_all(scratch)?;
    let path = scratch.join("case.safetensors");
    let block = 4096u64;

    let mut rng = Rng::new(seed);
    let seeds = seed_corpus();
    let mut report = FuzzReport::default();

    for i in 0..iterations {
        let bytes = match rng.below(10) {
            0..=4 => {
                let (json, payload, forced) = gen_case(&mut rng);
                assemble(&json, &payload, forced)
            }
            5..=6 => {
                let base = rng.pick(&seeds).clone();
                mutate(&mut rng, &base)
            }
            7..=8 => {
                // Replay a known-valid seed verbatim. Without this, randomly
                // generated headers almost never produce a *valid* file with a
                // degenerate tensor (zero-element, scalar, unaligned offset),
                // so bugs that only fire on valid-but-degenerate input would be
                // missed — exactly what happened before this branch existed.
                rng.pick(&seeds).clone()
            }
            _ => {
                // Purely random bytes, or random text wrapped in a header.
                if rng.below(2) == 0 {
                    let n = rng.below(4096) as usize;
                    (0..n).map(|_| rng.below(256) as u8).collect()
                } else {
                    let n = rng.below(256) as usize;
                    let s: String = (0..n)
                        .map(|_| (b' ' + rng.below(95) as u8) as char)
                        .collect();
                    assemble(&s, &[0u8; 32], None)
                }
            }
        };

        let result = catch_unwind(AssertUnwindSafe(|| run_case(&path, &bytes, block, do_reads)));
        match result {
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".to_string());
                let entry = format!(
                    "iter {i} (seed {seed}, {} bytes): {msg}",
                    bytes.len()
                );
                if verbose || report.panics.len() < 10 {
                    eprintln!("PANIC  {entry}");
                }
                report.panics.push(entry);
            }
            Ok((parsed, read_ok)) => {
                if parsed {
                    report.parsed_ok += 1;
                } else {
                    report.rejected += 1;
                }
                if read_ok {
                    report.reads_executed += 1;
                }
            }
        }
        report.iterations += 1;
    }

    let _ = std::fs::remove_file(&path);
    Ok(report)
}

/// Kind of dtype values the generator emits; kept for reporting so a reader of
/// the CLI output knows the mutation space is not just numeric.
pub fn dtype_token_count() -> usize {
    DTYPES.len()
}

/// Confirm the dtype table round-trips for every known name.
pub fn dtype_table_is_consistent() -> bool {
    DTYPES
        .iter()
        .filter_map(|s| Dtype::parse(s))
        .all(|d| Dtype::parse(d.as_str()) == Some(d))
}
