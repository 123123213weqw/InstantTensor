//! Synthetic self-tests: malformed and adversarial safetensors files.
//!
//! These do not need any real model on disk. Each case writes a small file to a
//! scratch directory and asserts that the loader either parses it correctly or
//! rejects it with an error — never panics, never allocates absurdly, never
//! wraps an arithmetic result.
//!
//! The adversarial cases matter because a release build has overflow checks
//! off: `a1 - a0` on inverted bounds, or `data_start + end`, would otherwise
//! wrap into a multi-exabyte length.

use std::io;
use std::path::{Path, PathBuf};

use crate::header::{read_header, shape_numel};
use crate::plan::{ceil_to, floor_to, plan_model, plan_shard};
use crate::reader::{read_files, ReadConfig};

pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

impl Check {
    fn ok(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), pass: true, detail: detail.into() }
    }

    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), pass: false, detail: detail.into() }
    }

    fn assert(name: impl Into<String>, cond: bool, detail: impl Into<String>) -> Self {
        let d = detail.into();
        if cond {
            Self::ok(name, d)
        } else {
            Self::fail(name, d)
        }
    }
}

fn build(header_json: &str, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + header_json.len() + payload.len());
    v.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    v.extend_from_slice(header_json.as_bytes());
    v.extend_from_slice(payload);
    v
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> io::Result<PathBuf> {
    let p = dir.join(name);
    std::fs::write(&p, bytes)?;
    Ok(p)
}

/// One entry of a synthetic header.
struct T {
    name: &'static str,
    dtype: &'static str,
    shape: &'static str,
    start: u64,
    end: u64,
}

fn json_of(ts: &[T]) -> String {
    let entries: Vec<String> = ts
        .iter()
        .map(|t| {
            format!(
                r#""{}":{{"dtype":"{}","shape":{},"data_offsets":[{},{}]}}"#,
                t.name, t.dtype, t.shape, t.start, t.end
            )
        })
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// Expect `read_header` to reject the file.
fn expect_reject(dir: &Path, name: &str, bytes: &[u8]) -> Check {
    let p = match write(dir, name, bytes) {
        Ok(p) => p,
        Err(e) => return Check::fail(name, format!("could not write: {e}")),
    };
    match read_header(&p) {
        Ok(h) => Check::fail(
            name,
            format!("accepted a malformed file: {} tensors, buffer {}", h.tensors.len(), h.buffer_bytes()),
        ),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => {
            let msg = e.to_string();
            let msg = msg.split_once(": ").map(|(_, r)| r).unwrap_or(&msg).to_string();
            Check::ok(name, format!("rejected: {msg}"))
        }
        Err(e) => Check::fail(name, format!("wrong error kind {:?}: {e}", e.kind())),
    }
}

/// Run every synthetic case. `dir` is created if missing.
pub fn run(dir: &Path) -> io::Result<Vec<Check>> {
    std::fs::create_dir_all(dir)?;
    let mut out = Vec::new();

    // ---------------------------------------------------------------- valid
    let payload = vec![0xABu8; 8192];
    let good = json_of(&[
        T { name: "a", dtype: "F32", shape: "[1024]", start: 0, end: 4096 },
        T { name: "b", dtype: "F32", shape: "[1024]", start: 4096, end: 8192 },
    ]);
    let gp = write(dir, "valid_two.safetensors", &build(&good, &payload))?;
    match read_header(&gp) {
        Ok(h) => {
            let tb = h.tensor_bytes();
            let buf = h.buffer_bytes();
            let n_ok = h.tensors.len() == 2;
            let plan = plan_shard(&h, 4096, 0);
            let covered: u64 = plan.iter().map(|r| r.len).sum();
            out.push(Check::assert(
                "valid: two contiguous tensors",
                n_ok && tb == 8192 && buf == 8192 && plan.len() == 1 && covered >= 8192,
                format!(
                    "tensors={} tensor_bytes={tb} buffer={buf} ranges={} covered={covered}",
                    h.tensors.len(),
                    plan.len()
                ),
            ));
        }
        Err(e) => out.push(Check::fail("valid: two contiguous tensors", format!("rejected valid file: {e}"))),
    }

    // ------------------------------------------------- header framing errors
    // header_len == 0
    {
        let mut v = 0u64.to_le_bytes().to_vec();
        v.extend_from_slice(&[0u8; 16]);
        out.push(expect_reject(dir, "hdr_zero.safetensors", &v));
    }

    // header_len overruns the file
    {
        let mut v = 4096u64.to_le_bytes().to_vec();
        v.extend_from_slice(b"{}");
        out.push(expect_reject(dir, "hdr_overrun.safetensors", &v));
    }
    // absurd header_len
    {
        let mut v = (1u64 << 40).to_le_bytes().to_vec();
        v.extend_from_slice(b"{}");
        out.push(expect_reject(dir, "hdr_absurd.safetensors", &v));
    }
    // file shorter than 8 bytes
    out.push(expect_reject(dir, "hdr_short.safetensors", b"GG"));
    // bad JSON
    out.push(expect_reject(dir, "json_bad.safetensors", &build("{not json", &[])));
    // header is an array, not an object
    out.push(expect_reject(dir, "json_array.safetensors", &build("[1,2,3]", &[])));
    // entry is not an object
    out.push(expect_reject(
        dir,
        "entry_not_obj.safetensors",
        &build(r#"{"a":42}"#, &[]),
    ));
    // __metadata__ is not an object
    out.push(expect_reject(
        dir,
        "meta_not_obj.safetensors",
        &build(r#"{"__metadata__":7}"#, &[]),
    ));

    // ------------------------------------------------------- entry-level errs
    out.push(expect_reject(
        dir,
        "dtype_unknown.safetensors",
        &build(&json_of(&[T { name: "a", dtype: "F7", shape: "[4]", start: 0, end: 16 }]), &[0u8; 16]),
    ));
    out.push(expect_reject(
        dir,
        "missing_shape.safetensors",
        &build(r#"{"a":{"dtype":"F32","data_offsets":[0,16]}}"#, &[0u8; 16]),
    ));
    out.push(expect_reject(
        dir,
        "offsets_not_pair.safetensors",
        &build(r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0]}}"#, &[0u8; 16]),
    ));
    out.push(expect_reject(
        dir,
        "end_before_start.safetensors",
        &build(&json_of(&[T { name: "a", dtype: "F32", shape: "[4]", start: 100, end: 10 }]), &[0u8; 256]),
    ));
    out.push(expect_reject(
        dir,
        "end_beyond_buffer.safetensors",
        &build(&json_of(&[T { name: "a", dtype: "F32", shape: "[256]", start: 0, end: 1024 }]), &[0u8; 8]),
    ));
    out.push(expect_reject(
        dir,
        "shape_dtype_mismatch.safetensors",
        &build(&json_of(&[T { name: "a", dtype: "F32", shape: "[10]", start: 0, end: 20 }]), &[0u8; 20]),
    ));
    out.push(expect_reject(
        dir,
        "shape_noninteger.safetensors",
        &build(r#"{"a":{"dtype":"F32","shape":["x"],"data_offsets":[0,0]}}"#, &[]),
    ));

    // ------------------------------------------------ adversarial / overflow
    // Offset at u64::MAX: must be rejected by the bounds check, not wrapped.
    out.push(expect_reject(
        dir,
        "offset_u64max.safetensors",
        &build(
            &json_of(&[T { name: "a", dtype: "F32", shape: "[4]", start: 0, end: u64::MAX }]),
            &[0u8; 16],
        ),
    ));
    // Shape product overflows u64.
    out.push(expect_reject(
        dir,
        "shape_overflow.safetensors",
        &build(
            &json_of(&[T { name: "a", dtype: "F32", shape: "[18446744073709551615,2]", start: 0, end: 0 }]),
            &[],
        ),
    ));
    // Offset just past the end of the buffer by one byte.
    out.push(expect_reject(
        dir,
        "off_by_one.safetensors",
        &build(&json_of(&[T { name: "a", dtype: "F32", shape: "[4]", start: 0, end: 17 }]), &[0u8; 16]),
    ));

    // ---------------------------------------------------------- valid corner
    // Empty tensor (shape with a zero dim) is legal.
    {
        let j = json_of(&[T { name: "z", dtype: "F32", shape: "[0]", start: 0, end: 0 }]);
        let p = write(dir, "empty_tensor.safetensors", &build(&j, &[]))?;
        match read_header(&p) {
            Ok(h) => {
                let plan = plan_shard(&h, 4096, 0);
                out.push(Check::assert(
                    "valid: zero-element tensor",
                    h.tensors.len() == 1 && h.tensor_bytes() == 0 && plan.is_empty(),
                    format!("tensors={} tensor_bytes={} ranges={}", h.tensors.len(), h.tensor_bytes(), plan.len()),
                ));
            }
            Err(e) => out.push(Check::fail("valid: zero-element tensor", format!("rejected: {e}"))),
        }
    }

    // Scalar (shape []) is legal and takes one element.
    {
        let j = json_of(&[T { name: "s", dtype: "F32", shape: "[]", start: 0, end: 4 }]);
        let p = write(dir, "scalar.safetensors", &build(&j, &[0u8; 4]))?;
        match read_header(&p) {
            Ok(h) => out.push(Check::assert(
                "valid: scalar tensor shape []",
                h.tensors.len() == 1 && h.tensor_bytes() == 4,
                format!("tensor_bytes={}", h.tensor_bytes()),
            )),
            Err(e) => out.push(Check::fail("valid: scalar tensor shape []", format!("rejected: {e}"))),
        }
    }

    // ------------------------------------------------- merging / gap semantics
    {
        let gap = 1u64 << 20;
        let j = json_of(&[
            T { name: "a", dtype: "F32", shape: "[1024]", start: 0, end: 4096 },
            T { name: "b", dtype: "F32", shape: "[1024]", start: 4096 + gap, end: 8192 + gap },
        ]);
        let payload = vec![0u8; (8192 + gap) as usize];
        let p = write(dir, "gapped.safetensors", &build(&j, &payload))?;
        match read_header(&p) {
            Ok(h) => {
                let tight = plan_shard(&h, 4096, 0);
                let loose = plan_shard(&h, 4096, gap * 2);
                let amp = tight.iter().map(|r| r.len).sum::<u64>() as f64 / h.tensor_bytes() as f64;
                out.push(Check::assert(
                    "gap: merge respects max_gap",
                    tight.len() == 2 && loose.len() == 1,
                    format!("ranges tight={} loose={} amp={amp:.4}x", tight.len(), loose.len()),
                ));
            }
            Err(e) => out.push(Check::fail("gap: merge respects max_gap", format!("rejected: {e}"))),
        }
    }

    // ------------------------------------------------- arithmetic primitives
    out.push(Check::assert(
        "arithmetic: ceil_to saturates, floor_to safe, no panic",
        {
            let a = ceil_to(u64::MAX, 4096);
            let b = ceil_to(0, 4096);
            let c = floor_to(u64::MAX, 4096);
            let d = ceil_to(1, 0); // b is clamped to >= 1
            a >= c && b == 0 && d == 1
        },
        format!(
            "ceil(MAX,4096)={} ceil(0,4096)={} floor(MAX,4096)={} ceil(1,0)={}",
            ceil_to(u64::MAX, 4096),
            ceil_to(0, 4096),
            floor_to(u64::MAX, 4096),
            ceil_to(1, 0)
        ),
    ));

    out.push(Check::assert(
        "arithmetic: shape_numel overflow -> None",
        shape_numel(&[u64::MAX, 2]).is_none()
            && shape_numel(&[2, 3, 4]) == Some(24)
            && shape_numel(&[]) == Some(1),
        format!("MAX*2={:?} 2*3*4={:?} []={:?}", shape_numel(&[u64::MAX, 2]), shape_numel(&[2, 3, 4]), shape_numel(&[])),
    ));

    // ------------------------------------------------- API misuse is an error
    {
        let files = vec![gp.clone()];
        let plan = vec![(7usize, crate::plan::Range { offset: 0, len: 4096 })];
        let r = read_files(&files, &plan, &ReadConfig::default());
        out.push(Check::assert(
            "api: out-of-range shard index rejected",
            matches!(r, Err(ref e) if e.kind() == io::ErrorKind::InvalidInput),
            match r {
                Ok(_) => "accepted an out-of-range shard index".to_string(),
                Err(e) => format!("rejected: {e}"),
            },
        ));
    }
    {
        let files = vec![gp.clone()];
        let plan = vec![(0usize, crate::plan::Range { offset: 0, len: 0 })];
        let r = read_files(&files, &plan, &ReadConfig::default());
        out.push(Check::assert(
            "api: zero-length range rejected",
            matches!(r, Err(ref e) if e.kind() == io::ErrorKind::InvalidInput),
            match r {
                Ok(_) => "accepted a zero-length range".to_string(),
                Err(e) => format!("rejected: {e}"),
            },
        ));
    }
    {
        let r = read_files(std::slice::from_ref(&gp), &[], &ReadConfig::default());
        out.push(Check::assert(
            "api: empty plan is a no-op",
            matches!(r, Ok(ref s) if s.bytes == 0 && s.requests == 0),
            match r {
                Ok(s) => format!("bytes={} requests={}", s.bytes, s.requests),
                Err(e) => format!("errored: {e}"),
            },
        ));
    }

    // ------------------------------------------------- plan shape invariants
    {
        let headers = vec![read_header(&gp)?];
        let plan = plan_model(&headers, 4096, 0);
        let no_zero_len = plan.iter().all(|(_, r)| r.len > 0);
        let in_range = plan.iter().all(|(i, _)| *i < headers.len());
        let no_overlap = plan
            .windows(2)
            .all(|w| w[0].1.end() <= w[1].1.offset);
        out.push(Check::assert(
            "plan: no zero-length, in-range, non-overlapping",
            no_zero_len && in_range && no_overlap,
            format!("{} ranges; zero_len_ok={no_zero_len} in_range={in_range} sorted={no_overlap}", plan.len()),
        ));
    }

    Ok(out)
}
