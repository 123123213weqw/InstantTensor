//! A minimal safetensors reader, scoped to loading a real Qwen3.5 checkpoint.
//!
//! # Why this is not the streaming loader
//!
//! `rust-qwen-engine/crates/stloader` exists to measure and minimise *IO cost*: it
//! plans block-aligned merged requests, issues them with `io_uring` and `O_DIRECT`,
//! and reports amplification. This module has a different job: turn a checkpoint on
//! disk into the `f32` tensors the forward pass wants, with enough validation that a
//! malformed or mismatched file fails loudly instead of producing quietly wrong
//! numbers. Keeping it here also keeps `qwen35-forward` self-contained.
//!
//! # What is validated, and why
//!
//! The safetensors format is a `u64` header length, a JSON header, then a flat data
//! buffer. Three things can disagree, and each has caught a real class of bug:
//!
//! * `end - start == numel(shape) * dtype.size()` — offsets, shape and dtype must
//!   agree. This is the check that catches a truncated or hand-edited header.
//! * every tensor's byte range must lie inside the file.
//! * the header length must be consistent with the file size.
//!
//! A reader that skips these reads the wrong bytes and returns plausible numbers.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Element types this reader decodes. Qwen3.5 ships `BF16`; `F32` appears in
/// small tensors and in converted checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Dtype::F32),
            "F16" => Some(Dtype::F16),
            "BF16" => Some(Dtype::Bf16),
            _ => None,
        }
    }

    pub fn size(self) -> u64 {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
        }
    }

    /// Decode `bytes` into `f32`. `bytes.len()` must be an exact multiple of the
    /// element size; the caller has already checked it against the shape.
    pub fn decode(self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        let esz = self.size() as usize;
        if !bytes.len().is_multiple_of(esz) {
            return Err(format!(
                "{}: {} bytes is not a multiple of the {esz}-byte element size",
                self.as_str(),
                bytes.len()
            ));
        }
        let n = bytes.len() / esz;
        let mut out = Vec::with_capacity(n);
        match self {
            Dtype::F32 => {
                for i in 0..n {
                    let b = &bytes[i * 4..i * 4 + 4];
                    out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
            Dtype::Bf16 => {
                // bfloat16 is the top 16 bits of an f32: same exponent, 8 fewer
                // mantissa bits. Widening is a left shift, with no rounding --
                // every bf16 is exactly representable as f32.
                for i in 0..n {
                    let b = &bytes[i * 2..i * 2 + 2];
                    let bits = u16::from_le_bytes([b[0], b[1]]) as u32;
                    out.push(f32::from_bits(bits << 16));
                }
            }
            Dtype::F16 => {
                for i in 0..n {
                    let b = &bytes[i * 2..i * 2 + 2];
                    let h = u16::from_le_bytes([b[0], b[1]]);
                    out.push(f16_to_f32(h));
                }
            }
        }
        Ok(out)
    }
}

/// IEEE 754 half -> single. Handles subnormals, infinities and NaN so that a
/// checkpoint containing them does not silently become zeros.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal: normalise it
                let mut e = 127 - 15 - 10;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                (sign << 31) | (((e + 1) as u32) << 23) | ((f & 0x3ff) << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// One tensor's location inside a shard.
#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute range in the file (the header offset is already folded in).
    pub start: u64,
    pub end: u64,
}

impl Entry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One `.safetensors` file.
#[derive(Debug, Clone)]
pub struct Shard {
    pub path: PathBuf,
    pub file_size: u64,
    pub data_start: u64,
    pub tensors: BTreeMap<String, Entry>,
}

impl Shard {
    /// Parse the header and validate it against the file size.
    pub fn open(path: impl AsRef<Path>) -> Result<Shard, String> {
        let path = path.as_ref().to_path_buf();
        let file_size = fs::metadata(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();

        // The header length is the first 8 bytes, little-endian u64.
        let mut head = [0u8; 8];
        {
            use std::io::Read;
            let mut f = fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            f.read_exact(&mut head)
                .map_err(|e| format!("{}: cannot read the 8-byte header length: {e}", path.display()))?;
        }
        let header_len = u64::from_le_bytes(head);

        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| format!("{}: header length {header_len} overflows", path.display()))?;
        if data_start > file_size {
            return Err(format!(
                "{}: header claims {header_len} bytes but the file is only {file_size} bytes",
                path.display()
            ));
        }

        let raw = {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            f.seek(SeekFrom::Start(8)).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut buf = vec![0u8; header_len as usize];
            f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
            buf
        };

        let doc: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| format!("{}: header is not valid JSON: {e}", path.display()))?;
        let obj = doc
            .as_object()
            .ok_or_else(|| format!("{}: header is not a JSON object", path.display()))?;

        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = v.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| {
                format!("{}: {name} has no dtype", path.display())
            })?;
            let dtype = Dtype::parse(dtype_str).ok_or_else(|| {
                format!("{}: {name} has unsupported dtype {dtype_str}", path.display())
            })?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("{}: {name} has no shape", path.display()))?
                .iter()
                .map(|x| x.as_u64().map(|n| n as usize).unwrap_or(usize::MAX))
                .collect();
            if shape.contains(&usize::MAX) {
                return Err(format!("{}: {name} has a non-integer shape", path.display()));
            }
            let offs = v
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("{}: {name} has no data_offsets", path.display()))?;
            if offs.len() != 2 {
                return Err(format!("{}: {name} data_offsets is not a pair", path.display()));
            }
            let rel_start = offs[0].as_u64().ok_or_else(|| format!("{}: {name} bad offset", path.display()))?;
            let rel_end = offs[1].as_u64().ok_or_else(|| format!("{}: {name} bad offset", path.display()))?;

            // Offsets must be ordered. `rel_end - rel_start` on a reversed pair
            // underflows into an enormous length, which is how a reader ends up
            // allocating terabytes.
            if rel_end < rel_start {
                return Err(format!(
                    "{}: {name} data_offsets are reversed ({rel_start} > {rel_end})",
                    path.display()
                ));
            }
            let span = rel_end - rel_start;

            let numel: u64 = shape.iter().map(|&d| d as u64).product();
            let expected = numel
                .checked_mul(dtype.size())
                .ok_or_else(|| format!("{}: {name} numel*size overflows", path.display()))?;
            if span != expected {
                return Err(format!(
                    "{}: {name} spans {span} bytes but shape {:?} of {} needs {expected}",
                    path.display(),
                    shape,
                    dtype.as_str()
                ));
            }

            let start = data_start
                .checked_add(rel_start)
                .ok_or_else(|| format!("{}: {name} start overflows", path.display()))?;
            let end = data_start
                .checked_add(rel_end)
                .ok_or_else(|| format!("{}: {name} end overflows", path.display()))?;
            if end > file_size {
                return Err(format!(
                    "{}: {name} ends at {end} but the file is {file_size} bytes",
                    path.display()
                ));
            }

            tensors.insert(name.clone(), Entry { dtype, shape, start, end });
        }

        Ok(Shard { path, file_size, data_start, tensors })
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.tensors.get(name)
    }
}

/// Every shard of a model directory, with tensors addressable by name.
///
/// The whole file contents are held in memory. For a 1.75 GB checkpoint that is
/// well within budget, and it turns "decode one tensor" into a slice instead of a
/// syscall per tensor. It is also the honest thing to do for a first correctness
/// pass: swapping in a streaming reader later changes IO, not numerics.
pub struct Model {
    pub shards: Vec<Shard>,
    /// tensor name -> (shard index, entry)
    index: BTreeMap<String, (usize, Entry)>,
    /// Kept for the lifetime of the decode; see the struct comment.
    bufs: Vec<Vec<u8>>,
}

impl Model {
    /// Open every `*.safetensors` file in `dir`.
    ///
    /// The `model.safetensors.index.json` file, when present, is *not* trusted to
    /// decide what to load: the shards themselves are enumerated and their headers
    /// are the authority. The index is only cross-checked afterwards, because a
    /// stale index would otherwise silently omit tensors.
    pub fn open(dir: impl AsRef<Path>) -> Result<Model, String> {
        let dir = dir.as_ref();
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("{}: no *.safetensors files", dir.display()));
        }

        let mut shards: Vec<Shard> = Vec::new();
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        // Annotated: the first `index.insert` comes after the duplicate check, so
        // inference has nothing to go on at the `shards[*prev]` use below.
        let mut index: BTreeMap<String, (usize, Entry)> = BTreeMap::new();
        for p in paths {
            let shard = Shard::open(&p)?;
            let si = shards.len();
            for (name, entry) in &shard.tensors {
                if let Some((prev, _)) = index.get(name) {
                    return Err(format!(
                        "{name} appears in both {} and {}",
                        shards[*prev].path.display(),
                        p.display()
                    ));
                }
                index.insert(name.clone(), (si, entry.clone()));
            }
            let buf = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if buf.len() as u64 != shard.file_size {
                return Err(format!("{}: file changed while reading", p.display()));
            }
            bufs.push(buf);
            shards.push(shard);
        }

        Ok(Model { shards, index, bufs })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.index.get(name).map(|(_, e)| e)
    }

    /// Decode one tensor to `f32`, checking the element count against the shape.
    pub fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let (si, e) = self
            .index
            .get(name)
            .ok_or_else(|| format!("no tensor named {name}"))?;
        let bytes = &self.bufs[*si][e.start as usize..e.end as usize];
        let v = e.dtype.decode(bytes).map_err(|err| format!("{name}: {err}"))?;
        if v.len() != e.numel() {
            return Err(format!(
                "{name}: decoded {} values but the shape {:?} needs {}",
                v.len(),
                e.shape,
                e.numel()
            ));
        }
        Ok(v)
    }

    /// Number of tensors whose name contains `needle`, for reporting.
    pub fn count_containing(&self, needle: &str) -> usize {
        self.index.keys().filter(|k| k.contains(needle)).count()
    }

    /// Total bytes of tensor data across all shards.
    pub fn tensor_bytes(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.tensors.values().map(|e| e.end - e.start).sum::<u64>())
            .sum()
    }

    /// Cross-check against `model.safetensors.index.json` when present.
    ///
    /// Returns the number of tensors the index lists but the shards do not have.
    /// Loading does not depend on the index, so a stale index is reported rather
    /// than obeyed.
    pub fn index_disagreement(&self, dir: impl AsRef<Path>) -> Result<Vec<String>, String> {
        let p = dir.as_ref().join("model.safetensors.index.json");
        if !p.exists() {
            return Ok(Vec::new());
        }
        let raw = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let doc: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", p.display()))?;
        let map = doc
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| format!("{}: no weight_map", p.display()))?;
        let mut missing = Vec::new();
        for name in map.keys() {
            if !self.contains(name) {
                missing.push(name.clone());
            }
        }
        Ok(missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("st_test_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Build a valid single-tensor file and read it back.
    fn write_file(dir: &Path, name: &str, dtype: Dtype, shape: &[usize], data: &[u8]) -> PathBuf {
        let header = serde_json::json!({
            name: { "dtype": dtype.as_str(), "shape": shape, "data_offsets": [0, data.len()] }
        });
        let hs = serde_json::to_string(&header).unwrap();
        let p = dir.join("model.safetensors");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&(hs.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hs.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        p
    }

    #[test]
    fn bf16_widening_is_exact() {
        // 1.0 in bf16 is 0x3F80; widening gives exactly 1.0f32.
        let b = 0x3F80u16.to_le_bytes();
        assert_eq!(Dtype::Bf16.decode(&b).unwrap(), vec![1.0f32]);
        // -2.0 is 0xC000
        let b = 0xC000u16.to_le_bytes();
        assert_eq!(Dtype::Bf16.decode(&b).unwrap(), vec![-2.0f32]);
        // zero
        let b = 0x0000u16.to_le_bytes();
        assert_eq!(Dtype::Bf16.decode(&b).unwrap(), vec![0.0f32]);
        // bfloat16 is 1 sign + 8 exponent + **7** mantissa bits, so the smallest
        // step above 1.0 is 2^-7, not 2^-8. 0x3F81 is 1 + 1/128.
        let b = 0x3F81u16.to_le_bytes();
        assert!(
            (Dtype::Bf16.decode(&b).unwrap()[0] - (1.0 + 1.0 / 128.0)).abs() < 1e-9,
            "expected 1 + 2^-7"
        );
        // Widening must be bit-exact: a bf16 value is always representable in f32,
        // so decoding then re-encoding has to round-trip without loss.
        for bits in [0x0000u16, 0x3F80, 0x3F81, 0xBF80, 0x7F7F, 0x0001, 0x8000, 0x4049] {
            let got = Dtype::Bf16.decode(&bits.to_le_bytes()).unwrap()[0];
            assert_eq!(
                (got.to_bits() >> 16) as u16,
                bits,
                "0x{bits:04X} did not round-trip through f32"
            );
        }
    }

    #[test]
    fn f16_handles_special_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);      // 1.0
        assert_eq!(f16_to_f32(0xC000), -2.0);     // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0);      // +0
        assert_eq!(f16_to_f32(0x8000), -0.0);     // -0
        assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95_f32); // 1/3 rounded to f16
    }

    #[test]
    fn reads_a_valid_file() {
        let d = tmpdir("valid");
        let data: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_file(&d, "w", Dtype::F32, &[3], &data);
        let m = Model::open(&d).unwrap();
        assert_eq!(m.tensor("w").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(m.entry("w").unwrap().shape, vec![3]);
    }

    /// `end - start` must equal `numel * dtype.size()`.
    #[test]
    fn rejects_shape_that_disagrees_with_byte_span() {
        let d = tmpdir("span");
        let data = vec![0u8; 8]; // says 2 f32 but claims 3
        write_file(&d, "w", Dtype::F32, &[3], &data);
        let err = Shard::open(d.join("model.safetensors")).unwrap_err();
        assert!(err.contains("spans 8 bytes"), "{err}");
    }

    /// A reversed offset pair must not underflow into a huge length.
    #[test]
    fn rejects_reversed_offsets() {
        let d = tmpdir("rev");
        let header = serde_json::json!({
            "w": { "dtype": "F32", "shape": [2], "data_offsets": [8, 0] }
        });
        let hs = serde_json::to_string(&header).unwrap();
        let p = d.join("model.safetensors");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&(hs.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hs.as_bytes()).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        drop(f);
        let err = Shard::open(&p).unwrap_err();
        assert!(err.contains("reversed"), "{err}");
    }

    #[test]
    fn rejects_tensor_past_end_of_file() {
        let d = tmpdir("oob");
        let header = serde_json::json!({
            "w": { "dtype": "F32", "shape": [100], "data_offsets": [0, 400] }
        });
        let hs = serde_json::to_string(&header).unwrap();
        let p = d.join("model.safetensors");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&(hs.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hs.as_bytes()).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        drop(f);
        let err = Shard::open(&p).unwrap_err();
        assert!(err.contains("ends at"), "{err}");
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let d = tmpdir("dtype");
        let header = serde_json::json!({
            "w": { "dtype": "I8", "shape": [1], "data_offsets": [0, 1] }
        });
        let hs = serde_json::to_string(&header).unwrap();
        let p = d.join("model.safetensors");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&(hs.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hs.as_bytes()).unwrap();
        f.write_all(&[0u8]).unwrap();
        drop(f);
        let err = Shard::open(&p).unwrap_err();
        assert!(err.contains("unsupported dtype"), "{err}");
    }

    #[test]
    fn missing_tensor_is_reported_by_name() {
        let d = tmpdir("missing");
        write_file(&d, "w", Dtype::F32, &[1], &1.0f32.to_le_bytes());
        let m = Model::open(&d).unwrap();
        let err = m.tensor("absent").unwrap_err();
        assert!(err.contains("absent"), "{err}");
    }
}
