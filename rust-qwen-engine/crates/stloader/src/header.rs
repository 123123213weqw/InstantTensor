//! safetensors header parsing and validation.
//!
//! Layout of a `.safetensors` file:
//!
//! ```text
//! [0..8)      u64 LE header size N
//! [8..8+N)    JSON header (UTF-8)
//! [8+N..EOF)  tensor byte buffer
//! ```
//!
//! A tensor entry's `data_offsets` are relative to the start of the byte
//! buffer, so its absolute file offset is `8 + N + start`.
//!
//! # Validation
//!
//! Every header is validated before any payload read is planned. After a
//! successful parse the following invariants hold, and `plan`/`reader` rely on
//! them:
//!
//! * `end >= start`, and `end <= buffer_size` where `buffer_size = file_size - data_start`;
//! * `end - start == numel(shape) * dtype.size()`, so offsets, shape and dtype agree;
//! * `data_start + end <= file_size`, which cannot overflow because
//!   `data_start + buffer_size == file_size`.
//!
//! A malformed file is rejected with an `InvalidData` error rather than being
//! silently planned, which is what makes `abs_range` infallible downstream.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Maximum plausible header size, as a sanity bound against corrupt files.
const MAX_HEADER_BYTES: u64 = 1 << 28; // 256 MiB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
    F8E4M3,
    F8E5M2,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I16" => Self::I16,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            _ => return None,
        })
    }

    pub fn size(self) -> u64 {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool | Self::F8E4M3 | Self::F8E5M2 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range relative to the start of the tensor data buffer.
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn nbytes(&self) -> u64 {
        // Invariant: end >= start, established at parse time.
        self.end - self.start
    }

    pub fn numel(&self) -> u64 {
        shape_numel(&self.shape).unwrap_or(0)
    }

    /// Absolute byte range within the file.
    ///
    /// Infallible: `read_header` guarantees `data_start + end <= file_size`,
    /// so neither addition can overflow.
    pub fn abs_range(&self, data_start: u64) -> (u64, u64) {
        (data_start + self.start, data_start + self.end)
    }
}

/// Product of a tensor shape, or `None` on overflow.
pub fn shape_numel(shape: &[u64]) -> Option<u64> {
    let mut n: u64 = 1;
    for &d in shape {
        n = n.checked_mul(d)?;
    }
    Some(n)
}

#[derive(Debug)]
pub struct ShardHeader {
    pub path: PathBuf,
    pub file_size: u64,
    pub header_len: u64,
    /// Offset of the tensor data buffer: `8 + header_len`.
    pub data_start: u64,
    pub metadata: Option<Map<String, Value>>,
    pub tensors: Vec<TensorInfo>,
}

impl ShardHeader {
    /// Sum of all tensor payload bytes (excludes header, gaps and padding).
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }

    /// Size of the tensor data buffer: `file_size - data_start`.
    pub fn buffer_bytes(&self) -> u64 {
        self.file_size.saturating_sub(self.data_start)
    }
}

fn invalid<T>(msg: impl Into<String>) -> std::io::Result<T> {
    Err(std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into()))
}

fn read_u64_le(f: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Parse and validate one shard's header. Only the header is read; tensor
/// payloads are not touched.
pub fn read_header(path: &Path) -> std::io::Result<ShardHeader> {
    let mut f = File::open(path)?;
    let file_size = f.metadata()?.len();

    if file_size < 8 {
        return invalid(format!("{}: file shorter than 8-byte header length", path.display()));
    }

    let header_len = read_u64_le(&mut f)?;
    if header_len == 0 {
        return invalid(format!("{}: header length is zero", path.display()));
    }
    if header_len > MAX_HEADER_BYTES {
        return invalid(format!(
            "{}: header length {header_len} exceeds {MAX_HEADER_BYTES}",
            path.display()
        ));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: header length overflows", path.display()),
            )
        })?;
    if data_start > file_size {
        return invalid(format!(
            "{}: header length {header_len} overruns file of {file_size} bytes",
            path.display()
        ));
    }
    let buffer_size = file_size - data_start;

    let mut raw = vec![0u8; header_len as usize];
    f.seek(SeekFrom::Start(8))?;
    f.read_exact(&mut raw)?;

    let parsed: Value = serde_json::from_slice(&raw).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: bad JSON header: {e}", path.display()),
        )
    })?;

    let obj = parsed.as_object().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: header is not a JSON object", path.display()),
        )
    })?;

    let mut metadata = None;
    let mut tensors = Vec::with_capacity(obj.len());

    for (name, v) in obj.iter() {
        if name == "__metadata__" {
            match v.as_object() {
                Some(m) => metadata = Some(m.clone()),
                None => {
                    return invalid(format!(
                        "{}: __metadata__ is not an object",
                        path.display()
                    ))
                }
            }
            continue;
        }

        let t = v.as_object().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} is not an object", path.display()),
            )
        })?;

        let dtype_str = t.get("dtype").and_then(Value::as_str).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} missing dtype", path.display()),
            )
        })?;
        let dtype = Dtype::parse(dtype_str).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} unknown dtype {dtype_str}", path.display()),
            )
        })?;

        let shape_arr = t.get("shape").and_then(Value::as_array).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} missing shape", path.display()),
            )
        })?;
        let mut shape: Vec<u64> = Vec::with_capacity(shape_arr.len());
        for d in shape_arr {
            match d.as_u64() {
                Some(v) => shape.push(v),
                None => {
                    return invalid(format!(
                        "{}: entry {name} shape has a non-integer dimension {d}",
                        path.display()
                    ))
                }
            }
        }

        let offs = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}: entry {name} missing data_offsets", path.display()),
                )
            })?;
        if offs.len() != 2 {
            return invalid(format!("{}: entry {name} data_offsets is not a pair", path.display()));
        }
        let start = offs[0].as_u64().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} data_offsets[0] is not a u64", path.display()),
            )
        })?;
        let end = offs[1].as_u64().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} data_offsets[1] is not a u64", path.display()),
            )
        })?;

        if end < start {
            return invalid(format!(
                "{}: entry {name} has end {end} < start {start}",
                path.display()
            ));
        }
        if end > buffer_size {
            return invalid(format!(
                "{}: entry {name} ends at {end} beyond buffer of {buffer_size} bytes",
                path.display()
            ));
        }

        // Offsets, shape and dtype must agree. This is the check that catches a
        // header which is well-formed JSON but describes the wrong byte spans.
        let numel = shape_numel(&shape).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} shape product overflows u64", path.display()),
            )
        })?;
        let expected = numel.checked_mul(dtype.size()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: entry {name} byte size overflows u64", path.display()),
            )
        })?;
        if expected != end - start {
            return invalid(format!(
                "{}: entry {name} declares {} bytes but shape {:?} x {} needs {}",
                path.display(),
                end - start,
                shape,
                dtype.as_str(),
                expected
            ));
        }

        tensors.push(TensorInfo { name: name.clone(), dtype, shape, start, end });
    }

    // Deterministic order; the range planner relies on it.
    tensors.sort_by_key(|t| (t.start, t.end));

    Ok(ShardHeader { path: path.to_path_buf(), file_size, header_len, data_start, metadata, tensors })
}

/// All `*.safetensors` files in `dir`, sorted by name.
pub fn discover_shards(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            out.push(p);
        }
    }
    out.sort();
    if out.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{}: no .safetensors files", dir.display()),
        ));
    }
    Ok(out)
}

/// Read every shard's header in `dir`.
pub fn read_model(dir: &Path) -> std::io::Result<Vec<ShardHeader>> {
    discover_shards(dir)?.iter().map(|p| read_header(p)).collect()
}

/// The `model.safetensors.index.json` weight map, when present.
///
/// Returns `tensor name -> shard file name`.
pub fn read_index(dir: &Path) -> std::io::Result<Option<BTreeMap<String, String>>> {
    let p = dir.join("model.safetensors.index.json");
    if !p.exists() {
        return Ok(None);
    }
    let raw = std::fs::read(&p)?;
    let v: Value = serde_json::from_slice(&raw).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: bad index json: {e}", p.display()),
        )
    })?;
    let wm = match v.get("weight_map").and_then(Value::as_object) {
        Some(m) => m,
        None => return Ok(None),
    };
    let mut out = BTreeMap::new();
    for (k, val) in wm {
        if let Some(s) = val.as_str() {
            out.insert(k.clone(), s.to_string());
        }
    }
    Ok(Some(out))
}

/// Cross-check `index.json` against the tensors actually present in shards.
///
/// Returns `(listed, found, missing_from_shards, not_in_index)`.
pub fn cross_check_index(
    headers: &[ShardHeader],
    index: &BTreeMap<String, String>,
) -> (usize, usize, Vec<String>, Vec<String>) {
    let present: std::collections::BTreeSet<&str> =
        headers.iter().flat_map(|h| h.tensors.iter().map(|t| t.name.as_str())).collect();
    let listed: std::collections::BTreeSet<&str> = index.keys().map(String::as_str).collect();
    let missing = listed.difference(&present).map(|s| s.to_string()).collect();
    let extra = present.difference(&listed).map(|s| s.to_string()).collect();
    (listed.len(), present.len(), missing, extra)
}
