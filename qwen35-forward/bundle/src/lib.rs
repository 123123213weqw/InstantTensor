//! Reader and comparator for `qwen35-golden-v1` bundles.
//!
//! A bundle is raw little-endian f32 plus a JSON manifest. The contract for a
//! from-scratch implementation is deliberately narrow: **write your
//! intermediates in the same layout, then run `bundlecmp golden/ yours/`** and
//! get a per-tensor error report.
//!
//! Layout:
//!
//! ```text
//! <bundle>/manifest.json
//! <bundle>/weights/<name>.f32          raw f32, C-contiguous, LE
//! <bundle>/intermediates/<name>.f32
//! <bundle>/units/<name>.f32
//! ```
//!
//! A tensor's name in the manifest is the module path with `.` replaced by `__`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct TensorEntry {
    pub group: String,
    pub name: String,
    pub file: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub numel: usize,
    pub nbytes: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GreedyStep {
    pub step: usize,
    pub input_len: usize,
    pub argmax_token: u32,
    pub topk_tokens: Vec<u32>,
    pub topk_logits: Vec<f32>,
    pub logit_sum: f32,
    pub logit_max: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Greedy {
    pub steps: Vec<GreedyStep>,
    pub final_ids: Vec<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Source {
    pub model: String,
    pub transformers: Option<String>,
    pub torch: Option<String>,
    pub seed: u64,
    pub params: u64,
    #[serde(default = "one")]
    pub ssm_gain: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct Manifest {
    pub schema: String,
    pub byte_order: String,
    pub dtype: String,
    pub source: Source,
    pub config: serde_json::Value,
    pub prompt_ids: Vec<u32>,
    pub tensors: Vec<TensorEntry>,
    pub greedy: Greedy,
}

#[derive(Debug)]
pub struct Bundle {
    pub root: PathBuf,
    pub manifest: Manifest,
    index: BTreeMap<String, TensorEntry>,
}

impl Bundle {
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let raw = std::fs::read(root.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_slice(&raw).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: bad manifest: {e}", root.display()),
            )
        })?;

        if manifest.schema != "qwen35-golden-v1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: unexpected schema {:?}", root.display(), manifest.schema),
            ));
        }
        if manifest.byte_order != "little" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: byte_order is {:?}, expected little", root.display(), manifest.byte_order),
            ));
        }
        if manifest.dtype != "f32" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: dtype is {:?}, expected f32", root.display(), manifest.dtype),
            ));
        }

        let mut index = BTreeMap::new();
        for t in &manifest.tensors {
            if index.insert(t.name.clone(), t.clone()).is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}: duplicate tensor name {}", root.display(), t.name),
                ));
            }
        }

        Ok(Self { root, manifest, index })
    }

    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.index.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    /// Read a tensor and validate its length against the manifest.
    pub fn read(&self, name: &str) -> std::io::Result<Vec<f32>> {
        let e = self.index.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("bundle has no tensor {name}"),
            )
        })?;
        self.read_entry(e)
    }

    pub fn read_entry(&self, e: &TensorEntry) -> std::io::Result<Vec<f32>> {
        let path = self.root.join(&e.file);
        let raw = std::fs::read(&path)?;
        if raw.len() != e.nbytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: {} bytes on disk, manifest says {}",
                    path.display(),
                    raw.len(),
                    e.nbytes
                ),
            ));
        }
        let mut out = Vec::with_capacity(e.numel);
        out.extend(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        debug_assert_eq!(out.len(), e.numel);
        Ok(out)
    }

    /// Write a tensor in the bundle's own layout, so an implementation can
    /// produce a candidate bundle with the same reader.
    pub fn write(&self, group: &str, name: &str, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
        let dir = self.root.join(group);
        std::fs::create_dir_all(&dir)?;
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let rel = format!("{group}/{name}.f32");
        std::fs::write(self.root.join(&rel), &bytes)?;
        let _ = shape;
        Ok(())
    }
}

/// Per-tensor error report.
#[derive(Debug, Clone)]
pub struct TensorDiff {
    pub name: String,
    pub shape: Vec<usize>,
    pub max_abs: f32,
    pub max_abs_index: usize,
    pub golden_at_max: f32,
    pub candidate_at_max: f32,
    pub golden_scale: f32,
    /// `max_abs / max(|golden|)`, the number that is actually comparable
    /// across tensors of different magnitude.
    pub max_rel: f32,
}

impl TensorDiff {
    pub fn describe(&self) -> String {
        format!(
            "{:<52} abs={:<11.3e} rel={:<11.3e} at[{:<7}] golden={:<12.6} cand={:<12.6}",
            self.name, self.max_abs, self.max_rel, self.max_abs_index,
            self.golden_at_max, self.candidate_at_max
        )
    }
}

/// Compare one tensor; `None` when the two agree bit for bit.
pub fn diff(name: &str, shape: &[usize], golden: &[f32], cand: &[f32]) -> Option<TensorDiff> {
    if golden.len() != cand.len() {
        return Some(TensorDiff {
            name: name.to_string(),
            shape: shape.to_vec(),
            max_abs: f32::INFINITY,
            max_abs_index: 0,
            golden_at_max: f32::NAN,
            candidate_at_max: f32::NAN,
            golden_scale: 0.0,
            max_rel: f32::INFINITY,
        });
    }
    let mut worst = 0f32;
    let mut worst_i = 0usize;
    let mut scale = 0f32;
    for (i, (g, c)) in golden.iter().zip(cand.iter()).enumerate() {
        if !g.is_finite() || !c.is_finite() {
            return Some(TensorDiff {
                name: name.to_string(),
                shape: shape.to_vec(),
                max_abs: f32::INFINITY,
                max_abs_index: i,
                golden_at_max: *g,
                candidate_at_max: *c,
                golden_scale: 0.0,
                max_rel: f32::INFINITY,
            });
        }
        let d = (g - c).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
        scale = scale.max(g.abs());
    }
    if worst == 0.0 {
        return None;
    }
    Some(TensorDiff {
        name: name.to_string(),
        shape: shape.to_vec(),
        max_abs: worst,
        max_abs_index: worst_i,
        golden_at_max: golden[worst_i],
        candidate_at_max: cand[worst_i],
        golden_scale: scale,
        max_rel: if scale > 0.0 { worst / scale } else { f32::INFINITY },
    })
}

/// Check that a manifest's recorded token ids agree with its own logits.
///
/// The comparator must not simply trust `greedy.final_ids`: a candidate could
/// record correct ids while its logits are entirely wrong (that is exactly what
/// a `x*w` instead of `x*(1+w)` norm bug produces — all-zero logits with an
/// unrelated id list). Deriving `argmax` from the candidate's own last-step
/// logits closes that hole.
///
/// Returns the indices where the recorded id disagrees with `argmax(logits)`.
pub fn check_token_selfconsistency(b: &Bundle) -> Vec<(usize, u32, u32)> {
    let steps = &b.manifest.greedy.steps;
    let mut bad = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        let name = format!("greedy_step{:02}__logits", s.step);
        let Ok(logits) = b.read(&name) else { continue };
        if logits.is_empty() {
            continue;
        }
        let mut best = 0usize;
        for (j, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = j;
            }
        }
        if best as u32 != s.argmax_token {
            bad.push((i, s.argmax_token, best as u32));
        }
    }
    bad
}

#[derive(Debug, Clone)]
pub struct TokenDiff {
    pub first_divergence: Option<usize>,
    pub golden_ids: Vec<u32>,
    pub candidate_ids: Vec<u32>,
    pub max_logit_delta: f32,
}

pub fn diff_tokens(golden: &Manifest, cand: &Manifest) -> TokenDiff {
    let g = &golden.greedy.final_ids;
    let c = &cand.greedy.final_ids;
    let first = (0..g.len().min(c.len())).find(|&i| g[i] != c[i]);
    let first = first.or(if g.len() != c.len() {
        Some(g.len().min(c.len()))
    } else {
        None
    });

    let mut max_logit = 0f32;
    for (gs, cs) in golden.greedy.steps.iter().zip(cand.greedy.steps.iter()) {
        for (a, b) in gs.topk_logits.iter().zip(cs.topk_logits.iter()) {
            max_logit = max_logit.max((a - b).abs());
        }
    }
    TokenDiff {
        first_divergence: first,
        golden_ids: g.clone(),
        candidate_ids: c.clone(),
        max_logit_delta: max_logit,
    }
}
