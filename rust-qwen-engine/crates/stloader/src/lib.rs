//! `stloader` — a Rust safetensors reader built for cold-start load throughput.
//!
//! Pipeline:
//!
//! 1. [`header`] parses shard headers (8-byte LE length + JSON) without
//!    touching tensor payloads.
//! 2. [`plan`] turns tensor byte ranges into block-aligned read requests,
//!    merging adjacent tensors so `O_DIRECT` alignment padding does not turn
//!    into extra I/O.
//! 3. [`reader`] executes those requests with `io_uring` at bounded depth into
//!    a fixed pool of aligned buffers.
//! 4. [`cache`] evicts and inspects the page cache so cold-load numbers are
//!    trustworthy.

pub mod aligned;
pub mod cache;
pub mod fuzz;
pub mod header;
pub mod plan;
pub mod reader;
pub mod selftest;

pub use aligned::AlignedBuf;
pub use header::{
    cross_check_index, discover_shards, read_header, read_index, read_model, shape_numel, Dtype,
    ShardHeader, TensorInfo,
};
pub use plan::{
    ceil_to, floor_to, plan_model, plan_model_unmerged, plan_shard, plan_shard_unmerged, Range,
};
pub use reader::{read_files, read_small_range, ReadConfig, ReadStats};

/// Model-wide totals, useful for reporting and for cross-checking against
/// other tools.
#[derive(Debug, Default, Clone)]
pub struct ModelSummary {
    pub shards: usize,
    pub tensors: usize,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub tensor_bytes: u64,
    pub planned_bytes: u64,
    pub planned_requests: usize,
    pub unaligned_offsets: usize,
    pub unaligned_sizes: usize,
}

impl ModelSummary {
    pub fn amplification(&self) -> f64 {
        if self.tensor_bytes == 0 {
            0.0
        } else {
            self.planned_bytes as f64 / self.tensor_bytes as f64
        }
    }
}

/// Summarise a model directory, including the aligned-read plan.
pub fn summarize(headers: &[ShardHeader], block: u64, max_gap: u64) -> ModelSummary {
    let plan = plan_model(headers, block, max_gap);
    summarize_plan(headers, block, &plan)
}

/// Summarise a model directory against an explicit plan.
pub fn summarize_plan(
    headers: &[ShardHeader],
    block: u64,
    plan: &[(usize, Range)],
) -> ModelSummary {
    let mut s = ModelSummary {
        shards: headers.len(),
        planned_requests: plan.len(),
        ..Default::default()
    };
    for h in headers {
        s.tensors += h.tensors.len();
        s.file_bytes += h.file_size;
        s.header_bytes += h.header_len;
        s.tensor_bytes += h.tensor_bytes();
        for t in &h.tensors {
            let (abs_s, abs_e) = t.abs_range(h.data_start);
            if abs_s % block != 0 {
                s.unaligned_offsets += 1;
            }
            if (abs_e - abs_s) % block != 0 {
                s.unaligned_sizes += 1;
            }
        }
    }
    s.planned_bytes = plan.iter().map(|(_, r)| r.len).sum();
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("stloader_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("create scratch");
        d
    }

    #[test]
    fn synthetic_robustness_cases_all_hold() {
        let dir = scratch("selftest");
        let checks = selftest::run(&dir).expect("selftest runs");
        let failed: Vec<&selftest::Check> = checks.iter().filter(|c| !c.pass).collect();
        for c in &failed {
            eprintln!("FAIL {} :: {}", c.name, c.detail);
        }
        assert!(
            failed.is_empty(),
            "{} of {} robustness cases failed",
            failed.len(),
            checks.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a file whose tensor bytes are a known pattern and whose tensor
    /// offsets are deliberately not block-aligned, then read one tensor back
    /// through the O_DIRECT path and compare byte for byte.
    ///
    /// This exercises the padding/slicing logic end to end rather than just the
    /// parser.
    #[test]
    fn round_trip_reads_exact_bytes() {
        let dir = scratch("roundtrip");
        const PAYLOAD: u64 = 8492;

        // 300 bytes of JSON-ish padding ahead of the payload makes data_start
        // land off a 4096 boundary on purpose.
        let name_a = "pad_a";
        let name_b = "pattern_b";
        let header = format!(
            r#"{{"{name_a}":{{"dtype":"U8","shape":[300],"data_offsets":[0,300]}},"{name_b}":{{"dtype":"U8","shape":[8192],"data_offsets":[300,8492]}}}}"#
        );

        let mut payload = Vec::new();
        payload.extend((0..300u32).map(|i| (i % 251) as u8));
        let pattern: Vec<u8> = (0..8192u32).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        payload.extend_from_slice(&pattern);

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(header.as_bytes());
        file_bytes.extend_from_slice(&payload);

        let path = dir.join("rt.safetensors");
        std::fs::write(&path, &file_bytes).expect("write");

        let h = header::read_header(&path).expect("parse");
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.buffer_bytes(), 8492);

        let t = h.tensors.iter().find(|t| t.name == name_b).expect("find");
        let (off, end) = t.abs_range(h.data_start);
        assert_ne!(off % 4096, 0, "test intent: tensor offset must be unaligned");

        let got = reader::read_small_range(&path, off, (end - off) as usize, true, 4096)
            .expect("O_DIRECT read");
        assert_eq!(got.len(), 8192);
        assert_eq!(got, pattern, "O_DIRECT bytes differ from the written pattern");

        let got_buf = reader::read_small_range(&path, off, (end - off) as usize, false, 4096)
            .expect("buffered read");
        assert_eq!(got, got_buf, "O_DIRECT and buffered disagree");

        // The plan rounds outward, so it can ask for up to a full block past EOF.
        // That is deliberate: clamping the length to file_size would make it
        // non-block-aligned and O_DIRECT would reject the request with EINVAL.
        // The final read therefore returns a short count at EOF, which is why
        // `short_reads` is tracked.
        let headers = vec![h];
        let plan = plan::plan_model(&headers, 4096, 0);
        let files = vec![path.clone()];
        let file_size = std::fs::metadata(&path).expect("stat").len();
        let planned: u64 = plan.iter().map(|(_, r)| r.len).sum();
        let stats = reader::read_files(&files, &plan, &ReadConfig::default()).expect("read_files");

        assert_eq!(stats.requests, 1, "one contiguous payload should need one request");
        assert_eq!(stats.short_reads, 1, "the final request should end at EOF");
        assert_eq!(stats.bytes, file_size, "reading from offset 0 must cover the whole file");
        assert!(planned >= PAYLOAD, "planned extent {planned} must cover the {PAYLOAD}-byte payload");
        assert!(
            planned <= file_size + 4096,
            "planned extent {planned} should not exceed the file plus one block"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A header that claims more bytes than the file holds must be rejected
    /// before any read is planned.
    #[test]
    fn truncated_file_is_rejected() {
        let dir = scratch("truncated");
        let header = r#"{"a":{"dtype":"U8","shape":[100000],"data_offsets":[0,100000]}}"#;
        let mut b = Vec::new();
        b.extend_from_slice(&(header.len() as u64).to_le_bytes());
        b.extend_from_slice(header.as_bytes());
        b.extend_from_slice(&[0u8; 16]); // far shorter than 100000
        let p = dir.join("short.safetensors");
        std::fs::write(&p, &b).expect("write");

        let err = header::read_header(&p).expect_err("must reject");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
