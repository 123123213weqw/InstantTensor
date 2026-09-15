# rust-qwen-engine — the loading layer

A Rust rewrite of the safetensors weight-loading layer. **The goal is not to
make the disk faster; it is to remove Python start-up cost.**

## Why

Measured breakdown of a 7.2B cold start against a Python baseline:

| Stage | Time | Share | Cause |
|---|---|---|---|
| `import torch` | 0.665 s | 13.5% | Python interpreter + C-extension import |
| `load_repo` | 1.871 s | 37.9% | Python module import machinery |
| Building 447 quantized modules | 0.666 s | 13.5% | Python object construction |
| **IT read** | 1.682 s | 34.1% | Disk I/O, **already at 98% of the device ceiling** |
| Total | **4.933 s** | | |

The fixed cost `F ≈ 2.8 s` is entirely Python runtime. The 34.1% spent on I/O is
already at the disk ceiling (4.25 / 4.33 GB/s). **Rust removes 3.202 s (64.9%)
and contributes nothing to I/O.**

```
today:  F 2.80s + W 2.74s = 5.54s  ->  1.58x
Rust:   F 0.40s + W 2.74s = 3.14s  ->  ~2.8x   (conservative projection)
Rust:   F 0.01s + W 2.74s = 2.75s  ->  ~3.2x   (measured; see "Start-up cost")
```

Acceptance target: **7.2B cold start, 5.54 s → 3.2 s.** The final number has to
be taken on the 4080 (4.33 GB/s device); development and verification happen on
a V100 (2.2 GB/s), where the proportions differ.

## Layout

```
crates/stloader/         library
  header.rs   safetensors header parsing: 8-byte LE length + JSON;
              multiple shards and index.json
  plan.rs     tensor byte ranges -> block-aligned read requests (merging)
  reader.rs   io_uring + O_DIRECT, bounded depth, pooled buffers
  cache.rs    posix_fadvise eviction + mincore verification
  aligned.rs  explicitly aligned buffer allocation
crates/stbench/          CLI: list / read / verify / selftest / fuzz
tools/                   Python reference tools (GGUF inspection, alignment)
```

### Why read requests must be merged

safetensors packs tensors back-to-back, but the header length is only guaranteed
to be 8-byte aligned, so a tensor's start offset is almost never 4096-aligned.
`O_DIRECT` requires both offset and length to be block-aligned, which forces
every tensor to be rounded outward.

Measured on `Qwen3.8-27B` (18 shards, 1199 tensors, 51.747 GiB):

```
internal gaps = 0B    head = 0B    tail = 0B      <- perfectly packed
unaligned offset: 1199/1199 (100.0%)              <- header length is not a multiple of 4096
unaligned size:    533/1199 (44.5%)
requests: naive=1199  ->  coalesced=18             <- one per shard
read amplification = 1.0000x                      <- zero waste
```

**Merging wastes nothing.** That is why `plan.rs` exists; without merging this
degenerates to 1199 requests.

## Verified

| Item | Method | Result |
|---|---|---|
| Header parsing correctness | Rust output vs an independent Python implementation | **identical** (1199 tensors, 51.747 GiB, 18 requests, 1.0000x, BF16×1199, all 1199 index.json entries present) |
| Header parsing speed | all headers of 18 shards | **2.6 ms** |
| O_DIRECT data correctness | 40 smallest tensors, direct vs buffered byte-for-byte | **0 mismatches** |
| index.json cross-check | `weight_map` vs tensors actually in shards | 1199 entries, **0 missing** |

Comparing the smallest tensors is deliberate: they are the ones **least likely
to be block-aligned**, so they exercise the rounding logic hardest.

## Usage

```bash
cargo build --release

# tensor inventory + alignment stats + read plan
stbench list <model_dir>

# cold-cache throughput (--cold evicts via posix_fadvise and reports mincore residency)
stbench read <model_dir> --depth 32 --chunk-mb 1 --cold
stbench read <model_dir> --depth 32 --chunk-mb 1 --cold --buffered

# control: one request per tensor, exposing the alignment penalty
stbench read <model_dir> --cold --per-tensor

# correctness: O_DIRECT vs buffered, byte for byte
stbench verify <model_dir> --tensors 40
```

## Requirements

- Linux kernel ≥ 5.6 for `io_uring`, ≥ 5.15 recommended; needs
  `kernel.io_uring_disabled = 0`
- A filesystem that supports `O_DIRECT`

## Measured

Development and verification host: Linux, kernel 6.8, rustc 1.95, single NVMe.
Test model: `Qwen3.8-27B` (18 shards, 1199 tensors, 51.747 GiB, BF16).
Every run is preceded by `posix_fadvise(DONTNEED)` + `sync`, and timing only
starts once **mincore confirms `residency = 0.000`**.

### Throughput

| Configuration | Time | GB/s | Requests |
|---|---|---|---|
| `dd` single-stream O_DIRECT (3.2 GiB reference) | 3.148 s | **1.10** | — |
| Rust depth=8, chunk=1 MiB | 31.139 s | 1.78 | 53004 |
| **Rust depth=32, chunk=1 MiB** | **25.569 s** | **2.17** | 53004 |
| Rust depth=64, chunk=1 MiB | 25.522 s | 2.18 | 53004 |
| Rust depth=32, chunk=8 MiB | 25.501 s | 2.18 | 6634 |
| Rust depth=32, chunk=1 MiB, **buffered** | 36.622 s | **1.52** | 53004 |
| Rust **one request per tensor** (control) | 25.786 s | 2.16 | 54081 (1.0001x) |

Conclusions:

1. **O_DIRECT is 43% faster than buffered** (2.17 vs 1.52 GB/s) — cold loads must
   bypass the page cache.
2. **Depth saturates at 32** (32→64 is +0.5%) and **chunk size is flat**
   (1 MiB vs 8 MiB is +0.5%) — the parameter landscape has no slope here.
3. **2.17 GB/s exceeds the 2.0 GB/s previously taken as this device's ceiling**;
   single-stream `dd` reaches only 1.10 GB/s, so concurrency is worth roughly 2×.
4. Peak RSS 0.03–0.07 GiB (the depth × chunk buffer pool); no memory growth.

### Start-up cost (the point of this tree)

| | Measured |
|---|---|
| Rust process start + parsing all 18 shard headers | **wall 0.00–0.01 s**, maxRSS **2–2.5 MB** |
| Header parsing alone | 2.5–9.6 ms |
| (baseline) Python `import torch` + `load_repo` | **2.80 s** |

**Rust takes the fixed cost F from 2.80 s to roughly 0.01 s.** Projected onto a
7.2B cold start:

```
today:  F 2.80s + W 2.74s = 5.54s  ->  1.58x
Rust:   F 0.01s + W 2.74s = 2.75s  ->  ~3.2x
```

### Cross-model validation

| Model | Shards | Tensors | Size | Requests | Amplification | dtype |
|---|---|---|---|---|---|---|
| `Qwen3.8-27B` | 18 | 1199 | 51.747 GiB | 18 | 1.0000x | BF16×1199 |
| `DeepSeek-R1-Distill-Qwen-7B` | 2 | 339 | 14.185 GiB | 2 | 1.0000x | BF16×339 |
| `rwkv7-g1g-1.5b-hf` | 6 | 795 | 2.845 GiB | 6 | 1.0000x | F16×795 |

For all three, the number of `index.json` entries **exactly matches** the number
of tensors in the shards, with none missing. DeepSeek-7B cold read: **2.21 GB/s**.

### A correction worth recording

The expected finding was that merging adjacent tensors was the performance
lever. It is not: **the tensors in these three models are perfectly packed**
(`internal gaps = 0B`), and reads are split by chunk size anyway, so **merged and
unmerged throughput are nearly the same** (2.16 vs 2.17 GB/s).

The value of merging is in the plan: it takes requests from 1199 down to 18, not
throughput. Merging would only become a throughput lever on a file with **large
holes or heavy fragmentation**, and none of the test models has that.

## Not done yet

This tree only gets bytes off the disk at device speed and verifies them. It does
not yet:

1. **Hand anything to a consumer** — how weights reach device memory and in what
   layout they are passed to a hand-written Qwen compute path (a pinned-buffer +
   `cudaMemcpyAsync` pipeline, or a direct copy into device memory).
2. **Build quantized weights** — if the r1mm8/w8row path is used, that
   construction cost (0.666 s) has to be added back into the budget.
3. **End-to-end acceptance on the 4080** — the `5.54 s → 3.2 s` target is stated
   for a 4.33 GB/s device; this host is a 2.2 GB/s V100, so the ratios differ.

## Robustness

### Thirteen problems found by self-review and fixed

The first version only proved that **valid** files read correctly; it never asked
what happens with **invalid** ones. Self-review turned up the following:

| # | Problem | Consequence | Fix |
|---|---|---|---|
| 1 | `a1 - a0` in `plan.rs` could **underflow** | release builds have overflow checks off → wraps to a huge u64 → **enormous allocation / OOM** | skip when `a1 <= a0`; never emit |
| 2 | `last.offset + last.len + max_gap` overflowed | the merge decision wrapped, wrongly merging or refusing | `saturating_add` |
| 3 | `data_start + self.end` unchecked | same class as 1 | enforced as a parse-time invariant |
| 4 | `end <= buffer_size` unvalidated | a malformed header could plan a read past the file | rejected at parse time |
| 5 | `offsets` not cross-checked against `shape × dtype` | valid JSON describing the wrong byte spans would be silently accepted | parse-time cross-check |
| 6 | `ceil_to` could overflow | large inputs wrapped | `saturating_mul` |
| 7 | `ReadConfig.keep_open` was a **dead field** | advertised a capability that did not exist | removed |
| 8 | `read_files` **panicked on an out-of-bounds index** for a mismatched plan | crash instead of error | `InvalidInput` error |
| 9 | `submit_and_wait` treated `EINTR` as fatal | a signal would be reported as a read failure | retry |
| 10 | **A zero-element tensor emitted a spurious read request** (reading header padding) | wasted I/O | skip `nbytes() == 0` |
| 11 | `shape` parsing reported `u64::MAX` as "non-integer" | misleading error | distinguish the two cases |
| 12 | No tests at all | — | see below |
| 13 | `verify` only covered tensors ≤ 64 KiB | the large tensors holding nearly all the bytes were never checked | added large-tensor coverage |

**Number 10 was caught by a test I wrote, not by review** — which is the point of
having tests.

### Tests

```bash
cargo test --release        # 3 tests, including 28 synthetic cases
stbench selftest            # 28 synthetic robustness cases, printed individually
```

`stbench selftest` **generates malformed safetensors on the spot** and asserts
behaviour, so it needs no real model:

```
[PASS] hdr_zero                          rejected: header length is zero
[PASS] hdr_overrun                       rejected: header length 4096 overruns file of 10 bytes
[PASS] hdr_absurd                        rejected: header length 1099511627776 exceeds 268435456
[PASS] hdr_short                         rejected: file shorter than 8-byte header length
[PASS] json_bad                          rejected: bad JSON header
[PASS] json_array                        rejected: header is not a JSON object
[PASS] entry_not_obj                     rejected: entry a is not an object
[PASS] meta_not_obj                      rejected: __metadata__ is not an object
[PASS] dtype_unknown                     rejected: unknown dtype F7
[PASS] missing_shape                     rejected: missing shape
[PASS] offsets_not_pair                  rejected: data_offsets is not a pair
[PASS] end_before_start                  rejected: end 10 < start 100
[PASS] end_beyond_buffer                 rejected: ends at 1024 beyond buffer of 8 bytes
[PASS] shape_dtype_mismatch              rejected: declares 20 bytes but shape [10] x F32 needs 40
[PASS] shape_noninteger                  rejected: shape has a non-integer dimension "x"
[PASS] offset_u64max                     rejected: ends at 18446744073709551615 beyond buffer of 16
[PASS] shape_overflow                    rejected: shape product overflows u64
[PASS] off_by_one                        rejected: ends at 17 beyond buffer of 16 bytes
[PASS] valid: two contiguous tensors     tensors=2 tensor_bytes=8192 ranges=1
[PASS] valid: zero-element tensor        tensors=1 tensor_bytes=0 ranges=0
[PASS] valid: scalar tensor shape []     tensor_bytes=4
[PASS] gap: merge respects max_gap       ranges tight=2 loose=1
[PASS] arithmetic: ceil_to saturates, floor_to safe, no panic
[PASS] arithmetic: shape_numel overflow -> None
[PASS] api: out-of-range shard index rejected
[PASS] api: zero-length range rejected
[PASS] api: empty plan is a no-op
[PASS] plan: no zero-length, in-range, non-overlapping

28 cases, 28 passed, 0 failed
```

There is also a **known-pattern round-trip test**: it builds a file with 300
bytes of padding followed by an 8192-byte deterministic pattern, arranged so the
tensor offset is **deliberately unaligned**, reads it back through `O_DIRECT`,
and compares byte for byte — an end-to-end check of the alignment and slicing
logic.

### Corruption tests on real files

Variants built from a real 512 MiB shard (`rwkv7-g1g-1.5b-hf`):

| Variant | Result |
|---|---|
| payload truncated to half | `entry lm_head.weight ends at 268435456 beyond buffer of 268435344 bytes` |
| **payload short by exactly one byte** | `entry model.embeddings.weight ends at 536870912 beyond buffer of 536870911 bytes` |
| header only, no payload | `ends at 268435456 beyond buffer of 0 bytes` |
| intact (control) | passes, `tensors=2 planned: 1 requests amplification 1.0000x` |
| `read` on a truncated file | also fails at parse time; never enters the read path |
| entirely random bytes | `header length 9636161184222581920 exceeds 268435456` |
| empty file | `file shorter than 8-byte header length` |
| directory with no `.safetensors` | `no .safetensors files` |

**All fail cleanly — no panic, no OOM, no enormous allocation.** A one-byte
truncation is detected exactly.

### A design fact that has to be recorded

The plan rounds **up past end-of-file**, and the read is closed out by a short
count at EOF (the 18 `short_reads` in the benchmark are the 18 shards). This is
the only correct choice under `O_DIRECT`: **clamping the length to `file_size`
would break block alignment and the kernel returns `EINVAL`.**

### Built-in fuzzer (`stbench fuzz`)

Hand-written cases only cover what their author thought of. This crate carries a
mutational fuzzer that depends on no external tooling — no nightly, no libFuzzer,
no sanitizer:

```bash
cargo build --profile fuzz
stbench fuzz --iters 1000000 --seed 1 --read
```

It does two things, and both are necessary:

1. **Catches panics.** The whole parse/plan/read pipeline runs inside
   `catch_unwind`, so an out-of-bounds index or an unwrap becomes a recorded
   failure rather than killing the run.
2. **Checks a real safety oracle.** Not crashing is not enough. **If a header
   parses**, the plan derived from it must satisfy:
   - every range block-aligned on both ends and non-empty;
   - every range inside `[0, ceil(file_size, block)]` → **cannot read outside the file**;
   - ranges sorted and non-overlapping;
   - **every tensor's absolute byte span fully contained in some range** → cannot skip a tensor;
   - **every range covers at least some tensor bytes** → cannot read nothing.

The last is an *efficiency* check; the first four are *safety* checks. Safety
checks alone miss a whole class of bug (see below).

#### Validating the oracle itself

"Fuzzing came back clean" has two readings: the code is fine, or **the oracle is
blind**. So bugs were injected deliberately to see whether it complains:

| Injected bug | Caught? | Diagnostic |
|---|---|---|
| removed the `.min(file_ceil)` clamp | **no (correct)** | parsing already guarantees `abs_e <= file_size`, so `ceil_to(abs_e) <= file_ceil` always holds — the clamp is **redundant**, not a bug |
| `a0` used `ceil_to` instead of `floor_to` | **yes** | `tensor a [127,4223) not contained in range [4096,8192)` |
| same, other manifestation | **yes** | `tensor w [96,104) has no covering range` |
| removed the zero-element skip | **missed before, caught on iteration 3 after** | `range [0, 4096) covers no tensor bytes (wasted read)` |

#### A real lesson: the fuzzer and the hand-written cases cover different things

After injecting the "removed zero-element skip" bug:

| | Result |
|---|---|
| **hand-written `selftest`** | **caught it** — `[FAIL] valid: zero-element tensor  tensors=1 tensor_bytes=0 ranges=1` |
| **fuzzer (before the improvement)** | **missed it** — randomly generated headers **almost never** produce the combination "valid file + degenerate tensor (zero-element / scalar / unaligned offset)" |

The fix was not to change the oracle but to **give the fuzzer a strategy that
replays the seed corpus verbatim**, without mutation. After that the same bug was
caught on iteration 3.

So: **28 hand-written cases are not a substitute for a fuzzer; the two are
complementary.** The hand-written cases encode "semantic boundaries I know
about"; the fuzzer covers "structural malformation I did not think of".

#### Large runs

| Version | Scale | Parsed | Correctly rejected | Reads executed | Result |
|---|---|---|---|---|---|
| before | 3 × 400,000 = 1.2M (with reads) | 157k | 1.04M | 157k | clean |
| **after** | **8 × 250,000 = 2.0M (with reads)** | **625,729** | **1,374,271** | **625,729** | **clean** |

The eight seeds (11/22/33/44/55/66/77/88) each ran independently and all reported
`RESULT: clean (no panics, no invariant violations)` at roughly 3,200 cases/s.
Also reported: `dtype table consistent: true` (the dtype names and the enum
round-trip in both directions).

`panic = "abort"` conflicts with `catch_unwind`, so the fuzzer must use the
`fuzz` profile (`inherits = "release"` + `panic = "unwind"`). This is not
optional — the wrong profile turns a caught panic into process termination and
the fuzzer becomes decorative.

### Still not done

- **No coverage guidance.** The built-in fuzzer mutates at random; unlike
  `cargo-fuzz` / libFuzzer it does not evolve toward new code paths, so its
  ability to find deeply buried bugs is weaker, and runs are minutes rather than
  long-lived fuzzing.
- 28 hand-written cases, 8 real-file variants and 2M mutations — but **none of
  them is a full-byte comparison**.
- **Not every byte is covered.** `verify` compares the 48 smallest tensors and
  the first 4 MiB of the 4 largest, not all 51.7 GiB.
- **Single thread, single ring.** No multi-ring, no multi-threading, no NUMA or
  CPU affinity.
- **`--gap` is unverified on a real file with holes** (all three test models have
  zero gaps between tensors).
- **The shard set is not cross-checked against the file names in `index.json`**;
  only the tensor-name sets are compared.
- **No mmap path to compare against** (`--buffered` is `read(2)`, not mmap).
- **No concurrent-load test** (two processes loading the same model).

## Related

- [`../qwen35-forward/`](../qwen35-forward/) — golden-reference generator and
  comparator for a Qwen3.5 forward pass (the other tree in this repository)
- [`../docs/loader-internals.md`](../docs/loader-internals.md) — internals of
  Siphon itself (Python/C++)
- [`../docs/benchmark.md`](../docs/benchmark.md) — Siphon's original H200
  benchmarks

### Cold-cache methodology

Every throughput number here was taken with a **cold cache**: `posix_fadvise`
with `DONTNEED` followed by `sync`, then a `mincore` sample to confirm residency
is `0.000` before the clock starts (`stbench read --cold` does both and prints
the result). A number taken without this step is meaningless — the page cache
turns it into a memory-bandwidth test.
