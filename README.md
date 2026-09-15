# Siphon

A fast safetensors weight loader, plus two companion trees that study the two
ends of the same problem: **getting weights off the disk**, and **proving a
from-scratch forward pass is numerically right**.

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

## What is in this repository

| Tree | Language | What it does |
|---|---|---|
| [`siphon/`](siphon/) + [`csrc/`](csrc/) | Python + C++ | The loader itself: `safe_open`, six I/O backends, automatic backend selection. Consumes safetensors on the host and delivers tensors to the GPU. |
| [`rust-qwen-engine/`](rust-qwen-engine/) | Rust | The same question asked from the other side: a safetensors reader on `io_uring` + `O_DIRECT`, built to measure how much of cold-start cost is the language runtime rather than the disk. |
| [`qwen35-forward/`](qwen35-forward/) | Python + Rust | Golden-reference generator and comparator for validating a hand-written Qwen3.5 forward pass, down to individual tensors and tokens. |

**The three trees share no code.** `rust-qwen-engine/` and `qwen35-forward/`
build with `cargo` and need no CUDA; they do not affect the extension's build,
its dependencies, or its runtime. Install and build instructions below apply to
the Python/C++ loader unless stated otherwise.

## The loader

### Measured

| Model | GPU | Backend | Load time (s) | Throughput (GB/s) | Speed-up |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1×H200 | `safetensors` | 57.4 | 1.1 | 1× |
| Qwen3-30B-A3B | 1×H200 | **Siphon** | 1.77 | 35 | **32.4×** |
| DeepSeek-R1 | 8×H200 | `safetensors` | 160 | 4.3 | 1× |
| DeepSeek-R1 | 8×H200 | **Siphon** | 15.3 | 45 | **10.5×** |

See [`docs/benchmark.md`](docs/benchmark.md) for the full set.

### Why it is faster

- **Direct I/O by default.** Skips page-cache allocation on cold start, which
  matters for large models and tight memory budgets.
- **Tuned I/O size and concurrency.** Chunk size, queue depth and worker count
  are derived from the storage device instead of being fixed constants.
- **Pipelining and prefetching.** The read, staging and transfer stages overlap
  rather than running one after another.
- **Distributed loading.** A `torch.distributed` (NCCL) process group
  coordinates the ranks, so a TP/PP/EP/CP/DP layout loads faster than each rank
  loading independently.
- **Six backends.** GPUDirect Storage, `io_uring`, Linux AIO, buffered variants
  of each, and `mmap` — picked per filesystem and per device capability.

### When it is worth using

Any of the following:

- Storage bandwidth ≥ 5 GB/s, where the loader can actually reach it.
- The model cannot stay cached in host memory — KV-cache offloading has taken
  the RAM, loads are infrequent, or several models are swapped in and out.
- The checkpoint is heavily sharded (TP=8 and up), so each GPU's reads are small
  and non-contiguous.
- Loading from `tmpfs`.

### Install

> **The package is not on PyPI under this name.** PyPI's `siphon` is an
> unrelated project ([Unidata/siphon](https://github.com/Unidata/siphon)). The
> name is taken, so `pip install siphon` installs that project, not this one.
> Install from source.

```bash
git clone https://github.com/123123213weqw/siphon.git
cd siphon
./checkout_submodules.sh     # dlpack, pybind11, libaio, liburing, atomic_queue
pip install .
```

Set `DEBUG=1` in the environment for a debug build.

**Requirements**

- Python ≥ 3.9, PyTorch ≥ 2.8.0
- CUDA or ROCm
- `URING` and `URING_BUFFERED` need Linux ≥ 5.6; ≥ 5.15 recommended
- The C++ extension vendors its dependencies as git submodules, so
  `./checkout_submodules.sh` must run before the first build

### Quickstart

```python
from siphon import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

Yielded tensors own their memory by default (`copy=True`).

**Multi-file.** Passing a list lets the backend plan the reads as a group,
which is faster than opening the shards one at a time:

```python
files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
tensors = {}
with safe_open(files, framework="pt", device=0) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

**Distributed.**

```python
import torch
import torch.distributed as dist
from siphon import safe_open

dist.init_process_group(backend="nccl")
process_group = dist.GroupMember.WORLD

with safe_open(files, framework="pt", device=torch.cuda.current_device(),
               process_group=process_group) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

A subgroup from `dist.new_group` also works, which lets independent groups load
independently — with TP=8 and PP=2, the two TP groups can each use their own.
For cross-node runs, per-node subgroups are sometimes faster than the world
group. The world group is the right default for most cases.

**Zero-copy.** `copy=False` skips the per-tensor clone and yields views into the
internal ring buffer:

```python
with safe_open(files, framework="pt", device=0, copy=False) as f:
    for name, tensor in f.tensors():
        model_param[name].copy_(tensor)
```

Two rules, and breaking either corrupts data silently:

1. Consume each tensor before the next one is yielded. `list(f.tensors())` and
   friends are unsafe when `buffer_size < total_tensor_size`.
2. Do not hold a reference past the `with` block; the buffer is freed on exit.

A `UserWarning` fires when `copy=False` and `buffer_size < total_tensor_size`.
Both `copy` and `buffer_size` are public attributes of the `safe_open` object.

`tests/test.py` is a full benchmark harness (TP/PP grouping, checksums, and
more).

### Backend selection

By default the backend is chosen automatically. You can pin candidates with the
`backend` argument, or with `SIPHON_BACKEND` when `backend=None`. Candidates are
tried in order and the first one both supported by the filesystem and available
on the machine wins.

```python
from siphon import Backend, BackendPolicy, safe_open

safe_open("model.safetensors", framework="pt", device=0, backend=Backend.URING)
safe_open("model.safetensors", framework="pt", device=0,
          backend=[Backend.URING, Backend.AIO])
safe_open("model.safetensors", framework="pt", device=0,
          backend=BackendPolicy.BUFFERED)
```

`BackendPolicy.BUFFERED` expands to
`[URING_BUFFERED, AIO_BUFFERED, MMAP]`. `SIPHON_BACKEND` takes the same names,
comma-separated:

```bash
SIPHON_BACKEND=URING,AIO
SIPHON_BACKEND=BUFFERED
```

**In-memory filesystems** (`tmpfs`, `ramfs`; backends `MMAP`, `URING_BUFFERED`,
`AIO_BUFFERED`): use `MMAP`. The others are usually slower for memory-backed
files.

**Regular filesystems** — direct or buffered:

- **Direct I/O** (`AIO`, `URING`, `CUFILE`) suits a model loaded once for a
  long-running job. It avoids page-cache cold-start cost and keeps the cache
  clean. `URING` is often the fastest on modern kernels; `AIO` has the widest
  platform support; `CUFILE` needs GPUDirect Storage and its throughput can be
  offset by cuFile initialisation cost.
- **Buffered I/O** (`AIO_BUFFERED`, `URING_BUFFERED`, `MMAP`) suits the same
  model loaded repeatedly in a short window, where later reads hit the page
  cache. The first read is normally slower than Direct I/O.

**With no backend specified**, the files are inspected first:

- `tmpfs`/`ramfs` files → `MMAP`.
- Regular disk files → the loader probes how much of each file is already
  page-cache resident and picks the matching family. Direct I/O deliberately
  bypasses the page cache, so without this probe a warm checkpoint would be
  re-read from the device.

  - Everything resident → Buffered, `[URING_BUFFERED, AIO_BUFFERED, MMAP]`.
  - Otherwise → Direct, `[URING, AIO]`.

  Residency is measured with `mincore(2)` over bounded windows sampled across
  each file, so the probe costs a fixed number of syscalls instead of one per
  page — the difference matters on multi-GB checkpoints.

  With a warm page cache this cut reload time by roughly **2.8–3.3×** across
  1.5B–13.3B checkpoints on a V100, while leaving cold loads unchanged.

`SIPHON_CACHE_RESIDENT_THRESHOLD` is the ratio required to take the Buffered
path (default `0.8`). A value above `1.0` can never be met, which disables the
probe and restores always-Direct-I/O. An explicit `backend` always wins — the
probe only runs when `backend=None` and `SIPHON_BACKEND` is unset, so a pinned
backend is never overridden.

If no candidate can be used, the error lists the reason each was rejected.

### Environment variables

Set before the first `safe_open` call. An explicit argument to `safe_open`
takes precedence over the matching variable.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SIPHON_BACKEND` | Comma-separated backend or policy candidates, e.g. `URING,AIO` or `BUFFERED`. | Selected from the filesystem type. |
| `SIPHON_BUFFER_SIZE` | Requested logical GPU tensor ring-buffer size, in bytes. Constrains `io_depth`, but may be enlarged to fit the largest tensor. | Derived from tensor sizes and I/O settings. |
| `SIPHON_CHUNK_SIZE` | File I/O chunk size, in bytes. | Derived from the backend. |
| `SIPHON_CONCURRENCY` | Worker threads for `MMAP` and `CUFILE`; other backends ignore it. | Derived from the backend. |
| `SIPHON_IO_DEPTH` | Maximum rank-local I/O operations in flight. Higher values can raise throughput and staging-memory use; the maximum is 1024. | Derived from the backend. |
| `SIPHON_MAX_FREE_MEM_USAGE` | Maximum fraction of currently free GPU memory the logical device buffer may use. | `0.5` |
| `SIPHON_CACHE_RESIDENT_THRESHOLD` | Residency ratio required before automatic selection prefers Buffered I/O. Above `1.0` disables the probe. | `0.8` |
| `SIPHON_CACHE_BUFFER` | `1` caches pinned host staging buffers across loader opens. Cached memory stays pinned until process cleanup. | `0` |
| `SIPHON_DEBUG` | `1` prints backend selection, buffer sizes, timings and throughput. | `0` |

The I/O capacity a configuration demands is
`round_up(chunk_size, page_size) * io_depth * world_size`. With `buffer_size`
omitted, the loader takes the larger of that and the tensor-layout
recommendation. If both `buffer_size` and `io_depth` are set they must be
compatible; if only `buffer_size` is set, `io_depth` is reduced as needed. The
device allocation carries a small extra alignment guard beyond the logical
`buffer_size`. If the final logical buffer would exceed the device-memory
budget, opening fails before the native allocation is attempted.

```bash
SIPHON_BACKEND=BUFFERED \
SIPHON_IO_DEPTH=32 \
SIPHON_DEBUG=1 python load_model.py
```

### Internals

[`docs/loader-internals.md`](docs/loader-internals.md) is a long walk through
the C++ loader. A Chinese translation is at
[`docs/loader-internals.zh-CN.md`](docs/loader-internals.zh-CN.md).

### API reference

```python
from siphon import Backend, BackendPolicy, safe_open
```

Build the HTML docs with:

```bash
cd docs && make html      # output in docs/build/html/
```

## The Rust reader

[`rust-qwen-engine/`](rust-qwen-engine/) — a safetensors reader on `io_uring` +
`O_DIRECT`, written to answer a question the Python loader cannot: **how much of
cold-start cost is the language runtime rather than the disk?**

Measured on a 51.7 GiB / 18-shard / 1199-tensor checkpoint, single NVMe, page
cache evicted and residency confirmed at `0.000` before timing:

| | |
|---|---|
| Throughput | **2.17 GB/s** (single-stream `dd` reaches 1.10; buffered `read(2)` 1.52) |
| Read plan | **18 aligned requests** cover the whole model, **1.0000×** amplification |
| Process start + all 18 headers parsed | **0.01 s**, 2.5 MB RSS |
| Python baseline, import path alone | **2.80 s** |

It ships a built-in mutational fuzzer with a safety oracle rather than a
crash-only check: every planned read must be block-aligned, inside the file, and
cover real tensor bytes, and every tensor byte must fall inside some read. See
the tree's README for the 28 synthetic malformed-file cases, the corrupted-file
variants, and the 2M-input fuzz runs.

## The forward golden harness

[`qwen35-forward/`](qwen35-forward/) — validates a hand-written Qwen3.5 forward
pass against a reference generated from the official implementation.

The reference is a tiny synthetic model (~368k parameters, pure CPU, about a
second to generate), which is enough because the gated delta rule's math does
not depend on dimension. Comparison works at three levels, ordered by cost:
isolated delta-rule units, per-tensor intermediates, then the greedy token
trace.

Two committed bundles make the comparator usable without a Python environment.
The tree's README documents a trap confirmed by test: `Qwen3_5RMSNorm` computes
`x * (1 + w)` with a zero-initialised weight, so writing it as `x * w` zeroes
every activation in the network and still runs to completion, producing only
garbage.

## Repository layout

```
siphon/                    Python package: Backend, BackendPolicy, safe_open
csrc/                      C++ extension and vendored third-party (submodules)
docs/                      Loader internals, benchmark notes, Sphinx API docs
tests/                     Loader tests and the benchmark harness
rust-qwen-engine/          Rust safetensors reader (independent cargo workspace)
qwen35-forward/            Forward golden harness (Python + independent cargo workspace)
```

## Origin, attribution and license

This project began as a fork of
**[InstantTensor](https://github.com/scitix/InstantTensor)** by ScitiX AI, and
is developed here as its own line of work. The original code is Apache-2.0
licensed; that license and the attribution above are retained as required.

Subsequent changes in this repository — including the page-cache probe in
automatic backend selection, and the two companion trees — are likewise
Apache-2.0. See [`LICENSE`](LICENSE).
