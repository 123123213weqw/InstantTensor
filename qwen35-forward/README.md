# qwen35-forward — a reference harness for a hand-written forward pass

A **golden-reference generator** and **comparator** for writing a Qwen3.5
forward pass from scratch. Pure CPU: no GPU, no large model download.

## Why it exists

The largest risk in writing a forward pass by hand is not failing to write it —
it is **writing it wrong and believing it is right.** Without something to
compare against, there is no way to tell.

Some wrong implementations do not error and do not crash. They just turn the
output into garbage. One real example was caught while building this:

```
golden (x*(1+w)) absmax = 2.97936
buggy  (x*w)     absmax = 0          <- every activation zeroed
logits: golden absmax=0.541   buggy absmax=0
```

`Qwen3_5RMSNorm` computes **`x * (1 + w)`** with `w` **zero-initialised**.
Writing it as `x * w` zeroes the whole network; the model still runs all 22
tokens and produces nothing but garbage.

## Where this sits

Part of the [Siphon repository](../README.md), which also carries the loader
itself and [`rust-qwen-engine/`](../rust-qwen-engine/), a Rust safetensors
reader. The three trees share no code.

## Layout

```
golden/              reference generator + validator (Python)
  gen_golden.py       generates a bundle
  validate_golden.py  checks the bundle itself is trustworthy
                      (reproducible / well-formed / discriminative)
  README.md           detailed docs: format, the two silent traps, sensitivity table
bundle/              Rust reader + comparator
  bundlecmp summary|show|compare|selftest
golden_tiny/         committed bundle (gain=1.0, for per-tensor comparison)
golden_sensitive/    committed bundle (gain=300, for token-trace comparison)
```

### The two committed bundles

| | `golden_tiny` | `golden_sensitive` |
|---|---|---|
| `ssm_gain` | 1.0 (official init) | 300 |
| Tensors | 259 | 259 |
| Greedy steps | 16 | 16 |
| Purpose | **per-tensor comparison** (sensitive at any gain) | **token-trace comparison** (at gain=1 the trace is nearly blind to the recurrent path) |
| Size | 2.3 MB / 260 files | 2.3 MB / 260 files |

Why two are needed is shown in "The trade-off, in numbers" below.

**Both are regenerable artifacts.** They are committed only so the comparator
works without a Python environment. Change `gen_golden.py` and you must
regenerate, then run `validate_golden.py`.

## Quick start

**Both bundles are committed, so this works with no Python environment:**

```bash
# build the comparator and use the committed bundles directly
cargo build --release
./target/release/bundlecmp summary golden_tiny
./target/release/bundlecmp selftest golden_tiny
```

To compare your own implementation, write your intermediates in the same layout
and point the comparator at them:

```bash
./target/release/bundlecmp compare golden_tiny <yours>
```

To regenerate (needs `transformers` + `torch`, about a second):

```bash
python golden/gen_golden.py --out golden_tiny --tokens 16
python golden/gen_golden.py --out golden_sensitive --ssm-gain 300 --tokens 16

# verify the reference itself is trustworthy -- mandatory after changing the generator
python golden/validate_golden.py --bundle golden_tiny
```

## Three levels of comparison

Order matters: **start with the cheapest, and introduce one unknown at a time.**

| Level | Directory | What is compared | Sensitivity | Use |
|---|---|---|---|---|
| **Unit** | `units/` | the delta rule alone: feed `q/k/v/g/beta`, compare `out` / `state` | high | **do this first.** No GGUF, no loading, no CUDA |
| **Per-tensor** | `intermediates/` | every operator's output in every layer | **high** | locate *which layer, which operator* |
| **Per-token** | `manifest.greedy` | the argmax sequence plus each step's top-k logits | **low** | end-to-end health check |

## The trade-off, in numbers

Perturb the SSM decay parameter `A_log` by 1% and measure:

| ssm_gain | per-tensor relative change | **greedy tokens changed** |
|---|---|---|
| 1 (official init) | 5.6e-04 | **0 / 16** |
| 30 | 1.3e-03 | **0 / 16** |
| 100 | 3.2e-03 | **0 / 16** |
| **300** | 1.5e+00 | **4 / 16** |

**At real weight scale the recurrent path contributes only 0.113% of the
residual**, so "change a weight by 1%, not one token moves" is inevitable. It
takes `ssm_gain=300` to make the token trace sensitive, and at that point the
recurrent path outweighs the residual, which is no longer realistic.

→ So: **`golden_tiny` (gain=1) validates per-tensor; `golden_sensitive`
(gain=300) validates the token trace.**

## The implementer's contract

Produce a bundle in the same layout, then
`bundlecmp compare <golden> <yours>`.

```
<bundle>/manifest.json
<bundle>/weights/<name>.f32          raw little-endian f32, C-contiguous
<bundle>/intermediates/<name>.f32
<bundle>/units/<name>.f32
```

Naming rule: **replace `.` in the module path with `__`**

```
model.layers.0.linear_attn.out_proj  ->  model__layers__0__linear_attn__out_proj
```

**Avoiding `.npy` is deliberate** — the consumer is Rust, and a hand-rolled npy
parser would be a needless source of disagreement.

## The comparator was verified to actually catch bugs

| Test | Result |
|---|---|
| `selftest` (against itself) | **PASS** — 259 tensors, 0 non-zero; the comparator is reflexive |
| self vs self | **PASS** — all 259 tensors bit-identical |
| tiny vs sensitive | **FAIL** — reports first divergence at token 6, `rel=2.990e2` (exactly gain−1=299, so the arithmetic is self-consistent) |
| the **`x*w` instead of `x*(1+w)`** bug | **FAIL** — 104/123 tensors differ, `rel=1.000`, pinpointing `input_layernorm` |

### A hole this testing exposed (now fixed)

On the first run of the `x*w` bug, the **token trace reported "identical"** —
because what was being compared was the **token ids copied out of the candidate's
own manifest**, while the candidate's logits were all zero.

**The real hole: the comparator trusted the token ids a candidate recorded about
itself.** An implementation can record correct ids while its logits are entirely
wrong.

The fix: the comparator now **derives argmax from the candidate's own
`greedy_stepNN__logits`** and reconciles that against the recorded ids:

```
!! CANDIDATE token ids disagree with its own logits at 1 of 16 steps:
     step 0: manifest says 68, its logits say 0
   (recorded ids are not trustworthy; the logits are the ground truth)
```

## Two traps that silently kill the network

### 1. Normalisation has **two** conventions, and getting it wrong is silent

| Norm | Formula | Weight init | Measured |
|---|---|---|---|
| `Qwen3_5RMSNorm` (`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm`) | `x * (1 + w)` | **zero** | all 0 |
| `Qwen3_5RMSNormGated` (`linear_attn.norm`) | `w * x`, then multiply by `silu(gate)` | one | all 1 |

A comment in the source: `Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)`

### 2. `q_proj` output is doubled, and half of it is a gate

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
...
attn_output = attn_output * torch.sigmoid(gate)
```

**`config.json` says `output_gate_type: "swish"` while the code uses `sigmoid`**
— read the source, not the config.

## Confidence in the reference (measured)

```
[PASS] 260 files byte-identical across runs
[PASS] 259 tensors well-formed (no NaN/Inf, no all-zero activations)
[PASS] eps=0 is bit-identical run to run
[PASS] delta-rule self-check: recurrent vs chunked agree to 1e-8 / 1e-7
```

The delta-rule self-check validates **the reference implementation itself**:

```
delta B1_H2_T6_K16_V16: recurrent vs chunked  out 2.235e-08  state 5.960e-08
delta B1_H2_T1_K16_V16: recurrent vs chunked  out 1.118e-08  state 5.960e-08
delta B2_H3_T5_K8_V8 : recurrent vs chunked  out 2.980e-08  state 1.192e-07
```

## Known limitations

- **The token trace is insensitive to the recurrent path** (quantified above).
  Not a bug — an inevitable consequence of real weight scale.
- **No cache.** Greedy re-runs the whole forward each step, deliberately
  isolating the *math* from cache bookkeeping: if the trace diverges, the cause
  is the forward, not the cache. Cache semantics need their own reference.
- **A tiny model with random weights** (367,952 parameters). It validates
  whether the math is implemented correctly, not whether the model is capable.
- **`linear_num_value_heads=4` / `linear_num_key_heads=2`, deliberately ratio=2**,
  so the `query.repeat_interleave(...)` GQA branch is exercised (the 27B has
  ratio=3 and takes it; the 2B has ratio=1 and does not — developing only on the
  2B would miss this path entirely).
- Validating a real model needs a separate reference (`--model-dir`), and a 27B
  in fp16 needs about 52 GiB of memory.
