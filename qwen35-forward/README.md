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

## Layout

```
golden/              reference generator + validator (Python)
  gen_golden.py       generates a bundle
  validate_golden.py  checks the bundle itself is trustworthy
                      (reproducible / well-formed / discriminative)
  README.md           detailed docs: format, the two silent traps, sensitivity table
bundle/              Rust reader + comparator
  bundlecmp summary|show|compare|selftest
delta/               the gated delta rule, checked against the unit golden
  deltacheck <bundle>
gdn/                 one decoder layer: gated delta net + SwiGLU MLP + residuals
  gdncheck <bundle> [--layer N] [--input TENSOR]
golden_tiny/         committed bundle (gain=1.0, for per-tensor comparison)
golden_sensitive/    committed bundle (gain=300, for token-trace comparison)
```

## Step one: the gated delta rule

`delta/` is the first piece of the forward pass that this harness validates. It
is the right place to start because it is **48 of the 64 layers** in qwen35, it is
pure math with no GGUF, no model loading and no CUDA, and it has a closed form at
`T=1` that gives a target independent of the reference implementation.

```bash
cargo build --release
./target/release/deltacheck golden_tiny
cargo test -p deltarule
```

The rule, per `(batch, head)` and per time step:

```text
state  = state * exp(g_t)                 // decay
kv_mem = (state * k_t).sum(over K)        // read out
delta  = (v_t - kv_mem) * beta_t          // prediction error
state  = state + outer(k_t, delta)        // rank-1 correction
out_t  = (state * q_t).sum(over K)        // read out
```

with `l2norm` over the head dimension applied to `q` and `k` first, and `q` then
scaled by `1/sqrt(K)`.

### Result

```
   B1_H2_T1_K16_V16       recurrent out   abs=9.313e-9    rel=4.207e-7    ok
   B1_H2_T1_K16_V16       recurrent state abs=0.000e0     rel=0.000e0     ok
   B1_H2_T1_K16_V16         chunked out   abs=9.313e-9    rel=4.207e-7    ok
   B1_H2_T6_K16_V16       recurrent out   abs=2.980e-8    rel=1.607e-7    ok
   B1_H2_T6_K16_V16       recurrent state abs=2.980e-8    rel=7.690e-8    ok
   B1_H2_T6_K16_V16         chunked out   abs=5.215e-8    rel=2.812e-7    ok
   B2_H3_T5_K8_V8         recurrent out   abs=2.980e-8    rel=8.736e-8    ok
   B2_H3_T5_K8_V8         recurrent state abs=5.960e-8    rel=6.458e-8    ok
   B2_H3_T5_K8_V8           chunked out   abs=7.451e-8    rel=2.184e-7    ok

   RESULT: PASS (3 cases, both recurrent and chunked forms)
```

Errors sit at `1e-8..1e-7`, which is exactly where the reference's own two forms
agree with each other. The implementation is essentially exact.

**Both forms are checked.** The recurrent form is what decode uses and the
chunked form is what prefill uses; they are computed independently by the
reference, so passing both means satisfying two targets for one operator.

### The check was verified to have teeth

An assertion nobody has seen fail is not evidence. Seven bugs were injected and
all seven were caught:

| Injected bug | Caught? |
|---|---|
| drop the `l2norm` on `q`/`k` | **FAIL (3 of 3 cases)** |
| drop the `1/sqrt(K)` scaling of `q` | **FAIL (3 of 3 cases)** |
| use `g` instead of `exp(g)` for decay | **FAIL (2 of 3 cases)** |
| drop `beta` from `delta` | **FAIL (3 of 3 cases)** |
| drop the state decay entirely | **FAIL (2 of 3 cases)** |
| index the rank-1 update wrongly | **FAIL (3 of 3 cases)** |
| drop `k` from `kv_mem` | **FAIL (2 of 3 cases)** |

Failure diagnostics are actionable — the error jumps by seven orders of
magnitude and the offending element is named:

```
   B1_H2_T6_K16_V16       recurrent out   abs=2.594e-1    rel=1.398e0     FAIL
        worst at index 125  mine=0.07389838 golden=-0.1854713
```

### A limitation the mutation test exposed

The three `T=1`, `T=6` and `T=5` cases do not have equal power. The decay, the
`kv_mem` read-out and the state update all involve the state, and **at `T=1` the
state starts at zero, so every one of those bugs is invisible**. That is why
three of the seven injected bugs fail only 2 of 3 cases: the `T=1` case passes
regardless.

`T=1` still earns its place — it is the only case with a closed form (see
`t1_matches_closed_form`) — but it cannot substitute for a `T>1` case.

### Tests beyond the golden

`cargo test -p deltarule` (5 tests) includes two checks that do **not** depend on
the reference implementation:

- **`t1_matches_closed_form`** — at `T=1` with a zero initial state the
  recurrence collapses to `out[vi] = (k·q) * beta * v[vi]` and
  `state[ki,vi] = k[ki] * v[vi] * beta`. A mistake inherited from the reference
  cannot hide behind this target.
- **`strong_decay_forgets_history`** — as `g → -inf` the state is wiped each
  step, so every step must reduce to its own contribution.

## Step two: the gated delta net block

`gdn/` is the shell around the rule — the projections, the convolution, the
gating and the output norm — validated against every intermediate the bundle
captures for `layers.0`, in chain order.

```bash
cargo build --release
./target/release/gdncheck golden_tiny
cargo test -p gdn
```

### Result: 17 of 17 checks pass

```
   1. input_layernorm        abs=4.768e-7    rel=1.600e-7    ok
   2. in_proj_qkv            abs=1.788e-7    rel=3.348e-7    ok
   3. conv input             abs=1.788e-7    rel=3.348e-7    ok
   4. conv + silu            abs=1.863e-9    rel=1.378e-7    ok
   5. in_proj_z              abs=1.192e-7    rel=2.590e-7    ok
   6. in_proj_b              abs=8.941e-8    rel=2.683e-7    ok
   7. in_proj_a              abs=5.960e-8    rel=1.932e-7    ok
   8. q (post-GQA)           abs=1.048e-9    rel=1.030e-7    ok
   9. k (post-GQA)           abs=1.164e-9    rel=8.612e-8    ok
  10. v                      abs=1.863e-9    rel=1.588e-7    ok
  11. g (decay)              abs=1.907e-6    rel=9.592e-8    ok
  12. beta (gate)            abs=0.000e0     rel=0.000e0     ok
  13. delta rule out         abs=1.746e-10   rel=4.164e-7    ok
  14. delta rule state       abs=5.239e-10   rel=1.780e-7    ok
  15. gated norm             abs=1.304e-8    rel=2.752e-7    ok
  16. out_proj               abs=1.630e-9    rel=4.834e-7    ok
  17. block output           abs=1.630e-9    rel=4.834e-7    ok

   RESULT: PASS (17 checks)
```

Errors are at `1e-7` or below, which is f32 rounding for a chain this long.

### The check was verified to have teeth

Seven bugs injected, and the **first failing check names the operator** in every
case that did not simply crash:

| Injected bug | First failing check |
|---|---|
| gated norm uses `(1+w)` instead of `w` | **15. gated norm** |
| `input_layernorm` uses `w` instead of `(1+w)` | **1. input_layernorm** |
| `g` drops the `exp` on `A_log` | **11. g (decay)** |
| `beta` drops the sigmoid | **12. beta (gate)** |
| convolution becomes non-causal | **4. conv + silu** |
| rmsnorm forgets to divide by `dim` | **1. input_layernorm** |
| GQA head expansion skipped | crashed (see below) |

That ordering is the point of checking intermediates one by one rather than only
comparing the block's output: a wrong `g` is reported as a wrong `g`, not as "the
block output differs".

### A crash the mutation run exposed

Skipping the GQA expansion made the delta rule index past the end of `q` and
panic with a bare `index out of bounds: the len is 192 but the index is 192` — no
mention of which operand or what shape was expected.

The rule now validates every operand against the declared shape **before**
indexing, and `forward_prepared` validates before it normalises (the normalisation
indexes using `B*T*H` rows, so it fails earliest). The same injection now reports:

```
assertion `left == right` failed: delta rule: q length vs shape
  Shape { b: 1, t: 6, h: 4, k: 16, v: 16 }
```

### Two conventions, both easy to invert

Both are checked by a dedicated unit test, because neither raises an error when
wrong:

* `input_layernorm` is `Qwen3_5RMSNorm`: **`x * (1 + w)`**, weight zero-initialised.
* `linear_attn.norm` is `Qwen3_5RMSNormGated`: **`w * x_hat * silu(z)`**, plain `w`.

### Where the GQA expansion sits

`repeat_interleave` on the head axis happens **before** the delta rule, so the
rule sees `num_v_heads`, not `num_k_heads` — confirmed by the captured operands
being `[1, 6, 4, 16]` while `num_k_heads` is 2. The implementation does the same,
and the skip-GQA injection above is what proves the ordering matters.

### A note on the block input

`gdncheck` needs the block's *input*, not the layernorm's output. For layer 0 that
is `model__embed_tokens`. Layers above 0 would need the previous layer's
residual-stream output, which the bundle does not capture (only submodule outputs
are hooked), so those require an explicit `--input <tensor>`; `gdncheck` refuses
rather than silently normalising an already-normalised tensor — which is exactly
the mistake the first run made.

## Step three: the full decoder layer

`gdn/src/layer.rs` adds the other half of the layer — the SwiGLU MLP and the two
residual connections — and `gdncheck` now runs **24 checks covering one complete
decoder layer**:

```
   1. input_layernorm        abs=4.768e-7    ok
   ...
  17. block output           abs=1.630e-9    ok
  18. post_attention_layernorm   abs=2.384e-7    ok
  19. mlp gate_proj              abs=8.941e-8    ok
  20. mlp up_proj                abs=9.965e-8    ok
  21. mlp swiglu product         abs=1.490e-8    ok
  22. mlp down_proj              abs=2.328e-9    ok
  23. mlp output                 abs=2.328e-9    ok
  24. layer output (both residuals)  abs=3.725e-9    ok

   RESULT: PASS (24 checks)
```

Every error is at `1e-7` or below — f32 rounding across the full layer.

### The MLP

`Qwen3_5MLP.forward` is one SwiGLU expression, no biases:

```python
down_proj(act_fn(gate_proj(x)) * up_proj(x))     # hidden_act = "silu"
```

The elementwise product `silu(gate) * up` is never bound to a name in the
reference and no module produces it, so hooks on `gate_proj`/`up_proj` yield only
its two *inputs*. `SwigluSpy` records the product itself, because that is where a
SwiGLU implementation most plausibly goes wrong.

### The residuals

```python
residual = hidden_states                    # BEFORE the layernorm  (pre-norm)
hidden_states = input_layernorm(hidden_states)
hidden_states = linear_attn(...)
hidden_states = residual + hidden_states    # first residual

residual = hidden_states                    # the UPDATED value, not the layer input
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states    # second residual
```

No scale, no dropout, no gate: plain `x + f(x)`.

### Seven more injected bugs, all caught

| Injected bug | First failing check |
|---|---|
| both branches get `silu` | **21. mlp swiglu product** |
| `silu` dropped entirely | **21. mlp swiglu product** |
| `gate` and `up` swapped | **21. mlp swiglu product** |
| `gate_proj` given `up_proj`'s weights | **19. mlp gate_proj** |
| residual adds the post-norm value | **18. post_attention_layernorm** |
| second residual uses the original input | **24. layer output** |
| second residual dropped | **24. layer output** |

**The last two fail only check 24 and nothing else.** That is the argument for
capturing the decoder layer's own output: without it, both residual bugs are
completely invisible — every submodule would still match exactly.

## The two committed bundles

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
