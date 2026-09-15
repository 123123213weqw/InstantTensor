# Qwen3.5 golden reference

A **reference to compare against** for writing a Qwen3.5 forward pass by hand.
Pure CPU: no GPU, no large model download.

## The one-line conclusion (measured)

**Per-tensor comparison is the primary instrument; the token trace is an
end-to-end health check, not a debugging tool.**

Perturbing `A_log` (the SSM decay parameter) by 1%:

| ssm_gain | per-tensor relative change | **greedy tokens changed** |
|---|---|---|
| 1 (official init) | 5.6e-04 | **0 / 16** |
| 30 | 1.3e-03 | **0 / 16** |
| 100 | 3.2e-03 | **0 / 16** |
| **300** | 1.5e+00 | **4 / 16** |

**At real weight scale the recurrent path contributes only 0.113% of the
residual**, so "change a weight by 1%, not one token moves" is inevitable.
`ssm_gain=300` does make the trace sensitive, but there the recurrent path
outweighs the residual, which is **no longer realistic**.

→ **The default `golden_tiny` (`ssm_gain=1.0`, official init) is for per-tensor
comparison** — per-tensor is sensitive at any gain (5.6e-04 already reflects a 1%
weight perturbation).
→ **A second bundle `golden_sensitive` (`--ssm-gain 300`) exists solely for token
trace acceptance**, because it changes 4 of 16 tokens.

Both use the same weight export format, so no Rust-side code changes.

## Quick start

```bash
# 1. generate the reference (about one second)
python golden/gen_golden.py --out golden_tiny --tokens 16

# 2. generate a token-trace-sensitive variant (for end-to-end acceptance)
python golden/gen_golden.py --out golden_sensitive --ssm-gain 300 --tokens 16

# 3. verify the reference is itself trustworthy (mandatory)
python golden/validate_golden.py --bundle golden_tiny
```

## What validation reports

`validate_golden.py` checks three things. **A reference that is not reproducible,
or not discriminative, is worse than none** — it produces confident, meaningless
verdicts.

```
1) reproducibility
   [PASS] all 260 files byte-identical across runs
2) well-formedness
   [PASS] 259 tensors checked
     note: 21 weight tensors are all zero; 21 are RMSNorm weights
           (expected: Qwen3_5RMSNorm uses x*(1+w) with zero init)
3) discrimination
   [PASS] eps=0 is bit-identical (deterministic)
   [PASS] eps=1e-2 moves per-tensor values by up to 5.127e-06
   [NOTE] eps=1e-2 changed 0/16 greedy tokens: the token trace alone
          cannot validate the recurrent path here
```

Generation also self-checks that **the two forms of the delta rule agree**,
which validates the reference implementation before it is used as a target:

```
delta B1_H2_T6_K16_V16: recurrent vs chunked  out 2.235e-08  state 5.960e-08
delta B1_H2_T1_K16_V16: recurrent vs chunked  out 1.118e-08  state 5.960e-08
delta B2_H3_T5_K8_V8 : recurrent vs chunked  out 2.980e-08  state 1.192e-07
```

## Two traps that silently kill the network (confirmed from source)

### 1. Normalisation has two *different* conventions, and getting it wrong is silent

```python
class Qwen3_5RMSNorm:
    self.weight = nn.Parameter(torch.zeros(dim))     # <- zero init
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # <- (1 + w)
```

| Norm | Formula | Weight init | Measured |
|---|---|---|---|
| `Qwen3_5RMSNorm` (`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm`) | `x * (1 + w)` | **zero** | all 0 |
| `Qwen3_5RMSNormGated` (`linear_attn.norm`) | `w * x`, then by `silu(gate)` | one | all 1 |

**Writing `x * w` zeroes every norm output**; the model still runs to completion
and produces only garbage. A comment in the source:
`Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)`.

→ This also explains the validator's initial false positive about "21 all-zero
weights": those are **correct**, and the check was naive. It is now reported as a
note.

### 2. `q_proj` output is doubled, and half of it is a gate

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
...
attn_output = attn_output * torch.sigmoid(gate)
```

And **`config.json` says `output_gate_type: "swish"` while the code uses
`sigmoid`** — read the source, not the config.

## Bundle structure

```
<out>/
  manifest.json       all metadata: config, tensor inventory, token trace, self-check
  weights/            109 weight tensors (fp32 raw LE)
  intermediates/      per-layer activations (captured by hooks) + per-step logits
  units/              standalone delta-rule reference
                      (q/k/v/g/beta + both recurrent and chunked outputs)
```

### Format: raw little-endian f32 plus a JSON manifest

**Avoiding `.npy` is deliberate** — the consumer is Rust, and a hand-rolled npy
parser would be a needless source of disagreement. Raw + JSON is unambiguous and
easy to diff. Reading it:

```rust
let raw = std::fs::read(bundle.join(&entry.file))?;
let vals: Vec<f32> = raw.chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();
// entry.shape gives the reshape target
```

### Tensor naming

`.` in the module path becomes `__`, so a file name can express the path:

```
model.layers.0.linear_attn.out_proj  ->  model__layers__0__linear_attn__out_proj
```

### The delta-rule units use the real model's layout

`Qwen3_5GatedDeltaNet.forward` passes `[B, T, H, K]` — the rule's first statement
is a `transpose(1, 2)` into `[B, H, T, K]`. An earlier version of the generator
passed `[B, H, T, K]` instead. The arithmetic stayed self-consistent, but the
recorded shapes had the `T` and `H` axes swapped, which would have been a trap
for anyone validating a real forward pass against this bundle.

The units now follow the model:

```
q, k, v        [B, T, H, *]
g, beta        [B, T, H]
out            [B, T, H, V]
state          [B, H, K, V]
```

### Two functions are captured from the module namespace, not from hooks

Two tensors an implementation needs cannot be captured with `register_forward_hook`,
because the reference never invokes the module that holds the parameters:

**`causal_conv1d_fn`.** `Qwen3_5GatedDeltaNet.forward` calls the **module-level**
function and passes `self.conv1d.weight` / `self.conv1d.bias` as arguments:

```python
mixed_qkv = causal_conv1d_fn(
    mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
    activation=self.activation, **kwargs)
```

`self.conv1d` is only a parameter container — it is never called — so a hook on
the `nn.Conv1d` module is dead. `ModuleSpy` rebinds the module-level name
instead, capturing both the input and the output, and records the call count so a
missing capture is loud rather than silent.

**The delta rule itself.** `forward` calls `torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule`. The **`torch_` prefix matters**: the
`@use_kernel_func_from_hub_with_fallback` decorator's first argument is the
*kernel* name (`"chunk_gated_delta_rule"`, unprefixed), which is a different
string and is not what `forward` resolves. These are neither instance nor class
attributes, so the module namespace is again what has to be wrapped.

Capturing those operands matters because they are the delta rule applied to
**real** block inputs rather than the synthetic problem in `units/`. An
implementation can verify the rule in place before writing the surrounding
projections.

#### One thing that capture revealed

The captured `q`/`k` are `[1, 6, 4, 16]` and `g`/`beta` are `[1, 6, 4]` —
**4 heads, not 2.** The GQA `repeat_interleave` that expands
`num_k_heads=2 → num_v_heads=4` happens *before* the delta rule is called, so the
rule always sees the expanded head count. An implementation that applies the
expansion after the rule, or not at all, will not match these tensors.

#### Capture is observation only

Verified, not asserted: reloading the recorded weights and recomputing the last
logits reproduces the bundle's `greedy_step00__logits` with

```
max abs diff = 0.000e+00
bundle argmax = 68   recomputed argmax = 68
```

Bit-identical, so the spies do not perturb the reference.

### What each block now provides

`layers.0`, in manifest (capture) order:

```
input_layernorm                                 [1, 6, 64]
linear_attn.in_proj_qkv                         [1, 6, 128]
linear_attn.in_proj_z                           [1, 6, 64]
linear_attn.in_proj_b                           [1, 6, 4]
linear_attn.in_proj_a                           [1, 6, 4]
causal_conv1d_fn_call0_in / _out                [1, 128, 6]   <- module-level spy
delta_torch_chunk_gated_delta_rule_0__{q,k,v}   [1, 6, 4, 16]
delta_torch_chunk_gated_delta_rule_0__{g,beta}  [1, 6, 4]
delta_torch_chunk_gated_delta_rule_0__out       [1, 6, 4, 16]
delta_torch_chunk_gated_delta_rule_0__state     [1, 4, 16, 16]
linear_attn.norm                                [24, 16]      = (B*T) x H x head_v_dim
linear_attn.out_proj                            [1, 6, 64]
linear_attn                                     [1, 6, 64]
post_attention_layernorm                        [1, 6, 64]
mlp.{gate,up}_proj                              [1, 6, 128]
mlp.down_proj                                   [1, 6, 64]
mlp                                             [1, 6, 64]
```

The conv and delta-rule captures carry a **call index**, not a layer number.
They are ordered by execution, which for a single forward pass is layer order, but
the index is not tied to the layer: with a KV/SSM cache and single-token decode,
`causal_conv1d_update` fires instead and the numbering shifts.

Note the convolution is **not** in the chain position the code might suggest. It
consumes `in_proj_qkv`, so it sits between that projection and the delta rule, but
its capture appears after `in_proj_b`/`in_proj_a` because those projections are
computed earlier in `forward` and their module hooks fire first.

## Three levels of comparison

| Level | What is compared | Sensitivity | Use |
|---|---|---|---|
| **Unit** (`units/`) | the delta rule alone: feed `q/k/v/g/beta`, compare `out` and `state` | high | **do this first.** No GGUF, no model loading, no CUDA |
| **Per-tensor** (`intermediates/`) | every operator's output in every layer | high (at any gain) | locate *which layer, which operator* |
| **Per-token** (`greedy`) | the `argmax` sequence plus each step's top-k logits | **low** (see table above) | end-to-end health check, and final real-model acceptance |

Each entry of `manifest.greedy.steps[i]` carries `argmax_token`, `topk_tokens`,
`topk_logits`, `logit_sum` and `logit_max`; the full logit vector for each step is
also written to `intermediates/greedy_stepNN__logits.f32`.

## The tiny model configuration (367,952 parameters)

```python
hidden_size=64, intermediate_size=128, num_hidden_layers=8,
num_attention_heads=2, num_key_value_heads=1, head_dim=32,
layer_types = 4-layer cycle (SSM,SSM,SSM,full_attn) x 2,
linear_conv_kernel_dim=4, linear_num_key_heads=2, linear_num_value_heads=4,
linear_key_head_dim=16, linear_value_head_dim=16,
vocab_size=256, rms_norm_eps=1e-6, partial_rotary_factor=0.25
```

**Why a tiny model suffices**: the delta rule's math does not depend on
dimension. Validating the math with 64-dimensional fake weights and validating it
with 5120-dimensional real weights test the same thing — but the former is ten
thousand times faster and needs no GPU or download.

`linear_num_value_heads=4` against `linear_num_key_heads=2` **keeps ratio=2
deliberately**, so the `query.repeat_interleave(...)` GQA branch is taken (the
27B has ratio=3 and takes it; the 2B has ratio=1 and does not — developing only
on the 2B would miss this path entirely).

## Arguments

```
--out DIR          output directory
--tokens N         greedy steps (default 16)
--seq N            prompt length (default 6)
--topk N           top-k recorded per step (default 5)
--seed N           weight initialisation seed (default 0)
--ssm-gain F       scale linear_attn.out_proj (default 1.0)
```

## Known limitations

- **The token trace is insensitive to the recurrent path** (quantified above).
  Not a bug — an inevitable consequence of real weight scale.
- **No cache in the token trace.** Greedy re-runs the whole forward each step,
  deliberately isolating the *math* from cache bookkeeping: if the trace
  diverges, the cause is the forward, not the cache. Cache semantics need their
  own reference.
- **Random weights in a tiny model.** It validates whether the math is
  implemented correctly, not whether the model is capable. Real-model acceptance
  needs a separate reference (`--model-dir`), and a 27B in fp16 needs about 52 GiB
  of memory.
- **`torch.use_deterministic_algorithms` is not enabled.** CPU fp32 was measured
  to be bit-identical run to run on these operators (eps=0 verified
  bit-identical), but that must be re-confirmed on other hardware or another BLAS
  backend.
