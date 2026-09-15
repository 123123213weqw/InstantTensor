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
gdn/                 the whole model
  lib.rs             the gated delta net mixer (conv, GQA, rule, gated norm)
  attention.rs       the full-attention mixer (per-head q/gate split, rope, GQA)
  layer.rs           one layer: input norm, either mixer, MLP, both residuals
  model.rs           embedding, the layer stack, final norm, head, greedy decode
  loader.rs          golden bundle -> weights, and the layer -> capture-index mapping
  safetensors.rs     a self-contained safetensors reader (bf16/f16/f32)
  real.rs            a real checkpoint -> weights, with config cross-checks
  gdncheck <bundle>  [--layer N] [--chain] [--model] [--verbose]   (golden)
  qwenrun  <model-dir> [--prompt ids] [--tokens N] [--compare FILE]  (real)
tools/ref_qwen35.py  the transformers reference: logits, per-layer dumps, greedy
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

## Step four: the whole stack, layer by layer

`gdncheck` now runs over every layer, not just layer 0:

```bash
./target/release/gdncheck golden_tiny            # every verifiable layer
./target/release/gdncheck golden_tiny --layer 4  # one layer, all 24 checks
./target/release/gdncheck golden_tiny --chain    # feed each output forward
./target/release/gdncheck golden_tiny --verbose  # all checks for every layer
```

```
   layer type              first failing / worst check               abs  layer-out
   0     linear_attention  worst: 11. g (decay)                 1.907e-6   3.725e-9  ok
   1     linear_attention  worst: 11. g (decay)                 1.907e-6   3.725e-9  ok
   2     linear_attention  worst: 11. g (decay)                 1.907e-6   5.588e-9  ok
   3     full_attention    — not implemented (full_attention) —
   4     linear_attention  worst: 11. g (decay)                 1.907e-6   7.451e-9  ok
   5     linear_attention  worst: 11. g (decay)                 1.907e-6   5.588e-9  ok
   6     linear_attention  worst: 11. g (decay)                 1.907e-6   7.451e-9  ok
   7     full_attention    — not implemented (full_attention) —

   RESULT: PASS (6 layers)
```

Six of the eight layers pass, with different weights and different inputs than layer
0 — so the block generalises rather than happening to fit one layer.

### Why the interface already existed

Layer *N*'s output is layer *N-1*'s input, and step three added a hook on the
decoder layer itself. That is the whole inter-layer interface: no new capture was
needed to stack layers.

### Two ways to check a layer, and both are needed

**Isolated** (the default) feeds each layer the input the *reference* produced for
it. An error in layer 2 cannot contaminate the verdict on layer 4, so a failure
names the layer that caused it.

**Chained** (`--chain`) feeds each layer's own output into the next. It localises
worse — one bad layer shows up in all the layers after it — but it is the only way
to show the stack works end to end. It answers the question the isolated mode
cannot: does rounding accumulate?

```
   layer-output error, chained vs isolated (the drift check)
     layer 0   chained=3.725e-9    isolated=3.725e-9    ratio=1.00
     layer 1   chained=7.451e-9    isolated=3.725e-9    ratio=2.00
     layer 2   chained=9.313e-9    isolated=5.588e-9    ratio=1.67
     layer 4   chained=7.451e-9    isolated=7.451e-9    ratio=1.00
     layer 5   chained=7.451e-9    isolated=5.588e-9    ratio=1.33
     layer 6   chained=1.490e-8    isolated=7.451e-9    ratio=2.00
```

At most 2x, and still in the `1e-8` range three layers deep. Nothing accumulates.

A full-attention layer breaks a chain: its output cannot be computed, so the layer
after it has no input this implementation can produce. `--chain` therefore runs over
maximal *runs* of linear layers — `[[0, 1, 2], [4, 5, 6]]` here — starting each run
from the golden input of its first layer.

### The capture index is not the layer index

This is the bug that made a single-layer checker look correct. The convolution and
delta-rule captures are named by **position among the linear-attention layers**:

```text
causal_conv1d_fn_call<K>_in / _out
delta_torch_chunk_gated_delta_rule_<K>__{q,k,v,g,beta,out,state}
```

`K` counts only the layers that run the delta rule. The linear layers here are
0, 1, 2, 4, 5, 6, so **layer 4's convolution is `call3`, not `call4`**. Indexing by
layer number is correct for layers 0-2 and wrong from layer 4 on — precisely the
kind of bug that survives a test that only ever ran layer 0. `loader::LayerCapture`
is now the single place the mapping lives, and `ssm_ordinal` is unit-tested against
the real layer-type pattern.

Reintroducing the bug proves it is load-bearing:

```
   4     linear_attention  FAIL 3. conv input (channels-first)    1.030e+0   FAIL (9 checks)
   5     linear_attention  FAIL 3. conv input (channels-first)    8.206e-1   FAIL (9 checks)
   6     linear_attention  FAIL 3. conv input (channels-first) MISSING causal_conv1d_fn_call6_in
```

### A silent skip that mutation testing exposed

The first version of that run showed layers 4 and 5 failing but **layer 6 passing**
under the same bug. Layer 6 looked up `causal_conv1d_fn_call6_*`, which does not
exist, and a missing golden tensor was treated as *not a failure* — so nine checks
disappeared without a word.

A tensor that is absent for a **full-attention** layer is genuinely not applicable.
A tensor that is absent for a **linear-attention** layer means the bundle is missing
something it should have. That distinction is now explicit, and the same mutation
reports `MISSING causal_conv1d_fn_call6_in` instead of a cheerful pass.

### Four more injected bugs, all caught

| Injected bug | First failing check | Layers affected |
|---|---|---|
| capture indexed by layer instead of ordinal | 3. conv input | 4, 5, 6 |
| input normalised twice | 2. in_proj_qkv | all 6 |
| first residual dropped | 18. post_attention_layernorm | all 6 |
| SwiGLU `gate`/`up` swapped | 21. mlp swiglu product | all 6 |

### What is now blocked, and on what

Layers **3 and 7 are `full_attention`** and are reported as not implemented rather
than guessed at. They need `Qwen3_5Attention`, which is a separate code path:
`q_proj` emits **2x** `num_attention_heads * head_dim` with the second half used as
a sigmoid gate, plus `q_norm`/`k_norm` and a partial rotary embedding. Until that
exists, a chain cannot cross layer 3, and `--chain` starts a new run after it.

## Step five: the whole model, and the tokens it generates

`gdncheck --model` runs embedding, all eight layers, the final norm and the head, then
decodes greedily and compares the generated tokens against the reference's.

```bash
./target/release/gdncheck golden_tiny --model     # logits and the token trace
./target/release/gdncheck golden_tiny --layer 3   # one attention layer, all checks
./target/release/gdncheck golden_tiny --chain     # 0 -> 7, all eight layers
```

```
   mode: whole model (embedding -> all layers -> final norm -> head)
    1. embedding                                        abs=0.000e0     rel=0.000e0     ok
    2. layer 0 output                                   abs=5.588e-9    rel=9.204e-8    ok
    ...
    9. layer 7 output                                   abs=2.794e-8    rel=3.059e-7    ok
   10. final norm                                       abs=9.537e-7    rel=2.766e-7    ok
   11. logits (all positions)                           abs=2.086e-7    rel=3.853e-7    ok

   greedy trace (16 steps)
     step  0  argmax want=68   got=68   ok   logits abs=1.565e-7  top5 same
     ...
     step 15  argmax want=75   got=75   ok   logits abs=1.639e-7  top5 same

     tokens: 16/16 match   worst logit abs=2.980e-7
     full token sequence: identical
```

All 16 generated tokens match, the top-5 ranking matches at every step, and the full
21-token sequence is identical. Both bundles pass, including the `ssm_gain=300` one
that amplifies the recurrent path.

### Full attention

`Qwen3_5Attention` is a separate code path and, per the reference, a different one in
three places that are each silent when wrong:

* **The query/gate split is per head.** `q_proj` emits `num_heads * head_dim * 2`
  values, which are *viewed* as `[B, T, num_heads, head_dim*2]` and chunked on the
  last axis. Head `h` takes columns `h*2D .. h*2D+D` as its query and
  `h*2D+D .. (h+1)*2D` as its gate. Splitting the flat output in half -- "the first
  half is the query" -- produces plausible output and is wrong. Injecting that
  version fails at check 3 (`q_norm`) and drops the token trace to 1/16.
* **The gate is `sigmoid`, not the `swish` the config claims.** The config key is
  `output_gate_type`; the code is `attn_output * torch.sigmoid(gate)`. Injecting
  `swish` is not caught by any shape, only by the values: 1/16 tokens.
* **Only part of the head rotates.** `partial_rotary_factor` is 0.25, so with
  `head_dim = 32` only the first 8 dimensions rotate. `rotate_half` splits those 8
  into 4+4 and returns `cat(-x2, x1)` -- the half convention, not the interleaved one.

### Text-only MRoPE is plain RoPE

`Qwen3_5TextRotaryEmbedding` builds three interleaved frequency rows (temporal,
height, width) and overwrites row 0 from rows 1 and 2 at interleaved indices. For
text-only input all three rows come from the same `arange`, so the interleave is a
no-op. Checked, not assumed: `max |cos - plain| = 0.000e+00`, exactly zero. The
builder is also compared against the reference's own captured tables at run time
(`cos abs=5.960e-8  sin abs=2.980e-8`).

### The gate needed a tensor that does not exist

`Qwen3_5Attention.forward` ends with

```python
attn_output = attn_output * torch.sigmoid(gate)
attn_output = self.o_proj(attn_output)
return attn_output, attn_weights
```

so the tensor the module *returns* is post-`o_proj`. The post-gate, pre-`o_proj`
tensor is never bound to a name and no module produces it, so a forward hook cannot
see it -- yet it is where the most falsifiable line in the block happens. A forward
hook receives `(module, inputs, output)`, so `InputSpy` records `o_proj`'s **input**
and captures it without touching the computation.

This was found by getting it wrong first: the check initially compared against the
recorder's `out0`, which is the module output, and failed at check 7 while checks 8
and 9 (both post-`o_proj`) passed -- which is what pointed at the mislabelling.

### A rope bug that a single-head test cannot see

The position for row `r` of a `[B, T, H, D]` tensor is `(r / heads) % T`, not
`r % T`. The wrong form scrambles which position each head is rotated by while
leaving every tensor *shape* unchanged, so it survived until the attention layer was
compared: 10/16 tokens instead of 16/16, failing at check 7.

`r % T` is correct when there is one head, so the existing single-head unit test
could not distinguish them. `rope_uses_the_position_not_the_row` now exercises two
heads over three positions.

### Two bugs no golden bundle could catch, and what was done about them

Mutation testing found two injections that passed everything:

**GQA head ordering.** `repeat_kv` expands as `hidden[:, :, None].expand(b, kvh,
n_rep, ...)` then reshapes, so query head `j` reads kv head `j / n_rep`. The wrong
ordering is *identical* when `num_key_value_heads == 1`, which is what the tiny bundle
has. The real 27B model has a ratio of 3. This cannot be fixed in the bundle without
changing its config, so it is pinned by `gqa_head_mapping_is_contiguous_per_kv_head`,
which uses two kv heads. That test fails under the injection; no golden check does.

**A doubled final norm.** `Qwen3_5RMSNorm.__init__` is `nn.Parameter(torch.zeros(dim))`
and its forward is `output * (1.0 + weight)`, so with the reference's own
initialisation `1 + w == 1` and the norm is a *pure* normalisation. Normalising twice
is then a no-op, because a normalised vector already has unit RMS. A model shell that
applied the final norm twice passed all 11 model checks and all 16 tokens.

That one *is* fixable in the artifact, so it was: the generator gained
`--randomize-norms`, which fills plain RMSNorm weights with `N(0, 0.5)` and gated ones
with `1 + N(0, 0.5)`. The operator and every code path are unchanged; only the values
are. Both committed bundles were regenerated with it. The same injection now reports

```
   10. final norm    abs=3.599e0    rel=6.538e-1    FAIL
   11. logits        abs=3.381e-1   rel=5.065e-1    FAIL
```

with every layer still passing -- because the layers do not include the final norm,
which is exactly why the model-level check has to exist.

### Tolerance and the amplified bundle

`--ssm-gain` multiplies `linear_attn.out_proj`, so the recurrent path's contribution
to the residual -- and its absolute rounding error -- scale by the same factor.
Holding the tolerance fixed made `golden_sensitive` fail on arithmetic that is
proportionally identical (1.54e-5 against a 1e-5 bound, with 16/16 tokens matching).
The tolerance is now `1e-5 * max(1, ssm_gain)` and is printed, so the amplified bundle
is judged at the same *relative* stringency rather than a looser or stricter one.

### Six more injected bugs, all caught

| Injected bug | First failing check |
|---|---|
| output gate uses `swish` | attention check 7 |
| rope applied with the row index as the position | attention check 7 |
| causal mask removed | attention check 7 |
| `1/sqrt(head_dim)` scaling dropped | attention check 7 |
| query/gate split front-and-back | attention check 3 (`q_norm`) |
| final norm applied twice | model check 10 (`final norm`) |

Plus, from the earlier steps, the capture-index and residual bugs, all still caught.

## Step six: a real checkpoint

Everything above validates against a synthetic model. This step runs a real one:

```bash
cargo build --release
./target/release/qwenrun /path/to/Qwen3.5-0.8B --prompt 9419 --tokens 10
./target/release/qwenrun /path/to/Qwen3.5-0.8B --prompt 9419 --compare ref/logits_last.f32
python tools/ref_qwen35.py /path/to/Qwen3.5-0.8B ref --prompt 9419 --greedy 10
```

`Qwen/Qwen3.5-0.8B` is the smallest official `qwen3_5` checkpoint: 24 layers, 18
`linear_attention` + 6 `full_attention`, hidden 1024, vocab 248320, one 1.63 GiB
bf16 shard.

### It reproduces the reference token for token

```
   loaded in 4.5s
   prefix `model.language_model.`  config from text_config  1 shard(s)  488 tensors
   dtypes: BF16=452 F32=36
   unused sub-trees: multi-token-prediction head=15 vision tower=153
   head: tied to embed_tokens
   24 layers (18 linear_attention, 6 full_attention), hidden=1024 vocab=248320

   compare vs ref/logits_last.f32
     max abs diff      1.717e-5  (tolerance 3e-5)
     argmax            mine=11 ref=11
     top-10 ordering  identical
     => PASS

   greedy:  11,271,40,1044,3133,440,264,12654,5148,421
   ref:     11,271,40,1044,3133,440,264,12654,5148,421
```

Same tokens, all ten steps, on a checkpoint the engine has never seen.

### What a real checkpoint needs that the golden bundle did not

| | golden bundle | real checkpoint |
|---|---|---|
| names | `model__layers__0__linear_attn__in_proj_qkv__weight` | `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` |
| dtype | `f32` | `bf16` (452 tensors) and `f32` (36) |
| layout | one file per tensor | one flat shard |
| extras | none | a vision tower (153 tensors) and an MTP head (15) |
| head | a separate `lm_head` | tied to `embed_tokens` |

So `real.rs` maps *structure* rather than strings, and `safetensors.rs` is a
self-contained reader with the same validation discipline as the streaming loader:
`end - start == numel * dtype.size()`, every range inside the file, and reversed
offsets rejected — a reversed pair underflows into an enormous length, which is how
a reader ends up allocating terabytes.

Dimensions are read off tensor shapes and then **cross-checked against the config**.
A disagreement is an error, not something to average over:

```
   check!("num_v_heads", num_v_heads_cfg, num_v_heads);
   check!("key_dim = num_k_heads * head_k_dim", num_k_heads_cfg * head_k_dim_cfg, key_dim);
   check!("q_proj out = num_heads * head_dim * 2", num_heads * head_dim * 2, q_out);
```

### The disagreement was mine, and it was an accumulation bug

The first run matched `argmax` and the top-10 ordering but differed by **1.016e-4**
in the logits — four times the reference's own CPU-vs-GPU disagreement. Rather than
wave that off as "float noise", the per-layer dump localised it: the error was
*bounded and gradual* (5e-7 to 2e-5 per layer, no jumps), which rules out a wrong
operator and points at precision.

The cause was the dot products. A naive `f32` loop accumulates `n` products with
worst-case error `n * eps`; for `n = 3584` that is a relative error around 8e-4. BLAS
does not accumulate that way — blocked or pairwise reductions grow like `log n * eps`,
roughly two orders of magnitude smaller. So "same computation, different order" was
wrong: a sequential loop is a measurably worse computation.

Accumulating in `f64` fixes it, and the measurement is unambiguous:

| | vs float64 ground truth |
|---|---|
| **this engine** | **5.901e-06** |
| reference, GPU f32 | 1.621e-05 |
| reference, CPU f32 | 2.623e-05 |

**The engine is now 2.7x closer to the true value than the GPU reference and 4.4x
closer than the CPU one.** Against the GPU reference the logits differ by 1.717e-5,
which is below the 2.46e-5 by which the reference disagrees with *itself* — so what
remains is the reference's error, not this engine's.

That also settles the tolerance. A `bf16` checkpoint loaded into an `f32` reference
is still an `f32` computation, and `f32` does not reproduce itself across
implementations. A bound tighter than the reference's own spread is not a test of
correctness; it demands that this engine agree with the reference more closely than
the reference agrees with itself. `qwenrun` defaults to `3e-5` and documents why.

Two gates are reported separately, because they fail for different reasons:
`argmax` plus top-k ordering is the functional test (differing tokens mean a wrong
operator), and the value bound is the numerical one.

### The one-piece-of-work question: bf16 storage

The checkpoint stores `bf16`. Widening to `f32` is exact — bfloat16 is the top 16
bits of an `f32`, so every bf16 value is representable and no rounding occurs. That is
checked by a round-trip test, and it is why loading into an `f32` reference removes
the storage dtype from the comparison instead of adding to it.

### Performance, honestly

Roughly 0.3 s/token for a 0.8B model with no KV or recurrent cache, so each step
re-runs the whole sequence:

```
   forward: 1.51s for 5 token(s)  (0.303s/token)
     step  0  len=1    -> 11     (0.94s elapsed)
     ...
     step  9  len=10   -> 421    (9.88s elapsed)
```

Matmuls are spread over up to 32 threads by row, which is bit-identical to the
serial path (a unit test asserts this bit for bit, because a reduction-order change
would make results irreproducible). This is a correctness-first implementation: the
flops are roughly 1.6 GFLOP/token, so there is a lot of headroom. Caching the
recurrent and KV state is the obvious next step and removes the quadratic
re-processing, but it changes what is *stored*, not what is *computed*.

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
