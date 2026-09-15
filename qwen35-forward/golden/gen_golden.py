#!/usr/bin/env python3
# coding=utf-8
"""Qwen3.5 golden reference generator.

Produces a **self-contained** golden bundle that a from-scratch forward
implementation can be validated against without Python, a GPU, or the real
model:

* `weights/`       — the exact fp32 weights, as raw little-endian f32
* `intermediates/` — intermediate tensors of each decoder block, via hooks
* `units/`         — isolated golden for the gated-delta-rule math
* `manifest.json`  — shapes, dtypes, file names, config, and the token trace

Design notes
------------
**Format.** Raw little-endian f32 plus shapes in JSON, deliberately not `.npy`:
the consuming side is Rust, and a hand-rolled npy parser is a needless source of
disagreement. Raw + a JSON manifest is unambiguous and trivial to diff.

**Everything is fp32.** A bf16 reference would make a *correct* f32
implementation look wrong by ~1e-2. The golden must be strictly more precise
than the thing under test, otherwise "it passes" carries no information.

**No KV/SSM cache in the token trace.** Greedy decode re-runs the full forward
each step. That isolates the *math* from cache bookkeeping: if the trace
diverges, the cause is the forward, not the cache. Cache semantics need their
own golden.

**The delta rule is dumped twice.** `torch_recurrent_gated_delta_rule` (decode
form) and `torch_chunk_gated_delta_rule` (prefill form) must agree; asserting
that validates the reference itself before it is used as a target, and gives the
implementation two independent targets for one operator.

Usage
-----
    python gen_golden.py --out golden_tiny --tokens 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# tiny synthetic config: identical math, ~100k params, no download, no GPU
# --------------------------------------------------------------------------- #

def tiny_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    n_layers = 8
    # Same 1-in-4 pattern as the real model, so both layer kinds are exercised.
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(n_layers)
    ]
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=layer_types,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
        vocab_size=256,
        tie_word_embeddings=False,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        partial_rotary_factor=0.25,
    )


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #

class Bundle:
    """Accumulates raw f32 tensors plus their manifest entries."""

    def __init__(self, out: Path):
        self.out = out
        self.entries: list[dict] = []
        for g in ("weights", "intermediates", "units"):
            (out / g).mkdir(parents=True, exist_ok=True)

    def write(self, group: str, name: str, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            return
        a = t.detach().to(torch.float32).cpu().contiguous()
        if a.numel() == 0:
            return
        fname = f"{group}/{name}.f32"
        path = self.out / fname
        path.write_bytes(a.numpy().tobytes(order="C"))
        self.entries.append({
            "group": group,
            "name": name,
            "file": fname,
            "shape": list(a.shape),
            "dtype": "f32",
            "numel": int(a.numel()),
            "nbytes": int(a.numel() * 4),
        })


def safe_name(s: str) -> str:
    return s.replace(".", "__")


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #

def randomize_norms(model: torch.nn.Module, seed: int) -> int:
    """Give every RMSNorm a non-trivial scale.

    `Qwen3_5RMSNorm.__init__` is `nn.Parameter(torch.zeros(dim))` and its forward is
    `output * (1.0 + weight)`. With the reference's own initialisation that collapses
    to `1.0` -- a pure normalisation. Two things follow, both bad for a test bundle:

      * a *scale* error in the `(1 + w)` convention is invisible. (The convention
        itself is still distinguishable: `x * w` would zero every activation.)
      * normalising twice is a no-op, because a normalised vector already has unit
        RMS. A model shell that applied the final norm twice would pass every check.

    Filling `w` with `N(0, 0.5)` keeps the operator and every code path identical
    while making both observable. `Qwen3_5RMSNormGated` computes `w * x` and starts
    at ones, so its weights are centred on 1 instead; the plain RMSNorm starts at
    zero, so its weights are centred on 0.
    """
    g = torch.Generator().manual_seed(seed + 977)
    n = 0
    for _name, mod in model.named_modules():
        cls = type(mod).__name__
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.nn.Parameter) or w.ndim != 1:
            continue
        if cls.endswith("RMSNormGated"):
            new = 1.0 + torch.randn(w.shape, generator=g) * 0.5
        elif cls.endswith("RMSNorm"):
            new = torch.randn(w.shape, generator=g) * 0.5
        else:
            continue
        with torch.no_grad():
            w.copy_(new.to(w.dtype))
        n += 1
    return n


def dump_weights(bundle: Bundle, model: torch.nn.Module) -> int:
    sd = model.state_dict()
    for k, v in sorted(sd.items()):
        bundle.write("weights", safe_name(k), v)
    total = sum(v.numel() for v in sd.values())
    print(f"  weights: {len(sd)} tensors, {total} params")
    return total


# --------------------------------------------------------------------------- #
# per-block intermediates
# --------------------------------------------------------------------------- #

# Only leaf-ish modules are hooked. Hooking every ancestor would duplicate the
# same activation under several names and make the bundle hard to read.
#
# `conv1d` is deliberately absent: `Qwen3_5GatedDeltaNet.forward` calls the
# module-level `causal_conv1d_fn` and only passes `self.conv1d.weight`/`.bias` as
# arguments, so the `nn.Conv1d` module is never invoked and a hook on it is dead.
# `ModuleSpy` captures that tensor instead.
# `layers.<N>` is included so a decoder layer's *output* is captured. Without it
# there is no way to check the two residual adds, and no way to obtain layer N's
# input from layer N-1's output -- which is what makes layers above 0 checkable at
# all.
HOOK_SUFFIXES = (
    "input_layernorm", "post_attention_layernorm", "mlp", "linear_attn",
    "self_attn", "norm", "in_proj_qkv", "in_proj_z", "in_proj_a",
    "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
    "q_norm", "k_norm", "gate_proj", "up_proj", "down_proj", "embed_tokens",
    # The rotary embedding returns `(cos, sin)`, which the recorder splits into
    # `rotary_emb__out0` / `__out1`. Capturing them isolates the rotation from the
    # attention arithmetic: a mismatch in the attention output can be attributed to
    # the rotation or to the softmax, rather than to "attention" as a whole.
    "rotary_emb", "lm_head", "norm",
)


def _is_decoder_layer(name: str) -> bool:
    """True for `...layers.<N>` but not for anything nested inside it."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit()


class ModuleSpy:
    """Record calls to a module-level function, without altering them.

    `Qwen3_5GatedDeltaNet.forward` does **not** call `self.conv1d`. It calls the
    module-level `causal_conv1d_fn`, passing `self.conv1d.weight` and
    `self.conv1d.bias` as arguments:

        mixed_qkv = causal_conv1d_fn(
            mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
            activation=self.activation, **kwargs)

    So a forward hook on the `nn.Conv1d` module never fires and the convolution's
    output is invisible to hook-based capture. Replacing the module-level name is
    what actually intercepts the call.

    The kernel-hub decorators are no-ops unless `USE_HUB_KERNELS` is set, in which
    case `_kernels_enable` swaps the callable inside `_kernels_use_kernelized_func`
    rather than rebinding the module attribute; the caller reports which path was
    taken so a missing capture is loud instead of silent.

    Records both the **input** and the **output**, since the input is the
    `in_proj_qkv` projection (already captured) and the output is what the
    implementation actually has to reproduce.
    """

    def __init__(self, bundle: Bundle, module, func_name: str, layers: list[int] | None = None,
                 expect_calls: bool = True):
        self.bundle = bundle
        self.module = module
        self.func_name = func_name
        self.orig = getattr(module, func_name)
        self.layers = layers
        self.expect_calls = expect_calls
        self.calls = 0
        self._index = 0

    def __enter__(self):
        bundle, layers = self.bundle, self.layers
        orig = self.orig
        state = self

        def spy(hidden_states, weight, bias=None, activation=None, **kw):
            out = orig(hidden_states, weight, bias=bias, activation=activation, **kw)
            state.calls += 1
            li = state._index
            if layers is None or li in layers:
                bundle.write("intermediates", f"{safe_name(state.func_name)}_call{li}_in", hidden_states)
                bundle.write("intermediates", f"{safe_name(state.func_name)}_call{li}_out", out)
            state._index += 1
            return out

        setattr(self.module, self.func_name, spy)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.func_name, self.orig)
        # If the call site bypassed our wrapper (kernel-hub path), `calls` stays
        # zero: report it rather than shipping a bundle missing the tensor.
        if self.calls == 0 and self.expect_calls:
            print(f"  !! WARNING: {self.func_name} spy captured nothing; the call "
                  f"site may be routed through the kernel hub")
        return False


class HookRecorder:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.handles = []
        self.count = 0

    def _hook(self, label: str):
        def fn(_module, _inputs, output):
            # Attention returns a tuple; record each tensor element.
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        self.bundle.write("intermediates", f"{label}__out{i}", o)
                        self.count += 1
            elif isinstance(output, torch.Tensor):
                self.bundle.write("intermediates", label, output)
                self.count += 1
        return fn

    def attach(self, model: torch.nn.Module) -> list[str]:
        hooked = []
        for name, mod in model.named_modules():
            if not name:
                continue
            if not (any(name.endswith(s) for s in HOOK_SUFFIXES) or _is_decoder_layer(name)):
                continue
            self.handles.append(mod.register_forward_hook(self._hook(safe_name(name))))
            hooked.append(name)
        return hooked

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------------------------------------------------------- #
# module-input capture
# --------------------------------------------------------------------------- #

class InputSpy:
    """Record the **input** of selected modules.

    `Qwen3_5Attention.forward` ends with

        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    so the tensor the module returns is *post*-`o_proj`. The post-gate,
    pre-`o_proj` tensor is never bound to a name, no module produces it, and a
    forward hook cannot see it -- but it is exactly where the output gate is
    applied, which is the single most falsifiable line in the whole attention
    block (the config says `swish`, the code says `sigmoid`).

    A forward hook receives `(module, inputs, output)`, so recording `inputs[0]` of
    `o_proj` captures it without touching the computation.
    """

    def __init__(self, bundle: Bundle, model, module_suffix: str, out_suffix: str = "__in"):
        self.bundle = bundle
        self.model = model
        self.module_suffix = module_suffix
        self.out_suffix = out_suffix
        self.handles = []
        self.count = 0

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if not name.endswith(self.module_suffix):
                continue
            self.handles.append(mod.register_forward_hook(self._mk(name)))
        return self

    def _mk(self, name: str):
        label = safe_name(name) + self.out_suffix
        bundle = self.bundle
        state = self

        def fn(_m, inputs, _out):
            if inputs and isinstance(inputs[0], torch.Tensor):
                bundle.write("intermediates", label, inputs[0])
                state.count += 1
        return fn

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


# --------------------------------------------------------------------------- #
# SwiGLU intermediate capture
# --------------------------------------------------------------------------- #

class SwigluSpy:
    """Capture `silu(gate_proj(x)) * up_proj(x)`, the elementwise product.

    `Qwen3_5MLP.forward` is a single expression:

        down_proj(act_fn(gate_proj(x)) * up_proj(x))

    so the product is never bound to a name and no module produces it -- hooks on
    `gate_proj` and `up_proj` give the two *inputs* to the multiplication but not
    its result. That result is where a SwiGLU implementation most plausibly goes
    wrong (wrong branch activated, or the order swapped), so it is worth recording
    explicitly. The spy recomputes nothing: it re-derives the product from the two
    tensors the projections returned.
    """

    def __init__(self, bundle: Bundle, model, act_fn):
        self.bundle = bundle
        self.model = model
        self.act_fn = act_fn
        self.handles = []
        self.count = 0
        # keyed by layer prefix, holding whichever of gate/up has arrived.
        self._pending: dict[str, dict] = {}

    def __enter__(self):
        for name, mod in self.model.named_modules():
            if not name.endswith("mlp"):
                continue
            gate = getattr(mod, "gate_proj", None)
            up = getattr(mod, "up_proj", None)
            if gate is None or up is None:
                continue
            self.handles.append(gate.register_forward_hook(self._mk(name, "gate")))
            self.handles.append(up.register_forward_hook(self._mk(name, "up")))
        return self

    def _mk(self, layer_name: str, which: str):
        prefix = safe_name(layer_name.rsplit(".", 1)[0])
        holder = self._pending

        def fn(_m, _i, out):
            slot = holder.setdefault(prefix, {})
            slot[which] = out
            if "gate" in slot and "up" in slot:
                gate, up = slot.pop("gate"), slot.pop("up")
                product = self.act_fn(gate) * up
                self.bundle.write("intermediates", f"{prefix}__mlp__swiglu_product", product)
                self.count += 1
        return fn

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


# --------------------------------------------------------------------------- #
# delta-rule operand capture (per block)
# --------------------------------------------------------------------------- #

class DeltaOperandSpy:
    """Capture the exact `q/k/v/g/beta` a block feeds into the delta rule.

    The unit golden in `units/` is a synthetic, isolated delta-rule problem. This
    is the same operator with *real* inputs from a real block, which is what an
    implementation needs in order to extend outward from the rule: it can verify
    the rule in place before the surrounding projections are written.

    `Qwen3_5GatedDeltaNet.forward` calls the **module-level**
    `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` by name.
    The `torch_` prefix matters: the `@use_kernel_func_from_hub_with_fallback`
    decorator's first argument is the *kernel* name (`"chunk_gated_delta_rule"`,
    without the prefix), which is a different string and is not what `forward`
    resolves. They are neither instance nor class attributes, so the module
    namespace is what has to be wrapped.

    The operands already have `l2norm` and the `1/sqrt(K)` scale applied by the
    caller when `use_qk_l2norm_in_kernel=True`. Recording them post-preparation
    means a caller-side implementation must not apply those twice.
    """

    def __init__(self, bundle: Bundle, module, captured: dict,
                 names=("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")):
        self.bundle = bundle
        self.module = module
        self.captured = captured
        self.names = names
        self.saved: list[tuple[str, object]] = []
        self.calls = 0

    def __enter__(self):
        for name in self.names:
            fn = getattr(self.module, name, None)
            if fn is None:
                continue
            self.saved.append((name, fn))
            setattr(self.module, name, self._wrap(name, fn))
        return self

    def _wrap(self, fname: str, fn):
        bundle, cap, state = self.bundle, self.captured, self
        counter = {"n": 0}

        def spy(query, key, value, g=None, beta=None, **kw):
            idx = counter["n"]
            counter["n"] += 1
            state.calls += 1
            tag = f"delta_{fname}_{idx}"
            for nm, t in (("q", query), ("k", key), ("v", value), ("g", g), ("beta", beta)):
                if isinstance(t, torch.Tensor):
                    bundle.write("intermediates", f"{safe_name(tag)}__{nm}", t)
            res = fn(query, key, value, g=g, beta=beta, **kw)
            out, st = res if isinstance(res, tuple) else (res, None)
            bundle.write("intermediates", f"{safe_name(tag)}__out", out)
            if isinstance(st, torch.Tensor):
                bundle.write("intermediates", f"{safe_name(tag)}__state", st)
            return res
        return spy

    def __exit__(self, *exc):
        for name, fn in self.saved:
            setattr(self.module, name, fn)
        if self.calls == 0:
            print("  !! WARNING: delta operand spy captured nothing")
        return False


# --------------------------------------------------------------------------- #
# delta-rule unit golden
# --------------------------------------------------------------------------- #

def dump_delta_rule(bundle: Bundle, m, seed: int = 1234) -> dict:
    """Golden for the gated delta rule, in the layout the real model uses.

    `Qwen3_5GatedDeltaNet.forward` passes `[B, T, H, K]`, not `[B, H, T, K]`:
    the rule's first statement is `x.transpose(1, 2)`, which turns `[B, T, H, K]`
    into `[B, H, T, K]`. Feeding `[B, H, T, K]` still produces self-consistent
    numbers but silently swaps the meaning of the T and H axes, which would be a
    trap for anyone validating a real forward pass against this bundle.

    Outputs therefore come back as:
      out   `[B, T, H, V]`
      state `[B, H, K, V]`
    """
    g = torch.Generator().manual_seed(seed)
    report = {}
    shapes = [(1, 2, 6, 16, 16, 4), (1, 2, 1, 16, 16, 4), (2, 3, 5, 8, 8, 4)]
    for B, H, T, K, V, CS in shapes:
        tag = f"B{B}_H{H}_T{T}_K{K}_V{V}"
        # [B, T, H, *] -- the real model's convention.
        q = torch.randn(B, T, H, K, generator=g)
        k = torch.randn(B, T, H, K, generator=g)
        v = torch.randn(B, T, H, V, generator=g)
        # g is a log-decay; keep it negative so exp(g) decays in (0, 1).
        gg = -torch.rand(B, T, H, generator=g) * 1.5
        beta = torch.rand(B, T, H, generator=g)

        for nm, t in (("q", q), ("k", k), ("v", v), ("g", gg), ("beta", beta)):
            bundle.write("units", f"delta_{tag}__{nm}", t)

        with torch.no_grad():
            rec_out, rec_state = m.torch_recurrent_gated_delta_rule(
                q.clone(), k.clone(), v.clone(), g=gg.clone(), beta=beta.clone(),
                initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            ck_out, ck_state = m.torch_chunk_gated_delta_rule(
                q.clone(), k.clone(), v.clone(), g=gg.clone(), beta=beta.clone(),
                chunk_size=CS, initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True)

        bundle.write("units", f"delta_{tag}__recurrent_out", rec_out)
        bundle.write("units", f"delta_{tag}__recurrent_state", rec_state)
        bundle.write("units", f"delta_{tag}__chunked_out", ck_out)
        bundle.write("units", f"delta_{tag}__chunked_state", ck_state)

        d_out = (rec_out - ck_out).abs().max().item()
        d_st = (rec_state - ck_state).abs().max().item()
        report[tag] = {
            "recurrent_vs_chunked_out_max_abs": d_out,
            "recurrent_vs_chunked_state_max_abs": d_st,
        }
        print(f"  delta {tag}: recurrent vs chunked  out {d_out:.3e}  state {d_st:.3e}")
    return report


# --------------------------------------------------------------------------- #
# greedy token trace
# --------------------------------------------------------------------------- #

@torch.no_grad()
def dump_token_trace(bundle: Bundle, model, input_ids: torch.Tensor,
                     steps: int, topk: int) -> dict:
    ids = input_ids.clone()
    trace = []
    for step in range(steps):
        logits = model(input_ids=ids).logits[0, -1].float()
        nxt = int(torch.argmax(logits).item())
        top = torch.topk(logits, topk)
        trace.append({
            "step": step,
            "input_len": int(ids.shape[1]),
            "argmax_token": nxt,
            "topk_tokens": [int(x) for x in top.indices.tolist()],
            "topk_logits": [float(x) for x in top.values.tolist()],
            "logit_sum": float(logits.sum().item()),
            "logit_max": float(logits.max().item()),
        })
        bundle.write("intermediates", f"greedy_step{step:02d}__logits", logits)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=ids.dtype)], dim=1)
    return {"steps": trace, "final_ids": [int(x) for x in ids[0].tolist()]}


# --------------------------------------------------------------------------- #

def build_model(cfg):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    try:
        model = M.Qwen3_5ForCausalLM(cfg)
        print("  model: Qwen3_5ForCausalLM")
        return model
    except Exception as e:  # noqa: BLE001
        print(f"  ForCausalLM unavailable ({type(e).__name__}: {e}); "
              f"using Qwen3_5TextModel + lm_head")

    class Wrapper(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.model = M.Qwen3_5TextModel(c)
            self.lm_head = torch.nn.Linear(c.hidden_size, c.vocab_size, bias=False)

        def forward(self, input_ids=None, **kw):
            h = self.model(input_ids=input_ids, **kw).last_hidden_state
            return type("Out", (), {"logits": self.lm_head(h)})()

    return Wrapper(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=None,
                    help="real HF model dir; omit for tiny synthetic")
    ap.add_argument("--tokens", type=int, default=16, help="greedy steps")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seq", type=int, default=6, help="prompt length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize-norms", action="store_true",
                    help="fill plain RMSNorm weights with N(0, 0.5) so that a scale "
                         "error or a doubled normalisation is observable")
    ap.add_argument("--ssm-gain", type=float, default=1.0,
                    help="scale linear_attn.out_proj so the recurrent path is "
                         "observable in the token trace (see note in main())")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(out)

    torch.manual_seed(args.seed)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    cfg = tiny_config()
    print(f"  tiny config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"layer_types={cfg.layer_types}")
    model = build_model(cfg).to(torch.float32).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params}")

    # Amplify the recurrent path. With default random init the SSM output is
    # ~0.1% of the residual, so the delta rule barely influences anything and a
    # wrong implementation would still reproduce the token trace exactly. Scaling
    # `out_proj` makes the recurrent path numerically comparable to the residual,
    # which is what gives the golden the power to detect a wrong delta rule.
    # This changes only the *test* model's weights; the math under test is
    # identical, and the scaled weights are what get written to the bundle.
    gain_lines = []
    if args.ssm_gain != 1.0:
        with torch.no_grad():
            n_scaled = 0
            for name, p in model.named_parameters():
                if name.endswith("linear_attn.out_proj.weight"):
                    p.mul_(args.ssm_gain)
                    n_scaled += 1
        gain_lines.append(f"  ssm_gain={args.ssm_gain} applied to {n_scaled} out_proj weights")

    if args.randomize_norms:
        n_norm = randomize_norms(model, args.seed)
        print(f"  randomize_norms: {n_norm} RMSNorm weight(s) reseeded "
              f"(plain N(0,0.5), gated 1+N(0,0.5))")
    dump_weights(bundle, model)

    print("  delta rule unit golden:")
    unit_report = dump_delta_rule(bundle, M)

    ids = torch.randint(0, cfg.vocab_size, (1, args.seq),
                        generator=torch.Generator().manual_seed(args.seed + 1))
    bundle.write("intermediates", "prompt_ids_as_f32", ids.to(torch.float32).reshape(-1))

    rec = HookRecorder(bundle)
    hooked = rec.attach(model)
    delta_cap: dict = {}
    conv_calls = 0
    conv_upd_calls = 0
    act_fn = M.ACT2FN[cfg.hidden_act]
    with torch.no_grad(), \
            ModuleSpy(bundle, M, "causal_conv1d_fn") as conv_spy, \
            ModuleSpy(bundle, M, "causal_conv1d_update",
                      expect_calls=False) as conv_upd_spy, \
            DeltaOperandSpy(bundle, M, delta_cap) as delta_spy, \
            SwigluSpy(bundle, model, act_fn) as swiglu_spy, \
            InputSpy(bundle, model, "self_attn.o_proj") as attn_gate_spy:
        out_main = model(input_ids=ids)
        conv_calls = conv_spy.calls
        conv_upd_calls = conv_upd_spy.calls
        delta_calls = delta_spy.calls
        swiglu_calls = swiglu_spy.count
        attn_gate_calls = attn_gate_spy.count
    rec.detach()
    print(f"  hooked {len(hooked)} modules -> {rec.count} tensors")
    print(f"  causal_conv1d_fn spy: {conv_calls} call(s); "
          f"causal_conv1d_update spy: {conv_upd_calls} call(s)")
    print(f"  delta rule spy: {delta_calls} call(s)")
    print(f"  swiglu product spy: {swiglu_calls} call(s)")
    print(f"  attn o_proj-input spy: {attn_gate_calls} call(s)")

    hidden = getattr(out_main, "last_hidden_state", None)
    if hidden is not None:
        bundle.write("intermediates", "last_hidden_state", hidden)

    print(f"  greedy trace: {args.tokens} steps")
    trace = dump_token_trace(bundle, model, ids, args.tokens, args.topk)

    import transformers
    manifest = {
        "schema": "qwen35-golden-v1",
        "byte_order": "little",
        "dtype": "f32",
        "source": {
            "model": "<tiny-synthetic>",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seed": args.seed,
            "params": n_params,
            "ssm_gain": args.ssm_gain,
            "randomize_norms": bool(args.randomize_norms),
        },
        "config": json.loads(json.dumps(cfg.to_dict(), default=str)),
        "prompt_ids": [int(x) for x in ids[0].tolist()],
        "delta_rule_selfcheck": unit_report,
        "hooked_modules": hooked,
        "tensors": bundle.entries,
        "greedy": trace,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(e["nbytes"] for e in bundle.entries)
    print(f"\n  bundle: {out}")
    print(f"  {len(bundle.entries)} tensors, {total / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
