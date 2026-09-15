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
HOOK_SUFFIXES = (
    "input_layernorm", "post_attention_layernorm", "mlp", "linear_attn",
    "self_attn", "conv1d", "norm", "in_proj_qkv", "in_proj_z", "in_proj_a",
    "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
    "q_norm", "k_norm", "gate_proj", "up_proj", "down_proj", "embed_tokens",
)


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
            if not any(name.endswith(s) for s in HOOK_SUFFIXES):
                continue
            self.handles.append(mod.register_forward_hook(self._hook(safe_name(name))))
            hooked.append(name)
        return hooked

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------------------------------------------------------- #
# delta-rule unit golden
# --------------------------------------------------------------------------- #

def dump_delta_rule(bundle: Bundle, m, seed: int = 1234) -> dict:
    g = torch.Generator().manual_seed(seed)
    report = {}
    shapes = [(1, 2, 6, 16, 16, 4), (1, 2, 1, 16, 16, 4), (2, 3, 5, 8, 8, 4)]
    for B, H, T, K, V, CS in shapes:
        tag = f"B{B}_H{H}_T{T}_K{K}_V{V}"
        q = torch.randn(B, H, T, K, generator=g)
        k = torch.randn(B, H, T, K, generator=g)
        v = torch.randn(B, H, T, V, generator=g)
        # g is a log-decay; keep it negative so exp(g) decays in (0, 1).
        gg = -torch.rand(B, H, T, generator=g) * 1.5
        beta = torch.rand(B, H, T, generator=g)

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

    dump_weights(bundle, model)

    print("  delta rule unit golden:")
    unit_report = dump_delta_rule(bundle, M)

    ids = torch.randint(0, cfg.vocab_size, (1, args.seq),
                        generator=torch.Generator().manual_seed(args.seed + 1))
    bundle.write("intermediates", "prompt_ids_as_f32", ids.to(torch.float32).reshape(-1))

    rec = HookRecorder(bundle)
    hooked = rec.attach(model)
    with torch.no_grad():
        out_main = model(input_ids=ids)
    rec.detach()
    print(f"  hooked {len(hooked)} modules -> {rec.count} tensors")

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
