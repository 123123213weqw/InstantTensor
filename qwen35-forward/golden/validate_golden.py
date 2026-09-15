#!/usr/bin/env python3
# coding=utf-8
"""Validate a golden bundle before trusting it as a target.

A golden that is *not reproducible* or *not discriminative* is worse than no
golden: it produces confident, meaningless verdicts. This checks three things:

1. **Reproducibility** — regenerate into a second directory and compare every
   file byte for byte. A non-deterministic golden cannot be a reference.
2. **Well-formedness** — no NaN/Inf, no all-zero tensors where values are
   expected, and every manifest entry matches its file's actual size.
3. **Discrimination** — perturb one weight by a relative epsilon and measure how
   far the outputs move. This yields the tolerance floor: a comparison tolerance
   *tighter* than the perturbation response is meaningless noise-chasing, and one
   *looser* than it cannot detect real bugs.

Usage
-----
    python validate_golden.py --bundle golden_tiny
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def safe_name(s: str) -> str:
    """Mirror of the generator's naming so bundle labels can be matched."""
    return s.replace(".", "__")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text())


def read_tensor(bundle: Path, entry: dict) -> np.ndarray:
    raw = np.fromfile(bundle / entry["file"], dtype="<f4")
    return raw.reshape(entry["shape"])


# --------------------------------------------------------------------------- #
# 1) reproducibility
# --------------------------------------------------------------------------- #

def check_reproducible(bundle: Path, gen: Path, seed: int, ssm_gain: float = 1.0) -> tuple[bool, str]:
    """Regenerate into a second directory and compare every file.

    `ssm_gain` has to be passed through: it scales `linear_attn.out_proj` at
    generation time, so regenerating `golden_sensitive` without it produces
    different weights and 183 downstream files differ. Omitting it was a bug in
    this validator, and it reported those as a reproducibility failure.
    """
    with tempfile.TemporaryDirectory() as td:
        second = Path(td) / "again"
        cmd = [sys.executable, str(gen), "--out", str(second), "--seed", str(seed)]
        if ssm_gain != 1.0:
            cmd += ["--ssm-gain", repr(ssm_gain)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"regeneration failed: {r.stderr.strip()[-400:]}"

        a = {p.relative_to(bundle) for p in bundle.rglob("*") if p.is_file()}
        b = {p.relative_to(second) for p in second.rglob("*") if p.is_file()}
        if a != b:
            return False, f"file sets differ: only-in-first={sorted(a - b)[:3]} only-in-second={sorted(b - a)[:3]}"

        # manifest.json embeds no timestamps, so it must match too.
        differing = []
        for rel in sorted(a):
            if sha256_file(bundle / rel) != sha256_file(second / rel):
                differing.append(str(rel))
        if differing:
            return False, f"{len(differing)} files differ: {differing[:5]}"
        return True, f"all {len(a)} files byte-identical across runs"


# --------------------------------------------------------------------------- #
# 2) well-formedness
# --------------------------------------------------------------------------- #

def check_wellformed(bundle: Path, man: dict) -> tuple[bool, list[str], list[str]]:
    """Returns (ok, problems, notes).

    Zero-valued weights are *reported as a note*, not a failure: `Qwen3_5RMSNorm`
    uses the Gemma convention `x * (1 + weight)` with `weight` initialised to
    zeros, so an all-zero norm weight is correct and has effective scale 1.
    Treating it as an error (as an earlier version of this script did) would be a
    false positive.
    """
    problems: list[str] = []
    notes: list[str] = []
    stats = []
    for e in man["tensors"]:
        p = bundle / e["file"]
        if not p.exists():
            problems.append(f"missing file {e['file']}")
            continue
        actual = p.stat().st_size
        if actual != e["nbytes"]:
            problems.append(f"{e['file']}: size {actual} != manifest {e['nbytes']}")
            continue
        a = read_tensor(bundle, e)
        if not np.isfinite(a).all():
            problems.append(f"{e['file']}: contains NaN/Inf")
        stats.append((e["group"], e["name"], float(np.abs(a).max()), float(np.abs(a).mean())))

    weights = [s for s in stats if s[0] == "weights"]
    zero_weights = [s[1] for s in weights if s[2] == 0.0]
    if zero_weights:
        norms = [n for n in zero_weights if "layernorm" in n or "norm__weight" in n]
        others = [n for n in zero_weights if n not in norms]
        notes.append(
            f"{len(zero_weights)} weight tensors are all zero; "
            f"{len(norms)} are RMSNorm weights (expected: Qwen3_5RMSNorm uses "
            f"x*(1+w) with zero init)"
        )
        if others:
            problems.append(f"{len(others)} non-norm weights are all zero: {others[:4]}")

    # A dead network would show up as all-zero activations everywhere.
    acts = [s for s in stats if s[0] == "intermediates"]
    dead = [s[1] for s in acts if s[2] == 0.0]
    if dead:
        problems.append(f"{len(dead)} intermediate tensors are entirely zero: {dead[:4]}")
    return (not problems), problems, notes


# --------------------------------------------------------------------------- #
# 3) discrimination
# --------------------------------------------------------------------------- #

def build_model(cfg_dict):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    cfg = Qwen3_5TextConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in Qwen3_5TextConfig().to_dict()
    })
    try:
        return M.Qwen3_5ForCausalLM(cfg), cfg
    except Exception:  # noqa: BLE001
        class Wrapper(torch.nn.Module):
            def __init__(self, c):
                super().__init__()
                self.model = M.Qwen3_5TextModel(c)
                self.lm_head = torch.nn.Linear(c.hidden_size, c.vocab_size, bias=False)

            def forward(self, input_ids=None, **kw):
                h = self.model(input_ids=input_ids, **kw).last_hidden_state
                return type("Out", (), {"logits": self.lm_head(h)})()
        return Wrapper(cfg), cfg


@torch.no_grad()
def probe(model, prompt_ids, steps):
    """Return the max|logit| path and the greedy token sequence."""
    ids = torch.tensor([prompt_ids])
    toks, last_logits = [], None
    for _ in range(steps):
        last_logits = model(input_ids=ids).logits[0, -1].float()
        nxt = int(torch.argmax(last_logits).item())
        toks.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return toks, last_logits


def apply_golden_weights(model, bundle, man):
    sd = dict(model.state_dict())
    for e in man["tensors"]:
        if e["group"] != "weights":
            continue
        k = e["name"].replace("__", ".")
        if k in sd:
            sd[k] = torch.from_numpy(read_tensor(bundle, e).copy())
    model.load_state_dict(sd, strict=False)


@torch.no_grad()
def probe_token(model, prompt_ids, steps):
    """Greedy decode without a cache; returns token ids and the last logits."""
    ids = torch.tensor([prompt_ids])
    toks, last = [], None
    for _ in range(steps):
        last = model(input_ids=ids).logits[0, -1].float()
        nxt = int(torch.argmax(last).item())
        toks.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return toks, last


class Capture:
    """Re-runs the forward and snapshots one tensor, for perturbation diffing."""

    def __init__(self, label: str):
        self.label = label
        self.value = None
        self.handle = None

    def attach(self, model):
        for name, mod in model.named_modules():
            if safe_name(name) == self.label:
                self.handle = mod.register_forward_hook(self._fn)
                return True
        return False

    def _fn(self, _m, _i, out):
        if isinstance(out, torch.Tensor):
            self.value = out.detach().float().clone()

    def detach(self):
        if self.handle:
            self.handle.remove()


def check_discrimination(bundle: Path, man: dict, epsilons, gain: float) -> list[dict]:
    """Perturb one weight by relative eps; measure per-tensor and token response.

    Two sensitivities matter and they differ by orders of magnitude:

    * **per-tensor** — how far does the tensor this layer produces move? This is
      the instrument that can actually validate the delta rule.
    * **token-level** — does the greedy sequence change? With random weights the
      recurrent path contributes ~0.1% of the residual, so this is nearly blind.
      Reporting both makes the limitation explicit rather than implicit.
    """
    torch.manual_seed(man["source"]["seed"])
    base, _cfg = build_model(man["config"])
    base = base.to(torch.float32).eval()
    apply_golden_weights(base, bundle, man)

    prompt = man["prompt_ids"]
    steps = len(man["greedy"]["steps"])

    # The layers whose output we watch: an SSM layer and a full-attention layer.
    probes = [
        "model__layers__0__linear_attn",
        "model__layers__0__mlp__down_proj",
        "model__layers__1__linear_attn",
        "model__layers__3__self_attn",
        "model__layers__7__self_attn",
    ]
    caps = {p: Capture(p) for p in probes}
    for p, c in caps.items():
        c.attach(base)
    base_toks, base_logits = probe_token(base, prompt, steps)
    base_vals = {p: c.value.clone() for p, c in caps.items() if c.value is not None}
    for c in caps.values():
        c.detach()

    target = next(e for e in man["tensors"]
                  if e["group"] == "weights" and e["name"].endswith("linear_attn__A_log"))

    rows = []
    for eps in epsilons:
        m2, _ = build_model(man["config"])
        m2 = m2.to(torch.float32).eval()
        apply_golden_weights(m2, bundle, man)
        with torch.no_grad():
            k = target["name"].replace("__", ".")
            w = dict(m2.named_parameters())[k]
            w.mul_(1.0 + eps)

        caps2 = {p: Capture(p) for p in probes}
        for p, c in caps2.items():
            c.attach(m2)
        toks2, logits2 = probe_token(m2, prompt, steps)
        new_vals = {p: c.value.clone() for p, c in caps2.items() if c.value is not None}
        for c in caps2.values():
            c.detach()

        per_tensor = {}
        for p in base_vals:
            if p in new_vals:
                d = (base_vals[p] - new_vals[p]).abs().max().item()
                ref = base_vals[p].abs().max().item() or 1.0
                per_tensor[p] = {"abs": d, "rel": d / ref}
        rows.append({
            "epsilon": eps,
            "max_abs_logit_delta": (base_logits - logits2).abs().max().item(),
            "greedy_tokens_changed": sum(1 for a, b in zip(base_toks, toks2) if a != b),
            "steps": len(base_toks),
            "per_tensor": per_tensor,
        })
        del m2
    return rows, base_vals


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--gen", default=None, help="generator script (for reproducibility)")
    ap.add_argument("--skip-discrimination", action="store_true")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="scale the recurrent out_proj to make the SSM path observable")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    if not (bundle / "manifest.json").exists():
        print(f"no manifest.json in {bundle}")
        return 2
    man = load_manifest(bundle)

    ok = True

    print("=" * 72)
    print("1) reproducibility")
    gen = Path(args.gen) if args.gen else Path(__file__).with_name("gen_golden.py")
    if gen.exists():
        gain = float(man["source"].get("ssm_gain", 1.0))
        good, msg = check_reproducible(bundle, gen, man["source"]["seed"], gain)
        print(f"   [{'PASS' if good else 'FAIL'}] {msg}")
        ok &= good
    else:
        print(f"   [SKIP] generator not found at {gen}")

    print("=" * 72)
    print("2) well-formedness")
    good, problems, notes = check_wellformed(bundle, man)
    print(f"   [{'PASS' if good else 'FAIL'}] {len(man['tensors'])} tensors checked")
    for n in notes:
        print(f"     note: {n}")
    for p in problems[:10]:
        print(f"     {p}")
    ok &= good

    print("=" * 72)
    print("3) discrimination (does the golden actually notice a wrong number?)")
    if args.skip_discrimination:
        print("   [SKIP]")
    else:
        rows, _base = check_discrimination(bundle, man, [0.0, 1e-4, 1e-3, 1e-2], args.gain)
        print(f"   perturbing {man['config'].get('num_hidden_layers')}-layer model A_log "
              f"by relative epsilon:")
        print(f"     {'eps':>7} | {'max|dlogit|':>12} | {'greedy':>7} | per-tensor max abs delta")
        first = rows[0]["per_tensor"]
        for r in rows:
            pt = max((v["abs"] for v in r["per_tensor"].values()), default=0.0)
            print(f"     {r['epsilon']:>7.0e} | {r['max_abs_logit_delta']:>12.3e} | "
                  f"{r['greedy_tokens_changed']:>3}/{r['steps']:<3} | {pt:.3e}")
        print()
        print("   per-tensor detail at eps=1e-2:")
        for k, v in sorted(rows[-1]["per_tensor"].items()):
            print(f"     {k:<48} abs={v['abs']:.3e}  rel={v['rel']:.3e}")

        zero = next((r for r in rows if r["epsilon"] == 0.0), None)
        if zero and (zero["max_abs_logit_delta"] != 0.0
                     or any(v["abs"] != 0.0 for v in zero["per_tensor"].values())):
            print(f"   [FAIL] eps=0 changed the output -> not deterministic")
            ok = False
        else:
            print("   [PASS] eps=0 is bit-identical (deterministic)")

        sens = max((v["abs"] for v in rows[-1]["per_tensor"].values()), default=0.0)
        if sens <= 0.0:
            print("   [FAIL] eps=1e-2 moves *no* watched tensor -> golden cannot "
                  "discriminate a wrong delta rule")
            ok = False
        else:
            print(f"   [PASS] eps=1e-2 moves per-tensor values by up to {sens:.3e} "
                  f"-> per-tensor comparison discriminates")
        tok = rows[-1]["greedy_tokens_changed"]
        if tok == 0:
            print(f"   [NOTE] eps=1e-2 changed 0/{rows[-1]['steps']} greedy tokens: "
                  f"the token trace alone cannot validate the recurrent path here")

    print("=" * 72)
    print("RESULT:", "all checks passed" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
