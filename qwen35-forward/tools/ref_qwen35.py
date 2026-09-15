#!/usr/bin/env python3
"""Reference logits for a real Qwen3.5 checkpoint, for `qwenrun --compare`.

    ref_qwen35.py <model-dir> <out-dir> [--prompt 1,2,3] [--dtype float32]

Writes
    <out-dir>/logits_last.f32     the last position's logits, raw little-endian f32
    <out-dir>/layers.f32          every decoder layer's output, concatenated
    <out-dir>/layer_shapes.json   shapes, so the blob can be sliced
    <out-dir>/ref.json            argmax, top-k, dtype, versions

Why float32: the checkpoint is bfloat16. Casting to f32 on load is exact (a bf16
value is always representable in f32), so it removes the storage dtype from the
comparison and leaves only arithmetic differences. Running the reference in bf16
would add its own rounding on top and make a real disagreement indistinguishable
from dtype noise.

Why the per-layer dump: if the logits differ, the useful question is *which layer*
diverged first. Dumping every layer output answers it without a second run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prompt", default="9419")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float64", "bfloat16"],
                    help="float64 is for establishing ground truth to measure how "
                         "much of a float32 disagreement is arithmetic rather than semantic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--greedy", type=int, default=0,
                    help="also decode this many tokens greedily and report them")
    ap.add_argument("--attn", default="eager",
                    help="attention implementation; eager is the most explicit "
                         "reference for a small sequence")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in args.prompt.split(",") if x.strip()]

    import transformers

    dtype = {"float32": torch.float32, "float64": torch.float64,
             "bfloat16": torch.bfloat16}[args.dtype]

    # The checkpoint is a `Qwen3_5ForConditionalGeneration` (it has a vision tower),
    # but the text-only path is all this needs. Loading the conditional-generation
    # class and feeding `input_ids` alone exercises the same language model.
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=dtype, attn_implementation=args.attn
    ).to(args.device).eval()

    print(f"  class            {type(model).__name__}")
    print(f"  dtype            {dtype}")
    print(f"  attn impl        {model.config._attn_implementation}")
    tc = getattr(model.config, "text_config", model.config)
    print(f"  layers           {tc.num_hidden_layers}  hidden {tc.hidden_size}  vocab {tc.vocab_size}")
    print(f"  layer_types      {tc.layer_types[:8]}...")
    print(f"  partial rotary   {tc.rope_parameters.get('partial_rotary_factor')}  "
          f"theta {tc.rope_parameters.get('rope_theta')}")

    input_ids = torch.tensor([ids], device=args.device)

    layer_outs = []
    handles = []
    for i, layer in enumerate(model.model.language_model.layers):
        def hook(_m, _inp, o, i=i):
            t = o[0] if isinstance(o, tuple) else o
            layer_outs.append((i, t.detach().float().cpu().numpy().astype(np.float32).copy()))
        handles.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        res = model(input_ids=input_ids)
    for h in handles:
        h.remove()

    if not layer_outs:
        print("  !! no layer hooks fired; the module path may differ", file=sys.stderr)
        return 2

    logits = res.logits[0, -1].detach().float().cpu().numpy().astype(np.float32)
    logits.tofile(out / "logits_last.f32")

    layer_outs.sort(key=lambda x: x[0])
    shapes = [list(t.shape) for _, t in layer_outs]
    blob = np.concatenate([t.reshape(-1) for _, t in layer_outs])
    blob.tofile(out / "layers.f32")
    (out / "layer_shapes.json").write_text(json.dumps(shapes))

    top = np.argsort(-logits)[:10]
    summary = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "attn_impl": model.config._attn_implementation,
        "prompt": ids,
        "seq_len": len(ids),
        "vocab": int(logits.shape[0]),
        "argmax": int(top[0]),
        "topk": [int(x) for x in top],
        "topk_logits": [float(logits[x]) for x in top],
        "logit_sum": float(logits.sum()),
        "n_layers_dumped": len(layer_outs),
    }
    (out / "ref.json").write_text(json.dumps(summary, indent=2))

    # Greedy decoding, matching the engine's loop: re-run the full sequence each
    # step with no KV cache, so the two implementations do the same amount and kind
    # of work and any difference is arithmetic, not scheduling.
    if args.greedy > 0:
        gen = []
        cur = list(ids)
        with torch.no_grad():
            for k in range(args.greedy):
                r = model(input_ids=torch.tensor([cur], device=args.device))
                nxt = int(torch.argmax(r.logits[0, -1]).item())
                gen.append(nxt)
                cur.append(nxt)
        summary["greedy"] = gen
        summary["greedy_full_ids"] = cur
        (out / "ref.json").write_text(json.dumps(summary, indent=2))
        print()
        print(f"  greedy ({args.greedy} steps, no cache)")
        for k, t in enumerate(gen):
            print(f"     step {k:>2}  len={len(ids)+k:<4} -> {t}")
        print(f"  generated ids    {','.join(map(str, gen))}")
        print(f"  full sequence    {','.join(map(str, cur))}")

    print()
    print(f"  prompt           {ids}")
    print(f"  argmax           {summary['argmax']}")
    print(f"  top-10           {summary['topk']}")
    print(f"  layers dumped    {len(layer_outs)}  -> {out/'layers.f32'}")
    print(f"  logits           -> {out/'logits_last.f32'}  ({logits.nbytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
