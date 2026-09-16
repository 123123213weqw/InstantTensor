#!/usr/bin/env python3
"""Build a tokenizer conformance corpus from the reference implementation.

    make_tok_corpus.py <model-dir> <out.json>

Writes `[{"text": ..., "ids": [...], "tokens": [...]}, ...]`.

The corpus exists so a tokenizer implementation can be judged by *output* rather
than by matching a regex dialect. It deliberately probes the places where the
Qwen2 pre-tokenizer is easy to get subtly wrong, since each of those is a place
where a hand-written splitter can agree on ordinary prose and disagree on real
input:

  * whitespace runs, in every position: leading, trailing, doubled, tabs, mixed
  * newlines, including runs, and CRLF
  * the contraction alternative, with and without a preceding word
  * letters adjacent to digits, and digits adjacent to punctuation
  * non-letter non-space symbol runs
  * scripts that are letters but not Latin, and scripts that are *not* `\\p{L}`
  * characters whose UTF-8 encoding is what actually reaches the BPE
  * special tokens, which are matched before the regex ever runs
  * the degenerate cases: empty, single byte, a lone space
"""

import argparse
import json
import sys
from pathlib import Path

# Fixed cases, chosen for the structure they exercise rather than their meaning.
CASES = [
    # --- degenerate ---
    "", " ", "  ", "\n", "\t", "a", ".", "0", "'", "|",
    # --- plain prose ---
    "Hello world", "The quick brown fox jumps over the lazy dog.",
    "one", "one two three four five",
    # --- whitespace in every position ---
    " leading", "  leading", "leading ", "leading  ", "  both  ",
    "a  b", "a   b", "a\tb", "a \t b", "\ta", "a\t",
    "a\nb", "a\n\nb", "a\n\n\nb", "\n\na", "a\r\nb", "a\r\n\r\nb",
    " \n ", "\t\t", "a \t\n b",
    # --- contractions ---
    "don't", "can't", "it's", "they're", "I've", "we'll", "he'd",
    "'s", "x's", "DON'T", "Don't", "rock'n'roll", "''", "'''",
    # --- digits and letters together ---
    "3.14159", "abc123", "123abc", "a1b2c3", "2024-01-15", "1,000,000",
    "0x1F", "1e-9", "v1.2.3", "100%",
    # --- symbol runs ---
    "a+b", "a++b", "+++", "->", "=>", "::", "...", "!!!", "?!",
    "((a))", "[1,2]", "{k: v}", "a|b", "\\n", "`code`", "$$$",
    # --- whitespace next to symbols ---
    "a + b", "a  +  b", "- item", " - item", "  -  item",
    # --- non-Latin letters ---
    "你好", "你好，世界", "日本語のテキスト", "한국어 텍스트",
    "Привет мир", "Γειά σου", "مرحبا بالعالم", "שלום עולם",
    "नमस्ते दुनिया", "ไทย", "ελληνικά",
    # --- emoji and symbols outside \p{L} ---
    "emoji 😀", "😀😀😀", "a😀b", "🇨🇳", "👨‍👩‍👧",
    "™©®", "→←↑↓", "½¼¾", "°C", "€100", "№5",
    # --- combining marks, which are \p{M} not \p{L} ---
    "e\u0301", "a\u0300b", "x\u0301\u0302\u0303", "cafe\u0301",
    # --- mixed scripts, the realistic hard case ---
    "mixed 中文 and English123", "Hello 世界 123 !!!", "a中b1c😀d",
    "中文English混合", "test-测试-2024",
    # --- case and camel ---
    "camelCaseWord", "PascalCase", "SCREAMING_SNAKE", "snake_case",
    "XMLHttpRequest", "iPhone", "eBay",
    # --- special tokens, matched before the regex ---
    "<|im_start|>user", "<|im_end|>", "<|endoftext|>",
    "a<|im_start|>b", "<|im_start|>system\nYou are helpful.<|im_end|>",
    "<tool_call>", "<think>reasoning</think>",
    # --- realistic code and JSON ---
    "def f(x): return x+1",
    '{"key": "value", "n": 42}',
    "for i in range(10):\n    print(i)",
    "https://example.com/path?a=1&b=2",
    "user@example.com",
    # --- long and repetitive ---
    "a" * 100, "ab" * 50, " " * 20 + "x" + " " * 20,
    "word " * 30,
    # --- canonically decomposable input, where NFC changes the bytes ---
    # A decomposition that recomposes: the tokenizer must see the composed form.
    "e\u0301", "E\u0301", "a\u0300b", "cafe\u0301", "n\u0303",
    "o\u0308", "u\u0308", "A\u030A", "\u00e9", "\u00c9",
    # Marks in both orders: canonical ordering must produce the same result.
    "a\u0323\u0301", "a\u0301\u0323", "q\u0323\u0307", "q\u0307\u0323",
    # Composition exclusions: these decompose and must NOT recompose.
    "\u0958", "\u09dc", "\u2adc", "\u0344", "a\u0958b",
    # Hangul: syllables, jamo in order, and a lone jamo.
    "\ud55c", "\uac00", "\ub098", "\u1112\u1161\u11ab",
    "\u1100\u1161", "\u11ab", "\ud55c\uad6d\uc5b4",
    # A combining mark with no starter.
    "\u0301a", "\u0301",
    # Marks on non-Latin bases, and mixed scripts.
    "\u4e2d\u0301", "\u0410\u0301", "cafe\u0301 \u4e2d\u6587",

    # --- realistic chat-ish text ---
    "Explain how a transformer works in one paragraph.",
    "What is 2+2?",
    "Write a poem about the sea.\n\nIt should rhyme.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "file"],
        help="auto = AutoTokenizer, which honours tokenizer_class and so drops "
        "\p{M}; file = load tokenizer.json directly, which keeps it",
    )
    args = ap.parse_args()

    if args.mode == "auto":
        # `AutoTokenizer` honours `tokenizer_class`, which for this checkpoint is a
        # Qwen2 variant that rebuilds the pre-tokenizer without `\\p{M}`.
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(args.model_dir)
        encode = lambda s: tk.encode(s, add_special_tokens=False)
        to_tokens = lambda ids: tk.convert_ids_to_tokens(ids)
        decode = lambda ids: tk.decode(ids, skip_special_tokens=False)
        label = "auto (AutoTokenizer, no-marks pattern)"
    else:
        # Loading `tokenizer.json` directly keeps the file's own pattern, which has
        # `\\p{M}`. The two disagree only where a combining mark follows a letter.
        from tokenizers import Tokenizer as RawTokenizer

        raw = RawTokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
        inv = raw.get_vocab()
        by_id = {v: k for k, v in inv.items()}
        encode = lambda s: raw.encode(s, add_special_tokens=False).ids
        to_tokens = lambda ids: [by_id.get(i, "") for i in ids]
        decode = lambda ids: raw.decode(ids, skip_special_tokens=False)
        label = "file (tokenizer.json, with-marks pattern)"

    out = []
    for text in CASES:
        ids = encode(text)
        toks = to_tokens(ids)
        out.append({"text": text, "ids": list(ids), "tokens": list(toks)})

    # Round-trip is a property the implementation must also have.
    bad = []
    for rec in out:
        back = decode(rec["ids"])
        rec["decoded"] = back
        if back != rec["text"]:
            bad.append((rec["text"], back))

    doc = {
        "mode": label,
        "generated_by": "tools/make_tok_corpus.py",
        "cases": out,
    }
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"  mode: {label}")
    print(f"  {len(out)} cases -> {args.out}")
    print(f"  decode round-trip mismatches: {len(bad)}")
    for t, b in bad[:8]:
        print(f"     {t!r} -> {b!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
