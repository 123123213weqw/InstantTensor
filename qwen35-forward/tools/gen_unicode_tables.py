#!/usr/bin/env python3
"""Generate `gdn/src/unicode_tables.rs` from Python's `unicodedata`.

    gen_unicode_tables.py > gdn/src/unicode_tables.rs

The tokenizer needs four things that Rust's standard library does not expose:

  * `\\p{L}`, `\\p{M}`, `\\p{N}` -- the Qwen2 pre-tokenizer splits on these general
    categories, and `char::is_alphabetic()` is *not* `\\p{L}`: it is the Alphabetic
    property, which also includes `Nl` (Roman numerals) and part of `Mn`.
  * canonical decomposition, combining classes and composition pairs, for NFC.

These are generated rather than pulled from a crate so the build stays offline and
the tables can be inspected and regenerated. `unicodedata.unidata_version` is
recorded in the output: a codepoint assigned in a later Unicode version than this
table will be classified as "other", which is the only way this can drift.

The tables are stored as ranges where the property is a pure set membership
(`L`, `M`, `N`), and as sorted `(key, value)` slices where a codepoint maps to a
value. Hangul syllables (U+AC00..U+D7A3, 11172 codepoints) carry an algorithmic
decomposition per UAX #15 and are excluded: they would otherwise be 85% of the
table for no information.
"""

import sys
import unicodedata as u

HANGUL_SBASE = 0xAC00
HANGUL_SCOUNT = 11172


def ranges(codepoints):
    """Collapse a sorted iterable of codepoints into inclusive ranges."""
    out = []
    for cp in codepoints:
        if out and cp == out[-1][1] + 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return [(a, b) for a, b in out]


def cat_prefix(ch, prefix):
    return u.category(ch).startswith(prefix)


def main() -> int:
    letters, marks, numbers = [], [], []
    for cp in range(0x110000):
        ch = chr(cp)
        cat = u.category(ch)
        if cat[0] == "L":
            letters.append(cp)
        elif cat[0] == "M":
            marks.append(cp)
        elif cat[0] == "N":
            numbers.append(cp)

    # --- canonical decomposition (single step, non-Hangul) ---
    decomp = {}
    for cp in range(0x110000):
        if HANGUL_SBASE <= cp < HANGUL_SBASE + HANGUL_SCOUNT:
            continue
        d = u.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue  # compatibility decomposition, or nothing
        parts = [int(x, 16) for x in d.split()]
        decomp[cp] = parts

    # --- canonical combining class ---
    ccc = {}
    for cp in range(0x110000):
        k = u.combining(chr(cp))
        if k:
            ccc[cp] = k

    # --- composition pairs, with exclusions already applied ---
    # A pair composes only if NFC of the decomposition returns the composed
    # character. That folds in the composition-exclusion table without needing to
    # read it separately.
    compose = {}
    for cp, parts in decomp.items():
        if len(parts) != 2:
            continue
        a, b = parts
        if u.normalize("NFC", chr(a) + chr(b)) == chr(cp):
            compose[(a, b)] = cp

    w = sys.stdout.write
    w("//! Unicode tables for the tokenizer. GENERATED -- do not edit.\n")
    w("//!\n")
    w("//! Produced by `tools/gen_unicode_tables.py`; run that script to regenerate.\n")
    w(f"//! Source: Python `unicodedata`, Unicode version {u.unidata_version}.\n")
    w("//!\n")
    w("//! Why these exist: the Qwen2 pre-tokenizer splits on the general categories\n")
    w("//! `\\p{L}`, `\\p{M}` and `\\p{N}`, and `char::is_alphabetic()` is not `\\p{L}` --\n")
    w("//! it is the Alphabetic property, which additionally covers `Nl` and part of\n")
    w("//! `Mn`. The NFC tables are here for the same reason: neither category\n")
    w("//! membership nor canonical composition is available in `core`.\n")
    w("\n")

    def emit_ranges(name, rs, doc):
        w(f"/// {doc}\n")
        w(f"pub static {name}: [(u32, u32); {len(rs)}] = [\n")
        for a, b in rs:
            w(f"    (0x{a:X}, 0x{b:X}),\n")
        w("];\n\n")

    emit_ranges("LETTER_RANGES", ranges(letters), "Inclusive ranges of General_Category L (all subcategories).")
    emit_ranges("MARK_RANGES", ranges(marks), "Inclusive ranges of General_Category M.")
    emit_ranges("NUMBER_RANGES", ranges(numbers), "Inclusive ranges of General_Category N.")

    def emit_map(name, items, doc):
        w(f"/// {doc}\n")
        w(f"pub static {name}: [(u32, u32); {len(items)}] = [\n")
        for k, v in items:
            w(f"    (0x{k:X}, 0x{v:X}),\n")
        w("];\n\n")

    # A one-step canonical decomposition is 1 or 2 codepoints. Emit the two shapes
    # separately so each is a flat sorted table with a single comparison.
    d1 = sorted((cp, parts[0]) for cp, parts in decomp.items() if len(parts) == 1)
    d2 = sorted((cp, parts[0], parts[1]) for cp, parts in decomp.items() if len(parts) == 2)
    emit_map("DECOMPOSE_1", d1, "One-step canonical decomposition to a single codepoint. Excludes Hangul, which is algorithmic.")
    w(f"/// One-step canonical decomposition to a pair, sorted by the first field.\n")
    w(f"pub static DECOMPOSE_2: [(u32, u32, u32); {len(d2)}] = [\n")
    for cp, a, b in d2:
        w(f"    (0x{cp:X}, 0x{a:X}, 0x{b:X}),\n")
    w("];\n\n")

    emit_map("CCC", sorted(ccc.items()), "Canonical combining class, for codepoints where it is non-zero.")

    comp = sorted(((a << 21) | b, cp) for (a, b), cp in compose.items())
    w("/// Canonical composition pairs, packed as `(a << 21) | b` so the table is a\n")
    w("/// flat sorted `u64` array. Composition exclusions are already applied.\n")
    w(f"pub static COMPOSE: [(u64, u32); {len(comp)}] = [\n")
    for k, v in comp:
        w(f"    (0x{k:X}, 0x{v:X}),\n")
    w("];\n\n")

    w(f"/// Unicode version the tables were generated from.\n")
    w(f"pub const UNICODE_VERSION: &str = \"{u.unidata_version}\";\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
