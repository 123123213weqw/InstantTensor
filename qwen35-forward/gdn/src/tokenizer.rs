//! Byte-level BPE tokenizer, matching `Qwen2Tokenizer` as shipped with Qwen3.5.
//!
//! # The pipeline
//!
//! ```text
//! text
//!   -> NFC                                     normalizer
//!   -> split out added tokens                  they never reach the regex or the BPE
//!   -> split each remaining run                the Qwen2 pre-tokenizer regex
//!   -> UTF-8 bytes -> byte-level characters    the reason there is no UNK
//!   -> BPE merges                              lowest merge rank first
//!   -> ids
//! ```
//!
//! # Why byte-level
//!
//! The pre-tokenizer's output is converted to UTF-8 bytes and each byte is mapped to
//! a printable character before the BPE ever sees it, so every one of the 256 possible
//! bytes is a token. A byte that is not printable in Latin-1 gets mapped up into
//! `U+0100..`. The observable consequences:
//!
//! * `"Hello world"` becomes `["Hello", "Ġworld"]` -- `Ġ` is byte `0x20`, a space.
//! * `"a\n\nb"` becomes `["a", "ĊĊ", "b"]` -- `Ċ` is byte `0x0A`.
//! * `"你好"` becomes 6 characters, because each CJK codepoint is 3 UTF-8 bytes. So
//!   the vocabulary contains no CJK character, only its bytes.
//! * There is no unknown token. `unk_token` is `null` and `byte_fallback` is false,
//!   because nothing can be out of vocabulary.
//!
//! # The pre-tokenizer is a regex, and its semantics had to be recovered by hand
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)       1  contractions, case-insensitive
//! |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   2  one optional non-letter/digit/newline, then letters
//! |\p{N}                             3  a single number-class character
//! | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*    4  optional space, symbol run, trailing newlines
//! |\s*[\r\n]+                        5  whitespace ending in newlines
//! |\s+(?!\S)                         6  whitespace not immediately before a non-space
//! |\s+                               7  whitespace
//! ```
//!
//! At each position the alternatives are tried **in order** and the first match wins,
//! with greedy quantifiers that backtrack -- Perl semantics, not longest-match. Three
//! observations pin down the parts that are easy to get wrong, all reproduced by the
//! committed corpus:
//!
//! * `"Hello world"` -> `["Hello", "Ġworld"]`. The space belongs to alternative 2,
//!   via its optional prefix, and is *not* a separate piece. Getting this wrong yields
//!   `["Hello", "Ġ", "world"]`, which is a different token sequence.
//! * `"a+b"` -> 2 tokens but `"a++b"` -> 3. Alternative 2's optional prefix takes
//!   **exactly one** character, so `"+b"` is one piece while `"++"` falls through to
//!   alternative 4. This is what fixes alternative 2's position ahead of 4.
//! * `"  leading"` -> `["Ġ", "Ġleading"]`. Alternative 6 is `\s+(?!\S)`: greedy `\s+`
//!   takes both spaces, the lookahead then fails because `l` follows, so the matcher
//!   backtracks to one space -- where the next character *is* a space and the
//!   lookahead succeeds. So it means "the whitespace run, minus its last character,
//!   unless the run reaches the end of input".
//!
//! That lookahead is why this is a hand-written scanner rather than a `regex` crate
//! call: the crate does not support lookaround. It is also why the pre-tokenizer
//! cannot be validated by comparing regex strings alone, and is instead validated by
//! comparing token ids on a committed corpus.

use std::collections::HashMap;
use std::path::Path;

use crate::unicode_gc::{is_letter, is_mark, is_number, nfc};

/// The pre-tokenizer pattern used by `tokenizer.json` and by
/// `transformers/models/qwen3_5/tokenization_qwen3_5.py`.
///
/// Alternative 2 is `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`, so combining marks join the
/// letters around them.
pub const PRETOKENIZE_REGEX_WITH_MARKS: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// The pattern used by `transformers/models/qwen2/tokenization_qwen2.py`.
///
/// Identical except that alternatives 2 and 4 use `\p{L}` and `[^\s\p{L}\p{N}]`, with
/// no `\p{M}`: combining marks are **not** letters here, so they fall through to the
/// symbol rule and become their own pieces.
///
/// # Why both exist
///
/// `Qwen/Qwen3.5-0.8B` ships both, and they disagree. Its `tokenizer_config.json` says
/// `"tokenizer_class": "Qwen2Tokenizer"`, and `transformers`' `AutoTokenizer` honours
/// that: it instantiates the Qwen2 class, which **rebuilds the pre-tokenizer from its
/// own hardcoded pattern** and so discards the `\p{M}` in `tokenizer.json`. Loading
/// `tokenizer.json` with the `tokenizers` library directly keeps `\p{M}`.
///
/// The two agree on ASCII, on CJK and on whitespace, and differ only where a combining
/// mark follows a letter. `tokcheck` verifies both against a corpus for each, so the
/// difference is measured rather than assumed.
pub const PRETOKENIZE_REGEX_NO_MARKS: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

// --------------------------------------------------------------------------- //
// byte-level character mapping
// --------------------------------------------------------------------------- //

/// GPT-2's byte-to-character map.
///
/// Printable Latin-1 bytes map to themselves; the remaining 68 map to `U+0100 + n` in
/// increasing byte order. The point is that every byte becomes a *printable, single*
/// character, so the BPE operates on a string with no whitespace or control
/// characters in it -- which is what lets `Ġ` stand for a space inside a merge table.
fn byte_to_char_table() -> [char; 256] {
    let mut printable: Vec<u32> = Vec::with_capacity(256);
    printable.extend(33u32..=126); // '!' .. '~'
    printable.extend(161u32..=172); // '¡' .. '¬'
    printable.extend(174u32..=255); // '®' .. 'ÿ'
    let base = printable.clone();

    let mut map = vec![0u32; 256];
    for &b in &base {
        map[b as usize] = b;
    }
    let mut n = 0u32;
    for b in 0u32..256 {
        if !base.contains(&b) {
            map[b as usize] = 256 + n;
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for b in 0..256 {
        out[b] = char::from_u32(map[b]).expect("mapped codepoint is valid");
    }
    out
}

// --------------------------------------------------------------------------- //
// pre-tokenizer
// --------------------------------------------------------------------------- //

/// The contraction alternative, matched case-insensitively at `c[i]`.
///
/// Only a small, fixed set, so this is a direct comparison rather than a table.
fn match_contraction(c: &[char], i: usize) -> Option<usize> {
    if c[i] != '\'' {
        return None;
    }
    let next = c.get(i + 1)?.to_ascii_lowercase();
    match next {
        's' | 't' | 'm' | 'd' => Some(i + 2),
        'r' => {
            if c.get(i + 2).map(|x| x.to_ascii_lowercase()) == Some('e') {
                Some(i + 3)
            } else {
                None
            }
        }
        'v' => {
            if c.get(i + 2).map(|x| x.to_ascii_lowercase()) == Some('e') {
                Some(i + 3)
            } else {
                None
            }
        }
        'l' => {
            if c.get(i + 2).map(|x| x.to_ascii_lowercase()) == Some('l') {
                Some(i + 3)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`
///
/// Note what the negated class excludes: `\r`, `\n`, letters and numbers -- but **not
/// marks**. A mark can therefore be either the optional prefix or part of the run, and
/// both readings give the same end position, so the ambiguity is harmless.
fn match_letters(c: &[char], i: usize, marks_join: bool) -> Option<usize> {
    let n = c.len();
    let is_lm = |x: char| is_letter(x) || (marks_join && is_mark(x));

    // With the optional prefix: one character that is not \r, \n, letter or number,
    // followed by at least one letter or mark.
    let c0 = c[i];
    // The optional prefix only helps if a letter or mark actually follows it.
    if c0 != '\r' && c0 != '\n' && !is_letter(c0) && !is_number(c0) && i + 1 < n && is_lm(c[i + 1]) {
        let mut j = i + 1;
        while j < n && is_lm(c[j]) {
            j += 1;
        }
        return Some(j);
    }
    // Without it.
    if is_lm(c0) {
        let mut j = i;
        while j < n && is_lm(c[j]) {
            j += 1;
        }
        return Some(j);
    }
    None
}

/// ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
///
/// The leading ` ?` is a literal space, not `\s`, so a tab cannot start this
/// alternative. Since `\r` and `\n` are whitespace they are excluded from the symbol
/// run, which is why the trailing `[\r\n]*` can pick them up without backtracking.
fn match_symbols(c: &[char], i: usize, marks_join: bool) -> Option<usize> {
    let n = c.len();
    // When marks do not join letters they are, by this rule's definition, symbols --
    // which is exactly what makes them their own pieces.
    let is_sym = |x: char| {
        !x.is_whitespace() && !is_letter(x) && !is_number(x) && !(marks_join && is_mark(x))
    };

    let start = if c[i] == ' ' { i + 1 } else { i };
    if start >= n || !is_sym(c[start]) {
        return None;
    }
    let mut j = start;
    while j < n && is_sym(c[j]) {
        j += 1;
    }
    while j < n && (c[j] == '\r' || c[j] == '\n') {
        j += 1;
    }
    Some(j)
}

/// `\s*[\r\n]+`
///
/// Greedy `\s*` wants to consume the whole whitespace run, but `[\r\n]+` then has
/// nothing to match, so the matcher backtracks to the **last** `\r` or `\n` inside the
/// run and takes the newline run from there. `"\n\na"` matches both newlines;
/// `" \n "` matches `" \n"` and leaves the trailing space.
fn match_ws_then_newline(c: &[char], i: usize) -> Option<usize> {
    let n = c.len();
    if !c[i].is_whitespace() {
        return None;
    }
    let mut k = i;
    while k < n && c[k].is_whitespace() {
        k += 1;
    }
    let j = (i..k).rev().find(|&x| c[x] == '\r' || c[x] == '\n')?;
    let mut e = j;
    while e < n && (c[e] == '\r' || c[e] == '\n') {
        e += 1;
    }
    Some(e)
}

/// `\s+(?!\S)`
///
/// The whitespace run, minus its last character -- unless the run reaches the end of
/// input, where the lookahead succeeds immediately and the whole run matches.
fn match_ws_lookahead(c: &[char], i: usize) -> Option<usize> {
    let n = c.len();
    if !c[i].is_whitespace() {
        return None;
    }
    let mut k = i;
    while k < n && c[k].is_whitespace() {
        k += 1;
    }
    if k == n {
        Some(k)
    } else if k >= i + 2 {
        Some(k - 1)
    } else {
        None
    }
}

/// `\s+`
fn match_whitespace(c: &[char], i: usize) -> Option<usize> {
    let n = c.len();
    if !c[i].is_whitespace() {
        return None;
    }
    let mut j = i;
    while j < n && c[j].is_whitespace() {
        j += 1;
    }
    Some(j)
}

/// Try the alternatives in order at char index `i`; return the end index.
///
/// Every character matches at least one alternative: whitespace matches 5, 6 or 7,
/// letters and marks match 2, numbers match 3, and anything else is a symbol and
/// matches 4. So this never fails, and the fallback return is unreachable.
fn match_alternative(c: &[char], i: usize, marks_join: bool) -> usize {
    let n = c.len();
    if let Some(j) = match_contraction(c, i) {
        return j;
    }
    if let Some(j) = match_letters(c, i, marks_join) {
        return j;
    }
    if is_number(c[i]) {
        return i + 1;
    }
    if let Some(j) = match_symbols(c, i, marks_join) {
        return j;
    }
    if let Some(j) = match_ws_then_newline(c, i) {
        return j;
    }
    if let Some(j) = match_ws_lookahead(c, i) {
        return j;
    }
    if let Some(j) = match_whitespace(c, i) {
        return j;
    }
    debug_assert!(i < n);
    i + 1
}

/// Split `text` into pieces, as byte ranges, with combining marks treated as part of
/// the letter run (the `tokenizer.json` behaviour).
pub fn pretokenize(text: &str) -> Vec<(usize, usize)> {
    pretokenize_with(text, true)
}

/// Split `text` into pieces, choosing whether combining marks join letter runs.
///
/// `marks_join = false` reproduces `Qwen2Tokenizer`, which is what
/// `AutoTokenizer` uses for this checkpoint.
pub fn pretokenize_with(text: &str, marks_join: bool) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let only: Vec<char> = chars.iter().map(|&(_, c)| c).collect();
    let n = only.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let j = match_alternative(&only, i, marks_join).min(n).max(i + 1);
        let start = chars[i].0;
        let end = if j < n { chars[j].0 } else { text.len() };
        out.push((start, end));
        i = j;
    }
    out
}

// --------------------------------------------------------------------------- //
// the tokenizer
// --------------------------------------------------------------------------- //

/// A loaded tokenizer.
pub struct Tokenizer {
    /// byte-level character -> id, for the 256 single-byte tokens.
    byte_char_id: [u32; 256],
    /// id -> token string.
    id_to_token: Vec<String>,
    /// `(left_id, right_id)` -> `(rank, merged_id)`, from the merge table.
    merge: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens, longest first so the longest match wins.
    added: Vec<(String, u32)>,
    char_to_byte: HashMap<char, u8>,
    /// Whether alternative 2 accepts `\p{M}`, i.e. whether a combining mark joins the
    /// letters around it. Decided by which of the two shipped patterns the checkpoint
    /// uses; see [`PRETOKENIZE_REGEX_NO_MARKS`].
    marks_join_letters: bool,
}

/// What the loader found, for reporting.
#[derive(Debug, Clone)]
pub struct TokenizerInfo {
    pub model_type: String,
    pub vocab_size: usize,
    pub merges: usize,
    pub added_tokens: usize,
    pub normalizer: String,
    pub pre_tokenizer: String,
    pub decoder: String,
    pub byte_characters_missing_from_vocab: usize,
    /// "with marks" or "without marks", naming which shipped pattern matched.
    pub pattern_variant: &'static str,
}

impl Tokenizer {
    /// Load from a `tokenizer.json`.
    ///
    /// The pipeline is validated rather than assumed: the model type, the normalizer,
    /// the pre-tokenizer's regex and the decoder are each checked against what this
    /// implementation does, because a tokenizer that silently applies a different
    /// pipeline produces the wrong ids and no error.
    pub fn from_file(path: impl AsRef<Path>) -> Result<(Tokenizer, TokenizerInfo), String> {
        let path = path.as_ref();
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let doc: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| format!("{}: {e}", path.display()))?;

        let model = doc
            .get("model")
            .ok_or_else(|| format!("{}: no `model`", path.display()))?;
        let model_type = model
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if model_type != "BPE" {
            return Err(format!(
                "{}: model type is `{model_type}`, this implementation is BPE",
                path.display()
            ));
        }
        if model.get("byte_fallback").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err(format!(
                "{}: byte_fallback is set; this implementation assumes byte-level \
                 characters are in the vocabulary",
                path.display()
            ));
        }
        if let Some(unk) = model.get("unk_token").and_then(|v| v.as_str()) {
            return Err(format!(
                "{}: unk_token is `{unk}`; byte-level BPE should need none",
                path.display()
            ));
        }

        // --- vocabulary ---
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("{}: no `model.vocab`", path.display()))?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0usize;
        for (tok, id) in vocab_obj {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("{}: vocab entry {tok:?} has a non-integer id", path.display()))?
                as usize;
            max_id = max_id.max(id);
            vocab.insert(tok.clone(), id as u32);
        }

        // --- added tokens (they are not in `vocab`) ---
        let mut added: Vec<(String, u32)> = Vec::new();
        if let Some(arr) = doc.get("added_tokens").and_then(|v| v.as_array()) {
            for a in arr {
                let content = a
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("{}: an added token has no content", path.display()))?;
                let id = a
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| format!("{}: added token {content:?} has no id", path.display()))?
                    as usize;
                max_id = max_id.max(id);
                added.push((content.to_string(), id as u32));
            }
        }
        // Longest first, so a scan takes the longest match at each position.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));

        // --- id -> token ---
        let mut id_to_token = vec![String::new(); max_id + 1];
        let mut have = vec![false; max_id + 1];
        for (tok, &id) in &vocab {
            id_to_token[id as usize] = tok.clone();
            have[id as usize] = true;
        }
        for (tok, id) in &added {
            if !have[*id as usize] {
                id_to_token[*id as usize] = tok.clone();
                have[*id as usize] = true;
            }
        }

        // --- byte-level tables, and the check that makes them safe ---
        let byte_to_char = byte_to_char_table();
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &ch) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(ch, b as u8);
        }
        let mut byte_char_id = [0u32; 256];
        let mut missing = 0usize;
        for b in 0..256 {
            let s = byte_to_char[b].to_string();
            match vocab.get(&s) {
                Some(&id) => byte_char_id[b] = id,
                None => missing += 1,
            }
        }
        // A byte-level tokenizer is only total if all 256 single-byte tokens exist.
        // Without this check an unseen byte would silently become id 0.
        if missing > 0 {
            return Err(format!(
                "{}: {missing} of the 256 byte-level characters are absent from the \
                 vocabulary; a byte-level BPE must contain all of them",
                path.display()
            ));
        }

        // --- merges ---
        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{}: no `model.merges`", path.display()))?;
        let mut merge: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(merges_arr.len());
        for (rank, m) in merges_arr.iter().enumerate() {
            // Newer `tokenizers` writes `["a", "b"]`; older writes the string `"a b"`.
            // Both are accepted, and which one was seen is not otherwise interesting.
            let (left, right) = if let Some(s) = m.as_str() {
                match s.split_once(' ') {
                    Some((a, b)) => (a.to_string(), b.to_string()),
                    None => {
                        return Err(format!(
                            "{}: merge {rank} is {s:?}, which has no space",
                            path.display()
                        ))
                    }
                }
            } else if let Some(pair) = m.as_array() {
                if pair.len() != 2 {
                    return Err(format!("{}: merge {rank} is not a pair", path.display()));
                }
                (
                    pair[0].as_str().unwrap_or("").to_string(),
                    pair[1].as_str().unwrap_or("").to_string(),
                )
            } else {
                return Err(format!("{}: merge {rank} has an unknown shape", path.display()));
            };
            let li = *vocab
                .get(&left)
                .ok_or_else(|| format!("{}: merge {rank}: {left:?} is not in the vocab", path.display()))?;
            let ri = *vocab
                .get(&right)
                .ok_or_else(|| format!("{}: merge {rank}: {right:?} is not in the vocab", path.display()))?;
            let merged = *vocab
                .get(&format!("{left}{right}"))
                .ok_or_else(|| {
                    format!("{}: merge {rank}: {left:?}+{right:?} has no vocabulary entry", path.display())
                })?;
            merge.insert((li, ri), (rank as u32, merged));
        }

        // --- pipeline validation ---
        let normalizer = doc
            .get("normalizer")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let ntype = doc
            .get("normalizer")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if ntype != "NFC" {
            return Err(format!(
                "{}: normalizer is `{ntype}`, this implementation performs NFC only",
                path.display()
            ));
        }

        let pre = doc
            .get("pre_tokenizer")
            .ok_or_else(|| format!("{}: no `pre_tokenizer`", path.display()))?;
        let subs = pre
            .get("pretokenizers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{}: `pre_tokenizer` is not a Sequence", path.display()))?;
        if subs.len() != 2 {
            return Err(format!(
                "{}: expected 2 pre-tokenizers (Split, ByteLevel), found {}",
                path.display(),
                subs.len()
            ));
        }
        let regex = subs[0]
            .get("pattern")
            .and_then(|p| p.get("Regex"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: the first pre-tokenizer is not a Regex Split", path.display()))?;
        // Either shipped variant is accepted; anything else is refused, because every
        // alternative is load-bearing and a near-miss would tokenize silently wrong.
        let (marks_join_letters, pattern_variant) = if regex == PRETOKENIZE_REGEX_WITH_MARKS {
            (true, "with marks")
        } else if regex == PRETOKENIZE_REGEX_NO_MARKS {
            (false, "without marks")
        } else {
            return Err(format!(
                "{}: the pre-tokenizer regex matches neither pattern this scanner \
                 implements.\n  checkpoint: {regex}\n  with marks:    {PRETOKENIZE_REGEX_WITH_MARKS}\n  \
                 without marks: {PRETOKENIZE_REGEX_NO_MARKS}",
                path.display()
            ));
        };
        let btype = subs[1].get("type").and_then(|v| v.as_str()).unwrap_or("");
        if btype != "ByteLevel" {
            return Err(format!(
                "{}: the second pre-tokenizer is `{btype}`, expected ByteLevel",
                path.display()
            ));
        }
        let dtype = doc
            .get("decoder")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if dtype != "ByteLevel" {
            return Err(format!(
                "{}: decoder is `{dtype}`, expected ByteLevel",
                path.display()
            ));
        }

        let info = TokenizerInfo {
            model_type,
            vocab_size: vocab.len(),
            merges: merge.len(),
            added_tokens: added.len(),
            normalizer,
            pre_tokenizer: regex.to_string(),
            decoder: dtype.to_string(),
            byte_characters_missing_from_vocab: missing,
            pattern_variant,
        };
        Ok((
            Tokenizer {
                byte_char_id,
                id_to_token,
                merge,
                added,
                char_to_byte,
                marks_join_letters,
            },
            info,
        ))
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// The vocabulary string for an id, before byte-level decoding.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// BPE over one pre-tokenized piece.
    ///
    /// The correct rule is: repeatedly take the adjacent pair with the **lowest merge
    /// rank**, merge those two symbols, and repeat -- one merge at a time, leftmost
    /// wins a tie. Merging every occurrence of the best pair in a single pass looks
    /// equivalent and is not, so the one-at-a-time form is used.
    fn bpe(&self, piece: &str) -> Vec<u32> {
        let mut syms: Vec<u32> = Vec::with_capacity(piece.len());
        for &b in piece.as_bytes() {
            syms.push(self.byte_char_id[b as usize]);
        }
        if syms.len() < 2 {
            return syms;
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                if let Some(&(rank, _)) = self.merge.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(br, _)| rank < br) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = self.merge[&(syms[i], syms[i + 1])].1;
            syms[i] = merged;
            syms.remove(i + 1);
        }
        syms
    }

    /// Encode, without any special tokens added (the checkpoint sets
    /// `add_bos_token` false, and there is no BOS token at all).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = nfc(text);
        let mut out = Vec::new();

        // Added tokens are matched on the normalized text and never reach the
        // pre-tokenizer or the BPE. Longest match wins.
        let mut i = 0usize;
        let mut plain_start = 0usize;
        while i < normalized.len() {
            if !normalized.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let hit = self
                .added
                .iter()
                .find(|(tok, _)| normalized[i..].starts_with(tok.as_str()));
            match hit {
                Some((tok, id)) => {
                    if plain_start < i {
                        self.encode_plain(&normalized[plain_start..i], &mut out);
                    }
                    out.push(*id);
                    i += tok.len();
                    plain_start = i;
                }
                None => {
                    // Advance one character.
                    i += normalized[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                }
            }
        }
        if plain_start < normalized.len() {
            self.encode_plain(&normalized[plain_start..], &mut out);
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for (a, b) in pretokenize_with(text, self.marks_join_letters) {
            out.extend(self.bpe(&text[a..b]));
        }
    }

    /// Whether combining marks join letter runs.
    pub fn marks_join_letters(&self) -> bool {
        self.marks_join_letters
    }

    /// Override the mark handling.
    ///
    /// Exists because one checkpoint ships both patterns and the choice is made
    /// outside `tokenizer.json`: `AutoTokenizer` reads `tokenizer_class` from
    /// `tokenizer_config.json` and lets that class rebuild the pre-tokenizer. A caller
    /// that knows which one it is emulating can say so; see
    /// [`Tokenizer::from_model_dir`], which reads the config and does it automatically.
    pub fn set_marks_join_letters(&mut self, v: bool) {
        self.marks_join_letters = v;
    }

    /// Load from a model directory, honouring `tokenizer_config.json` the way
    /// `AutoTokenizer` does.
    ///
    /// `tokenizer.json`'s own pattern uses `\p{M}`, but a checkpoint whose
    /// `tokenizer_class` is a Qwen2 variant is normally loaded through the class, which
    /// rebuilds the pre-tokenizer without `\p{M}`. Loading `tokenizer.json` directly
    /// and loading the model through `AutoTokenizer` therefore give different ids for
    /// text containing combining marks, and this picks the `AutoTokenizer` behaviour
    /// because that is what the model is served with.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<(Tokenizer, TokenizerInfo), String> {
        let dir = dir.as_ref();
        let (mut tk, info) = Tokenizer::from_file(dir.join("tokenizer.json"))?;

        let cfg_path = dir.join("tokenizer_config.json");
        let class = if cfg_path.exists() {
            let raw = std::fs::read(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
            let doc: serde_json::Value = serde_json::from_slice(&raw)
                .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
            doc.get("tokenizer_class")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };

        // A Qwen2 tokenizer class carries its own pattern, without `\p{M}`, and that
        // pattern is what actually runs. Anything else keeps the file's pattern.
        if class.starts_with("Qwen2") {
            tk.set_marks_join_letters(false);
        }
        Ok((tk, info))
    }

    /// Decode ids back to text.
    ///
    /// Errors are replaced rather than raised: a token may hold only part of a
    /// multi-byte character, so any prefix of a sequence is decodable and the result
    /// is the concatenation of the bytes actually present.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.token_str(id) else { continue };
            for ch in tok.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    // An added token is stored as plain text, not byte-level encoded.
                    bytes.extend_from_slice(ch.to_string().as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_is_a_bijection_over_256_values() {
        let t = byte_to_char_table();
        let mut seen = std::collections::HashSet::new();
        for c in t {
            assert!(seen.insert(c), "duplicate {c:?}");
            assert!(!c.is_whitespace(), "{c:?} is whitespace; the table must be printable");
            assert!(!c.is_control(), "{c:?} is a control character");
        }
        assert_eq!(seen.len(), 256);
        // Printable bytes map to themselves.
        assert_eq!(t[0x20], 'Ġ');
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[0x0A], 'Ċ');
        assert_eq!(t[0x09], 'ĉ');
    }

    // The pre-tokenizer tests use the byte-level spellings, because that is what the
    // pieces turn into: `Ġ` for a space, `Ċ` for a newline.

    fn pieces(s: &str) -> Vec<String> {
        let s = nfc(s);
        pretokenize(&s).into_iter().map(|(a, b)| s[a..b].to_string()).collect()
    }

    #[test]
    fn a_space_joins_the_following_word() {
        // Alternative 2's optional prefix takes the space, so it is not a piece of
        // its own. A separate "Ġ" would produce a different token sequence.
        assert_eq!(pieces("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pieces("world"), vec!["world"]);
    }

    #[test]
    fn the_optional_prefix_takes_exactly_one_character() {
        // "+b" is one piece; "++" is not a letter prefix and falls to alternative 4.
        assert_eq!(pieces("a+b"), vec!["a", "+b"]);
        assert_eq!(pieces("a++b"), vec!["a", "++", "b"]);
    }

    #[test]
    fn whitespace_run_keeps_its_last_character() {
        // The `(?!\S)` backtracking case.
        assert_eq!(pieces("  leading"), vec![" ", " leading"]);
        assert_eq!(pieces("a  b"), vec!["a", " ", " b"]);
        // At end of input the lookahead succeeds, so the whole run is one piece.
        assert_eq!(pieces("leading  "), vec!["leading", "  "]);
        assert_eq!(pieces(" "), vec![" "]);
    }

    #[test]
    fn newline_runs_are_their_own_piece() {
        assert_eq!(pieces("\n\na"), vec!["\n\n", "a"]);
        assert_eq!(pieces("a\r\n\r\nb"), vec!["a", "\r\n\r\n", "b"]);
        // Whitespace before a newline is absorbed by alternative 5.
        assert_eq!(pieces(" \n "), vec![" \n", " "]);
    }

    #[test]
    fn contractions_are_split_off_and_case_insensitive() {
        assert_eq!(pieces("don't"), vec!["don", "'t"]);
        assert_eq!(pieces("DON'T"), vec!["DON", "'T"]);
        assert_eq!(pieces("x's"), vec!["x", "'s"]);
        assert_eq!(pieces("they're"), vec!["they", "'re"]);
        // "'n" is not a contraction, so alternative 2 handles it as prefix+letters.
        assert_eq!(pieces("rock'n'roll"), vec!["rock", "'n", "'roll"]);
    }

    #[test]
    fn digits_are_single_character_pieces() {
        assert_eq!(pieces("abc123"), vec!["abc", "1", "2", "3"]);
        assert_eq!(pieces("123abc"), vec!["1", "2", "3", "abc"]);
        assert_eq!(pieces("3.14159"), vec!["3", ".", "1", "4", "1", "5", "9"]);
    }

    #[test]
    fn symbols_group_but_letters_and_digits_bound_them() {
        assert_eq!(pieces("((a))"), vec!["((", "a", "))"]);
        assert_eq!(pieces(" - item"), vec![" -", " item"]);
        assert_eq!(pieces("  -  item"), vec![" ", " -", " ", " item"]);
    }

    #[test]
    fn non_latin_letters_stay_together() {
        // `，` is punctuation, so alternative 2 may take it as its optional prefix and
        // then absorb the letters after it: the piece is `，世界`, not `，` + `世界`.
        //
        // This is worth being explicit about, because it shows a piece boundary is not
        // a token boundary. The reference produces three tokens here, not two, and it
        // does so because the merge table has no rule joining the bytes of `，` to the
        // bytes of `世` -- not because the pre-tokenizer separated them.
        assert_eq!(pieces("你好，世界"), vec!["你好", "，世界"]);
        // Cyrillic behaves like Latin: the space joins the following word, so this is
        // two pieces, not three.
        assert_eq!(pieces("Привет мир"), vec!["Привет", " мир"]);
        // A combining mark is \p{M}, which alternative 2 accepts in its run. NFC has
        // already composed it by this point, so the piece is the single char `à`.
        assert_eq!(pieces("a\u{300}b"), vec!["àb"]);
    }

    #[test]
    fn every_character_matches_some_alternative() {
        // The scanner's fallback is documented as unreachable; check that nothing in a
        // broad sample produces an empty or non-advancing piece.
        let s: String = (0u32..0x3000).filter_map(char::from_u32).collect();
        let p = pretokenize(&s);
        let total: usize = p.iter().map(|(a, b)| b - a).sum();
        assert_eq!(total, s.len(), "pieces do not cover the input");
        for (a, b) in p {
            assert!(b > a, "empty piece at {a}");
        }
    }
}
