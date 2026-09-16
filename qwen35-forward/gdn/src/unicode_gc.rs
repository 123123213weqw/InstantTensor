//! General-category lookups and NFC, backed by `unicode_tables`.
//!
//! # Why not the standard library
//!
//! The Qwen2 pre-tokenizer splits on `\p{L}`, `\p{M}` and `\p{N}`. Rust offers
//! `char::is_alphabetic()`, but that is the **Alphabetic property**, not `\p{L}`: it
//! also covers `Nl` (Roman numerals such as `Ⅻ`) and part of `Mn`. `char::is_numeric()`
//! is closer to `\p{N}` but still not defined as the general category. And there is no
//! standard API at all for `\p{M}`.
//!
//! So the categories come from generated tables. That also makes the Unicode version
//! explicit: `unicode_tables::UNICODE_VERSION` names the database the tables came
//! from, and a codepoint assigned later than that is classified as "other" here.

use crate::unicode_tables as t;

/// Binary search over sorted, disjoint, inclusive ranges.
fn in_ranges(rs: &[(u32, u32)], cp: u32) -> bool {
    let (mut lo, mut hi) = (0usize, rs.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (a, b) = rs[mid];
        if cp < a {
            hi = mid;
        } else if cp > b {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// `\p{L}`
#[inline]
pub fn is_letter(c: char) -> bool {
    in_ranges(&t::LETTER_RANGES, c as u32)
}

/// `\p{M}`
#[inline]
pub fn is_mark(c: char) -> bool {
    in_ranges(&t::MARK_RANGES, c as u32)
}

/// `\p{N}`
#[inline]
pub fn is_number(c: char) -> bool {
    in_ranges(&t::NUMBER_RANGES, c as u32)
}

/// Canonical combining class; 0 for a starter.
#[inline]
fn ccc(c: char) -> u32 {
    let cp = c as u32;
    match t::CCC.binary_search_by(|&(k, _)| k.cmp(&cp)) {
        Ok(i) => t::CCC[i].1,
        Err(_) => 0,
    }
}

// Hangul constants for the algorithmic decomposition (UAX #15 section 10).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT; // 588
const S_COUNT: u32 = 19 * N_COUNT; // 11172

/// Full canonical decomposition of one char, appended to `out`.
fn decompose_one(c: char, out: &mut Vec<char>) {
    let cp = c as u32;

    // Hangul syllables decompose arithmetically. They are 11172 of the 13233
    // decomposable codepoints, so keeping them out of the table is most of the
    // table's size.
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let si = cp - S_BASE;
        out.push(char::from_u32(L_BASE + si / N_COUNT).unwrap());
        out.push(char::from_u32(V_BASE + (si % N_COUNT) / T_COUNT).unwrap());
        let tj = si % T_COUNT;
        if tj != 0 {
            out.push(char::from_u32(T_BASE + tj).unwrap());
        }
        return;
    }

    if let Ok(i) = t::DECOMPOSE_1.binary_search_by(|&(k, _)| k.cmp(&cp)) {
        let to = char::from_u32(t::DECOMPOSE_1[i].1).unwrap();
        decompose_one(to, out);
        return;
    }
    if let Ok(i) = t::DECOMPOSE_2.binary_search_by(|&(k, _, _)| k.cmp(&cp)) {
        let (_, a, b) = t::DECOMPOSE_2[i];
        decompose_one(char::from_u32(a).unwrap(), out);
        decompose_one(char::from_u32(b).unwrap(), out);
        return;
    }
    out.push(c);
}

/// Canonical composition of a starter and a following character, if one exists.
///
/// Hangul is handled arithmetically **before** the table, and that ordering is not
/// cosmetic. The generator deliberately leaves Hangul out of the decomposition tables,
/// because 11172 of the 13233 decomposable codepoints are Hangul and their
/// decomposition is a formula. But the composition table is *derived from* the
/// decomposition table, so leaving Hangul out of one silently left it out of the other:
/// syllables decomposed and then failed to recompose. A unit test that round-trips
/// three jamo caught it.
#[inline]
fn compose(a: char, b: char) -> Option<char> {
    let (au, bu) = (a as u32, b as u32);

    // L + V -> LV syllable
    if (L_BASE..L_BASE + 19).contains(&au) && (V_BASE..V_BASE + V_COUNT).contains(&bu) {
        let l = au - L_BASE;
        let v = bu - V_BASE;
        return char::from_u32(S_BASE + (l * V_COUNT + v) * T_COUNT);
    }
    // LV + T -> LVT syllable
    if (S_BASE..S_BASE + S_COUNT).contains(&au)
        && (au - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&bu)
    {
        let tj = bu - T_BASE;
        return char::from_u32(au + tj);
    }

    let key = ((a as u64) << 21) | (b as u64);
    match t::COMPOSE.binary_search_by(|&(k, _)| k.cmp(&key)) {
        Ok(i) => char::from_u32(t::COMPOSE[i].1),
        Err(_) => None,
    }
}

/// Canonical composition, then full canonical decomposition, then canonical
/// ordering -- the three steps of UAX #15 `toNFC`.
///
/// The tokenizer applies this to its input before anything else, so
/// `decode(encode(x))` reproduces `nfc(x)` and **not** necessarily `x`. That is a
/// real property of the reference, not an artefact here: `"e\u{301}"` encodes to the
/// same single token as `"é"`.
pub fn nfc(s: &str) -> String {
    // Step 1: full canonical decomposition.
    let mut d: Vec<char> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_one(c, &mut d);
    }
    if d.is_empty() {
        return String::new();
    }

    // Step 2: canonical ordering. A stable insertion sort over runs of non-zero
    // combining class; starters (ccc 0) are never moved, which is what makes the
    // sort stable and correct.
    for i in 1..d.len() {
        let cc = ccc(d[i]);
        if cc == 0 {
            continue;
        }
        let mut j = i;
        while j > 0 {
            let prev = ccc(d[j - 1]);
            if prev == 0 || prev <= cc {
                break;
            }
            d.swap(j - 1, j);
            j -= 1;
        }
    }

    // Step 3: canonical composition. `last_class` starts at 256 when the first
    // character is itself a combining mark, so it can never compose with anything;
    // that matches the reference algorithm, which needs a starter to compose onto.
    let mut out: Vec<char> = Vec::with_capacity(d.len());
    let mut starter_pos = 0usize;
    let mut starter_ch = d[0];
    let mut last_class = if ccc(starter_ch) != 0 { 256 } else { 0 };
    out.push(starter_ch);

    for &ch in &d[1..] {
        let ch_class = ccc(ch);
        // A character is blocked from its starter when something with an equal or
        // higher combining class sits between them, which is what `last_class` tracks.
        if last_class < ch_class || last_class == 0 {
            if let Some(comp) = compose(starter_ch, ch) {
                starter_ch = comp;
                out[starter_pos] = comp;
                continue;
            }
        }
        if ch_class == 0 {
            starter_pos = out.len();
            starter_ch = ch;
        }
        last_class = ch_class;
        out.push(ch);
    }

    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_are_not_the_alphabetic_property() {
        // The distinction that forced these tables to exist.
        assert!(is_letter('a'));
        assert!(is_letter('中'));
        assert!(is_letter('Ω'));
        // U+216B ROMAN NUMERAL TWELVE is `Nl`: Alphabetic in Rust, but not `\p{L}`.
        assert!(!is_letter('\u{216B}'), "Nl must not count as a letter");
        // Written without braces: `assert!` treats its message as a format string,
        // so a bare `{N}` would be parsed as a named argument.
        assert!(is_number('\u{216B}'), "Nl is still in the number category");
        assert!('\u{216B}'.is_alphabetic(), "sanity: Rust calls it alphabetic");
        // Marks are their own category.
        assert!(is_mark('\u{301}'));
        assert!(!is_letter('\u{301}'));
        // No (Number, other): ½
        assert!(is_number('½'));
        assert!(!is_letter('½'));
        // Spaces are none of the three.
        assert!(!is_letter(' ') && !is_mark(' ') && !is_number(' '));
    }

    #[test]
    fn nfc_composes_simple_pairs() {
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("a\u{300}"), "à");
        assert_eq!(nfc("cafe\u{301}"), "café");
        // Already composed input is unchanged.
        assert_eq!(nfc("é"), "é");
        // ASCII and non-composing text pass through.
        assert_eq!(nfc("hello"), "hello");
        assert_eq!(nfc("你好"), "你好");
    }

    #[test]
    fn nfc_orders_combining_marks_by_class() {
        // U+0323 (ccc 220, below) then U+0301 (ccc 230, above) is already in order.
        let ordered = "a\u{323}\u{301}";
        assert_eq!(nfc(ordered), "ạ\u{301}");
        // Reversed input must be reordered before composing, so the *same* result
        // comes out. Without canonical ordering this yields ạ + combining acute.
        let reversed = "a\u{301}\u{323}";
        assert_eq!(nfc(reversed), nfc(ordered));
    }

    #[test]
    fn hangul_decomposes_and_recomposes() {
        // U+D55C HANGUL SYLLABLE HAN = U+1112 + U+1161 + U+11AB
        let d = nfc("\u{1112}\u{1161}\u{11AB}");
        assert_eq!(d, "한");
        // And a syllable stays itself.
        assert_eq!(nfc("한"), "한");
        // A two-jamo syllable with no trailing consonant.
        assert_eq!(nfc("\u{1100}\u{1161}"), "가");
    }

    /// Iterate the generated tables rather than a codepoint range: sampling a range
    /// found only 71 decomposable codepoints and would have missed almost everything.
    #[test]
    fn every_decomposable_codepoint_is_idempotent_and_hangul_round_trips() {
        let mut n = 0usize;
        let check = |c: char| {
            let once = nfc(&c.to_string());
            assert_eq!(nfc(&once), once, "U+{:04X} is not idempotent", c as u32);
        };
        for &(cp, _) in t::DECOMPOSE_1.iter() {
            check(char::from_u32(cp).unwrap());
            n += 1;
        }
        for &(cp, _, _) in t::DECOMPOSE_2.iter() {
            check(char::from_u32(cp).unwrap());
            n += 1;
        }
        // Hangul is not in the tables, and must survive NFC exactly.
        for cp in S_BASE..S_BASE + S_COUNT {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(
                nfc(&c.to_string()),
                c.to_string(),
                "Hangul U+{cp:04X} did not survive NFC"
            );
            n += 1;
        }
        assert!(n > 13000, "only {n} codepoints checked; the tables look short");
    }

    /// Characters with a canonical decomposition that are *excluded* from
    /// recomposition. `toNFC` must leave them decomposed.
    ///
    /// The generator derives the composition table with the rule "NFC of the
    /// decomposition returns the character", which folds in the exclusion list without
    /// needing to read it. These are its witnesses: if that rule were dropped, the
    /// composition table would gain entries for all four and this test would fail.
    #[test]
    fn composition_exclusions_stay_decomposed() {
        for cp in [0x0958u32, 0x09DC, 0x2ADC, 0x0344] {
            let c = char::from_u32(cp).unwrap();
            let got = nfc(&c.to_string());
            assert_ne!(got, c.to_string(), "U+{cp:04X} must not recompose");
            assert!(got.chars().count() > 1, "U+{cp:04X} must decompose");
            assert_eq!(nfc(&got), got, "U+{cp:04X} is not a fixed point");
        }
    }

    #[test]
    fn hangul_jamo_composes_algorithmically() {
        assert_eq!(nfc("\u{1112}\u{1161}\u{11AB}"), "한");
        assert_eq!(nfc("\u{1100}\u{1161}"), "가");
        assert_eq!(nfc("\u{B098}"), "나");
        // A lone trailing jamo has no starter to compose onto and stays.
        assert_eq!(nfc("\u{11AB}"), "\u{11AB}");
    }

    #[test]
    fn nfc_is_idempotent() {
        for s in ["e\u{301}", "a\u{301}\u{323}", "\u{1112}\u{1161}\u{11AB}", "café", "x"] {
            let once = nfc(s);
            assert_eq!(nfc(&once), once, "not idempotent for {s:?}");
        }
    }

    #[test]
    fn nfc_handles_a_leading_combining_mark() {
        // A mark with no starter cannot compose, and must survive.
        let s = "\u{301}a";
        assert_eq!(nfc(s), s);
    }
}
