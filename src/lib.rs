// abr-u-tokenizer  lib.rs
// V0.1.0 — U tokenizer for Phi-3
// Origin: Robin Macomber / Metatron Dynamics
// OOV handling: Option A (character-by-character, no information loss)
//
// TOKEN DOMAIN: D_U = 5,844 U units derived from strict local minima of Q(S)
// across 317,070 bounded structures in the abr-language-analysis corpus.
// Every token boundary has declared provenance from character participation
// patterns — NOT statistical compression.
//
// OOV RULE (Origin-declared):
//   If a character sequence has no P2 data and thus no Q(S) values,
//   emit each character as its own token (character-level fallback).
//   No information is lost.
//
// SPECIAL TOKEN IDs:
//   0 .. 5843  — U unit tokens (indexed from vocab file)
//   5844       — [CHAR] prefix marker for OOV character tokens
//   5845+      — individual ASCII/UTF-8 code points mapped to token IDs
//               via: 5845 + (codepoint as u32)
//
// This scheme keeps U unit IDs and character fallback IDs disjoint
// with no UNK collision.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// VOCABULARY
// ─────────────────────────────────────────────────────────────────────────────

/// The complete declared vocabulary derived from abr-language-analysis.
#[derive(Debug, Clone)]
pub struct UVocabulary {
    /// U unit string → token ID (0..5843)
    pub unit_to_id: HashMap<String, u32>,
    /// token ID → U unit string
    pub id_to_unit: HashMap<u32, String>,
    /// Number of U unit tokens (5,844)
    pub u_unit_count: u32,
}

impl UVocabulary {
    /// Build vocabulary from the contents of u_units_index.txt.
    ///
    /// Expected format (tab-separated):
    ///   <rank>\t<unit_string>\t<count>\t<source_structures>
    /// OR simpler:
    ///   <unit_string>\t<count>
    ///
    /// This parser handles both. The integer ID is assigned by sorted
    /// index position, not by the rank column, so the mapping is
    /// deterministic regardless of file ordering.
    pub fn from_index_file(contents: &str) -> Result<Self, String> {
        let mut units: Vec<(String, u64)> = Vec::new();

        for (line_num, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            // Accept layouts with 2+ columns; unit string is always
            // the first non-numeric column.
            let (unit, count) = match parts.len() {
                0 | 1 => {
                    return Err(format!("line {}: too few columns", line_num + 1));
                }
                2 => {
                    // unit \t count
                    let count = parts[1].parse::<u64>().unwrap_or(1);
                    (parts[0].to_string(), count)
                }
                _ => {
                    // rank \t unit \t count \t ...
                    // Detect: if parts[0] parses as integer, skip it
                    if parts[0].parse::<u64>().is_ok() {
                        let count = parts[2].parse::<u64>().unwrap_or(1);
                        (parts[1].to_string(), count)
                    } else {
                        let count = parts[1].parse::<u64>().unwrap_or(1);
                        (parts[0].to_string(), count)
                    }
                }
            };
            units.push((unit, count));
        }

        if units.is_empty() {
            return Err("No U units found in index file".to_string());
        }

        // Sort lexicographically so ID assignment is deterministic
        units.sort_by(|a, b| a.0.cmp(&b.0));

        let mut unit_to_id = HashMap::new();
        let mut id_to_unit = HashMap::new();

        for (id, (unit, _count)) in units.iter().enumerate() {
            let id = id as u32;
            unit_to_id.insert(unit.clone(), id);
            id_to_unit.insert(id, unit.clone());
        }

        let u_unit_count = units.len() as u32;

        Ok(UVocabulary {
            unit_to_id,
            id_to_unit,
            u_unit_count,
        })
    }

    /// Decode a token ID back to its string representation.
    pub fn decode_id(&self, id: u32) -> String {
        if id < self.u_unit_count {
            self.id_to_unit
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("[UNK:{}]", id))
        } else if id == self.u_unit_count {
            "[CHAR]".to_string()
        } else {
            // OOV character: id = u_unit_count + 1 + codepoint
            let cp = id - self.u_unit_count - 1;
            char::from_u32(cp)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("[CP:{}]", cp))
        }
    }

    /// Encode a single character as an OOV token ID (Option A).
    pub fn char_to_oov_id(&self, c: char) -> u32 {
        self.u_unit_count + 1 + (c as u32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Q(S) COMPUTATION
// ─────────────────────────────────────────────────────────────────────────────

/// Participation frequencies for a character position.
/// P1 = frequency of this char as position 1 in any bounded structure.
/// P2 = frequency of this char as position 2.
/// (Extended to P1..P6 in the full corpus; we use P1+P2 for boundary signal.)
#[derive(Debug, Clone)]
pub struct ParticipationProfile {
    pub p1: f64,
    pub p2: f64,
}

/// Q(S) participation signal for a bounded structure S = c_0 c_1 ... c_n.
///
/// Per the declared formula from abr-language-analysis:
///   Q(S)[i] = P1[c_i] + P2[c_{i+1}]   for i in 0..len-1
///
/// Local minima of Q(S) at interior positions (1..len-2) are U boundaries.
pub fn compute_q(
    chars: &[char],
    p1: &HashMap<char, f64>,
    p2: &HashMap<char, f64>,
) -> Vec<f64> {
    let n = chars.len();
    if n < 2 {
        return vec![];
    }
    let mut q = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let p1_i = p1.get(&chars[i]).copied().unwrap_or(0.0);
        let p2_next = p2.get(&chars[i + 1]).copied().unwrap_or(0.0);
        q.push(p1_i + p2_next);
    }
    q
}

/// Find interior local minima of Q(S).
/// Position i (1-indexed in Q) is a boundary iff:
///   Q[i] < Q[i-1]  AND  Q[i] < Q[i+1]
/// Returns the set of boundary indices in the original char array
/// (position after which to split).
pub fn find_u_boundaries(q: &[f64]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    if q.len() < 3 {
        return boundaries;
    }
    for i in 1..q.len() - 1 {
        if q[i] < q[i - 1] && q[i] < q[i + 1] {
            // boundary falls between char[i] and char[i+1]
            boundaries.push(i + 1); // split position in chars
        }
    }
    boundaries
}

// ─────────────────────────────────────────────────────────────────────────────
// TOKENIZER
// ─────────────────────────────────────────────────────────────────────────────

/// The declared U tokenizer.
/// Holds vocabulary + participation tables derived from the corpus.
pub struct UTokenizer {
    pub vocab: UVocabulary,
    /// P1 table: char → frequency as position-1 in bounded structures
    pub p1: HashMap<char, f64>,
    /// P2 table: char → frequency as position-2 in bounded structures
    pub p2: HashMap<char, f64>,
}

impl UTokenizer {
    pub fn new(
        vocab: UVocabulary,
        p1: HashMap<char, f64>,
        p2: HashMap<char, f64>,
    ) -> Self {
        UTokenizer { vocab, p1, p2 }
    }

    /// Tokenize a raw text string.
    ///
    /// Process:
    ///   1. Split input into whitespace-delimited words (bounded structures).
    ///   2. For each word:
    ///      a. Compute Q(S) over its characters.
    ///      b. Find interior local minima → U boundaries.
    ///      c. Partition at boundaries → candidate U units.
    ///      d. For each candidate:
    ///         - If found in vocab → emit its ID.
    ///         - Else (OOV, Option A) → emit each character as its own token.
    ///   3. Return flat Vec<u32> of token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();

            // Compute Q(S) and find boundaries
            let q = compute_q(&chars, &self.p1, &self.p2);
            let boundaries = find_u_boundaries(&q);

            // Partition chars at boundaries into segments
            let segments = partition_at_boundaries(&chars, &boundaries);

            for seg in &segments {
                let seg_str: String = seg.iter().collect();
                if let Some(&id) = self.vocab.unit_to_id.get(&seg_str) {
                    ids.push(id);
                } else {
                    // OOV Option A: emit character by character
                    for &c in seg.iter() {
                        ids.push(self.vocab.char_to_oov_id(c));
                    }
                }
            }
        }

        ids
    }

    /// Decode a sequence of token IDs back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.vocab.decode_id(id))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Tokenize and return the string segments (before ID lookup).
    /// Useful for inspection and Verifier review.
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let q = compute_q(&chars, &self.p1, &self.p2);
            let boundaries = find_u_boundaries(&q);
            let segs = partition_at_boundaries(&chars, &boundaries);
            for seg in segs {
                let s: String = seg.iter().collect();
                if self.vocab.unit_to_id.contains_key(&s) {
                    segments.push(s);
                } else {
                    // OOV: emit each character with [OOV] marker
                    for c in seg {
                        segments.push(format!("[OOV:{}]", c));
                    }
                }
            }
        }

        segments
    }
}

/// Partition a char slice at declared boundary positions.
fn partition_at_boundaries(chars: &[char], boundaries: &[usize]) -> Vec<Vec<char>> {
    let mut segments = Vec::new();
    let mut start = 0;

    for &b in boundaries {
        if b > start && b <= chars.len() {
            segments.push(chars[start..b].to_vec());
            start = b;
        }
    }
    // Final segment
    if start < chars.len() {
        segments.push(chars[start..].to_vec());
    }

    if segments.is_empty() && !chars.is_empty() {
        segments.push(chars.to_vec());
    }

    segments
}

// ─────────────────────────────────────────────────────────────────────────────
// PARTICIPATION TABLE LOADER
// ─────────────────────────────────────────────────────────────────────────────

/// Load P1 or P2 participation table from a two-column TSV:
///   <char>\t<frequency>
pub fn load_participation_table(contents: &str) -> HashMap<char, f64> {
    let mut table = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() == 2 {
            if let (Some(c), Ok(freq)) = (
                parts[0].chars().next(),
                parts[1].parse::<f64>(),
            ) {
                table.insert(c, freq);
            }
        }
    }
    table
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_vocab() -> UVocabulary {
        let contents = "ab\t10\ncd\t5\nef\t3\na\t20\nb\t15\nc\t8\nd\t4\ne\t6\nf\t2\n";
        UVocabulary::from_index_file(contents).unwrap()
    }

    fn mock_p_tables() -> (HashMap<char, f64>, HashMap<char, f64>) {
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        // Give 'a' high P1 and 'b' low P2 → Q drops at ab boundary
        p1.insert('a', 0.9);
        p1.insert('b', 0.1);
        p1.insert('c', 0.5);
        p1.insert('d', 0.2);
        p2.insert('a', 0.1);
        p2.insert('b', 0.05);
        p2.insert('c', 0.3);
        p2.insert('d', 0.8);
        (p1, p2)
    }

    #[test]
    fn test_vocab_load_roundtrip() {
        let vocab = mock_vocab();
        assert!(vocab.u_unit_count > 0);
        for (unit, &id) in &vocab.unit_to_id {
            assert_eq!(vocab.id_to_unit[&id], *unit);
        }
    }

    #[test]
    fn test_oov_id_disjoint_from_u_ids() {
        let vocab = mock_vocab();
        let oov_id = vocab.char_to_oov_id('z');
        assert!(oov_id >= vocab.u_unit_count + 1);
    }

    #[test]
    fn test_decode_oov_roundtrip() {
        let vocab = mock_vocab();
        let c = 'z';
        let id = vocab.char_to_oov_id(c);
        let decoded = vocab.decode_id(id);
        assert_eq!(decoded, "z");
    }

    #[test]
    fn test_q_computation_basic() {
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        p1.insert('a', 1.0);
        p1.insert('b', 0.5);
        p2.insert('a', 0.2);
        p2.insert('b', 0.8);
        let chars: Vec<char> = "ab".chars().collect();
        let q = compute_q(&chars, &p1, &p2);
        // Q[0] = P1['a'] + P2['b'] = 1.0 + 0.8 = 1.8
        assert!((q[0] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_find_boundaries_local_minima() {
        // Q = [1.0, 0.3, 0.8] → interior position 1 is a local min
        let q = vec![1.0, 0.3, 0.8];
        let b = find_u_boundaries(&q);
        assert_eq!(b, vec![2]); // split after char index 1
    }

    #[test]
    fn test_no_boundary_short_word() {
        let q = vec![0.5, 0.3]; // only 2 elements, no interior
        let b = find_u_boundaries(&q);
        assert!(b.is_empty());
    }

    #[test]
    fn test_partition_at_boundaries() {
        let chars: Vec<char> = "abcd".chars().collect();
        let boundaries = vec![2]; // split: "ab" | "cd"
        let segs = partition_at_boundaries(&chars, &boundaries);
        assert_eq!(segs.len(), 2);
        let s0: String = segs[0].iter().collect();
        let s1: String = segs[1].iter().collect();
        assert_eq!(s0, "ab");
        assert_eq!(s1, "cd");
    }

    #[test]
    fn test_encode_known_unit() {
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        // "ab" is in vocab — should encode to its ID
        let ids = tok.encode("ab");
        assert!(!ids.is_empty());
        let id = *ids.first().unwrap();
        assert!(id < vocab.u_unit_count);
    }

    #[test]
    fn test_encode_oov_falls_back_to_chars() {
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1.clone(), p2.clone());
        // "zz" will not be in vocab — should emit 2 OOV char tokens
        let ids = tok.encode("zz");
        assert_eq!(ids.len(), 2);
        for &id in &ids {
            assert!(id >= vocab.u_unit_count + 1);
        }
    }

    #[test]
    fn test_decode_roundtrip_oov_chars() {
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        let text = "zz";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size_matches_declared() {
        // When built from real u_units_index.txt, should be 5,844
        // Here we just confirm mock count is consistent
        let vocab = mock_vocab();
        assert_eq!(vocab.u_unit_count, vocab.id_to_unit.len() as u32);
    }

    #[test]
    fn test_participation_table_loader() {
        let contents = "a\t0.9\nb\t0.3\n# comment\nc\t0.5\n";
        let table = load_participation_table(contents);
        assert!((table[&'a'] - 0.9).abs() < 1e-10);
        assert!((table[&'b'] - 0.3).abs() < 1e-10);
        assert!((table[&'c'] - 0.5).abs() < 1e-10);
    }
}
