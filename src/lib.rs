// abr-u-tokenizer  lib.rs
// V0.1.1 — U tokenizer for Phi-3
// Origin: Robin Macomber / Metatron Dynamics
//
// COMPLETE TILING (Origin-declared):
//   Every character in the input stream tiles into the output.
//   No character is discarded. No position is fabricated.
//   Spaces and all characters are treated identically to their
//   treatment in the relational processing — as observed characters
//   in the stream, tiling as codepoint tokens when they have no
//   full P1/P2 coverage.
//
// TOKEN DOMAIN:
//   0 .. (u_unit_count-1)  — U unit tokens, derived from strict local
//                            minima of Q(S) across 317,070 bounded
//                            structures. Every boundary has declared
//                            provenance from character participation
//                            patterns — NOT statistical compression.
//   u_unit_count+          — codepoint tokens: ID = u_unit_count + codepoint
//                            No gap, no phantom [CHAR] marker.
//
// DECODE ROUNDTRIP INVARIANT (Verifier-required):
//   decode(encode(S)) == S  for all S, including spaces and all characters.
//
// VERIFIER CORRECTIONS FROM V0.1.0:
//   F1 — Missing P1 or P2 now triggers full character-by-character tiling
//        of the bounded structure BEFORE Q(S) is computed. Absence of
//        observation is never substituted with 0.0.
//   F2 — Input is processed as a flat character stream. Spaces tile as
//        codepoint tokens, identical to all other characters. No whitespace
//        splitting discards dividers.
//   F3 — [CHAR] phantom marker removed. Codepoint path is the sole and
//        complete OOV representation. ID scheme is contiguous.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// VOCABULARY
// ─────────────────────────────────────────────────────────────────────────────

/// The complete declared vocabulary derived from abr-language-analysis.
#[derive(Debug, Clone)]
pub struct UVocabulary {
    /// U unit string → token ID (0..u_unit_count-1)
    pub unit_to_id: HashMap<String, u32>,
    /// token ID → U unit string
    pub id_to_unit: HashMap<u32, String>,
    /// Number of U unit tokens (5,844 from declared corpus)
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
    /// ID assignment is by sorted lexicographic index — deterministic
    /// regardless of file ordering.
    pub fn from_index_file(contents: &str) -> Result<Self, String> {
        let mut units: Vec<(String, u64)> = Vec::new();

        for (line_num, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            let (unit, count) = match parts.len() {
                0 | 1 => {
                    return Err(format!("line {}: too few columns", line_num + 1));
                }
                2 => {
                    let count = parts[1].parse::<u64>().unwrap_or(1);
                    (parts[0].to_string(), count)
                }
                _ => {
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
    /// IDs 0..u_unit_count-1 → U unit string.
    /// IDs u_unit_count+     → codepoint: id - u_unit_count = codepoint.
    pub fn decode_id(&self, id: u32) -> String {
        if id < self.u_unit_count {
            self.id_to_unit
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("[UNK:{}]", id))
        } else {
            // Codepoint token: id = u_unit_count + codepoint
            let cp = id - self.u_unit_count;
            char::from_u32(cp)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("[CP:{}]", cp))
        }
    }

    /// Encode a single character as a codepoint token ID.
    /// ID = u_unit_count + (codepoint as u32).
    /// Contiguous with U unit IDs, no gap.
    pub fn char_to_codepoint_id(&self, c: char) -> u32 {
        self.u_unit_count + (c as u32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PARTICIPATION COVERAGE CHECK
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether a bounded structure has complete P1 and P2 coverage
/// for all positions required by the declared Q(S) formula:
///   Q(S)[i] = P1[c_i] + P2[c_{i+1}]   for i in 0..len-1
///
/// Required observations:
///   P1 for every character at position 0..len-1
///   P2 for every character at position 1..len
///
/// Returns true only if ALL required observations are present.
/// Absence of observation is NEVER substituted with 0.0.
pub fn has_complete_coverage(
    chars: &[char],
    p1: &HashMap<char, f64>,
    p2: &HashMap<char, f64>,
) -> bool {
    if chars.len() < 2 {
        return false; // cannot compute Q(S) for single char or empty
    }
    for i in 0..chars.len() - 1 {
        if !p1.contains_key(&chars[i]) {
            return false;
        }
        if !p2.contains_key(&chars[i + 1]) {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Q(S) COMPUTATION
// ─────────────────────────────────────────────────────────────────────────────

/// Q(S) participation signal for a bounded structure with COMPLETE coverage.
///
/// PRECONDITION: has_complete_coverage(chars, p1, p2) == true.
/// Call site is responsible for this check. No unwrap_or substitution here.
///
/// Q(S)[i] = P1[c_i] + P2[c_{i+1}]   for i in 0..len-1
pub fn compute_q(
    chars: &[char],
    p1: &HashMap<char, f64>,
    p2: &HashMap<char, f64>,
) -> Vec<f64> {
    let n = chars.len();
    debug_assert!(n >= 2, "compute_q called on structure with < 2 chars");
    let mut q = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        // Both lookups are guaranteed present by has_complete_coverage.
        // Panic here is correct — it means the precondition was violated.
        let p1_i    = *p1.get(&chars[i]).expect("P1 missing — precondition violated");
        let p2_next = *p2.get(&chars[i + 1]).expect("P2 missing — precondition violated");
        q.push(p1_i + p2_next);
    }
    q
}

/// Find interior local minima of Q(S).
/// Position i is a boundary iff Q[i] < Q[i-1] AND Q[i] < Q[i+1].
/// Returns split positions in the original char array.
pub fn find_u_boundaries(q: &[f64]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    if q.len() < 3 {
        return boundaries;
    }
    for i in 1..q.len() - 1 {
        if q[i] < q[i - 1] && q[i] < q[i + 1] {
            boundaries.push(i + 1);
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
    /// P1 table: char → observed frequency as position-1
    pub p1: HashMap<char, f64>,
    /// P2 table: char → observed frequency as position-2
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

    /// Encode a bounded structure (a word — characters only, no spaces).
    ///
    /// Process:
    ///   1. Check complete P1/P2 coverage for all positions.
    ///   2. If coverage is incomplete → tile character-by-character.
    ///      Q(S) is NOT computed. No fabricated zeros.
    ///   3. If coverage is complete → compute Q(S), find strict interior
    ///      local minima, partition into candidate U units.
    ///   4. For each candidate:
    ///      - Known U unit → emit its ID.
    ///      - Unknown → tile character-by-character (codepoint tokens).
    fn encode_structure(&self, chars: &[char], out: &mut Vec<u32>) {
        if chars.is_empty() {
            return;
        }

        // Single character: tile directly, no Q(S) possible
        if chars.len() == 1 {
            out.push(self.vocab.char_to_codepoint_id(chars[0]));
            return;
        }

        // F1 FIX: check coverage BEFORE computing Q(S).
        // Missing P1 or P2 → tile entire structure character-by-character.
        if !has_complete_coverage(chars, &self.p1, &self.p2) {
            for &c in chars {
                out.push(self.vocab.char_to_codepoint_id(c));
            }
            return;
        }

        // Coverage confirmed — compute Q(S) and find boundaries
        let q = compute_q(chars, &self.p1, &self.p2);
        let boundaries = find_u_boundaries(&q);
        let segments = partition_at_boundaries(chars, &boundaries);

        for seg in &segments {
            let seg_str: String = seg.iter().collect();
            if let Some(&id) = self.vocab.unit_to_id.get(&seg_str) {
                out.push(id);
            } else {
                // Unknown candidate U unit → character-by-character
                for &c in seg.iter() {
                    out.push(self.vocab.char_to_codepoint_id(c));
                }
            }
        }
    }

    /// Tokenize a raw text string.
    ///
    /// COMPLETE TILING: every character in the input tiles into the output.
    /// Spaces and all characters are observed stream members — they tile
    /// as codepoint tokens, consistent with their treatment in the
    /// relational processing.
    ///
    /// Process:
    ///   Walk the input character by character.
    ///   Accumulate runs of non-space characters as bounded structures.
    ///   On space (or any whitespace): flush the accumulated structure,
    ///   then emit the space as its own codepoint token.
    ///   At end of input: flush any remaining structure.
    ///
    /// DECODE ROUNDTRIP: decode(encode(S)) == S for all S.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut current: Vec<char> = Vec::new();

        for c in text.chars() {
            if c.is_whitespace() {
                // Flush accumulated bounded structure
                if !current.is_empty() {
                    self.encode_structure(&current, &mut ids);
                    current.clear();
                }
                // Tile the space/whitespace character itself
                ids.push(self.vocab.char_to_codepoint_id(c));
            } else {
                current.push(c);
            }
        }
        // Flush final structure
        if !current.is_empty() {
            self.encode_structure(&current, &mut ids);
        }

        ids
    }

    /// Decode a sequence of token IDs back to the original string.
    /// Roundtrip invariant: decode(encode(S)) == S.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| self.vocab.decode_id(id))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Tokenize and return string segments for inspection and Verifier review.
    /// Spaces appear as their literal character in the segment list.
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current: Vec<char> = Vec::new();

        let flush = |chars: &[char], segments: &mut Vec<String>, tok: &UTokenizer| {
            if chars.is_empty() { return; }
            if chars.len() == 1 {
                segments.push(format!("[CP:{}]", chars[0]));
                return;
            }
            if !has_complete_coverage(chars, &tok.p1, &tok.p2) {
                for &c in chars {
                    segments.push(format!("[CP:{}]", c));
                }
                return;
            }
            let q = compute_q(chars, &tok.p1, &tok.p2);
            let boundaries = find_u_boundaries(&q);
            let segs = partition_at_boundaries(chars, &boundaries);
            for seg in segs {
                let s: String = seg.iter().collect();
                if tok.vocab.unit_to_id.contains_key(&s) {
                    segments.push(s);
                } else {
                    for c in seg {
                        segments.push(format!("[CP:{}]", c));
                    }
                }
            }
        };

        for c in text.chars() {
            if c.is_whitespace() {
                flush(&current, &mut segments, self);
                current.clear();
                segments.push(c.to_string()); // space as itself
            } else {
                current.push(c);
            }
        }
        flush(&current, &mut segments, self);

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
    fn test_codepoint_id_disjoint_from_u_ids() {
        let vocab = mock_vocab();
        let cp_id = vocab.char_to_codepoint_id('z');
        assert!(cp_id >= vocab.u_unit_count);
    }

    #[test]
    fn test_decode_codepoint_roundtrip() {
        let vocab = mock_vocab();
        let c = 'z';
        let id = vocab.char_to_codepoint_id(c);
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
        let q = vec![1.0, 0.3, 0.8];
        let b = find_u_boundaries(&q);
        assert_eq!(b, vec![2]);
    }

    #[test]
    fn test_no_boundary_short_word() {
        let q = vec![0.5, 0.3];
        let b = find_u_boundaries(&q);
        assert!(b.is_empty());
    }

    #[test]
    fn test_partition_at_boundaries() {
        let chars: Vec<char> = "abcd".chars().collect();
        let boundaries = vec![2];
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
        let ids = tok.encode("ab");
        assert!(!ids.is_empty());
        let id = *ids.first().unwrap();
        assert!(id < vocab.u_unit_count);
    }

    #[test]
    fn test_encode_missing_coverage_tiles_chars() {
        // F1: "zz" has no P1/P2 data — must tile character-by-character
        // WITHOUT computing Q(S)
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        let ids = tok.encode("zz");
        assert_eq!(ids.len(), 2);
        for &id in &ids {
            assert!(id >= vocab.u_unit_count);
        }
    }

    #[test]
    fn test_missing_p1_triggers_char_fallback() {
        // F1: Q(S)[i] = P1[c_i] + P2[c_{i+1}]
        // For "ba": needs P1['b'] (position 0). P1['b'] absent → char fallback.
        // Note: for "ab", P1['b'] is NOT required — 'b' is only at position 1.
        let vocab = mock_vocab();
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        // 'b' has no P1 entry; 'a' does
        p1.insert('a', 0.9);
        p2.insert('a', 0.1);
        p2.insert('b', 0.05);
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        // "ba" — P1['b'] missing (position 0) → full char fallback, no Q(S)
        let ids = tok.encode("ba");
        assert_eq!(ids.len(), 2);
        for &id in &ids {
            assert!(id >= vocab.u_unit_count);
        }
    }

    #[test]
    fn test_missing_p2_triggers_char_fallback() {
        // F1: structure where P2 is missing for a character
        let vocab = mock_vocab();
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        p1.insert('a', 0.9);
        p1.insert('b', 0.1);
        p2.insert('a', 0.1);
        // P2['b'] missing
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        // "ab" needs P2['b'] for Q[0] — absent → char fallback
        let ids = tok.encode("ab");
        assert_eq!(ids.len(), 2);
        for &id in &ids {
            assert!(id >= vocab.u_unit_count);
        }
    }

    #[test]
    fn test_space_tiles_as_codepoint() {
        // F2: spaces must tile as codepoint tokens
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        let ids = tok.encode("a b");
        // space codepoint = 32, id = u_unit_count + 32
        let space_id = vocab.char_to_codepoint_id(' ');
        assert!(ids.contains(&space_id));
    }

    #[test]
    fn test_decode_roundtrip_with_spaces() {
        // F2: full roundtrip including spaces
        let vocab = mock_vocab();
        let (p1, p2) = mock_p_tables();
        let tok = UTokenizer::new(vocab.clone(), p1, p2);
        let text = "ab cd";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
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

    #[test]
    fn test_has_complete_coverage_true() {
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        p1.insert('a', 0.9);
        p1.insert('b', 0.1);
        p2.insert('a', 0.1);
        p2.insert('b', 0.05);
        let chars: Vec<char> = "ab".chars().collect();
        assert!(has_complete_coverage(&chars, &p1, &p2));
    }

    #[test]
    fn test_has_complete_coverage_false_missing_p2() {
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        p1.insert('a', 0.9);
        p1.insert('b', 0.1);
        p2.insert('a', 0.1);
        // p2['b'] missing
        let chars: Vec<char> = "ab".chars().collect();
        assert!(!has_complete_coverage(&chars, &p1, &p2));
    }

    #[test]
    fn test_has_complete_coverage_false_missing_p1() {
        // For "ba": needs P1['b'] at position 0. P1['b'] absent → false.
        // "ab" would NOT trigger this — 'b' is only at position 1 in "ab".
        let mut p1 = HashMap::new();
        let mut p2 = HashMap::new();
        p1.insert('a', 0.9);
        // p1['b'] missing
        p2.insert('a', 0.1);
        p2.insert('b', 0.05);
        let chars: Vec<char> = "ba".chars().collect();
        assert!(!has_complete_coverage(&chars, &p1, &p2));
    }
}
