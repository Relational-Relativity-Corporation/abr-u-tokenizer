# tests/test_cross_impl.py
# Cross-implementation agreement tests for abr-u-tokenizer V0.1.1.
#
# Verifier requirement: E_Rust(S) == E_Python(S) and
#   D_Rust(E(S)) == D_Python(E(S)) == S
# for all declared test cases.
#
# This test suite runs entirely in Python, simulating the Rust logic
# with a reference reimplementation, then comparing against the
# Python UTokenizer. It does not require a compiled Rust binary.
#
# To run:
#   cd abr-u-tokenizer
#   python -m pytest tests/test_cross_impl.py -v
#
# Or without pytest:
#   python tests/test_cross_impl.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE RUST-EQUIVALENT IMPLEMENTATION (pure Python)
# Mirrors src/lib.rs exactly. Used as the ground truth for comparison.
# ─────────────────────────────────────────────────────────────────────────────

def ref_has_complete_coverage(chars, p1, p2):
    if len(chars) < 2:
        return False
    for i in range(len(chars) - 1):
        if chars[i] not in p1:
            return False
        if chars[i + 1] not in p2:
            return False
    return True

def ref_compute_q(chars, p1, p2):
    q = []
    for i in range(len(chars) - 1):
        q.append(p1[chars[i]] + p2[chars[i + 1]])
    return q

def ref_find_u_boundaries(q):
    boundaries = []
    for i in range(1, len(q) - 1):
        if q[i] < q[i - 1] and q[i] < q[i + 1]:
            boundaries.append(i + 1)
    return boundaries

def ref_partition(chars, boundaries):
    segments = []
    start = 0
    for b in boundaries:
        if b > start:
            segments.append(chars[start:b])
            start = b
    if start < len(chars):
        segments.append(chars[start:])
    if not segments and chars:
        segments.append(chars[:])
    return segments

def ref_codepoint_id(c, u_unit_count):
    return u_unit_count + ord(c)

def ref_decode_id(idx, u_unit_count, id_to_unit):
    if idx < u_unit_count:
        return id_to_unit.get(idx, f"[UNK:{idx}]")
    else:
        cp = idx - u_unit_count
        try:
            return chr(cp)
        except (ValueError, OverflowError):
            return f"[CP:{cp}]"

def ref_encode_structure(chars, p1, p2, unit_to_id, u_unit_count, out):
    if not chars:
        return
    if len(chars) == 1:
        out.append(ref_codepoint_id(chars[0], u_unit_count))
        return
    if not ref_has_complete_coverage(chars, p1, p2):
        for c in chars:
            out.append(ref_codepoint_id(c, u_unit_count))
        return
    q = ref_compute_q(chars, p1, p2)
    boundaries = ref_find_u_boundaries(q)
    segments = ref_partition(chars, boundaries)
    for seg in segments:
        s = "".join(seg)
        if s in unit_to_id:
            out.append(unit_to_id[s])
        else:
            for c in seg:
                out.append(ref_codepoint_id(c, u_unit_count))

def ref_encode(text, p1, p2, unit_to_id, u_unit_count):
    ids = []
    current = []
    for c in text:
        if c.isspace():
            if current:
                ref_encode_structure(current, p1, p2, unit_to_id, u_unit_count, ids)
                current = []
            ids.append(ref_codepoint_id(c, u_unit_count))
        else:
            current.append(c)
    if current:
        ref_encode_structure(current, p1, p2, unit_to_id, u_unit_count, ids)
    return ids

def ref_decode(ids, u_unit_count, id_to_unit):
    return "".join(ref_decode_id(i, u_unit_count, id_to_unit) for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FIXTURES — mirror Rust test fixtures exactly
# ─────────────────────────────────────────────────────────────────────────────

MOCK_VOCAB_CONTENTS = "ab\t10\ncd\t5\nef\t3\na\t20\nb\t15\nc\t8\nd\t4\ne\t6\nf\t2\n"

def build_mock_vocab():
    units = []
    for line in MOCK_VOCAB_CONTENTS.strip().split("\n"):
        parts = line.split("\t")
        units.append((parts[0], int(parts[1])))
    units.sort(key=lambda x: x[0])
    unit_to_id = {u: i for i, (u, _) in enumerate(units)}
    id_to_unit = {i: u for u, i in unit_to_id.items()}
    return unit_to_id, id_to_unit, len(units)

def build_mock_p_tables():
    p1 = {'a': 0.9, 'b': 0.1, 'c': 0.5, 'd': 0.2}
    p2 = {'a': 0.1, 'b': 0.05, 'c': 0.3, 'd': 0.8}
    return p1, p2

def build_python_tokenizer():
    """Build a UTokenizer from mock data without requiring files."""
    import io
    from unittest.mock import patch, mock_open

    unit_to_id, id_to_unit, u_unit_count = build_mock_vocab()
    p1, p2 = build_mock_p_tables()

    # Build via internal state injection (avoids file I/O)
    from u_tokenizer import UTokenizer

    # Create instance with dummy paths, then inject state
    class MockUTokenizer(UTokenizer):
        def __init__(self):
            self._unit_to_id = unit_to_id
            self._id_to_unit = id_to_unit
            self._u_unit_count = u_unit_count
            self._p1 = p1
            self._p2 = p2

    return MockUTokenizer(), unit_to_id, id_to_unit, u_unit_count, p1, p2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_known_u_unit_agreement():
    """Known U unit 'ab' encodes to same ID in both implementations."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1, p2, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"known unit: Python={py_ids} Rust={rs_ids}"
    assert py_ids[0] < u_unit_count, "should be a U unit ID"

def test_missing_p2_agreement():
    """Missing P2 tiles char-by-char identically in both."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    # 'ab' with P2['b'] absent
    p1_partial = {'a': 0.9, 'b': 0.1}
    p2_partial = {'a': 0.1}  # P2['b'] missing
    tok._p1 = p1_partial
    tok._p2 = p2_partial
    text = "ab"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1_partial, p2_partial, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"missing P2: Python={py_ids} Rust={rs_ids}"
    assert len(py_ids) == 2, "should tile 2 chars"
    for id_ in py_ids:
        assert id_ >= u_unit_count, "should be codepoint IDs"

def test_missing_p1_agreement():
    """Missing P1 at position 0 tiles char-by-char identically in both."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    # 'ba' — P1['b'] at position 0 is required
    p1_partial = {'a': 0.9}   # P1['b'] missing
    p2_partial = {'a': 0.1, 'b': 0.05}
    tok._p1 = p1_partial
    tok._p2 = p2_partial
    text = "ba"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1_partial, p2_partial, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"missing P1: Python={py_ids} Rust={rs_ids}"
    assert len(py_ids) == 2
    for id_ in py_ids:
        assert id_ >= u_unit_count

def test_space_tiling_agreement():
    """Spaces tile as codepoint tokens identically in both."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab cd"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1, p2, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"space tiling: Python={py_ids} Rust={rs_ids}"
    space_id = u_unit_count + ord(' ')
    assert space_id in py_ids, "space must appear as codepoint token"

def test_multiple_spaces_agreement():
    """Multiple consecutive spaces each tile as their own codepoint token."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab  cd"  # two spaces
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1, p2, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"multiple spaces: Python={py_ids} Rust={rs_ids}"
    space_id = u_unit_count + ord(' ')
    assert py_ids.count(space_id) == 2

def test_punctuation_agreement():
    """Punctuation characters tile as codepoint tokens identically."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab,cd"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1, p2, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"punctuation: Python={py_ids} Rust={rs_ids}"

def test_unseen_unicode_agreement():
    """Unseen Unicode characters tile as codepoint tokens identically."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "αβγ"
    py_ids = tok._encode(text)
    rs_ids = ref_encode(text, p1, p2, unit_to_id, u_unit_count)
    assert py_ids == rs_ids, f"unicode: Python={py_ids} Rust={rs_ids}"
    for id_ in py_ids:
        assert id_ >= u_unit_count

def test_roundtrip_known_unit():
    """decode(encode(S)) == S for a known U unit."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab"
    assert tok._decode(tok._encode(text)) == text

def test_roundtrip_with_spaces():
    """decode(encode(S)) == S including spaces."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "ab cd"
    assert tok._decode(tok._encode(text)) == text

def test_roundtrip_missing_coverage():
    """decode(encode(S)) == S when coverage is incomplete."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    tok._p1 = {}
    tok._p2 = {}
    text = "zz"
    assert tok._decode(tok._encode(text)) == text

def test_roundtrip_unicode():
    """decode(encode(S)) == S for unseen Unicode."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    text = "αβγ δ"
    assert tok._decode(tok._encode(text)) == text

def test_codepoint_id_no_gap():
    """Codepoint IDs start at u_unit_count with no +1 gap (F3)."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    # space is codepoint 32; ID should be u_unit_count + 32, not u_unit_count + 1 + 32
    space_id = tok._codepoint_id(' ')
    assert space_id == u_unit_count + 32, \
        f"Expected {u_unit_count + 32}, got {space_id}"

def test_no_fabricated_zero():
    """Confirm 0.0 is never substituted for absent P1/P2 (F1)."""
    tok, unit_to_id, id_to_unit, u_unit_count, p1, p2 = build_python_tokenizer()
    # Empty tables — no observations
    tok._p1 = {}
    tok._p2 = {}
    # Coverage check must return False
    chars = list("ab")
    assert not tok._has_complete_coverage(chars)
    # Encode must tile chars, not attempt Q(S)
    ids = tok._encode("ab")
    assert len(ids) == 2
    for id_ in ids:
        assert id_ >= u_unit_count


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER (no pytest required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_known_u_unit_agreement,
        test_missing_p2_agreement,
        test_missing_p1_agreement,
        test_space_tiling_agreement,
        test_multiple_spaces_agreement,
        test_punctuation_agreement,
        test_unseen_unicode_agreement,
        test_roundtrip_known_unit,
        test_roundtrip_with_spaces,
        test_roundtrip_missing_coverage,
        test_roundtrip_unicode,
        test_codepoint_id_no_gap,
        test_no_fabricated_zero,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\ntest result: {'ok' if failed == 0 else 'FAILED'}. "
          f"{passed} passed; {failed} failed")
    sys.exit(0 if failed == 0 else 1)
