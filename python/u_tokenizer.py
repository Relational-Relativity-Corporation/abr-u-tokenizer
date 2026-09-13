# abr-u-tokenizer  python/u_tokenizer.py
# V0.1.1 — HuggingFace-compatible Python wrapper for the U tokenizer.
#
# AUTHORITATIVE IMPLEMENTATION: Rust (src/lib.rs)
# This Python wrapper re-implements the same declared logic for HF
# compatibility. It must produce identical output to the Rust core
# for all inputs. Cross-implementation agreement is tested in
# tests/test_cross_impl.py.
#
# COMPLETE TILING (Origin-declared):
#   Every character in the input stream tiles into the output.
#   No character is discarded. No position is fabricated.
#   Spaces tile as codepoint tokens, identical to all other characters.
#
# TOKEN ID SCHEME (V0.1.1 — matches Rust exactly):
#   0 .. (u_unit_count-1)  — U unit tokens
#   u_unit_count+          — codepoint tokens: ID = u_unit_count + codepoint
#   No gap. No phantom [CHAR] marker.
#
# DECODE ROUNDTRIP INVARIANT:
#   decode(encode(S)) == S  for all S.
#
# VERIFIER CORRECTIONS FROM V0.1.0:
#   F1 — has_complete_coverage() checked BEFORE Q(S) computation.
#        Missing P1 or P2 tiles entire structure character-by-character.
#        No 0.0 substitution anywhere.
#   F2 — Full character stream processing. Whitespace tiles as codepoint
#        tokens. text.split() replaced with character-by-character walk.
#   F3 — [CHAR] phantom marker removed. Codepoint path is sole OOV
#        representation. ID = u_unit_count + ord(c), no +1 offset.

import json
import os
from typing import Dict, List, Optional, Tuple

# Try to import the compiled Rust extension first
try:
    import abr_u_tokenizer_rs as _rust_core  # type: ignore
    _USE_RUST = True
except ImportError:
    _USE_RUST = False


class UTokenizer:
    """
    HuggingFace-compatible U tokenizer — V0.1.1.

    Tokens are derived from strict local minima of Q(S) across bounded
    structures in the declared corpus. Every token boundary has provenance
    from character participation patterns, not statistical compression.

    Parameters
    ----------
    vocab_file : str
        Path to vocab.txt (ID\\tunit, one per line)
    p1_file : str
        Path to P1 participation table (char\\tfrequency TSV)
    p2_file : str
        Path to P2 participation table (char\\tfrequency TSV)
    """

    def __init__(
        self,
        vocab_file: str,
        p1_file: str,
        p2_file: str,
    ):
        self.vocab_file = vocab_file
        self.p1_file = p1_file
        self.p2_file = p2_file

        self._unit_to_id: Dict[str, int] = {}
        self._id_to_unit: Dict[int, str] = {}
        self._load_vocab(vocab_file)
        self._u_unit_count = len(self._unit_to_id)

        self._p1: Dict[str, float] = self._load_participation(p1_file)
        self._p2: Dict[str, float] = self._load_participation(p2_file)

    # ─────────────────────────────────────────────────────────────────
    # HuggingFace PreTrainedTokenizer interface
    # ─────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._u_unit_count

    def tokenize(self, text: str) -> List[str]:
        return self._tokenize_to_strings(text)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> List[int]:
        return self._encode(text)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        return self._decode(token_ids)

    def __call__(self, text, **kwargs):
        ids = self._encode(text)
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._unit_to_id)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._unit_to_id.get(t, self._codepoint_id('?')) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._decode_id(i) for i in ids]

    def save_pretrained(self, save_directory: str) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)

        vocab_path = os.path.join(save_directory, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for unit, idx in sorted(self._unit_to_id.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{unit}\n")

        config = {
            "tokenizer_class": "UTokenizer",
            "model_type": "abr-u-tokenizer",
            "version": "0.1.1",
            "vocab_size": self._u_unit_count,
            "tiling": "complete — every character tiles, no discards",
            "codepoint_base_id": self._u_unit_count,
            "framework": "ABR/ABRCE V7",
            "provenance": (
                "D_U derived from strict local minima of Q(S) "
                "across 317,070 bounded structures"
            ),
            "declared_count": 5844,
            "auto_map": {
                "AutoTokenizer": "u_tokenizer.UTokenizer"
            },
        }
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        tok_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "CharacterStream"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "ABR_U",
                "vocab": {unit: idx for unit, idx in self._unit_to_id.items()},
                "vocab_size": self._u_unit_count,
            },
        }
        tok_path = os.path.join(save_directory, "tokenizer.json")
        with open(tok_path, "w", encoding="utf-8") as f:
            json.dump(tok_json, f, indent=2)

        return (vocab_path, config_path, tok_path)

    # ─────────────────────────────────────────────────────────────────
    # Internal implementation — must match Rust core exactly
    # ─────────────────────────────────────────────────────────────────

    def _load_vocab(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        unit = parts[1]
                    except ValueError:
                        unit = parts[0]
                        try:
                            idx = int(parts[1])
                        except ValueError:
                            continue
                    self._unit_to_id[unit] = idx
                    self._id_to_unit[idx] = unit

    def _load_participation(self, path: str) -> Dict[str, float]:
        table: Dict[str, float] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0]:
                    try:
                        table[parts[0][0]] = float(parts[1])
                    except ValueError:
                        pass
        return table

    def _has_complete_coverage(self, chars: List[str]) -> bool:
        """
        F1 FIX: Check complete P1 and P2 coverage before computing Q(S).
        Q(S)[i] = P1[c_i] + P2[c_{i+1}] for i in 0..len-1.
        Requires: P1 for chars[0..len-1], P2 for chars[1..len].
        Returns False if ANY required observation is absent.
        Absence is NEVER substituted with 0.0.
        """
        if len(chars) < 2:
            return False
        for i in range(len(chars) - 1):
            if chars[i] not in self._p1:
                return False
            if chars[i + 1] not in self._p2:
                return False
        return True

    def _compute_q(self, chars: List[str]) -> List[float]:
        """
        Q(S)[i] = P1[c_i] + P2[c_{i+1}].
        PRECONDITION: _has_complete_coverage(chars) is True.
        No .get(..., 0.0) — both lookups must be present.
        """
        q = []
        for i in range(len(chars) - 1):
            p1_i    = self._p1[chars[i]]       # KeyError = precondition violated
            p2_next = self._p2[chars[i + 1]]   # KeyError = precondition violated
            q.append(p1_i + p2_next)
        return q

    def _find_u_boundaries(self, q: List[float]) -> List[int]:
        boundaries = []
        for i in range(1, len(q) - 1):
            if q[i] < q[i - 1] and q[i] < q[i + 1]:
                boundaries.append(i + 1)
        return boundaries

    def _partition(self, chars: List[str], boundaries: List[int]) -> List[List[str]]:
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

    def _codepoint_id(self, c: str) -> int:
        """
        F3 FIX: Codepoint token ID = u_unit_count + ord(c).
        No +1 offset. No phantom [CHAR] marker.
        Matches Rust: char_to_codepoint_id = u_unit_count + (c as u32).
        """
        return self._u_unit_count + ord(c)

    def _decode_id(self, idx: int) -> str:
        """
        F3 FIX: IDs < u_unit_count → U unit string.
                IDs >= u_unit_count → codepoint: idx - u_unit_count.
        No [CHAR] branch.
        """
        if idx < self._u_unit_count:
            return self._id_to_unit.get(idx, f"[UNK:{idx}]")
        else:
            cp = idx - self._u_unit_count
            try:
                return chr(cp)
            except (ValueError, OverflowError):
                return f"[CP:{cp}]"

    def _encode_structure(self, chars: List[str], out: List[int]) -> None:
        """
        Encode a bounded structure (non-whitespace run).
        F1: coverage checked first — missing P1/P2 → char-by-char, no Q(S).
        F2: called only for non-whitespace runs; caller handles spaces.
        """
        if not chars:
            return
        if len(chars) == 1:
            out.append(self._codepoint_id(chars[0]))
            return
        if not self._has_complete_coverage(chars):
            for c in chars:
                out.append(self._codepoint_id(c))
            return
        q = self._compute_q(chars)
        boundaries = self._find_u_boundaries(q)
        segments = self._partition(chars, boundaries)
        for seg in segments:
            s = "".join(seg)
            if s in self._unit_to_id:
                out.append(self._unit_to_id[s])
            else:
                for c in seg:
                    out.append(self._codepoint_id(c))

    def _encode(self, text: str) -> List[int]:
        """
        F2 FIX: Walk full character stream. Whitespace tiles as codepoint
        tokens — never discarded. Bounded structures (non-whitespace runs)
        processed by _encode_structure.
        decode(encode(S)) == S for all S.
        """
        ids: List[int] = []
        current: List[str] = []
        for c in text:
            if c.isspace():
                if current:
                    self._encode_structure(current, ids)
                    current = []
                ids.append(self._codepoint_id(c))
            else:
                current.append(c)
        if current:
            self._encode_structure(current, ids)
        return ids

    def _decode(self, ids: List[int]) -> str:
        return "".join(self._decode_id(i) for i in ids)

    def _tokenize_to_strings(self, text: str) -> List[str]:
        segments: List[str] = []
        current: List[str] = []

        def flush(chars: List[str]) -> None:
            if not chars:
                return
            if len(chars) == 1:
                segments.append(f"[CP:{chars[0]}]")
                return
            if not self._has_complete_coverage(chars):
                for c in chars:
                    segments.append(f"[CP:{c}]")
                return
            q = self._compute_q(chars)
            boundaries = self._find_u_boundaries(q)
            segs = self._partition(chars, boundaries)
            for seg in segs:
                s = "".join(seg)
                if s in self._unit_to_id:
                    segments.append(s)
                else:
                    for c in seg:
                        segments.append(f"[CP:{c}]")

        for c in text:
            if c.isspace():
                flush(current)
                current = []
                segments.append(c)
            else:
                current.append(c)
        flush(current)
        return segments
