# abr-u-tokenizer  python/u_tokenizer.py
# HuggingFace-compatible Python wrapper for the U tokenizer.
#
# This wrapper implements the PreTrainedTokenizer interface so the
# U tokenizer can be dropped in wherever HF tokenizers are used,
# including with Phi-3 via AutoTokenizer.
#
# PURE PYTHON IMPLEMENTATION
# The Rust core is the authoritative implementation. This Python wrapper
# re-implements the same logic for HF compatibility without requiring
# a compiled Rust binary at inference time. When the PyO3 binding is
# available, it will be used instead.
#
# OOV handling: Option A (Origin-declared)
#   Characters with no P2 data emit as individual character tokens.

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
    HuggingFace-compatible U tokenizer.

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

        # Load vocabulary
        self._unit_to_id: Dict[str, int] = {}
        self._id_to_unit: Dict[int, str] = {}
        self._load_vocab(vocab_file)
        self._u_unit_count = len(self._unit_to_id)

        # Load participation tables
        self._p1: Dict[str, float] = self._load_participation(p1_file)
        self._p2: Dict[str, float] = self._load_participation(p2_file)

    # ─────────────────────────────────────────────────────────────────
    # HuggingFace PreTrainedTokenizer interface
    # ─────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of declared U unit tokens (5,844)."""
        return self._u_unit_count

    def tokenize(self, text: str) -> List[str]:
        """Return list of token strings (U units or [OOV:c] markers)."""
        return self._tokenize_to_strings(text)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> List[int]:
        """Encode text to list of token IDs."""
        return self._encode(text)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """Decode token IDs back to string."""
        return self._decode(token_ids)

    def __call__(self, text, **kwargs):
        """Minimal __call__ for use with model.generate() pipelines."""
        ids = self._encode(text)
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._unit_to_id)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._unit_to_id.get(t, self._oov_id('?')) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._decode_id(i) for i in ids]

    def save_pretrained(self, save_directory: str) -> Tuple[str, ...]:
        """
        Save tokenizer files compatible with AutoTokenizer.from_pretrained().
        Writes: vocab.txt, tokenizer_config.json, tokenizer.json
        """
        os.makedirs(save_directory, exist_ok=True)

        # vocab.txt
        vocab_path = os.path.join(save_directory, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for unit, idx in sorted(self._unit_to_id.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{unit}\n")

        # tokenizer_config.json
        config = {
            "tokenizer_class": "UTokenizer",
            "model_type": "abr-u-tokenizer",
            "version": "0.1.0",
            "vocab_size": self._u_unit_count,
            "oov_handling": "option_a_character_fallback",
            "oov_base_id": self._u_unit_count + 1,
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

        # tokenizer.json (minimal HF fast-tokenizer schema)
        tok_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
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
    # Internal implementation
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

    def _compute_q(self, chars: List[str]) -> List[float]:
        """Q(S)[i] = P1[c_i] + P2[c_{i+1}] for i in 0..len-1."""
        q = []
        for i in range(len(chars) - 1):
            p1_i = self._p1.get(chars[i], 0.0)
            p2_next = self._p2.get(chars[i + 1], 0.0)
            q.append(p1_i + p2_next)
        return q

    def _find_u_boundaries(self, q: List[float]) -> List[int]:
        """Interior local minima of Q → split positions in char array."""
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

    def _oov_id(self, c: str) -> int:
        """Token ID for an OOV character (Option A)."""
        return self._u_unit_count + 1 + ord(c)

    def _decode_id(self, idx: int) -> str:
        if idx < self._u_unit_count:
            return self._id_to_unit.get(idx, f"[UNK:{idx}]")
        elif idx == self._u_unit_count:
            return "[CHAR]"
        else:
            cp = idx - self._u_unit_count - 1
            try:
                return chr(cp)
            except (ValueError, OverflowError):
                return f"[CP:{cp}]"

    def _encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in text.split():
            chars = list(word)
            q = self._compute_q(chars)
            boundaries = self._find_u_boundaries(q)
            segments = self._partition(chars, boundaries)
            for seg in segments:
                s = "".join(seg)
                if s in self._unit_to_id:
                    ids.append(self._unit_to_id[s])
                else:
                    # OOV Option A: character by character
                    for c in seg:
                        ids.append(self._oov_id(c))
        return ids

    def _decode(self, ids: List[int]) -> str:
        return "".join(self._decode_id(i) for i in ids)

    def _tokenize_to_strings(self, text: str) -> List[str]:
        segments: List[str] = []
        for word in text.split():
            chars = list(word)
            q = self._compute_q(chars)
            boundaries = self._find_u_boundaries(q)
            segs = self._partition(chars, boundaries)
            for seg in segs:
                s = "".join(seg)
                if s in self._unit_to_id:
                    segments.append(s)
                else:
                    for c in seg:
                        segments.append(f"[OOV:{c}]")
        return segments
