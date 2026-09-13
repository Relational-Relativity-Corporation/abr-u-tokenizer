# abr-u-tokenizer  v0.2.0/observe.py
# Observation Harness — Phase 1 and Phase 2
#
# PURPOSE:
#   Pure structural comparison of T_Phi (Phi-3 BPE) vs T_U (U tokenizer)
#   on the same declared text corpus. No scoring. No optimization.
#   No U boundaries changed because of results.
#
# WHAT IS RECORDED (per sequence):
#   - Original character stream and length
#   - Phi-3 token sequence: count, spans [a_k, b_k), token strings
#   - U token sequence: count, spans, U-unit vs character-fallback breakdown
#   - Exact reconstruction from both (roundtrip verification)
#
# PHASE 2 — CONTROLLED INTERVENTIONS:
#   For declared pairs (S0, S1) differing by a known character operation,
#   record Delta_T_Phi and Delta_T_U independently.
#
# DECLARED QUESTION:
#   How does the same observable character structure map into T_Phi vs T_U?
#
# USAGE:
#   python observe.py \
#       --vocab   <path>/results/u_units_tokenizer/vocab.txt \
#       --p1      <path>/results/u_units_tokenizer/p1.tsv \
#       --p2      <path>/results/u_units_tokenizer/p2.tsv \
#       --corpus  <path>/corpus \
#       --phi3    <path-to-phi3-directory> \
#       --n       200 \
#       --out     results/observation_v0.2.0.json

import argparse
import json
import os
import sys
import random
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenSpan:
    """A single token with its character interval in the original stream."""
    token_str: str
    token_id: int
    char_start: int
    char_end: int        # exclusive
    kind: str            # "U_unit" | "char_fallback" | "phi3"

@dataclass
class TokenizationRecord:
    """Full tokenization record for one text sequence under one tokenizer."""
    tokenizer: str       # "U" or "Phi3"
    text: str
    char_len: int
    token_count: int
    tokens: List[TokenSpan]
    reconstruction: str
    roundtrip_ok: bool
    u_unit_count: int    # U tokenizer only (0 for Phi3)
    char_fallback_count: int  # U tokenizer only

@dataclass
class SequenceObservation:
    """Complete observation for one input sequence."""
    seq_id: str
    source: str          # corpus file name
    text: str
    u_record: TokenizationRecord
    phi3_record: Optional[TokenizationRecord]
    phi3_available: bool

@dataclass
class InterventionObservation:
    """Phase 2: controlled intervention pair."""
    pair_id: str
    operation: str       # "substitution" | "transposition" | "insertion" | "deletion"
    s0: str
    s1: str
    char_delta_description: str
    # Character positions where s0 and s1 differ
    diff_positions: List[int]
    # U tokenizer deltas
    u_s0_tokens: List[str]
    u_s1_tokens: List[str]
    u_delta_count: int   # abs(len(s1_tokens) - len(s0_tokens))
    u_boundary_shift: bool  # did any U boundary move?
    # Phi-3 deltas (None if Phi-3 unavailable)
    phi3_s0_tokens: Optional[List[str]]
    phi3_s1_tokens: Optional[List[str]]
    phi3_delta_count: Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# U TOKENIZER WITH SPAN TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def u_tokenize_with_spans(text: str, tok) -> TokenizationRecord:
    """
    Run the U tokenizer and record exact character spans for every token.
    Mirrors the V0.1.1 encode logic but tracks positions.
    """
    tokens: List[TokenSpan] = []
    u_unit_count = 0
    char_fallback_count = 0

    pos = 0
    current: List[str] = []
    current_start = 0

    def flush_structure(chars, start):
        nonlocal u_unit_count, char_fallback_count
        if not chars:
            return
        if len(chars) == 1:
            tok_id = tok._codepoint_id(chars[0])
            tokens.append(TokenSpan(
                token_str=chars[0],
                token_id=tok_id,
                char_start=start,
                char_end=start + 1,
                kind="char_fallback"
            ))
            char_fallback_count += 1
            return

        if not tok._has_complete_coverage(chars):
            for i, c in enumerate(chars):
                tok_id = tok._codepoint_id(c)
                tokens.append(TokenSpan(
                    token_str=c,
                    token_id=tok_id,
                    char_start=start + i,
                    char_end=start + i + 1,
                    kind="char_fallback"
                ))
                char_fallback_count += 1
            return

        q = tok._compute_q(chars)
        boundaries = tok._find_u_boundaries(q)
        segments = tok._partition(chars, boundaries)

        seg_start = start
        for seg in segments:
            s = "".join(seg)
            seg_end = seg_start + len(seg)
            if s in tok._unit_to_id:
                tokens.append(TokenSpan(
                    token_str=s,
                    token_id=tok._unit_to_id[s],
                    char_start=seg_start,
                    char_end=seg_end,
                    kind="U_unit"
                ))
                u_unit_count += 1
            else:
                for i, c in enumerate(seg):
                    tok_id = tok._codepoint_id(c)
                    tokens.append(TokenSpan(
                        token_str=c,
                        token_id=tok_id,
                        char_start=seg_start + i,
                        char_end=seg_start + i + 1,
                        kind="char_fallback"
                    ))
                    char_fallback_count += 1
            seg_start = seg_end

    for i, c in enumerate(text):
        if c.isspace():
            if current:
                flush_structure(current, current_start)
                current = []
            tok_id = tok._codepoint_id(c)
            tokens.append(TokenSpan(
                token_str=c,
                token_id=tok_id,
                char_start=i,
                char_end=i + 1,
                kind="char_fallback"
            ))
            char_fallback_count += 1
            current_start = i + 1
        else:
            if not current:
                current_start = i
            current.append(c)

    if current:
        flush_structure(current, current_start)

    reconstruction = "".join(t.token_str for t in tokens)
    roundtrip_ok = (reconstruction == text)

    return TokenizationRecord(
        tokenizer="U",
        text=text,
        char_len=len(text),
        token_count=len(tokens),
        tokens=tokens,
        reconstruction=reconstruction,
        roundtrip_ok=roundtrip_ok,
        u_unit_count=u_unit_count,
        char_fallback_count=char_fallback_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHI-3 TOKENIZER WITH SPAN TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def phi3_tokenize_with_spans(text: str, phi3_tokenizer) -> TokenizationRecord:
    """
    Run Phi-3's BPE tokenizer and record character spans.
    Uses offset_mapping to get exact character intervals.
    """
    encoding = phi3_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    tokens = []
    for tok_id, (start, end) in zip(input_ids, offsets):
        tok_str = phi3_tokenizer.decode([tok_id])
        tokens.append(TokenSpan(
            token_str=tok_str,
            token_id=tok_id,
            char_start=start,
            char_end=end,
            kind="phi3"
        ))

    reconstruction = phi3_tokenizer.decode(input_ids, skip_special_tokens=False)
    roundtrip_ok = (reconstruction == text)

    return TokenizationRecord(
        tokenizer="Phi3",
        text=text,
        char_len=len(text),
        token_count=len(tokens),
        tokens=tokens,
        reconstruction=reconstruction,
        roundtrip_ok=roundtrip_ok,
        u_unit_count=0,
        char_fallback_count=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_sentences(corpus_dir: str, n: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample n sentences from the declared corpus files.
    Returns list of (source_filename, sentence_text).
    Sentences: split on . ? ! followed by whitespace or end.
    Min length 20 chars, max 200 chars.
    """
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    if not files:
        raise ValueError(f"No .txt files found in {corpus_dir}")

    candidates = []
    for fname in sorted(files):
        path = os.path.join(corpus_dir, fname)
        try:
            text = open(path, encoding='utf-8', errors='replace').read()
        except Exception as e:
            print(f"  [WARN] Cannot read {fname}: {e}")
            continue
        # Split into sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            s = s.strip()
            s = re.sub(r'\s+', ' ', s)  # normalize whitespace
            if 20 <= len(s) <= 200:
                candidates.append((fname, s))

    if not candidates:
        raise ValueError("No valid sentences found in corpus")

    random.seed(seed)
    if len(candidates) <= n:
        selected = candidates
    else:
        selected = random.sample(candidates, n)

    print(f"  Sampled {len(selected)} sentences from {len(files)} corpus file(s)")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLED INTERVENTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_intervention_pairs(sentences: List[Tuple[str, str]]) -> List[dict]:
    """
    Build declared intervention pairs from the sampled sentences.
    Operations: substitution, transposition, insertion, deletion.
    Uses only sentences with at least 2 words and 10+ chars.
    """
    pairs = []
    eligible = [(src, s) for src, s in sentences if len(s) >= 10 and ' ' in s]
    random.seed(99)
    sample = random.sample(eligible, min(40, len(eligible)))

    for i, (src, s) in enumerate(sample):
        words = s.split()
        if len(words) < 3:
            continue

        # SUBSTITUTION: replace one word with a length-similar alternative
        # Use simple vowel swap in the middle word
        mid = len(words) // 2
        w = words[mid]
        if len(w) >= 3:
            # swap first vowel found
            vowels = 'aeiouAEIOU'
            new_w = w
            for j, c in enumerate(w):
                if c in vowels:
                    replacement = 'u' if c.lower() != 'u' else 'a'
                    if c.isupper():
                        replacement = replacement.upper()
                    new_w = w[:j] + replacement + w[j+1:]
                    break
            if new_w != w:
                new_words = words[:]
                new_words[mid] = new_w
                s1 = ' '.join(new_words)
                diff = [j for j, (a, b) in enumerate(zip(s, s1)) if a != b]
                pairs.append({
                    "pair_id": f"sub_{i:03d}",
                    "operation": "substitution",
                    "source": src,
                    "s0": s,
                    "s1": s1,
                    "char_delta_description": f"vowel substitution in word '{w}' → '{new_w}' at word position {mid}",
                    "diff_positions": diff,
                })

        # TRANSPOSITION: swap two adjacent characters in a word
        w2 = words[-1]
        if len(w2) >= 4:
            j = len(w2) // 2
            new_w2 = w2[:j] + w2[j+1] + w2[j] + w2[j+2:]
            new_words = words[:]
            new_words[-1] = new_w2
            s1 = ' '.join(new_words)
            diff = [k for k, (a, b) in enumerate(zip(s, s1 + ' ')) if a != b]
            pairs.append({
                "pair_id": f"trn_{i:03d}",
                "operation": "transposition",
                "source": src,
                "s0": s,
                "s1": s1,
                "char_delta_description": f"character transposition in word '{w2}' → '{new_w2}' (positions {j},{j+1})",
                "diff_positions": diff,
            })

    return pairs


def observe_intervention(pair: dict, tok, phi3_tokenizer) -> InterventionObservation:
    """Run both tokenizers on an intervention pair and record deltas."""
    s0, s1 = pair["s0"], pair["s1"]

    u_r0 = u_tokenize_with_spans(s0, tok)
    u_r1 = u_tokenize_with_spans(s1, tok)

    u_s0_toks = [t.token_str for t in u_r0.tokens]
    u_s1_toks = [t.token_str for t in u_r1.tokens]

    # Did any U boundary shift? Compare span sets.
    u_s0_spans = {(t.char_start, t.char_end) for t in u_r0.tokens}
    u_s1_spans = {(t.char_start, t.char_end) for t in u_r1.tokens}
    u_boundary_shift = (u_s0_spans != u_s1_spans)

    phi3_s0_toks = None
    phi3_s1_toks = None
    phi3_delta = None

    if phi3_tokenizer is not None:
        p0 = phi3_tokenize_with_spans(s0, phi3_tokenizer)
        p1 = phi3_tokenize_with_spans(s1, phi3_tokenizer)
        phi3_s0_toks = [t.token_str for t in p0.tokens]
        phi3_s1_toks = [t.token_str for t in p1.tokens]
        phi3_delta = abs(p1.token_count - p0.token_count)

    return InterventionObservation(
        pair_id=pair["pair_id"],
        operation=pair["operation"],
        s0=s0,
        s1=s1,
        char_delta_description=pair["char_delta_description"],
        diff_positions=pair["diff_positions"],
        u_s0_tokens=u_s0_toks,
        u_s1_tokens=u_s1_toks,
        u_delta_count=abs(len(u_s1_toks) - len(u_s0_toks)),
        u_boundary_shift=u_boundary_shift,
        phi3_s0_tokens=phi3_s0_toks,
        phi3_s1_tokens=phi3_s1_toks,
        phi3_delta_count=phi3_delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(observations: List[SequenceObservation],
                  interventions: List[InterventionObservation]) -> None:
    print("\n" + "═" * 60)
    print("OBSERVATION SUMMARY — abr-u-tokenizer V0.2.0")
    print("═" * 60)

    n = len(observations)
    phi3_available = any(o.phi3_available for o in observations)

    u_tokens_total = sum(o.u_record.token_count for o in observations)
    u_units_total  = sum(o.u_record.u_unit_count for o in observations)
    u_chars_total  = sum(o.u_record.char_fallback_count for o in observations)
    char_lens      = sum(o.u_record.char_len for o in observations)
    roundtrips_ok  = sum(1 for o in observations if o.u_record.roundtrip_ok)

    print(f"\nPHASE 1 — STRUCTURAL COMPARISON ({n} sequences)")
    print(f"  Total characters:         {char_lens}")
    print(f"  U tokens total:           {u_tokens_total}")
    print(f"  U unit tokens:            {u_units_total}  ({100*u_units_total/max(u_tokens_total,1):.1f}%)")
    print(f"  Char fallback tokens:     {u_chars_total}  ({100*u_chars_total/max(u_tokens_total,1):.1f}%)")
    print(f"  Avg U tokens/sequence:    {u_tokens_total/max(n,1):.1f}")
    print(f"  Roundtrip D(E(S))==S:     {roundtrips_ok}/{n}")

    if phi3_available:
        phi3_total = sum(o.phi3_record.token_count for o in observations if o.phi3_record)
        phi3_n = sum(1 for o in observations if o.phi3_record)
        print(f"\n  Phi-3 tokens total:       {phi3_total}")
        print(f"  Avg Phi-3 tokens/seq:     {phi3_total/max(phi3_n,1):.1f}")
        ratio = u_tokens_total / max(phi3_total, 1)
        print(f"  U/Phi-3 token ratio:      {ratio:.3f}")
    else:
        print(f"\n  Phi-3: not available (run with --phi3 to enable)")

    print(f"\nPHASE 2 — CONTROLLED INTERVENTIONS ({len(interventions)} pairs)")
    if interventions:
        subs  = [v for v in interventions if v.operation == "substitution"]
        trans = [v for v in interventions if v.operation == "transposition"]
        u_boundary_shifts = sum(1 for v in interventions if v.u_boundary_shift)
        u_delta_nonzero   = sum(1 for v in interventions if v.u_delta_count > 0)

        print(f"  Substitution pairs:       {len(subs)}")
        print(f"  Transposition pairs:      {len(trans)}")
        print(f"  U boundary shifts:        {u_boundary_shifts}/{len(interventions)}")
        print(f"  U token count changes:    {u_delta_nonzero}/{len(interventions)}")

        if phi3_available and interventions[0].phi3_delta_count is not None:
            phi3_shifts = sum(1 for v in interventions
                              if v.phi3_delta_count is not None and v.phi3_delta_count > 0)
            print(f"  Phi-3 token count changes:{phi3_shifts}/{len(interventions)}")

    print(f"\nSAMPLE OBSERVATIONS (first 5)")
    for o in observations[:5]:
        u = o.u_record
        print(f"\n  [{o.seq_id}] {o.source}")
        print(f"  TEXT:  {o.text[:80]}{'...' if len(o.text)>80 else ''}")
        print(f"  CHARS: {u.char_len}")
        print(f"  U TOK: {u.token_count}  (U-units: {u.u_unit_count}, fallback: {u.char_fallback_count})")
        print(f"  SEGS:  {' | '.join(t.token_str for t in u.tokens[:15])}{'...' if len(u.tokens)>15 else ''}")
        if o.phi3_record:
            p = o.phi3_record
            print(f"  PHI3:  {p.token_count} tokens: {' | '.join(t.token_str for t in p.tokens[:15])}{'...' if len(p.tokens)>15 else ''}")
        print(f"  RT OK: {u.roundtrip_ok}")

    if interventions:
        print(f"\nSAMPLE INTERVENTIONS (first 3)")
        for v in interventions[:3]:
            print(f"\n  [{v.pair_id}] {v.operation}")
            print(f"  S0: {v.s0[:70]}")
            print(f"  S1: {v.s1[:70]}")
            print(f"  Δ:  {v.char_delta_description}")
            print(f"  U  S0: {' | '.join(v.u_s0_tokens[:12])}")
            print(f"  U  S1: {' | '.join(v.u_s1_tokens[:12])}")
            print(f"  U  boundary shift: {v.u_boundary_shift}  token count Δ: {v.u_delta_count}")
            if v.phi3_s0_tokens:
                print(f"  P3 S0: {' | '.join(v.phi3_s0_tokens[:12])}")
                print(f"  P3 S1: {' | '.join(v.phi3_s1_tokens[:12])}")
                print(f"  P3 token count Δ: {v.phi3_delta_count}")


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def serialise_observation(o: SequenceObservation) -> dict:
    def rec(r: TokenizationRecord) -> dict:
        d = asdict(r)
        # tokens list: keep essential fields only for file size
        d['tokens'] = [
            {"s": t.token_str, "id": t.token_id,
             "a": t.char_start, "b": t.char_end, "k": t.kind}
            for t in r.tokens
        ]
        return d

    return {
        "seq_id": o.seq_id,
        "source": o.source,
        "text": o.text,
        "u": rec(o.u_record),
        "phi3": rec(o.phi3_record) if o.phi3_record else None,
        "phi3_available": o.phi3_available,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="abr-u-tokenizer V0.2.0 — Observation Harness"
    )
    parser.add_argument("--vocab",   required=True, help="Path to vocab.txt")
    parser.add_argument("--p1",      required=True, help="Path to p1.tsv")
    parser.add_argument("--p2",      required=True, help="Path to p2.tsv")
    parser.add_argument("--corpus",  required=True, help="Path to corpus directory")
    parser.add_argument("--phi3",    default=None,  help="Path to Phi-3 local directory (optional)")
    parser.add_argument("--n",       type=int, default=200, help="Number of sequences to sample")
    parser.add_argument("--out",     default="results/observation_v0.2.0.json")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print("abr-u-tokenizer V0.2.0 — Observation Harness")
    print("═" * 60)

    # ── Load U tokenizer ─────────────────────────────────────────────
    print("\n[1] Loading U tokenizer...")
    from u_tokenizer import UTokenizer
    tok = UTokenizer(
        vocab_file=args.vocab,
        p1_file=args.p1,
        p2_file=args.p2,
    )
    print(f"    Vocab size: {tok.vocab_size} U units")

    # ── Load Phi-3 (optional) ────────────────────────────────────────
    phi3_tokenizer = None
    if args.phi3:
        print(f"\n[2] Loading Phi-3 tokenizer from: {args.phi3}")
        try:
            from transformers import AutoTokenizer
            phi3_tokenizer = AutoTokenizer.from_pretrained(
                args.phi3, trust_remote_code=True
            )
            print(f"    Phi-3 vocab size: {phi3_tokenizer.vocab_size}")
        except Exception as e:
            print(f"    [WARN] Could not load Phi-3: {e}")
            print(f"    Continuing without Phi-3 comparison.")
    else:
        print("\n[2] Phi-3 not specified — U tokenizer only.")

    # ── Sample corpus ────────────────────────────────────────────────
    print(f"\n[3] Sampling {args.n} sentences from corpus: {args.corpus}")
    random.seed(args.seed)
    sentences = sample_sentences(args.corpus, args.n, seed=args.seed)

    # ── Phase 1: tokenize all sequences ─────────────────────────────
    print(f"\n[4] Phase 1 — Tokenizing {len(sentences)} sequences...")
    observations = []
    for i, (source, text) in enumerate(sentences):
        u_record = u_tokenize_with_spans(text, tok)

        phi3_record = None
        if phi3_tokenizer is not None:
            try:
                phi3_record = phi3_tokenize_with_spans(text, phi3_tokenizer)
            except Exception as e:
                pass

        observations.append(SequenceObservation(
            seq_id=f"seq_{i:04d}",
            source=source,
            text=text,
            u_record=u_record,
            phi3_record=phi3_record,
            phi3_available=(phi3_record is not None),
        ))

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(sentences)}...")

    # ── Phase 2: controlled interventions ───────────────────────────
    print(f"\n[5] Phase 2 — Building intervention pairs...")
    random.seed(args.seed)
    pairs = build_intervention_pairs(sentences)
    print(f"    {len(pairs)} pairs declared")

    interventions = []
    for pair in pairs:
        obs = observe_intervention(pair, tok, phi3_tokenizer)
        interventions.append(obs)

    # ── Print summary ────────────────────────────────────────────────
    print_summary(observations, interventions)

    # ── Write results ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    output = {
        "version": "0.2.0",
        "declared_question": (
            "How does the same observable character structure map "
            "into T_Phi versus T_U?"
        ),
        "n_sequences": len(observations),
        "n_interventions": len(interventions),
        "phi3_available": phi3_tokenizer is not None,
        "observations": [serialise_observation(o) for o in observations],
        "interventions": [asdict(v) for v in interventions],
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[6] Results written to: {args.out}")
    print("    Done.")


if __name__ == "__main__":
    main()
