# abr-u-tokenizer V0.1.0 — Build Instructions

## What This Builds

A tokenizer whose tokens ARE the 5,844 U units derived from the
abr-language-analysis corpus — not BPE, not statistical compression.
Every token boundary has declared provenance from character participation
patterns (Q(S) local minima).

**OOV handling: Option A (Origin-declared)**
Characters with no P2 data emit character-by-character. No information lost.

---

## Step 0 — File Placement

Copy these files into your abr-language-analysis repo:

```
abr-language-analysis\
  src\
    tokenizer\
      Cargo.toml          ← copy from abr-u-tokenizer/Cargo.toml
      src\
        lib.rs            ← copy from abr-u-tokenizer/src/lib.rs
        main.rs           ← copy from abr-u-tokenizer/src/main.rs
  python\
    u_tokenizer.py                  ← copy
    extract_participation_tables.py ← copy
    phi3_smoke_test.py              ← copy
```

Or keep as a standalone repo alongside abr-language-analysis.

---

## Step 1 — Extract Participation Tables

The tokenizer needs P1 and P2 frequency tables from your existing results.

```powershell
cd abr-language-analysis
python python\extract_participation_tables.py results results\u_units_tokenizer
```

Expected output:
```
P1 entries: ~52
P2 entries: ~52
Written: results\u_units_tokenizer\p1.tsv
Written: results\u_units_tokenizer\p2.tsv
```

---

## Step 2 — Build Vocabulary

```powershell
cd src\tokenizer
cargo run -- build-vocab \
    ..\..\results\u_units_index.txt \
    ..\..\results\u_units_tokenizer
```

Expected output:
```
Vocabulary built successfully.
  U units:       5844
  vocab.txt:     results\u_units_tokenizer\vocab.txt
  config:        results\u_units_tokenizer\tokenizer_config.json
```

---

## Step 3 — Run Tests

```powershell
cd src\tokenizer
cargo test
```

All 10 tests should pass. Any failure is a build error — stop and report.

---

## Step 4 — Smoke Test CLI Tokenizer

```powershell
cargo run -- stats ..\..\results\u_units_index.txt
```

Confirm total U units = 5,844.

```powershell
cargo run -- tokenize \
    ..\..\results\u_units_index.txt \
    ..\..\results\u_units_tokenizer\p1.tsv \
    ..\..\results\u_units_tokenizer\p2.tsv \
    "the relational structure of language"
```

Inspect segments and IDs. OOV tokens will show as `[OOV:c]` for any
character not segmented into a known U unit.

---

## Step 5 — Phi-3 Smoke Test (Phase 3)

Requires: `pip install transformers torch`
Requires: Phi-3 downloaded locally.

```powershell
cd python
python phi3_smoke_test.py \
    --vocab  ..\results\u_units_tokenizer\vocab.txt \
    --p1     ..\results\u_units_tokenizer\p1.tsv \
    --p2     ..\results\u_units_tokenizer\p2.tsv \
    --phi3   <path_to_your_phi3_directory>
```

Expected: `SMOKE TEST RESULT: PASS` with a clean forward-pass shape.

**Note on OC-UT-1 (expected open condition):**
OOV character IDs (>= 5845) exceed Phi-3's embedding range (~32,000).
The smoke test handles this by filtering to in-range IDs for the
forward pass. OC-UT-1 is declared open — embedding projection strategy
for OOV characters must be declared before fine-tuning work begins.

---

## Version Tag

After all tests pass: commit as V0.1.0.

Open conditions declared this build (V0.1.0):
- OC-UT-1: OOV character ID range vs Phi-3 embedding range — embedding
  projection or ID remapping strategy needed for fine-tuning phase.
