# extract_participation_tables.py
#
# Extracts P1 and P2 participation frequency tables from the
# abr-language-analysis results directory.
#
# These tables are required by the U tokenizer to compute Q(S).
# They record how often each character appears as position 1 (P1)
# or position 2 (P2) across all observed bounded structures.
#
# Usage:
#   python extract_participation_tables.py <results_dir> <output_dir>
#
# Inputs (from results_dir):
#   incident_k2.txt or similar — character-level incident records
#   OR the raw P-value files if already computed.
#
# Outputs (in output_dir):
#   p1.tsv   — char TAB frequency (normalized)
#   p2.tsv   — char TAB frequency (normalized)

import sys
import os
import re
from collections import defaultdict


def load_incident_file(path):
    """
    Load a character-level incident file.
    Expected format from abr-language-analysis:
      Each line: <structure>\t<k>\t<P1>\t<P2>\t...\t<Q_values>
    OR per-position records. We accept both and extract P1, P2 counts.

    Returns: (p1_counts, p2_counts) as dicts of char -> float
    """
    p1_counts = defaultdict(float)
    p2_counts = defaultdict(float)

    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return p1_counts, p2_counts

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            # First column: the bounded structure string
            structure = parts[0]
            if len(structure) < 2:
                continue
            # P1: character at position 0
            p1_counts[structure[0]] += 1.0
            # P2: character at position 1
            p2_counts[structure[1]] += 1.0

    return p1_counts, p2_counts


def normalize(counts):
    """Normalize counts to frequencies."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {c: v / total for c, v in counts.items()}


def write_tsv(table, path):
    """Write char TAB frequency TSV, sorted by frequency descending."""
    rows = sorted(table.items(), key=lambda x: -x[1])
    with open(path, "w", encoding="utf-8") as f:
        f.write("# char\tfrequency\n")
        for c, freq in rows:
            f.write(f"{c}\t{freq:.8f}\n")
    print(f"  Written: {path} ({len(rows)} entries)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_participation_tables.py <results_dir> <output_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning: {results_dir}")

    p1_total = defaultdict(float)
    p2_total = defaultdict(float)

    # Look for incident files — any file matching incident_k*.txt or P*.txt
    # or the u_units_index.txt (which contains the source structures)
    candidate_files = []
    for fname in os.listdir(results_dir):
        if fname.startswith("incident_k") and fname.endswith(".txt"):
            candidate_files.append(os.path.join(results_dir, fname))
        elif fname == "u_units_index.txt":
            candidate_files.append(os.path.join(results_dir, fname))

    if not candidate_files:
        # Try scanning for any .txt files with tab-separated content
        for fname in os.listdir(results_dir):
            if fname.endswith(".txt"):
                candidate_files.append(os.path.join(results_dir, fname))

    if not candidate_files:
        print(f"ERROR: No candidate files found in {results_dir}")
        print("Expected: incident_k*.txt or u_units_index.txt")
        sys.exit(1)

    print(f"Found {len(candidate_files)} candidate file(s):")
    for f in sorted(candidate_files):
        print(f"  {os.path.basename(f)}")
        p1, p2 = load_incident_file(f)
        for c, v in p1.items():
            p1_total[c] += v
        for c, v in p2.items():
            p2_total[c] += v

    if not p1_total:
        print("\nERROR: No P1 data extracted.")
        print("Check that results_dir contains incident_k*.txt files")
        print("from the abr-language-analysis run.")
        sys.exit(1)

    p1_norm = normalize(p1_total)
    p2_norm = normalize(p2_total)

    print(f"\nExtracted:")
    print(f"  P1 entries: {len(p1_norm)}")
    print(f"  P2 entries: {len(p2_norm)}")

    p1_path = os.path.join(output_dir, "p1.tsv")
    p2_path = os.path.join(output_dir, "p2.tsv")
    write_tsv(p1_norm, p1_path)
    write_tsv(p2_norm, p2_path)

    print("\nDone. Use these files with the U tokenizer:")
    print(f"  p1.tsv: {p1_path}")
    print(f"  p2.tsv: {p2_path}")


if __name__ == "__main__":
    main()
