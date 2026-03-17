#!/usr/bin/env bash
set -euo pipefail

IN="outputs/submissions/submission_classical.csv"
OUT="outputs/submissions/submission_classical_EUOS25.csv"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

python - <<'PY'
import pandas as pd
from pathlib import Path

IN = Path("outputs/submissions/submission_classical.csv")
OUT = Path("outputs/submissions/submission_classical_EUOS25.csv")

df = pd.read_csv(IN)

df_fixed = df[
    [
        "Transmittance340",
        "Transmittance450",
        "Fluorescence340_450",
        "Fluorescence480",
    ]
].copy()

df_fixed.columns = [
    "Transmittance(340)",
    "Transmittance(450)",
    "Fluorescence(340/450)",
    "Fluorescence(>480)",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
df_fixed.to_csv(OUT, index=False)

print("Saved:", str(OUT))
print("Shape:", df_fixed.shape)
print(df_fixed.head())
PY

echo
echo "Preview:"
head -n 5 "$OUT"
echo
echo "Line count (should be 29421 incl header):"
wc -l "$OUT"
