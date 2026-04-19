"""
Bootstrap 95% CI for the HF model's baseline→war distress shift.

Reclassifies the same 2,000+2,000 sample used in check_hf_baseline_vs_war.py
and computes a non-parametric bootstrap CI (2,000 resamples) over the shift
in the expanded distress ratio (sadness+fear+anger+disgust).

Output: bootstrap_hf_shift.txt
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAR_CSV      = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
BASELINE_CSV = os.path.join(SCRIPT_DIR, "reddit_baseline_comments_clean.csv")
OUT_FILE     = os.path.join(SCRIPT_DIR, "bootstrap_hf_shift.txt")

HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SAMPLE_N = 2000
N_BOOTSTRAP = 2000
SEED = 42
DISTRESS_STRICT   = {"sadness", "fear", "anger"}
DISTRESS_EXPANDED = {"sadness", "fear", "anger", "disgust"}


def main():
    rng = np.random.default_rng(SEED)

    print(f"Loading samples (seed={SEED})...")
    war_texts = (pd.read_csv(WAR_CSV)["body"].fillna("").astype(str)
                 .sample(n=SAMPLE_N, random_state=SEED).tolist())
    base_texts = (pd.read_csv(BASELINE_CSV)["body"].fillna("").astype(str)
                  .sample(n=SAMPLE_N, random_state=SEED).tolist())

    print(f"Loading HF model: {HF_MODEL}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = pipeline("text-classification", model=HF_MODEL, top_k=1, device=device)

    def classify(texts):
        out = pipe(texts, batch_size=32, truncation=True, max_length=128)
        return np.array([r[0]["label"] if isinstance(r, list) else r["label"]
                         for r in out])

    print(f"Classifying {SAMPLE_N} war comments...")
    war_labels = classify(war_texts)
    print(f"Classifying {SAMPLE_N} baseline comments...")
    base_labels = classify(base_texts)

    war_is_distress_strict = np.isin(war_labels, list(DISTRESS_STRICT))
    base_is_distress_strict = np.isin(base_labels, list(DISTRESS_STRICT))
    war_is_distress_exp = np.isin(war_labels, list(DISTRESS_EXPANDED))
    base_is_distress_exp = np.isin(base_labels, list(DISTRESS_EXPANDED))

    # Point estimates (pp)
    pt_strict = 100 * (war_is_distress_strict.mean() - base_is_distress_strict.mean())
    pt_exp    = 100 * (war_is_distress_exp.mean() - base_is_distress_exp.mean())

    # Bootstrap
    print(f"Running {N_BOOTSTRAP} bootstrap resamples...")
    shifts_strict = np.empty(N_BOOTSTRAP)
    shifts_exp = np.empty(N_BOOTSTRAP)
    n = SAMPLE_N
    for i in range(N_BOOTSTRAP):
        idx_w = rng.integers(0, n, n)
        idx_b = rng.integers(0, n, n)
        shifts_strict[i] = 100 * (war_is_distress_strict[idx_w].mean()
                                  - base_is_distress_strict[idx_b].mean())
        shifts_exp[i] = 100 * (war_is_distress_exp[idx_w].mean()
                               - base_is_distress_exp[idx_b].mean())

    ci_strict = np.percentile(shifts_strict, [2.5, 97.5])
    ci_exp    = np.percentile(shifts_exp, [2.5, 97.5])

    lines = []
    def P(*a):
        msg = " ".join(str(x) for x in a)
        print(msg); lines.append(msg)

    P("=" * 60)
    P("  BOOTSTRAP 95% CI — HF MODEL baseline→war distress shift")
    P("=" * 60)
    P(f"  n per group:       {SAMPLE_N}")
    P(f"  bootstrap samples: {N_BOOTSTRAP}")
    P(f"  seed:              {SEED}")
    P("")
    P(f"  Strict distress (sad+fear+anger):")
    P(f"    point estimate:   {pt_strict:+.2f} pp")
    P(f"    95% CI:           [{ci_strict[0]:+.2f}, {ci_strict[1]:+.2f}] pp")
    P(f"    contains zero:    {'YES' if ci_strict[0] <= 0 <= ci_strict[1] else 'NO'}")
    P("")
    P(f"  Expanded distress (+disgust):")
    P(f"    point estimate:   {pt_exp:+.2f} pp")
    P(f"    95% CI:           [{ci_exp[0]:+.2f}, {ci_exp[1]:+.2f}] pp")
    P(f"    contains zero:    {'YES' if ci_exp[0] <= 0 <= ci_exp[1] else 'NO'}")
    P("=" * 60)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {OUT_FILE}")

    # Return for downstream use
    return pt_strict, ci_strict, pt_exp, ci_exp


if __name__ == "__main__":
    main()
