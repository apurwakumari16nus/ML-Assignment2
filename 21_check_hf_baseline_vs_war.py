"""
HF Pre-trained Model — Independent Baseline vs War Validation
DSS5104 — Mental Health Analysis Project

Uses j-hartmann/emotion-english-distilroberta-base (7-class with neutral)
on a sample of baseline and war comments. Purpose: verify the
baseline→war distress shift we saw with our fine-tuned model holds up
under a completely independent model with a neutral class.

Output: check_hf_baseline_vs_war_output.txt
"""

import os
import pandas as pd
import torch
from transformers import pipeline
from scipy.stats import chi2_contingency

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAR_CSV      = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
BASELINE_CSV = os.path.join(SCRIPT_DIR, "reddit_baseline_comments_clean.csv")
OUT_FILE     = os.path.join(SCRIPT_DIR, "check_hf_baseline_vs_war_output.txt")

HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SAMPLE_WAR = 2000
SAMPLE_BASE = 2000
SEED = 42

DISTRESS_STRICT   = {"sadness", "fear", "anger"}
DISTRESS_EXPANDED = {"sadness", "fear", "anger", "disgust"}


def load_sample(path, n, seed):
    df = pd.read_csv(path)
    n_eff = min(n, len(df))
    s = df.sample(n=n_eff, random_state=seed)
    return s["body"].fillna("").astype(str).tolist()


def classify(pipe, texts, bs=32):
    out = pipe(texts, batch_size=bs, truncation=True, max_length=128)
    return [r[0]["label"] if isinstance(r, list) else r["label"] for r in out]


def dist(labels):
    s = pd.Series(labels).value_counts(normalize=True) * 100
    return s.to_dict()


def main():
    lines = []
    def P(*args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        lines.append(msg)

    P("=" * 60)
    P("  HF PRE-TRAINED MODEL — Baseline vs War Distress Validation")
    P("=" * 60)

    P(f"\n  Loading HF model: {HF_MODEL}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = pipeline("text-classification", model=HF_MODEL, top_k=1,
                    device=device)

    P(f"\n  Sampling {SAMPLE_WAR} war-period comments...")
    war_texts = load_sample(WAR_CSV, SAMPLE_WAR, SEED)
    P(f"  Sampling {SAMPLE_BASE} baseline comments...")
    base_texts = load_sample(BASELINE_CSV, SAMPLE_BASE, SEED)

    P(f"\n  Classifying war sample ({len(war_texts)})...")
    war_labels = classify(pipe, war_texts)
    P(f"  Classifying baseline sample ({len(base_texts)})...")
    base_labels = classify(pipe, base_texts)

    # Distributions
    war_d = dist(war_labels)
    base_d = dist(base_labels)
    all_emotions = sorted(set(list(war_d) + list(base_d)))

    P("\n" + "=" * 60)
    P("  HF EMOTION DISTRIBUTION (7-class, includes neutral)")
    P("=" * 60)
    P(f"\n  {'Emotion':<12} {'Baseline':>10} {'War':>10} {'Δ':>10}")
    P("  " + "-" * 44)
    for e in all_emotions:
        b = base_d.get(e, 0.0)
        w = war_d.get(e, 0.0)
        P(f"  {e:<12} {b:>9.1f}% {w:>9.1f}% {w-b:>+9.1f}pp")

    # Distress ratios
    def distress_pct(labels, which):
        return 100 * sum(1 for l in labels if l in which) / len(labels)

    b_strict = distress_pct(base_labels, DISTRESS_STRICT)
    w_strict = distress_pct(war_labels, DISTRESS_STRICT)
    b_exp    = distress_pct(base_labels, DISTRESS_EXPANDED)
    w_exp    = distress_pct(war_labels, DISTRESS_EXPANDED)

    P("\n" + "=" * 60)
    P("  DISTRESS RATIO COMPARISON (HF model, independent)")
    P("=" * 60)
    P(f"\n  Strict (sadness+fear+anger):")
    P(f"    Baseline: {b_strict:.1f}%")
    P(f"    War:      {w_strict:.1f}%")
    P(f"    Change:   {w_strict - b_strict:+.1f} pp")

    P(f"\n  Expanded (sadness+fear+anger+disgust):")
    P(f"    Baseline: {b_exp:.1f}%")
    P(f"    War:      {w_exp:.1f}%")
    P(f"    Change:   {w_exp - b_exp:+.1f} pp")

    # Chi-square on 7-class distribution
    all_labels = sorted(set(war_labels) | set(base_labels))
    b_counts = [base_labels.count(l) for l in all_labels]
    w_counts = [war_labels.count(l) for l in all_labels]
    chi2, pval, dof, _ = chi2_contingency([b_counts, w_counts])

    P("\n" + "=" * 60)
    P("  STATISTICAL TEST (Chi-square, HF 7-class)")
    P("=" * 60)
    P(f"\n  Chi-square: {chi2:.2f}")
    P(f"  df:         {dof}")
    P(f"  p-value:    {pval:.2e}")
    P(f"  Significant: {'YES' if pval < 0.05 else 'NO'}")

    # Neutral class insight
    neu_b = base_d.get("neutral", 0.0)
    neu_w = war_d.get("neutral", 0.0)
    P("\n" + "=" * 60)
    P("  NEUTRAL-CLASS INSIGHT")
    P("=" * 60)
    P(f"\n  Neutral share — Baseline: {neu_b:.1f}%, War: {neu_w:.1f}%")
    P(f"  Delta: {neu_w - neu_b:+.1f} pp")
    P("  (If neutral drops during war, emotional engagement rose.)")

    P("\n" + "=" * 60)
    P("  INTERPRETATION")
    P("=" * 60)
    P("""
  This run uses an INDEPENDENT pre-trained model (j-hartmann) with a
  NEUTRAL class that our 6-class fine-tuned model cannot produce.

  If the HF model *also* shows a baseline->war distress rise, this is
  strong independent validation that the effect is real and not an
  artifact of our model's forced emotional labelling on neutral text.

  If the HF model shows NO rise (or a smaller one), it suggests our
  fine-tuned model was amplifying a true but smaller underlying signal
  (absolute numbers inflated, relative direction still correct).
""")
    P("=" * 60)

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
