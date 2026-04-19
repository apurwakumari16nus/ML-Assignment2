"""
Baseline Comparison — War Period vs True Pre-War Peaceful Baseline
DSS5104 — Mental Health Analysis Project

Compares emotional patterns between:
  - BASELINE: Apr 15 - Jun 10, 2025 (pre-Twelve-Day-War; truly peaceful)
  - WAR:      Feb 1, 2026 - Apr 3, 2026 (2026 Iran-Israel-US conflict)

Applies the saved BERT model to baseline comments, then compares
distress ratios, emotion distributions, and statistical significance.

Prerequisites:
  - Run 02_reddit_fetch_baseline.py (baseline data)
  - Run 03_clean_data.py (cleans both war + baseline data)
  - Saved BERT model from 05_bert_train.py

Output: charts/baseline_*.png, baseline_comparison_results.txt
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "baseline_comparison_results.txt")

# War-period data (already classified)
WAR_BERT_COMMENTS = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")

# Baseline data (cleaned by 03_clean_data.py, fallback to raw)
BASELINE_POSTS    = os.path.join(SCRIPT_DIR, "reddit_baseline_posts_clean.csv")
BASELINE_COMMENTS = os.path.join(SCRIPT_DIR, "reddit_baseline_comments_clean.csv")
BASELINE_POSTS_RAW    = os.path.join(SCRIPT_DIR, "reddit_baseline_posts.csv")
BASELINE_COMMENTS_RAW = os.path.join(SCRIPT_DIR, "reddit_baseline_comments.csv")
BASELINE_BERT_OUT = os.path.join(SCRIPT_DIR, "reddit_baseline_comments_bert.csv")

# BERT model
BERT_MODEL_DIR = os.path.join(SCRIPT_DIR, "bert_emotion_model", "final")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DISTRESS_EMOTIONS = ["sadness", "fear", "anger"]

EMOTION_COLORS = {
    "anger":    "#e74c3c",
    "fear":     "#9b59b6",
    "sadness":  "#3498db",
    "joy":      "#2ecc71",
    "love":     "#e91e63",
    "surprise": "#f39c12",
}


def clean_text(text):
    """Basic cleaning for baseline comments (mirrors 02_clean_data.py logic)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # markdown links
    text = re.sub(r"[*_~`#>]", "", text)        # markdown formatting
    text = re.sub(r"&[a-z]+;", " ", text)       # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_baseline_with_bert(df_baseline):
    """Classify baseline comments using saved BERT model."""
    if os.path.exists(BASELINE_BERT_OUT):
        print(f"  Loading cached baseline classifications from {BASELINE_BERT_OUT}")
        return pd.read_csv(BASELINE_BERT_OUT)

    if not os.path.exists(BERT_MODEL_DIR):
        print(f"ERROR: BERT model not found at {BERT_MODEL_DIR}")
        print("Run 04_bert_train.py first.")
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("  Loading BERT model...")
    tok = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    model.eval()
    print(f"  Device: {device}")

    # Clean text
    df_baseline["cleaned_body"] = df_baseline["body"].apply(clean_text)
    df_baseline = df_baseline[df_baseline["cleaned_body"].str.len() >= 10].reset_index(drop=True)

    # Filter bots and deleted
    bot_authors = {"AutoModerator", "[deleted]", "RemindMeBot", "WikiTextBot"}
    df_baseline = df_baseline[~df_baseline["author"].isin(bot_authors)].reset_index(drop=True)

    texts = df_baseline["cleaned_body"].tolist()
    all_preds = []
    all_confs = []
    batch_size = 32

    print(f"  Classifying {len(texts)} baseline comments...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()

        all_preds.extend(preds)
        all_confs.extend(confs)

        if (start // batch_size) % 50 == 0:
            print(f"    Processed {start + len(batch)}/{len(texts)}")

    df_baseline["bert_emotion"] = [LABEL_NAMES[p] for p in all_preds]
    df_baseline["bert_confidence"] = all_confs

    df_baseline.to_csv(BASELINE_BERT_OUT, index=False)
    print(f"  Saved -> {BASELINE_BERT_OUT}")

    return df_baseline


def plot_emotion_comparison(war_dist, base_dist, save_path):
    """Side-by-side bar chart of emotion distributions."""
    emotions = LABEL_NAMES
    x = np.arange(len(emotions))
    width = 0.35

    war_pcts = [war_dist.get(e, 0) for e in emotions]
    base_pcts = [base_dist.get(e, 0) for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, base_pcts, width, label="Baseline (Apr-Jun 2025)",
                   color="#3498db", edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x + width/2, war_pcts, width, label="War Period (Feb+)",
                   color="#e74c3c", edgecolor="white", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Percentage of Comments (%)")
    ax.set_title("Emotion Distribution: Pre-Conflict Baseline vs War Period",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions])
    ax.legend()
    ax.set_ylim(0, max(max(war_pcts), max(base_pcts)) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_distress_comparison(war_distress, base_distress, save_path):
    """Bar chart comparing distress ratios."""
    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(["Baseline\n(Apr-Jun 2025)", "War Period\n(Feb+ 2026)"],
                  [base_distress, war_distress],
                  color=["#3498db", "#e74c3c"], edgecolor="white", width=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Distress Ratio (%)")
    ax.set_title("Distress Ratio: Baseline vs War Period",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  BASELINE COMPARISON — Pre-Conflict vs War Period")
    log("=" * 60)

    # ── Load war-period data ──
    if not os.path.exists(WAR_BERT_COMMENTS):
        log(f"\nERROR: {WAR_BERT_COMMENTS} not found. Run the main pipeline first.")
        return

    df_war = pd.read_csv(WAR_BERT_COMMENTS)
    log(f"\n  War-period comments: {len(df_war):,}")

    # ── Load and classify baseline data ──
    # Prefer cleaned file from 03_clean_data.py, fall back to raw
    baseline_file = BASELINE_COMMENTS
    if not os.path.exists(baseline_file):
        baseline_file = BASELINE_COMMENTS_RAW
    if not os.path.exists(baseline_file):
        log(f"\nERROR: No baseline comments found.")
        log("  Run 02_reddit_fetch_baseline.py first, then 03_clean_data.py.")
        return

    log(f"\n  Loading baseline data from {os.path.basename(baseline_file)}...")
    df_base_raw = pd.read_csv(baseline_file)
    log(f"  Baseline comments: {len(df_base_raw):,}")

    log("\n  Classifying baseline comments with BERT...")
    df_base = classify_baseline_with_bert(df_base_raw)
    log(f"  Classified baseline comments: {len(df_base):,}")

    # ── Emotion distributions ──
    log(f"\n{'=' * 60}")
    log("  EMOTION DISTRIBUTIONS")
    log(f"{'=' * 60}")

    war_dist = (df_war["bert_emotion"].value_counts(normalize=True) * 100).to_dict()
    base_dist = (df_base["bert_emotion"].value_counts(normalize=True) * 100).to_dict()

    log(f"\n  {'Emotion':<12} {'Baseline %':>12} {'War %':>12} {'Change':>12}")
    log(f"  {'─' * 48}")
    for emo in LABEL_NAMES:
        bp = base_dist.get(emo, 0)
        wp = war_dist.get(emo, 0)
        change = wp - bp
        arrow = "+" if change > 0 else ""
        log(f"  {emo:<12} {bp:>11.1f}% {wp:>11.1f}% {arrow}{change:>10.1f}pp")

    # ── Distress ratio ──
    war_distress = df_war["bert_emotion"].isin(DISTRESS_EMOTIONS).mean() * 100
    base_distress = df_base["bert_emotion"].isin(DISTRESS_EMOTIONS).mean() * 100

    log(f"\n  Distress Ratio (sadness + fear + anger):")
    log(f"    Baseline: {base_distress:.1f}%")
    log(f"    War:      {war_distress:.1f}%")
    log(f"    Change:   {'+' if war_distress > base_distress else ''}{war_distress - base_distress:.1f} percentage points")

    # ── Statistical test ──
    log(f"\n{'=' * 60}")
    log("  STATISTICAL SIGNIFICANCE")
    log(f"{'=' * 60}")

    # Chi-square test on emotion distributions
    war_counts = df_war["bert_emotion"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    base_counts = df_base["bert_emotion"].value_counts().reindex(LABEL_NAMES, fill_value=0)

    # Normalize to same total for chi-square
    contingency = pd.DataFrame({
        "baseline": base_counts,
        "war": war_counts,
    })
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency.T)

    log(f"\n  Chi-square test (baseline vs war emotion distributions):")
    log(f"    Chi-square statistic: {chi2:.2f}")
    log(f"    Degrees of freedom:   {dof}")
    log(f"    p-value:              {p_value:.2e}")
    log(f"    Significant at 0.05:  {'YES' if p_value < 0.05 else 'NO'}")

    # Mann-Whitney U on distress (binary: distress vs non-distress)
    war_binary = df_war["bert_emotion"].isin(DISTRESS_EMOTIONS).astype(int).values
    base_binary = df_base["bert_emotion"].isin(DISTRESS_EMOTIONS).astype(int).values

    u_stat, u_pval = stats.mannwhitneyu(war_binary, base_binary, alternative="two-sided")
    log(f"\n  Mann-Whitney U test (distress proportion):")
    log(f"    U statistic: {u_stat:,.0f}")
    log(f"    p-value:     {u_pval:.2e}")
    log(f"    Significant: {'YES' if u_pval < 0.05 else 'NO'}")

    # ── Per-subreddit comparison ──
    if "subreddit" in df_base.columns:
        log(f"\n{'=' * 60}")
        log("  PER-SUBREDDIT DISTRESS COMPARISON")
        log(f"{'=' * 60}")

        # Need to get subreddit for war comments via posts
        war_posts_file = os.path.join(SCRIPT_DIR, "reddit_posts_bert.csv")
        if os.path.exists(war_posts_file):
            war_posts = pd.read_csv(war_posts_file, usecols=["post_id", "subreddit"])
            df_war_sub = df_war.merge(war_posts[["post_id", "subreddit"]], on="post_id", how="left")
        else:
            df_war_sub = df_war

        if "subreddit" in df_war_sub.columns:
            # Get subreddit for baseline via posts
            if os.path.exists(BASELINE_POSTS):
                base_posts = pd.read_csv(BASELINE_POSTS, usecols=["post_id", "subreddit"])
                df_base_sub = df_base.merge(base_posts[["post_id", "subreddit"]], on="post_id", how="left")
            else:
                df_base_sub = df_base

            if "subreddit" in df_base_sub.columns:
                log(f"\n  {'Subreddit':<16} {'Base Distress':>14} {'War Distress':>14} {'Change':>10}")
                log(f"  {'─' * 54}")

                for sub in sorted(set(df_war_sub["subreddit"].dropna().unique()) |
                                  set(df_base_sub["subreddit"].dropna().unique())):
                    war_sub = df_war_sub[df_war_sub["subreddit"] == sub]
                    base_sub = df_base_sub[df_base_sub["subreddit"] == sub]

                    if len(war_sub) < 10 or len(base_sub) < 10:
                        continue

                    wd = war_sub["bert_emotion"].isin(DISTRESS_EMOTIONS).mean() * 100
                    bd = base_sub["bert_emotion"].isin(DISTRESS_EMOTIONS).mean() * 100
                    change = wd - bd
                    log(f"  {sub:<16} {bd:>13.1f}% {wd:>13.1f}% {'+' if change > 0 else ''}{change:>8.1f}pp")

    # ── Charts ──
    log(f"\n{'=' * 60}")
    log("  GENERATING CHARTS")
    log(f"{'=' * 60}")

    plot_emotion_comparison(war_dist, base_dist,
                            os.path.join(CHARTS_DIR, "baseline_emotion_comparison.png"))
    plot_distress_comparison(war_distress, base_distress,
                            os.path.join(CHARTS_DIR, "baseline_distress_comparison.png"))

    # ── Summary ──
    log(f"\n{'=' * 60}")
    log("  KEY FINDINGS")
    log(f"{'=' * 60}")
    log(f"""
  1. Baseline distress ratio: {base_distress:.1f}%
     War-period distress ratio: {war_distress:.1f}%
     Change: {'+' if war_distress > base_distress else ''}{war_distress - base_distress:.1f} percentage points

  2. The emotional distribution shift is {'statistically significant' if p_value < 0.05 else 'NOT statistically significant'}
     (Chi-square p = {p_value:.2e})

  3. This confirms that the war period shows {'elevated' if war_distress > base_distress else 'reduced'}
     emotional distress compared to the pre-conflict baseline.

  NOTE: Baseline data uses broader keywords than war-period data.
  The baseline captures general subreddit discourse, while war data
  is specifically filtered for conflict-related content. This means
  some of the distress difference may reflect topic selection rather
  than a true population-level emotional shift.
""")

    log("=" * 60)

    # Save
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
