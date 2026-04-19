"""
VADER vs BERT Comparison — Method Agreement Analysis
DSS5104 — Mental Health Analysis Project (Step 8)

Compares VADER (lexicon-based) and BERT (deep learning) sentiment results.
Shows where they agree, where they disagree, and why.

Input:  reddit_comments_scored.csv (VADER), reddit_comments_bert.csv (BERT)
Output: charts saved as PNG in charts/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VADER_FILE    = os.path.join(SCRIPT_DIR, "reddit_comments_scored.csv")
BERT_FILE     = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")
POSTS_FILE    = os.path.join(SCRIPT_DIR, "reddit_posts_bert.csv")
CHARTS_DIR    = os.path.join(SCRIPT_DIR, "charts")

# Map BERT 6 emotions to a 3-category scale for fair comparison with VADER
BERT_TO_SENTIMENT = {
    "anger":    "negative",
    "fear":     "negative",
    "sadness":  "negative",
    "joy":      "positive",
    "love":     "positive",
    "surprise": "neutral",
}


def load_and_merge():
    """Load both scored files and merge on comment_id."""
    print("Loading data...")

    df_vader = pd.read_csv(VADER_FILE)
    df_bert  = pd.read_csv(BERT_FILE)

    print(f"  VADER comments: {len(df_vader):,}")
    print(f"  BERT comments:  {len(df_bert):,}")

    # Merge on comment_id — keep only comments scored by both
    # Select only the columns we need from each to avoid conflicts
    vader_cols = ["comment_id", "vader_neg", "vader_neu", "vader_pos",
                  "vader_compound", "vader_label"]
    bert_cols  = ["comment_id", "post_id", "body", "bert_emotion", "bert_confidence",
                  "bert_sadness", "bert_joy", "bert_love",
                  "bert_anger", "bert_fear", "bert_surprise"]

    vader_cols = [c for c in vader_cols if c in df_vader.columns]
    bert_cols  = [c for c in bert_cols if c in df_bert.columns]

    df = pd.merge(df_vader[vader_cols], df_bert[bert_cols], on="comment_id", how="inner")

    # Map BERT emotions to pos/neg/neutral for comparison
    df["bert_sentiment"] = df["bert_emotion"].map(BERT_TO_SENTIMENT)

    print(f"  Merged comments: {len(df):,}")
    return df


def plot_confusion_matrix(df, save_path):
    """Confusion matrix: VADER label vs BERT sentiment."""
    labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(df["vader_label"], df["bert_sentiment"], labels=labels)

    # Normalize to percentages
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    im1 = ax1.imshow(cm, cmap="Blues")
    ax1.set_title("Agreement (counts)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    ax1.set_xticks(range(3))
    ax1.set_xticklabels([l.capitalize() for l in labels])
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([l.capitalize() for l in labels])
    ax1.set_xlabel("BERT")
    ax1.set_ylabel("VADER")

    # Percentages
    im2 = ax2.imshow(cm_pct, cmap="Blues")
    ax2.set_title("Agreement (%)", fontsize=12, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if cm_pct[i, j] > cm_pct.max() * 0.5 else "black")

    ax2.set_xticks(range(3))
    ax2.set_xticklabels([l.capitalize() for l in labels])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([l.capitalize() for l in labels])
    ax2.set_xlabel("BERT")
    ax2.set_ylabel("VADER")

    fig.suptitle("VADER vs BERT Sentiment Agreement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")

    return cm


def plot_compound_by_emotion(df, save_path):
    """Box plot of VADER compound score grouped by BERT emotion."""
    emotions = ["anger", "fear", "sadness", "surprise", "joy", "love"]
    emotions = [e for e in emotions if e in df["bert_emotion"].values]

    data = [df[df["bert_emotion"] == e]["vader_compound"].values for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(data, tick_labels=[e.capitalize() for e in emotions],
                    patch_artist=True, showfliers=False)

    colors = ["#e74c3c", "#9b59b6", "#3498db", "#f39c12", "#2ecc71", "#e91e63"]
    for patch, color in zip(bp["boxes"], colors[:len(emotions)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(0.05, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(-0.05, color="red", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_ylabel("VADER Compound Score")
    ax.set_title("VADER Compound Score by BERT Emotion Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_agreement_by_subreddit(df, save_path):
    """Bar chart of VADER-BERT agreement percentage per subreddit."""
    if "subreddit" not in df.columns:
        # Get subreddit from posts file via post_id
        try:
            df_posts = pd.read_csv(POSTS_FILE, usecols=["post_id", "subreddit"])
            df = df.merge(df_posts.drop_duplicates("post_id"), on="post_id", how="left")
        except Exception:
            print("  Cannot determine subreddits — skipping subreddit agreement chart.")
            return

    if "subreddit" not in df.columns or df["subreddit"].isna().all():
        print("  No subreddit data available — skipping subreddit agreement chart.")
        return

    df["agree"] = df["vader_label"] == df["bert_sentiment"]

    sub_agree = df.groupby("subreddit").agg(
        agreement=("agree", "mean"),
        total=("agree", "count"),
    )
    sub_agree["agreement"] *= 100
    sub_agree = sub_agree.sort_values("agreement", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub_agree) * 0.5)))

    bars = ax.barh([f"r/{s}" for s in sub_agree.index], sub_agree["agreement"],
                   color="steelblue", edgecolor="white")

    for bar, (_, row) in zip(bars, sub_agree.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{row['agreement']:.1f}% (n={int(row['total'])})",
                va="center", fontsize=9)

    ax.set_xlabel("Agreement (%)")
    ax.set_title("VADER-BERT Agreement by Subreddit", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def print_summary(df, cm):
    """Print detailed comparison summary."""
    print("\n" + "=" * 60)
    print("  VADER vs BERT COMPARISON SUMMARY")
    print("=" * 60)

    # Overall agreement
    df["agree"] = df["vader_label"] == df["bert_sentiment"]
    overall_agree = 100 * df["agree"].mean()
    print(f"\n  Overall agreement: {overall_agree:.1f}%")
    print(f"  Total comments compared: {len(df):,}")

    # Agreement by VADER label
    print(f"\n  Agreement by VADER label:")
    for label in ["negative", "neutral", "positive"]:
        subset = df[df["vader_label"] == label]
        if len(subset) > 0:
            agree = 100 * subset["agree"].mean()
            print(f"    {label:<10} {agree:.1f}% agreement  (n={len(subset):,})")

    # Agreement by BERT emotion
    print(f"\n  Agreement by BERT emotion:")
    for emotion in ["anger", "fear", "sadness", "joy", "love", "surprise"]:
        subset = df[df["bert_emotion"] == emotion]
        if len(subset) > 0:
            agree = 100 * subset["agree"].mean()
            print(f"    {emotion:<10} {agree:.1f}% agreement  (n={len(subset):,})")

    # Interesting disagreements
    print(f"\n  Notable disagreements:")

    # VADER positive but BERT says anger/fear/sadness
    false_pos = df[(df["vader_label"] == "positive") & (df["bert_sentiment"] == "negative")]
    print(f"    VADER=positive, BERT=negative: {len(false_pos):,} "
          f"({100 * len(false_pos) / len(df):.1f}%)")
    if len(false_pos) > 0:
        sample = false_pos.sample(min(3, len(false_pos)), random_state=42)
        for _, row in sample.iterrows():
            text = str(row.get("body", ""))[:80]
            print(f"      [{row['vader_compound']:+.3f} vs {row['bert_emotion']}] {text}...")

    # VADER negative but BERT says joy/love
    false_neg = df[(df["vader_label"] == "negative") & (df["bert_sentiment"] == "positive")]
    print(f"\n    VADER=negative, BERT=positive: {len(false_neg):,} "
          f"({100 * len(false_neg) / len(df):.1f}%)")
    if len(false_neg) > 0:
        sample = false_neg.sample(min(3, len(false_neg)), random_state=42)
        for _, row in sample.iterrows():
            text = str(row.get("body", ""))[:80]
            print(f"      [{row['vader_compound']:+.3f} vs {row['bert_emotion']}] {text}...")

    # Mean VADER compound per BERT emotion
    print(f"\n  Mean VADER compound by BERT emotion:")
    means = df.groupby("bert_emotion")["vader_compound"].mean().sort_values()
    for emotion, score in means.items():
        direction = "negative" if score < -0.05 else "positive" if score > 0.05 else "neutral"
        print(f"    {emotion:<10} {score:+.3f}  ({direction})")

    print("\n" + "=" * 60)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df = load_and_merge()

    print("\nGenerating comparison charts...")
    cm = plot_confusion_matrix(df, os.path.join(CHARTS_DIR, "comparison_confusion_matrix.png"))
    plot_compound_by_emotion(df, os.path.join(CHARTS_DIR, "comparison_compound_by_emotion.png"))
    plot_agreement_by_subreddit(df, os.path.join(CHARTS_DIR, "comparison_agreement_by_subreddit.png"))

    print_summary(df, cm)


if __name__ == "__main__":
    main()
