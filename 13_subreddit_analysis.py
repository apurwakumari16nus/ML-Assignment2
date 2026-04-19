"""
Subreddit Analysis — Cross-Community Emotion Comparison
DSS5104 — Mental Health Analysis Project (Step 7)

Compares how different subreddit communities react emotionally.
Is r/mentalhealth more distressed than r/worldnews?

Input:  reddit_comments_bert.csv, reddit_posts_bert.csv
Output: charts saved as PNG in charts/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
COMMENTS_FILE = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")
POSTS_FILE    = os.path.join(SCRIPT_DIR, "reddit_posts_bert.csv")
CHARTS_DIR    = os.path.join(SCRIPT_DIR, "charts")

DISTRESS_EMOTIONS = ["sadness", "fear", "anger"]
EMOTION_ORDER = ["anger", "fear", "sadness", "surprise", "joy", "love"]
EMOTION_COLORS = {
    "anger":    "#e74c3c",
    "fear":     "#9b59b6",
    "sadness":  "#3498db",
    "joy":      "#2ecc71",
    "love":     "#e91e63",
    "surprise": "#f39c12",
}


def load_data():
    """Load BERT-classified comments and posts, merge subreddit info."""
    print("Loading data...")

    df_comments = pd.read_csv(COMMENTS_FILE)
    print(f"  Comments: {len(df_comments):,}")

    try:
        df_posts = pd.read_csv(POSTS_FILE)
        print(f"  Posts:    {len(df_posts):,}")
    except FileNotFoundError:
        df_posts = pd.DataFrame()
        print("  Posts file not found — analyzing comments only.")

    # Comments may not have subreddit — get it from posts via post_id
    if "subreddit" not in df_comments.columns and not df_posts.empty:
        post_subs = df_posts[["post_id", "subreddit"]].drop_duplicates("post_id")
        df_comments = df_comments.merge(post_subs, on="post_id", how="left")
        print(f"  Merged subreddit info from posts -> {df_comments['subreddit'].notna().sum():,} matched")

    return df_comments, df_posts


def plot_emotion_heatmap(df, save_path):
    """Heatmap of emotion percentages per subreddit."""
    # Cross-tab: subreddit x emotion
    ct = pd.crosstab(df["subreddit"], df["bert_emotion"], normalize="index") * 100

    # Reorder columns
    emotions = [e for e in EMOTION_ORDER if e in ct.columns]
    ct = ct[emotions]

    # Sort subreddits by distress ratio (most distressed on top)
    distress_cols = [e for e in DISTRESS_EMOTIONS if e in ct.columns]
    ct["_distress"] = ct[distress_cols].sum(axis=1)
    ct = ct.sort_values("_distress", ascending=True)
    ct = ct.drop(columns=["_distress"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(ct) * 0.6)))

    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")

    # Labels
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels([e.capitalize() for e in ct.columns], fontsize=10)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([f"r/{s}" for s in ct.index], fontsize=10)

    # Add percentage text in each cell
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Emotion Distribution by Subreddit (%)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Percentage", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_distress_comparison(df, save_path):
    """Bar chart comparing distress ratio across subreddits."""
    df["is_distress"] = df["bert_emotion"].isin(DISTRESS_EMOTIONS)

    sub_stats = df.groupby("subreddit").agg(
        total=("is_distress", "count"),
        distress_pct=("is_distress", "mean"),
    )
    sub_stats["distress_pct"] *= 100
    sub_stats = sub_stats.sort_values("distress_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub_stats) * 0.5)))

    bars = ax.barh(
        [f"r/{s}" for s in sub_stats.index],
        sub_stats["distress_pct"],
        color=["#e74c3c" if pct > 60 else "#f39c12" if pct > 40 else "#2ecc71"
               for pct in sub_stats["distress_pct"]],
        edgecolor="white",
    )

    # Add value labels
    for bar, (_, row) in zip(bars, sub_stats.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{row['distress_pct']:.1f}% (n={int(row['total'])})",
                va="center", fontsize=9)

    ax.set_xlabel("Distress Ratio (sadness + fear + anger) %")
    ax.set_title("Distress Ratio by Subreddit", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_stacked_emotions(df, save_path):
    """Stacked horizontal bar chart of emotion breakdown per subreddit."""
    ct = pd.crosstab(df["subreddit"], df["bert_emotion"], normalize="index") * 100
    emotions = [e for e in EMOTION_ORDER if e in ct.columns]
    ct = ct[emotions]

    # Sort by anger (most angry on top)
    ct = ct.sort_values("anger", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(ct) * 0.5)))

    left = np.zeros(len(ct))
    for emotion in emotions:
        vals = ct[emotion].values
        ax.barh([f"r/{s}" for s in ct.index], vals, left=left,
                color=EMOTION_COLORS[emotion], label=emotion.capitalize(),
                edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_xlabel("Percentage (%)")
    ax.set_title("Emotion Breakdown by Subreddit", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_comment_volume(df, save_path):
    """Bar chart of comment volume per subreddit with avg score."""
    sub_stats = df.groupby("subreddit").agg(
        comments=("comment_id", "count"),
        avg_score=("score", "mean"),
    ).sort_values("comments", ascending=True)

    fig, ax1 = plt.subplots(figsize=(10, max(4, len(sub_stats) * 0.5)))

    ax1.barh([f"r/{s}" for s in sub_stats.index], sub_stats["comments"],
             color="steelblue", edgecolor="white")

    for i, (_, row) in enumerate(sub_stats.iterrows()):
        ax1.text(row["comments"] + max(sub_stats["comments"]) * 0.01, i,
                 f"{int(row['comments']):,}", va="center", fontsize=9)

    ax1.set_xlabel("Number of Comments")
    ax1.set_title("Comment Volume by Subreddit", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def print_summary(df):
    """Print detailed subreddit comparison."""
    print("\n" + "=" * 60)
    print("  SUBREDDIT ANALYSIS SUMMARY")
    print("=" * 60)

    df["is_distress"] = df["bert_emotion"].isin(DISTRESS_EMOTIONS)

    sub_stats = df.groupby("subreddit").agg(
        comments=("bert_emotion", "count"),
        distress_pct=("is_distress", "mean"),
        dominant=("bert_emotion", lambda x: x.value_counts().index[0]),
        avg_confidence=("bert_confidence", "mean"),
    )
    sub_stats["distress_pct"] *= 100
    sub_stats = sub_stats.sort_values("distress_pct", ascending=False)

    print(f"\n  {'Subreddit':<22} {'Comments':>8} {'Distress%':>10} {'Dominant':<10} {'Confidence':>10}")
    print("  " + "-" * 62)
    for sub, row in sub_stats.iterrows():
        print(f"  r/{sub:<20} {int(row['comments']):>8,} {row['distress_pct']:>9.1f}% "
              f"{row['dominant']:<10} {row['avg_confidence']:>9.3f}")

    # Key finding
    most_distressed = sub_stats.index[0]
    least_distressed = sub_stats.index[-1]
    print(f"\n  Most distressed:  r/{most_distressed} ({sub_stats.loc[most_distressed, 'distress_pct']:.1f}%)")
    print(f"  Least distressed: r/{least_distressed} ({sub_stats.loc[least_distressed, 'distress_pct']:.1f}%)")

    # Mental health subs vs news subs
    mh_subs = ["mentalhealth", "anxiety"]
    news_subs = ["worldnews", "geopolitics"]

    mh_data = df[df["subreddit"].isin(mh_subs)]
    news_data = df[df["subreddit"].isin(news_subs)]

    if len(mh_data) > 0 and len(news_data) > 0:
        mh_distress = 100 * mh_data["is_distress"].mean()
        news_distress = 100 * news_data["is_distress"].mean()
        print(f"\n  Mental health subs distress: {mh_distress:.1f}%")
        print(f"  News subs distress:          {news_distress:.1f}%")
        if mh_distress > news_distress:
            print(f"  -> Mental health subs show {mh_distress - news_distress:.1f}pp MORE distress")
        else:
            print(f"  -> News subs show {news_distress - mh_distress:.1f}pp MORE distress")

    print("\n" + "=" * 60)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df_comments, df_posts = load_data()

    print("\nGenerating subreddit charts...")
    plot_emotion_heatmap(df_comments, os.path.join(CHARTS_DIR, "subreddit_emotion_heatmap.png"))
    plot_distress_comparison(df_comments, os.path.join(CHARTS_DIR, "subreddit_distress_comparison.png"))
    plot_stacked_emotions(df_comments, os.path.join(CHARTS_DIR, "subreddit_emotion_stacked.png"))
    plot_comment_volume(df_comments, os.path.join(CHARTS_DIR, "subreddit_comment_volume.png"))

    print_summary(df_comments)


if __name__ == "__main__":
    main()
