"""
VADER Sentiment Scoring — Reddit Iran-Israel-US Conflict
DSS5104 — Mental Health Analysis Project (Step 3)

Reads cleaned posts and comments, scores each text with VADER,
and saves the results with sentiment columns appended.

VADER gives 4 scores per text:
  - neg, neu, pos  (proportions that sum to 1.0)
  - compound        (-1 to +1, overall sentiment)

Output: reddit_posts_scored.csv, reddit_comments_scored.csv

Install: pip install vaderSentiment
"""

import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
POSTS_CLEAN     = os.path.join(SCRIPT_DIR, "reddit_posts_clean.csv")
COMMENTS_CLEAN  = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
POSTS_SCORED    = os.path.join(SCRIPT_DIR, "reddit_posts_scored.csv")
COMMENTS_SCORED = os.path.join(SCRIPT_DIR, "reddit_comments_scored.csv")

# Compound score thresholds (standard VADER convention)
POS_THRESHOLD =  0.05
NEG_THRESHOLD = -0.05


def score_texts(texts, analyzer):
    """
    Score a pandas Series of text strings with VADER.
    Returns a DataFrame with neg, neu, pos, compound columns.
    """
    scores = texts.fillna("").apply(lambda t: analyzer.polarity_scores(t))
    return pd.DataFrame(scores.tolist())


def classify_compound(compound):
    """Map compound score to a sentiment label."""
    if compound >= POS_THRESHOLD:
        return "positive"
    elif compound <= NEG_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def score_posts(analyzer):
    """Score posts using title + selftext combined."""
    print("Loading cleaned posts...")
    df = pd.read_csv(POSTS_CLEAN)
    print(f"  {len(df):,} posts loaded")

    # Combine title and body for scoring — title always exists, body may be empty
    df["text_for_scoring"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["text_for_scoring"] = df["text_for_scoring"].str.strip()

    print("  Scoring with VADER...")
    sentiment = score_texts(df["text_for_scoring"], analyzer)
    df["vader_neg"]      = sentiment["neg"]
    df["vader_neu"]      = sentiment["neu"]
    df["vader_pos"]      = sentiment["pos"]
    df["vader_compound"] = sentiment["compound"]
    df["vader_label"]    = df["vader_compound"].apply(classify_compound)

    # Drop the temporary scoring column
    df = df.drop(columns=["text_for_scoring"])

    df.to_csv(POSTS_SCORED, index=False)
    print(f"  Saved -> {POSTS_SCORED}")

    return df


def score_comments(analyzer):
    """Score comments using the body text."""
    print("\nLoading cleaned comments...")
    try:
        df = pd.read_csv(COMMENTS_CLEAN)
    except FileNotFoundError:
        print("  No cleaned comments file found. Skipping.")
        return pd.DataFrame()

    print(f"  {len(df):,} comments loaded")

    print("  Scoring with VADER...")
    sentiment = score_texts(df["body"], analyzer)
    df["vader_neg"]      = sentiment["neg"]
    df["vader_neu"]      = sentiment["neu"]
    df["vader_pos"]      = sentiment["pos"]
    df["vader_compound"] = sentiment["compound"]
    df["vader_label"]    = df["vader_compound"].apply(classify_compound)

    df.to_csv(COMMENTS_SCORED, index=False)
    print(f"  Saved -> {COMMENTS_SCORED}")

    return df


def print_summary(df_posts, df_comments):
    """Print sentiment distribution summary."""
    print("\n" + "=" * 60)
    print("  VADER SENTIMENT SUMMARY")
    print("=" * 60)

    if not df_posts.empty:
        print("\n── Posts ──")
        print(f"  Total: {len(df_posts):,}")
        print(f"  Mean compound score: {df_posts['vader_compound'].mean():.3f}")
        print(f"\n  Sentiment distribution:")
        dist = df_posts["vader_label"].value_counts()
        for label, count in dist.items():
            pct = 100 * count / len(df_posts)
            print(f"    {label:<10} {count:>6,}  ({pct:.1f}%)")

        print(f"\n  By subreddit (mean compound):")
        sub_scores = df_posts.groupby("subreddit")["vader_compound"].agg(["mean", "count"])
        sub_scores = sub_scores.sort_values("mean")
        for sub, row in sub_scores.iterrows():
            bar = "+" * int(max(0, row["mean"]) * 20) or "-" * int(abs(min(0, row["mean"])) * 20)
            print(f"    r/{sub:<20} {row['mean']:+.3f}  (n={int(row['count'])})")

    if not df_comments.empty:
        print("\n── Comments ──")
        print(f"  Total: {len(df_comments):,}")
        print(f"  Mean compound score: {df_comments['vader_compound'].mean():.3f}")
        print(f"\n  Sentiment distribution:")
        dist = df_comments["vader_label"].value_counts()
        for label, count in dist.items():
            pct = 100 * count / len(df_comments)
            print(f"    {label:<10} {count:>6,}  ({pct:.1f}%)")

        # Top 5 most negative comments (potential distress signals)
        print(f"\n  Most negative comments (potential distress):")
        most_neg = df_comments.nsmallest(5, "vader_compound")
        for _, row in most_neg.iterrows():
            text = str(row["body"])[:100]
            print(f"    [{row['vader_compound']:+.3f}] {text}...")

        # Top 5 most positive comments
        print(f"\n  Most positive comments:")
        most_pos = df_comments.nlargest(5, "vader_compound")
        for _, row in most_pos.iterrows():
            text = str(row["body"])[:100]
            print(f"    [{row['vader_compound']:+.3f}] {text}...")

    print("\n" + "=" * 60)


def main():
    print("VADER Sentiment Scoring")
    print(f"Input:  {POSTS_CLEAN}")
    print(f"        {COMMENTS_CLEAN}\n")

    analyzer = SentimentIntensityAnalyzer()

    df_posts = score_posts(analyzer)
    df_comments = score_comments(analyzer)

    print_summary(df_posts, df_comments)


if __name__ == "__main__":
    main()
