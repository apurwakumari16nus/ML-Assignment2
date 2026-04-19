"""
Data Cleaning — Reddit Iran-Israel-US Conflict Posts & Comments
DSS5104 — Mental Health Analysis Project

Reads the raw CSVs from reddit_fetch_all.py, cleans both posts and comments,
and saves cleaned versions ready for analysis.

Input:  reddit_iran_israel_posts.csv, reddit_iran_israel_comments.csv
Output: reddit_posts_clean.csv, reddit_comments_clean.csv
"""

import pandas as pd
import re
import os
from datetime import datetime, timezone

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
POSTS_RAW      = os.path.join(SCRIPT_DIR, "reddit_iran_israel_posts.csv")
COMMENTS_RAW   = os.path.join(SCRIPT_DIR, "reddit_iran_israel_comments.csv")
POSTS_CLEAN    = os.path.join(SCRIPT_DIR, "reddit_posts_clean.csv")
COMMENTS_CLEAN = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")

DATE_START = datetime(2026, 2, 1, tzinfo=timezone.utc)

# Markers that indicate deleted/removed content
DELETED_MARKERS = {"[deleted]", "[removed]", "[deleted by user]", ""}

# Bot accounts to exclude
BOT_AUTHORS = {"AutoModerator", "[deleted]", "None", "bot", "RemindMeBot",
               "AutoNewspaperAdmin", "SaveVideo", "stabbot"}

# Minimum text length to keep (very short = noise)
MIN_TEXT_LENGTH = 15


def load_data():
    """Load raw CSVs with proper datetime parsing."""
    print("Loading raw data...")

    df_posts = pd.read_csv(POSTS_RAW, parse_dates=["created_utc"])
    print(f"  Posts loaded:    {len(df_posts):,} rows")

    try:
        df_comments = pd.read_csv(COMMENTS_RAW, parse_dates=["created_utc"])
        print(f"  Comments loaded: {len(df_comments):,} rows")
    except FileNotFoundError:
        print("  No comments file found — cleaning posts only.")
        df_comments = pd.DataFrame()

    return df_posts, df_comments


def clean_text(text):
    """Clean a single text string — remove markdown, URLs, excess whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown links [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove markdown bold/italic markers
    text = re.sub(r"[*_]{1,3}", "", text)
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove block quotes
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Remove HTML entities
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#x200B;", "", text)  # zero-width space
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_posts(df):
    """Clean the posts dataframe."""
    print("\nCleaning posts...")
    n_start = len(df)

    # 1. Drop duplicates
    df = df.drop_duplicates("post_id")
    print(f"  After dedup:              {len(df):,}")

    # 2. Enforce date range
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df[df["created_utc"] >= DATE_START]
    print(f"  After date filter (>=Feb 2026): {len(df):,}")

    # 3. Remove deleted/bot authors
    df["author"] = df["author"].astype(str).str.strip()
    df = df[~df["author"].isin(BOT_AUTHORS)]
    print(f"  After removing bots:      {len(df):,}")

    # 4. Clean text fields
    df["title"]    = df["title"].apply(clean_text)
    df["selftext"] = df["selftext"].apply(clean_text)

    # 5. Remove posts where both title and selftext are empty/deleted
    df["selftext_clean"] = df["selftext"].replace(DELETED_MARKERS, "")
    df = df[~((df["title"].str.len() < 5) & (df["selftext_clean"].str.len() < 5))]
    print(f"  After removing empty:     {len(df):,}")

    # 6. Add useful derived columns
    df["text_length"]   = df["selftext_clean"].str.len()
    df["has_body"]      = df["text_length"] > 0
    df["date"]          = df["created_utc"].dt.date
    df["week"]          = df["created_utc"].dt.isocalendar().week.astype(int)

    # 7. Drop helper column
    df = df.drop(columns=["selftext_clean"])

    # 8. Drop the keyword column (was only for fetching)
    if "keyword" in df.columns:
        df = df.drop(columns=["keyword"])

    print(f"  Final posts: {len(df):,} (removed {n_start - len(df):,})")
    return df.reset_index(drop=True)


def clean_comments(df, valid_post_ids):
    """Clean the comments dataframe."""
    print("\nCleaning comments...")
    n_start = len(df)

    if df.empty:
        print("  No comments to clean.")
        return df

    # 1. Drop duplicates
    df = df.drop_duplicates("comment_id")
    print(f"  After dedup:              {len(df):,}")

    # 2. Keep only comments belonging to valid (cleaned) posts
    df = df[df["post_id"].isin(valid_post_ids)]
    print(f"  After matching to posts:  {len(df):,}")

    # 3. Enforce date range
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df[df["created_utc"] >= DATE_START]
    print(f"  After date filter:        {len(df):,}")

    # 4. Remove deleted/bot authors
    df["author"] = df["author"].astype(str).str.strip()
    df = df[~df["author"].isin(BOT_AUTHORS)]
    print(f"  After removing bots:      {len(df):,}")

    # 5. Clean body text
    df["body"] = df["body"].apply(clean_text)

    # 6. Remove deleted/empty/very short comments
    df = df[~df["body"].isin(DELETED_MARKERS)]
    df = df[df["body"].str.len() >= MIN_TEXT_LENGTH]
    print(f"  After removing short/empty: {len(df):,}")

    # 7. Add derived columns
    df["text_length"] = df["body"].str.len()
    df["date"]        = df["created_utc"].dt.date

    print(f"  Final comments: {len(df):,} (removed {n_start - len(df):,})")
    return df.reset_index(drop=True)


def print_summary(df_posts, df_comments):
    """Print a summary of cleaned data."""
    print("\n" + "=" * 55)
    print("  CLEANING SUMMARY")
    print("=" * 55)

    print(f"\n  Posts:    {len(df_posts):,}")
    print(f"  Comments: {len(df_comments):,}")

    if not df_posts.empty:
        print(f"\n  Date range: {df_posts['created_utc'].min().date()} -> {df_posts['created_utc'].max().date()}")
        print(f"  Subreddits: {df_posts['subreddit'].nunique()}")
        print(f"  Unique post authors:    {df_posts['author'].nunique()}")
        print(f"  Avg post score:         {df_posts['score'].mean():.1f}")

    if not df_comments.empty:
        print(f"  Unique commenters:      {df_comments['author'].nunique()}")
        print(f"  Avg comment length:     {df_comments['text_length'].mean():.0f} chars")
        print(f"  Avg comment score:      {df_comments['score'].mean():.1f}")

    if not df_posts.empty:
        print("\n  Posts by subreddit:")
        sub_stats = df_posts.groupby("subreddit").agg(
            posts=("post_id", "count"),
            avg_score=("score", "mean"),
        ).sort_values("posts", ascending=False)
        for sub, row in sub_stats.iterrows():
            print(f"    r/{sub:<20} {row['posts']:>5} posts  (avg score: {row['avg_score']:.0f})")

    print("\n" + "=" * 55)


def clean_baseline_data():
    """Clean baseline (pre-conflict) data using the same rules."""
    baseline_posts_raw = os.path.join(SCRIPT_DIR, "reddit_baseline_posts.csv")
    baseline_comments_raw = os.path.join(SCRIPT_DIR, "reddit_baseline_comments.csv")
    baseline_posts_clean = os.path.join(SCRIPT_DIR, "reddit_baseline_posts_clean.csv")
    baseline_comments_clean = os.path.join(SCRIPT_DIR, "reddit_baseline_comments_clean.csv")

    if not os.path.exists(baseline_posts_raw):
        print("\n  No baseline data found — skipping baseline cleaning.")
        return

    print("\n" + "=" * 55)
    print("  CLEANING BASELINE DATA (Apr 15 - Jun 10, 2025)")
    print("=" * 55)

    df_posts = pd.read_csv(baseline_posts_raw, parse_dates=["created_utc"])
    print(f"\n  Baseline posts loaded: {len(df_posts):,}")

    # Clean posts (same logic but no date cutoff — baseline has its own date range)
    df_posts = df_posts.drop_duplicates("post_id")
    df_posts["created_utc"] = pd.to_datetime(df_posts["created_utc"], utc=True)
    df_posts["author"] = df_posts["author"].astype(str).str.strip()
    df_posts = df_posts[~df_posts["author"].isin(BOT_AUTHORS)]
    df_posts["title"] = df_posts["title"].apply(clean_text)
    df_posts["selftext"] = df_posts["selftext"].apply(clean_text)
    df_posts["selftext_clean"] = df_posts["selftext"].replace(DELETED_MARKERS, "")
    df_posts = df_posts[~((df_posts["title"].str.len() < 5) & (df_posts["selftext_clean"].str.len() < 5))]
    df_posts = df_posts.drop(columns=["selftext_clean"])
    df_posts["date"] = df_posts["created_utc"].dt.date
    print(f"  Cleaned baseline posts: {len(df_posts):,}")

    df_posts.to_csv(baseline_posts_clean, index=False)
    print(f"  Saved -> {baseline_posts_clean}")

    # Clean comments
    if os.path.exists(baseline_comments_raw):
        df_comments = pd.read_csv(baseline_comments_raw, parse_dates=["created_utc"])
        print(f"\n  Baseline comments loaded: {len(df_comments):,}")

        df_comments = df_comments.drop_duplicates("comment_id")
        df_comments["created_utc"] = pd.to_datetime(df_comments["created_utc"], utc=True)
        df_comments = df_comments[df_comments["post_id"].isin(set(df_posts["post_id"]))]
        df_comments["author"] = df_comments["author"].astype(str).str.strip()
        df_comments = df_comments[~df_comments["author"].isin(BOT_AUTHORS)]
        df_comments["body"] = df_comments["body"].apply(clean_text)
        df_comments = df_comments[~df_comments["body"].isin(DELETED_MARKERS)]
        df_comments = df_comments[df_comments["body"].str.len() >= MIN_TEXT_LENGTH]
        df_comments["text_length"] = df_comments["body"].str.len()
        df_comments["date"] = df_comments["created_utc"].dt.date
        print(f"  Cleaned baseline comments: {len(df_comments):,}")

        df_comments.to_csv(baseline_comments_clean, index=False)
        print(f"  Saved -> {baseline_comments_clean}")


def main():
    df_posts, df_comments = load_data()

    df_posts_clean = clean_posts(df_posts)
    df_comments_clean = clean_comments(df_comments, set(df_posts_clean["post_id"]))

    # Save war-period data
    df_posts_clean.to_csv(POSTS_CLEAN, index=False)
    print(f"\nSaved -> {POSTS_CLEAN}")

    if not df_comments_clean.empty:
        df_comments_clean.to_csv(COMMENTS_CLEAN, index=False)
        print(f"Saved -> {COMMENTS_CLEAN}")

    print_summary(df_posts_clean, df_comments_clean)

    # Also clean baseline data if it exists
    clean_baseline_data()


if __name__ == "__main__":
    main()
