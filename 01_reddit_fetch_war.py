"""
Reddit Full Data Fetcher — Anonymous Mode
DSS5104 — Iran-Israel-US Conflict Mental Health Project

Fetches posts AND their comments for given keywords using pagination.
Filters to Feb 2026 onwards only.
No credentials needed — uses public Reddit JSON API.

Data is written incrementally to CSV (batch append) to avoid memory issues
and to preserve progress if the script crashes or gets rate-limited.
"""

import requests
import pandas as pd
import csv
import os
import time
from datetime import datetime, timezone

HEADERS = {"User-Agent": "DSS5104-research/0.1"}

SUBREDDITS = ["worldnews", "geopolitics", "iran", "israel", "middleeast", "mentalhealth", "anxiety"]
KEYWORDS   = ["Iran Israel war", "Iran attack Israel", "Iran missile Israel", "Iran Israel conflict",
              "Iran US war", "Iran Israel US"]

# Only keep posts from Feb 1, 2026 onwards
DATE_CUTOFF = datetime(2026, 2, 1, tzinfo=timezone.utc)

# Output files — saved in the same directory as this script
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
POSTS_FILE     = os.path.join(SCRIPT_DIR, "reddit_iran_israel_posts.csv")
COMMENTS_FILE  = os.path.join(SCRIPT_DIR, "reddit_iran_israel_comments.csv")

# CSV column definitions
POST_COLUMNS = [
    "subreddit", "post_id", "title", "selftext", "score",
    "upvote_ratio", "num_comments", "created_utc", "author",
    "author_flair", "link_flair", "url", "is_self",
]
COMMENT_COLUMNS = [
    "post_id", "comment_id", "author", "body", "score",
    "created_utc", "parent_id",
]


def init_csv(filepath, columns):
    """Write CSV header if the file doesn't exist yet."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv(filepath, columns, rows):
    """Append a batch of rows to a CSV file."""
    if not rows:
        return
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerows(rows)


def fetch_comments(post_id, max_retries=2):
    """
    Fetch all top-level and nested comments for a single post.
    Returns a flat list of comment dicts.
    """
    url = f"https://www.reddit.com/comments/{post_id}.json"
    comments = []

    for _ in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params={"limit": 500}, timeout=15)

            if r.status_code == 429:
                print(f"      Rate limited on comments — waiting 60s...")
                time.sleep(60)
                continue

            r.raise_for_status()
            data = r.json()

            if len(data) < 2:
                break

            _extract_comments(data[1]["data"]["children"], post_id, comments)
            break

        except Exception as e:
            print(f"      Error fetching comments for {post_id}: {e}")
            break

    return comments


def _extract_comments(children, post_id, results):
    """Recursively flatten the comment tree."""
    for child in children:
        if child["kind"] != "t1":
            continue
        d = child["data"]
        results.append({
            "post_id":     post_id,
            "comment_id":  d["id"],
            "author":      str(d.get("author", "")),
            "body":        d.get("body", ""),
            "score":       d.get("score", 0),
            "created_utc": datetime.fromtimestamp(d["created_utc"], tz=timezone.utc).isoformat(),
            "parent_id":   d.get("parent_id", ""),
        })
        if d.get("replies") and isinstance(d["replies"], dict):
            _extract_comments(d["replies"]["data"]["children"], post_id, results)


def fetch_and_save_posts(subreddit, keyword, seen_post_ids, max_pages=10):
    """
    Fetch posts from a subreddit for a keyword using pagination.
    Writes each page directly to CSV — nothing accumulates in memory.
    Returns count of new posts saved.
    """
    url        = f"https://www.reddit.com/r/{subreddit}/search.json"
    after      = None
    page       = 0
    saved      = 0
    hit_cutoff = False

    while page < max_pages and not hit_cutoff:
        params = {
            "q":           keyword,
            "restrict_sr": 1,
            "sort":        "new",
            "limit":       100,
            "t":           "all",
        }
        if after:
            params["after"] = after

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)

            if r.status_code == 429:
                print(f"    Rate limited — waiting 60s...")
                time.sleep(60)
                continue

            r.raise_for_status()
            data  = r.json()["data"]
            posts = data["children"]

            if not posts:
                break

            # Build this page's rows, dedup, and write immediately
            page_rows = []
            for p in posts:
                d = p["data"]
                post_time = datetime.fromtimestamp(d["created_utc"], tz=timezone.utc)

                if post_time < DATE_CUTOFF:
                    hit_cutoff = True
                    break

                if d["id"] in seen_post_ids:
                    continue

                seen_post_ids.add(d["id"])
                page_rows.append({
                    "subreddit":    subreddit,
                    "post_id":      d["id"],
                    "title":        d["title"],
                    "selftext":     d.get("selftext", ""),
                    "score":        d["score"],
                    "upvote_ratio": d.get("upvote_ratio"),
                    "num_comments": d["num_comments"],
                    "created_utc":  post_time.isoformat(),
                    "author":       str(d.get("author")),
                    "author_flair": d.get("author_flair_text"),
                    "link_flair":   d.get("link_flair_text"),
                    "url":          f"https://reddit.com{d['permalink']}",
                    "is_self":      d.get("is_self"),
                })

            # Write this page to disk and free memory
            append_csv(POSTS_FILE, POST_COLUMNS, page_rows)
            saved += len(page_rows)

            after = data.get("after")
            page += 1

            print(f"    Page {page}: +{len(page_rows)} new posts (saved so far: {saved})")

            if not after:
                break

            time.sleep(2)

        except Exception as e:
            print(f"    Error: {e}")
            break

    return saved


def main():
    print("Reddit Full Fetcher — Posts + Comments (batch append mode)")
    print(f"Date filter: {DATE_CUTOFF.date()} -> now")
    print(f"Output dir:  {SCRIPT_DIR}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize CSV files with headers
    init_csv(POSTS_FILE, POST_COLUMNS)
    init_csv(COMMENTS_FILE, COMMENT_COLUMNS)

    seen_post_ids = set()
    total_posts = 0
    total_comments = 0

    # If resuming, load already-fetched post IDs to skip duplicates
    if os.path.getsize(POSTS_FILE) > 0:
        try:
            existing = pd.read_csv(POSTS_FILE, usecols=["post_id"])
            seen_post_ids = set(existing["post_id"].astype(str))
            total_posts = len(seen_post_ids)
            print(f"  Resuming — {total_posts} posts already in file\n")
        except Exception:
            pass

    # ── Step 1: Fetch posts — written per page inside the function ──────
    for sub in SUBREDDITS:
        for kw in KEYWORDS:
            print(f"  r/{sub} + '{kw}'")
            saved = fetch_and_save_posts(sub, kw, seen_post_ids, max_pages=10)
            total_posts += saved
            print(f"  -> {saved} new posts saved (total: {total_posts})\n")
            time.sleep(2)

    print(f"\nTotal unique posts: {total_posts}")

    if total_posts == 0:
        print("No posts found. Exiting.")
        return

    # ── Step 2: Fetch comments in batches of BATCH_SIZE posts ──────────
    BATCH_SIZE = 10  # flush to disk every 10 posts
    print(f"\nFetching comments for {total_posts} posts (flushing every {BATCH_SIZE} posts)...")

    # Read back all post IDs and their comment counts
    df_posts = pd.read_csv(POSTS_FILE, usecols=["post_id", "subreddit", "num_comments"])

    comment_buffer = []
    posts_in_buffer = 0

    for i, (_, row) in enumerate(df_posts.iterrows(), 1):
        if row["num_comments"] == 0:
            continue

        print(f"  [{i}/{len(df_posts)}] {row['post_id']} "
              f"(r/{row['subreddit']}, ~{row['num_comments']} comments)")

        comments = fetch_comments(row["post_id"])
        comment_buffer.extend(comments)
        posts_in_buffer += 1

        print(f"    -> {len(comments)} comments (buffer: {len(comment_buffer)})")
        time.sleep(2)

        # Flush buffer every BATCH_SIZE posts
        if posts_in_buffer >= BATCH_SIZE:
            append_csv(COMMENTS_FILE, COMMENT_COLUMNS, comment_buffer)
            total_comments += len(comment_buffer)
            print(f"    [FLUSH] Wrote {len(comment_buffer)} comments to disk (total: {total_comments})")
            comment_buffer = []
            posts_in_buffer = 0

    # Flush remaining comments
    if comment_buffer:
        append_csv(COMMENTS_FILE, COMMENT_COLUMNS, comment_buffer)
        total_comments += len(comment_buffer)
        print(f"    [FLUSH] Wrote final {len(comment_buffer)} comments to disk (total: {total_comments})")

    # ── Step 3: Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"\n  Posts file:    {POSTS_FILE}")
    print(f"  Comments file: {COMMENTS_FILE}")
    print(f"  Total posts:    {total_posts}")
    print(f"  Total comments: {total_comments}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
