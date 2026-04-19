"""
Reddit Baseline Data Fetcher — True Pre-War Peaceful Baseline
DSS5104 — Mental Health Analysis Project

Scrapes Reddit posts + comments from April 15, 2025 to June 10, 2025
— a 57-day window ending 3 days BEFORE the June 13, 2025 Israeli
strike on Iranian nuclear sites (Twelve-Day War). This captures a
TRUE peaceful baseline uncontaminated by either the June 2025 or
Feb 2026 Iran-Israel wars.

Previously this script used Dec 2025 - Jan 2026, but that window
sits only 5-6 months after the Twelve-Day War ceasefire (June 24,
2025) and is therefore contaminated by residual post-war discourse.

Output: reddit_baseline_posts.csv, reddit_baseline_comments.csv
"""

import requests
import pandas as pd
import csv
import os
import time
from datetime import datetime, timezone

HEADERS = {"User-Agent": "DSS5104-research/0.1"}

# Same subreddits as the main pipeline
SUBREDDITS = ["worldnews", "geopolitics", "iran", "israel", "middleeast",
              "mentalhealth", "anxiety"]

# BROADER keywords — capture general discourse, not just conflict
# For news subs: general Middle East / geopolitical topics
# For mental health subs: general mental health discussion
KEYWORDS_NEWS = [
    "Iran",
    "Israel",
    "Middle East",
    "Iran nuclear",
    "geopolitics",
]
KEYWORDS_MENTAL_HEALTH = [
    "anxiety",
    "mental health",
    "depression",
    "stress",
    "feeling",
]

# Baseline period: Apr 15, 2025 to Jun 11, 2025
# (ends 2 days before June 13, 2025 Israeli strike on Iran nuclear sites)
DATE_START = datetime(2025, 4, 15, tzinfo=timezone.utc)
DATE_END   = datetime(2025, 6, 11, tzinfo=timezone.utc)  # exclusive

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
POSTS_FILE    = os.path.join(SCRIPT_DIR, "reddit_baseline_posts.csv")
COMMENTS_FILE = os.path.join(SCRIPT_DIR, "reddit_baseline_comments.csv")

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
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv(filepath, columns, rows):
    if not rows:
        return
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerows(rows)


def fetch_comments(post_id, max_retries=2):
    url = f"https://www.reddit.com/comments/{post_id}.json"
    comments = []

    for _ in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params={"limit": 500}, timeout=15)
            if r.status_code == 429:
                print(f"      Rate limited — waiting 60s...")
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
    url   = f"https://www.reddit.com/r/{subreddit}/search.json"
    after = None
    page  = 0
    saved = 0

    while page < max_pages:
        params = {
            "q": keyword,
            "restrict_sr": 1,
            "sort": "new",
            "limit": 100,
            "t": "all",
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

            page_rows = []
            hit_old = False
            for p in posts:
                d = p["data"]
                post_time = datetime.fromtimestamp(d["created_utc"], tz=timezone.utc)

                # Only keep posts within baseline window
                if post_time < DATE_START:
                    hit_old = True
                    break
                if post_time >= DATE_END:
                    continue  # too new, skip

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

            append_csv(POSTS_FILE, POST_COLUMNS, page_rows)
            saved += len(page_rows)

            after = data.get("after")
            page += 1
            print(f"    Page {page}: +{len(page_rows)} new posts (saved: {saved})")

            if not after or hit_old:
                break
            time.sleep(2)

        except Exception as e:
            print(f"    Error: {e}")
            break

    return saved


def main():
    print("Reddit Baseline Fetcher — Pre-Conflict Period")
    print(f"Date window: {DATE_START.date()} -> {DATE_END.date()} (exclusive)")
    print(f"Output dir:  {SCRIPT_DIR}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    init_csv(POSTS_FILE, POST_COLUMNS)
    init_csv(COMMENTS_FILE, COMMENT_COLUMNS)

    seen_post_ids = set()
    total_posts = 0
    total_comments = 0

    # Resume support
    if os.path.exists(POSTS_FILE) and os.path.getsize(POSTS_FILE) > 0:
        try:
            existing = pd.read_csv(POSTS_FILE, usecols=["post_id"])
            seen_post_ids = set(existing["post_id"].astype(str))
            total_posts = len(seen_post_ids)
            if total_posts > 0:
                print(f"  Resuming — {total_posts} posts already in file\n")
        except Exception:
            pass

    # Fetch posts — use appropriate keywords per subreddit type
    mental_health_subs = {"mentalhealth", "anxiety"}

    for sub in SUBREDDITS:
        keywords = KEYWORDS_MENTAL_HEALTH if sub in mental_health_subs else KEYWORDS_NEWS
        for kw in keywords:
            print(f"  r/{sub} + '{kw}'")
            saved = fetch_and_save_posts(sub, kw, seen_post_ids, max_pages=10)
            total_posts += saved
            print(f"  -> {saved} new posts (total: {total_posts})\n")
            time.sleep(2)

    print(f"\nTotal baseline posts: {total_posts}")

    if total_posts == 0:
        print("No posts found. Exiting.")
        return

    # Fetch comments
    BATCH_SIZE = 10
    print(f"\nFetching comments for {total_posts} posts...")

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

        if posts_in_buffer >= BATCH_SIZE:
            append_csv(COMMENTS_FILE, COMMENT_COLUMNS, comment_buffer)
            total_comments += len(comment_buffer)
            print(f"    [FLUSH] Wrote {len(comment_buffer)} comments (total: {total_comments})")
            comment_buffer = []
            posts_in_buffer = 0

    if comment_buffer:
        append_csv(COMMENTS_FILE, COMMENT_COLUMNS, comment_buffer)
        total_comments += len(comment_buffer)
        print(f"    [FLUSH] Wrote final {len(comment_buffer)} comments (total: {total_comments})")

    print("\n" + "=" * 55)
    print("  BASELINE DATA SUMMARY")
    print("=" * 55)
    print(f"\n  Period:       Apr 15, 2025 — Jun 10, 2025 (pre-Twelve-Day-War)")
    print(f"  Posts file:   {POSTS_FILE}")
    print(f"  Comments:     {COMMENTS_FILE}")
    print(f"  Total posts:  {total_posts}")
    print(f"  Total comments: {total_comments}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
