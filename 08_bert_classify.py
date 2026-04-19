"""
BERT Emotion Classification — Apply to Reddit Data
DSS5104 — Mental Health Analysis Project (Step 4b)

Uses the fine-tuned BERT model (from bert_train.py) to classify
cleaned Reddit posts and comments into 6 emotion categories:
  sadness, joy, love, anger, fear, surprise

Processes in batches to handle large datasets without memory issues.

Output: reddit_posts_bert.csv, reddit_comments_bert.csv

Install (same as training):
  pip install transformers torch
"""

import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(SCRIPT_DIR, "bert_emotion_model", "final")
POSTS_CLEAN     = os.path.join(SCRIPT_DIR, "reddit_posts_clean.csv")
COMMENTS_CLEAN  = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
POSTS_OUT       = os.path.join(SCRIPT_DIR, "reddit_posts_bert.csv")
COMMENTS_OUT    = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")

BATCH_SIZE     = 32       # texts per batch for inference
MAX_SEQ_LENGTH = 128


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon GPU
    return torch.device("cpu")


def load_model():
    """Load the fine-tuned model, tokenizer, and label map."""
    print(f"Loading model from {MODEL_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    # Keys are strings from JSON, convert to int
    label_map = {int(k): v for k, v in label_map.items()}

    device = get_device()
    model.to(device)
    model.eval()

    print(f"  Device: {device}")
    print(f"  Labels: {list(label_map.values())}")

    return model, tokenizer, label_map, device


def classify_batch(texts, model, tokenizer, label_map, device):
    """
    Classify a list of text strings.
    Returns list of dicts with emotion label and confidence scores.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
    pred_ids = probs.argmax(axis=1)

    results = []
    for i in range(len(texts)):
        results.append({
            "bert_emotion":     label_map[int(pred_ids[i])],
            "bert_confidence":  float(probs[i][pred_ids[i]]),
            "bert_sadness":     float(probs[i][0]),
            "bert_joy":         float(probs[i][1]),
            "bert_love":        float(probs[i][2]),
            "bert_anger":       float(probs[i][3]),
            "bert_fear":        float(probs[i][4]),
            "bert_surprise":    float(probs[i][5]),
        })

    return results


def classify_dataframe(df, text_column, model, tokenizer, label_map, device, desc=""):
    """
    Classify all rows in a dataframe, processing in batches.
    Returns the dataframe with BERT columns appended.
    """
    all_results = []
    texts = df[text_column].fillna("").astype(str).tolist()
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]

        results = classify_batch(batch_texts, model, tokenizer, label_map, device)
        all_results.extend(results)

        if (start // BATCH_SIZE) % 50 == 0 or end == total:
            print(f"  {desc} [{end:,}/{total:,}] ({100*end/total:.0f}%)")

    bert_df = pd.DataFrame(all_results)
    return pd.concat([df.reset_index(drop=True), bert_df], axis=1)


def classify_posts(model, tokenizer, label_map, device):
    """Classify posts using title + selftext."""
    print("\nClassifying posts...")
    df = pd.read_csv(POSTS_CLEAN)
    print(f"  {len(df):,} posts loaded")

    # Combine title and body for classification
    df["text_for_bert"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["text_for_bert"] = df["text_for_bert"].str.strip()

    df = classify_dataframe(df, "text_for_bert", model, tokenizer, label_map, device, desc="Posts")
    df = df.drop(columns=["text_for_bert"])

    df.to_csv(POSTS_OUT, index=False)
    print(f"  Saved -> {POSTS_OUT}")
    return df


def classify_comments(model, tokenizer, label_map, device):
    """Classify comments using the body text."""
    print("\nClassifying comments...")
    try:
        df = pd.read_csv(COMMENTS_CLEAN)
    except FileNotFoundError:
        print("  No cleaned comments file found. Skipping.")
        return pd.DataFrame()

    print(f"  {len(df):,} comments loaded")

    df = classify_dataframe(df, "body", model, tokenizer, label_map, device, desc="Comments")

    df.to_csv(COMMENTS_OUT, index=False)
    print(f"  Saved -> {COMMENTS_OUT}")
    return df


def print_summary(df_posts, df_comments):
    """Print emotion distribution summary."""
    print("\n" + "=" * 60)
    print("  BERT EMOTION CLASSIFICATION SUMMARY")
    print("=" * 60)

    for name, df in [("Posts", df_posts), ("Comments", df_comments)]:
        if df.empty:
            continue

        print(f"\n── {name} ({len(df):,} total) ──")

        # Emotion distribution
        dist = df["bert_emotion"].value_counts()
        print(f"\n  Emotion distribution:")
        for emotion, count in dist.items():
            pct = 100 * count / len(df)
            bar = "#" * int(pct / 2)
            print(f"    {emotion:<10} {count:>6,}  ({pct:5.1f}%)  {bar}")

        # Mean confidence
        print(f"\n  Mean confidence: {df['bert_confidence'].mean():.3f}")

        # Emotion by subreddit (for posts) or overall (for comments)
        if "subreddit" in df.columns:
            print(f"\n  Dominant emotion by subreddit:")
            for sub in df["subreddit"].unique():
                sub_df = df[df["subreddit"] == sub]
                top = sub_df["bert_emotion"].value_counts().index[0]
                top_pct = 100 * (sub_df["bert_emotion"] == top).mean()
                print(f"    r/{sub:<20} {top:<10} ({top_pct:.0f}%)")

    # Mental health relevance — distress ratio
    if not df_comments.empty:
        distress_emotions = ["sadness", "fear", "anger"]
        distress_count = df_comments["bert_emotion"].isin(distress_emotions).sum()
        distress_pct = 100 * distress_count / len(df_comments)
        print(f"\n── Mental Health Signal ──")
        print(f"  Distress emotions (sadness + fear + anger): "
              f"{distress_count:,} ({distress_pct:.1f}%)")
        print(f"  This is your key metric for the mental health analysis.")

    print("\n" + "=" * 60)


def main():
    print("BERT Emotion Classification — Reddit Data")
    print(f"Model: {MODEL_DIR}\n")

    if not os.path.exists(MODEL_DIR):
        print("ERROR: Fine-tuned model not found!")
        print("Run 04_bert_train.py first to train the model.")
        return

    model, tokenizer, label_map, device = load_model()

    df_posts = classify_posts(model, tokenizer, label_map, device)
    df_comments = classify_comments(model, tokenizer, label_map, device)

    print_summary(df_posts, df_comments)


if __name__ == "__main__":
    main()
