"""
DistilBERT Emotion Classification — Apply to Reddit Data
DSS5104 — Mental Health Analysis Project

Uses the fine-tuned DistilBERT model (from 07_distilbert_train.py)
to classify Reddit posts and comments into 6 emotion categories.

Output: reddit_posts_distilbert.csv, reddit_comments_distilbert.csv
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(SCRIPT_DIR, "distilbert_emotion_model", "final")
COMMENTS_CLEAN  = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
COMMENTS_OUT    = os.path.join(SCRIPT_DIR, "reddit_comments_distilbert.csv")

LABEL_NAMES    = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BATCH_SIZE     = 32
MAX_SEQ_LENGTH = 128


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    print("DistilBERT Emotion Classification — Reddit Data")
    print(f"Model: {MODEL_DIR}\n")

    if not os.path.exists(MODEL_DIR):
        print("ERROR: DistilBERT model not found!")
        print("Run 07_distilbert_train.py first.")
        return

    # Load model
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    device = get_device()
    model.to(device)
    model.eval()
    print(f"  Device: {device}")

    # Classify comments
    print("\nClassifying comments...")
    df = pd.read_csv(COMMENTS_CLEAN)
    print(f"  {len(df):,} comments loaded")

    texts = df["body"].fillna("").astype(str).tolist()
    all_preds = []
    all_probs = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=1))
        all_probs.extend(probs)

        done = min(start + BATCH_SIZE, len(texts))
        if (start // BATCH_SIZE) % 200 == 0:
            print(f"    [{done:,}/{len(texts):,}] ({100*done/len(texts):.0f}%)")

    all_probs = np.array(all_probs)
    df["distilbert_emotion"] = [label_map[int(p)] for p in all_preds]
    df["distilbert_confidence"] = all_probs.max(axis=1)

    for i, name in enumerate(LABEL_NAMES):
        df[f"distilbert_{name}"] = all_probs[:, i]

    df.to_csv(COMMENTS_OUT, index=False)
    print(f"\n  Saved -> {COMMENTS_OUT}")

    # Summary
    print("\n" + "=" * 60)
    print("  DISTILBERT CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"\n  Total comments: {len(df):,}")
    dist = df["distilbert_emotion"].value_counts()
    for emotion, count in dist.items():
        pct = 100 * count / len(df)
        print(f"    {emotion:<10} {count:>6,}  ({pct:.1f}%)")

    distress = df["distilbert_emotion"].isin(["sadness", "fear", "anger"]).sum()
    print(f"\n  Distress ratio: {100*distress/len(df):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
