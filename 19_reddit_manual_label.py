"""
Domain Validation — Bulk Labeling + Evaluation
DSS5104 — Mental Health Analysis Project

Addresses DOMAIN MISMATCH: models trained on Saravia tweets but applied
to Reddit comments. This script samples ~200 comments and exports them
for bulk labeling (via spreadsheet or AI assistant), then evaluates.

WORKFLOW (3 steps):
  Step 1: python 13_reddit_manual_label.py --export
           → Creates reddit_unlabeled_sample.csv (200 comments)
           → Copy the CSV content, paste into Claude/ChatGPT/spreadsheet
           → Ask AI: "Label each comment with one of: sadness, joy, love,
             anger, fear, surprise. Fill the human_label column."
           → Save the result as reddit_manual_labels.csv

  Step 2: Place the labeled CSV back as reddit_manual_labels.csv
           (must have columns: comment_id, human_label)

  Step 3: python 13_reddit_manual_label.py --evaluate
           → Computes BERT / DistilBERT / TF-IDF F1 on labeled sample
           → Saves domain_validation_results.txt

Output: reddit_unlabeled_sample.csv, reddit_manual_labels.csv,
        domain_validation_results.txt
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files (model predictions on Reddit)
BERT_COMMENTS       = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")
DISTILBERT_COMMENTS = os.path.join(SCRIPT_DIR, "reddit_comments_distilbert.csv")
TFIDF_COMMENTS      = os.path.join(SCRIPT_DIR, "reddit_comments_tfidf.csv")

# Output
UNLABELED_FILE = os.path.join(SCRIPT_DIR, "reddit_unlabeled_sample.csv")
LABELS_FILE    = os.path.join(SCRIPT_DIR, "reddit_manual_labels.csv")
RESULTS_FILE   = os.path.join(SCRIPT_DIR, "domain_validation_results.txt")

VALID_EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def export_sample(n=200):
    """Sample n comments and export to CSV for bulk labeling."""
    if not os.path.exists(BERT_COMMENTS):
        print(f"ERROR: {BERT_COMMENTS} not found. Run 06_bert_classify.py first.")
        sys.exit(1)

    df = pd.read_csv(BERT_COMMENTS)
    print(f"Loaded {len(df):,} BERT-scored comments.")

    # Stratified sample: proportional to BERT emotion distribution
    # with minimum 5 per class to cover rare emotions
    sample_per_emotion = {}
    emotion_dist = df["bert_emotion"].value_counts(normalize=True)
    for emotion, pct in emotion_dist.items():
        count = max(5, int(round(n * pct)))
        sample_per_emotion[emotion] = count

    # Adjust total to match n
    total = sum(sample_per_emotion.values())
    if total > n:
        largest = max(sample_per_emotion, key=sample_per_emotion.get)
        sample_per_emotion[largest] -= (total - n)

    samples = []
    for emotion, count in sample_per_emotion.items():
        pool = df[df["bert_emotion"] == emotion]
        k = min(count, len(pool))
        samples.append(pool.sample(n=k, random_state=42))

    sampled = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

    # Get text column (could be 'body' or 'cleaned_body')
    text_col = "body" if "body" in sampled.columns else "cleaned_body"

    # Export for labeling — include comment_id, text, BERT prediction, empty label column
    export_df = pd.DataFrame({
        "row_number": range(1, len(sampled) + 1),
        "comment_id": sampled["comment_id"],
        "text": sampled[text_col].apply(lambda x: str(x)[:500]),  # truncate long ones
        "bert_prediction": sampled["bert_emotion"],
        "human_label": "",  # ← THIS IS WHAT NEEDS TO BE FILLED IN
    })

    export_df.to_csv(UNLABELED_FILE, index=False)
    print(f"\nExported {len(export_df)} comments -> {UNLABELED_FILE}")
    print(f"\nSampling breakdown (stratified by BERT prediction):")
    for emo, cnt in sampled["bert_emotion"].value_counts().items():
        print(f"  {emo:<12} {cnt:>4}")

    print(f"""
{'=' * 60}
  NEXT STEPS
{'=' * 60}

  1. Open {UNLABELED_FILE}

  2. Label the 'human_label' column. Two easy options:

     OPTION A — Use an AI assistant:
       Copy all rows and paste into Claude/ChatGPT with this prompt:
       "For each Reddit comment below, assign exactly ONE emotion label
        from: sadness, joy, love, anger, fear, surprise.
        Return a CSV with columns: comment_id, human_label"

     OPTION B — Label in a spreadsheet:
       Open the CSV in Excel/Google Sheets, fill the human_label column.

  3. Save the labeled file as: reddit_manual_labels.csv
     (must have columns: comment_id, human_label)

  4. Run: python 13_reddit_manual_label.py --evaluate

{'=' * 60}
""")


def evaluate_labels():
    """Compute F1 for BERT, DistilBERT, TF-IDF against human/AI labels."""
    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: {LABELS_FILE} not found.")
        print(f"Run --export first, label the comments, save as {LABELS_FILE}")
        sys.exit(1)

    labels = pd.read_csv(LABELS_FILE)

    # Normalize label column name (handle variations)
    label_col = None
    for col in ["human_label", "label", "emotion", "human_emotion"]:
        if col in labels.columns:
            label_col = col
            break
    if label_col is None:
        print("ERROR: No label column found. Expected 'human_label', 'label', or 'emotion'.")
        sys.exit(1)

    labels["human_label"] = labels[label_col].str.strip().str.lower()

    # Validate labels
    valid_mask = labels["human_label"].isin(VALID_EMOTIONS)
    invalid = labels[~valid_mask]
    if len(invalid) > 0:
        print(f"WARNING: {len(invalid)} rows have invalid labels (dropping them):")
        print(f"  Invalid values: {invalid['human_label'].unique()}")
    labels = labels[valid_mask].reset_index(drop=True)

    if len(labels) < 10:
        print(f"ERROR: Only {len(labels)} valid labeled comments. Need at least 10.")
        sys.exit(1)

    print(f"Evaluating on {len(labels)} labeled comments...\n")

    # Load BERT predictions for these comment_ids
    df_bert = pd.read_csv(BERT_COMMENTS)
    labels = labels.merge(
        df_bert[["comment_id", "bert_emotion"]],
        on="comment_id", how="left", suffixes=("", "_from_bert")
    )

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  DOMAIN VALIDATION — Model Performance on Reddit Comments")
    log("=" * 60)
    log(f"\n  Total labeled comments: {len(labels)}")
    log(f"  Labeling method: external (AI-assisted or manual)")
    log(f"\n  Label distribution:")
    for emo, cnt in labels["human_label"].value_counts().items():
        log(f"    {emo:<12} {cnt:>4} ({100*cnt/len(labels):.1f}%)")

    y_true = labels["human_label"].values

    # ── BERT ──
    log(f"\n{'─' * 50}")
    log("  BERT vs Human/AI Labels:")
    log(f"{'─' * 50}")
    if "bert_emotion" in labels.columns:
        y_bert = labels["bert_emotion"].values
        valid_idx = pd.notna(y_bert)
        if valid_idx.sum() > 0:
            y_t, y_b = y_true[valid_idx], y_bert[valid_idx]
            acc = accuracy_score(y_t, y_b)
            f1 = f1_score(y_t, y_b, average="weighted", zero_division=0)
            log(f"    Accuracy: {acc:.4f} ({100*acc:.1f}%)")
            log(f"    F1 (weighted): {f1:.4f}")
            log(f"    Saravia test F1: 0.9282  |  Domain gap: {0.9282 - f1:+.4f}")
            log(f"\n    Classification Report:")
            report = classification_report(y_t, y_b, zero_division=0)
            for line in report.split("\n"):
                log(f"    {line}")

    # ── DistilBERT ──
    if os.path.exists(DISTILBERT_COMMENTS):
        log(f"\n{'─' * 50}")
        log("  DistilBERT vs Human/AI Labels:")
        log(f"{'─' * 50}")
        df_db = pd.read_csv(DISTILBERT_COMMENTS)
        merged = labels.merge(df_db[["comment_id", "distilbert_emotion"]], on="comment_id", how="left")
        valid = merged.dropna(subset=["distilbert_emotion"])
        if len(valid) > 0:
            y_db = valid["distilbert_emotion"].values
            y_t = valid["human_label"].values
            acc = accuracy_score(y_t, y_db)
            f1 = f1_score(y_t, y_db, average="weighted", zero_division=0)
            log(f"    Accuracy: {acc:.4f} ({100*acc:.1f}%)")
            log(f"    F1 (weighted): {f1:.4f}")
            log(f"    Saravia test F1: 0.9332  |  Domain gap: {0.9332 - f1:+.4f}")
            log(f"\n    Classification Report:")
            report = classification_report(y_t, y_db, zero_division=0)
            for line in report.split("\n"):
                log(f"    {line}")

    # ── TF-IDF ──
    if os.path.exists(TFIDF_COMMENTS):
        log(f"\n{'─' * 50}")
        log("  TF-IDF vs Human/AI Labels:")
        log(f"{'─' * 50}")
        df_tf = pd.read_csv(TFIDF_COMMENTS)
        merged = labels.merge(df_tf[["comment_id", "tfidf_emotion"]], on="comment_id", how="left")
        valid = merged.dropna(subset=["tfidf_emotion"])
        if len(valid) > 0:
            y_tf = valid["tfidf_emotion"].values
            y_t = valid["human_label"].values
            acc = accuracy_score(y_t, y_tf)
            f1 = f1_score(y_t, y_tf, average="weighted", zero_division=0)
            log(f"    Accuracy: {acc:.4f} ({100*acc:.1f}%)")
            log(f"    F1 (weighted): {f1:.4f}")
            log(f"    Saravia test F1: 0.8189  |  Domain gap: {0.8189 - f1:+.4f}")
            log(f"\n    Classification Report:")
            report = classification_report(y_t, y_tf, zero_division=0)
            for line in report.split("\n"):
                log(f"    {line}")

    # ── Summary ──
    log(f"\n{'=' * 60}")
    log("  INTERPRETATION")
    log(f"{'=' * 60}")
    log("""
  These F1 scores represent model performance on the TARGET DOMAIN
  (Reddit comments about Iran-Israel conflict), not the training
  domain (Saravia tweets).

  Domain Gap = Saravia F1 - Reddit F1
    > 0 means the model performs WORSE on Reddit (expected)
    ≈ 0 means the model generalizes well
    < 0 means the model performs BETTER on Reddit (unlikely)

  A domain gap > 5 F1 points suggests significant distribution shift.
  A domain gap > 10 points warrants caution in interpreting results.

  NOTE: If labels were generated by an AI assistant, this is
  "LLM-as-judge" validation. While not equivalent to expert human
  annotation, it provides a useful estimate of domain performance
  and is increasingly accepted in NLP research (Zheng et al. 2023).
""")

    log("=" * 60)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Domain validation: export comments for labeling, then evaluate")
    parser.add_argument("--export", action="store_true",
                        help="Export 200 sampled comments to CSV for bulk labeling")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate existing labels against model predictions")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of comments to sample (default: 200)")
    args = parser.parse_args()

    if not args.export and not args.evaluate:
        print("Usage:")
        print("  Step 1: python 13_reddit_manual_label.py --export")
        print("  Step 2: Label the exported CSV (via AI or manually)")
        print("  Step 3: python 13_reddit_manual_label.py --evaluate")
        print("\nRun with --export to start.")
        return

    if args.export:
        export_sample(n=args.n)

    if args.evaluate:
        evaluate_labels()


if __name__ == "__main__":
    main()
