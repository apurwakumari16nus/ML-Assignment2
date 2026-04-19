"""
Error Analysis — Misclassification Patterns
DSS5104 — Mental Health Analysis Project

Examines BERT and TF-IDF misclassifications on the Saravia emotion test set:
  - Samples 30 misclassified examples per model
  - Categorizes failure modes (e.g., sarcasm, mixed emotions, short text)
  - Confusion matrix heatmaps for both models
  - Per-class precision/recall comparison

This demonstrates critical thinking about model limitations.

Output: charts/error_confusion_bert.png, charts/error_confusion_tfidf.png,
        error_analysis_results.txt
"""

import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score,
)
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR   = os.path.join(SCRIPT_DIR, "charts")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "error_analysis_results.txt")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_EXAMPLES  = 30   # number of misclassified examples to display


def plot_confusion(y_true, y_pred, title, save_path):
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABEL_NAMES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels([n.capitalize() for n in LABEL_NAMES], rotation=45, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_yticklabels([n.capitalize() for n in LABEL_NAMES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 50 else "black")

    plt.colorbar(im, ax=ax, label="% of true class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def categorize_error(text, true_label, pred_label):
    """Heuristic categorization of why a misclassification may have occurred."""
    text_lower = text.lower()
    word_count = len(text.split())

    if word_count <= 5:
        return "very short text"
    if any(w in text_lower for w in ["but", "however", "though", "although"]):
        return "mixed/contrasting emotions"
    if any(w in text_lower for w in ["lol", "haha", "lmao", "jk", "sarcasm"]):
        return "sarcasm/humor"
    if "!" in text and "?" in text:
        return "ambiguous punctuation"
    if true_label in [0, 3, 4] and pred_label in [0, 3, 4]:
        return "confused negative emotions"
    if true_label in [1, 2] and pred_label in [1, 2]:
        return "confused positive emotions"
    if word_count > 30:
        return "long/complex text"
    return "other"


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  ERROR ANALYSIS — Misclassification Patterns")
    log("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────────
    log("\nLoading emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")

    texts_test = dataset["test"]["text"]
    y_test = np.array(dataset["test"]["label"])

    # ── TF-IDF + LogReg predictions ──────────────────────────────────────
    log("\nTraining TF-IDF + LogReg for error analysis...")

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_test  = vectorizer.transform(texts_test)
    y_train = np.array(dataset["train"]["label"])

    lr = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    tfidf_preds = lr.predict(X_test)

    tfidf_acc = accuracy_score(y_test, tfidf_preds)
    tfidf_f1  = f1_score(y_test, tfidf_preds, average="weighted")
    log(f"  TF-IDF acc={tfidf_acc:.4f}, F1={tfidf_f1:.4f}")

    # ── Deep learning predictions (load saved models if available) ──────
    def load_and_predict(model_dir, model_label):
        """Load a saved transformer model and predict on test set."""
        if not os.path.exists(model_dir):
            log(f"\n  {model_label} model not found ��� skipping.")
            return None

        log(f"\nRunning {model_label} inference for error analysis...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)

        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
        mdl.to(device)
        mdl.eval()

        all_preds = []
        batch_size = 32

        for start in range(0, len(texts_test), batch_size):
            batch = texts_test[start:start + batch_size]
            inputs = tok(batch, padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = mdl(**inputs).logits

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

        preds = np.array(all_preds)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted")
        log(f"  {model_label} acc={acc:.4f}, F1={f1:.4f}")
        return preds, acc, f1

    bert_preds = None
    bert_f1 = 0
    bert_result = load_and_predict(
        os.path.join(SCRIPT_DIR, "bert_emotion_model", "final"), "BERT")
    if bert_result:
        bert_preds, bert_acc, bert_f1 = bert_result

    distilbert_preds = None
    distilbert_f1 = 0
    distilbert_result = load_and_predict(
        os.path.join(SCRIPT_DIR, "distilbert_emotion_model", "final"), "DistilBERT")
    if distilbert_result:
        distilbert_preds, distilbert_acc, distilbert_f1 = distilbert_result

    # ── Confusion matrices ───────────────────────────────────────────────
    log("\nGenerating confusion matrices...")

    plot_confusion(y_test, tfidf_preds,
                   "TF-IDF + LogReg — Confusion Matrix (Normalized)",
                   os.path.join(CHARTS_DIR, "error_confusion_tfidf.png"))

    if distilbert_preds is not None:
        plot_confusion(y_test, distilbert_preds,
                       "DistilBERT — Confusion Matrix (Normalized)",
                       os.path.join(CHARTS_DIR, "error_confusion_distilbert.png"))

    if bert_preds is not None:
        plot_confusion(y_test, bert_preds,
                       "BERT — Confusion Matrix (Normalized)",
                       os.path.join(CHARTS_DIR, "error_confusion_bert.png"))

    # ── Misclassified example analysis ───────────────────────────────────
    models = [("TF-IDF + LogReg", tfidf_preds)]
    if distilbert_preds is not None:
        models.append(("DistilBERT", distilbert_preds))
    if bert_preds is not None:
        models.append(("BERT", bert_preds))

    for model_name, preds in models:
        log(f"\n{'=' * 60}")
        log(f"  MISCLASSIFIED EXAMPLES — {model_name}")
        log(f"{'=' * 60}")

        wrong_mask = preds != y_test
        wrong_idx  = np.where(wrong_mask)[0]
        n_wrong    = len(wrong_idx)
        log(f"\n  Total misclassified: {n_wrong} / {len(y_test)} "
            f"({100 * n_wrong / len(y_test):.1f}%)")

        # Sample up to N_EXAMPLES
        np.random.seed(42)
        sample_idx = np.random.choice(wrong_idx,
                                      min(N_EXAMPLES, n_wrong),
                                      replace=False)

        # Categorize errors
        error_categories = {}
        log(f"\n  Sample of {len(sample_idx)} misclassified examples:")
        log("-" * 60)

        for i, idx in enumerate(sample_idx):
            text = texts_test[idx]
            true_label = LABEL_NAMES[y_test[idx]]
            pred_label = LABEL_NAMES[preds[idx]]
            category = categorize_error(text, y_test[idx], preds[idx])

            error_categories[category] = error_categories.get(category, 0) + 1

            # Show first 15 examples in detail
            if i < 15:
                text_short = text[:100] + ("..." if len(text) > 100 else "")
                log(f"\n  [{i+1}] True: {true_label:<10} Pred: {pred_label:<10} "
                    f"Type: {category}")
                log(f"      \"{text_short}\"")

        # Error category summary
        log(f"\n  Error Category Distribution ({model_name}):")
        log("-" * 40)
        for cat, count in sorted(error_categories.items(),
                                 key=lambda x: -x[1]):
            pct = 100 * count / len(sample_idx)
            log(f"    {cat:<30} {count:>3}  ({pct:.0f}%)")

    # ── Per-class comparison ─────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log("  PER-CLASS F1 COMPARISON")
    log(f"{'=' * 60}")

    header = f"\n  {'Emotion':<12} {'TF-IDF F1':>10}"
    if distilbert_preds is not None:
        header += f"  {'DistilB F1':>10}"
    if bert_preds is not None:
        header += f"  {'BERT F1':>10}  {'Delta':>8}"
    log(header)
    log("-" * len(header))

    tfidf_per_class = f1_score(y_test, tfidf_preds, average=None)
    distilbert_per_class = (f1_score(y_test, distilbert_preds, average=None)
                            if distilbert_preds is not None else None)
    bert_per_class = (f1_score(y_test, bert_preds, average=None)
                      if bert_preds is not None else None)

    for i, name in enumerate(LABEL_NAMES):
        line = f"  {name:<12} {tfidf_per_class[i]:>10.4f}"
        if distilbert_per_class is not None:
            line += f"  {distilbert_per_class[i]:>10.4f}"
        if bert_per_class is not None:
            delta = bert_per_class[i] - tfidf_per_class[i]
            sign = "+" if delta > 0 else ""
            line += f"  {bert_per_class[i]:>10.4f}  {sign}{delta:>7.4f}"
        log(line)

    footer = f"\n  {'WEIGHTED':>12} {tfidf_f1:>10.4f}"
    if distilbert_preds is not None:
        footer += f"  {distilbert_f1:>10.4f}"
    if bert_preds is not None:
        delta = bert_f1 - tfidf_f1
        sign = "+" if delta > 0 else ""
        footer += f"  {bert_f1:>10.4f}  {sign}{delta:>7.4f}"
    log(footer)

    # ── Key findings ─────────────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log("  KEY FINDINGS")
    log(f"{'=' * 60}")

    # Find hardest class for TF-IDF
    hardest_tfidf = LABEL_NAMES[np.argmin(tfidf_per_class)]
    log(f"\n  - Hardest class for TF-IDF: {hardest_tfidf} "
        f"(F1={tfidf_per_class[np.argmin(tfidf_per_class)]:.4f})")

    if distilbert_per_class is not None:
        hardest_db = LABEL_NAMES[np.argmin(distilbert_per_class)]
        log(f"  - Hardest class for DistilBERT: {hardest_db} "
            f"(F1={distilbert_per_class[np.argmin(distilbert_per_class)]:.4f})")

    if bert_per_class is not None:
        hardest_bert = LABEL_NAMES[np.argmin(bert_per_class)]
        log(f"  - Hardest class for BERT: {hardest_bert} "
            f"(F1={bert_per_class[np.argmin(bert_per_class)]:.4f})")

        biggest_gain_idx = np.argmax(bert_per_class - tfidf_per_class)
        log(f"  - BERT's biggest gain over TF-IDF: {LABEL_NAMES[biggest_gain_idx]} "
            f"(+{bert_per_class[biggest_gain_idx] - tfidf_per_class[biggest_gain_idx]:.4f})")

    if distilbert_per_class is not None and bert_per_class is not None:
        f1_gap = np.mean(bert_per_class) - np.mean(distilbert_per_class)
        log(f"  - DistilBERT vs BERT avg F1 gap: {f1_gap:.4f} "
            f"({'negligible' if abs(f1_gap) < 0.01 else 'modest' if abs(f1_gap) < 0.03 else 'significant'})")

    log("\n" + "=" * 60)

    # Save
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
