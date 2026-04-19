"""
Model Comparison — VADER vs TF-IDF vs BERT
DSS5104 — Mental Health Analysis Project

Side-by-side comparison of all five classification approaches:
  1. VADER (lexicon-based, no training)
  2. TF-IDF + Logistic Regression (classical ML)
  3. TF-IDF + Linear SVM (classical ML)
  4. DistilBERT fine-tuned (lightweight deep learning)
  5. BERT fine-tuned (deep learning)

Compares: accuracy, F1, training time, inference time, complexity.
Shows the full accuracy-speed tradeoff across 4 methods.

Output: charts/model_comparison_bar.png, charts/model_comparison_tradeoff.png,
        model_comparison_results.txt
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR    = os.path.join(SCRIPT_DIR, "charts")
RESULTS_FILE  = os.path.join(SCRIPT_DIR, "model_comparison_results.txt")

# Reddit scored files
VADER_COMMENTS      = os.path.join(SCRIPT_DIR, "reddit_comments_scored.csv")
BERT_COMMENTS       = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")
DISTILBERT_COMMENTS = os.path.join(SCRIPT_DIR, "reddit_comments_distilbert.csv")
TFIDF_COMMENTS      = os.path.join(SCRIPT_DIR, "reddit_comments_tfidf.csv")
POSTS_FILE          = os.path.join(SCRIPT_DIR, "reddit_posts_bert.csv")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def evaluate_vader_on_emotion_dataset():
    """VADER has no training — evaluate directly on the emotion test set."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    dataset = load_dataset("dair-ai/emotion")
    texts = dataset["test"]["text"]
    y_true = np.array(dataset["test"]["label"])

    analyzer = SentimentIntensityAnalyzer()

    # VADER produces compound score -> map to 3 categories
    # To compare with 6-class, map VADER to majority class per sentiment bucket
    # This is inherently unfair (VADER wasn't designed for 6-class) but shows the gap
    vader_labels = []
    t_start = time.time()
    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            vader_labels.append("positive")
        elif score <= -0.05:
            vader_labels.append("negative")
        else:
            vader_labels.append("neutral")
    inference_time = time.time() - t_start

    # Map 6 emotions to 3 for VADER comparison
    emotion_to_sentiment = {
        0: "negative",  # sadness
        1: "positive",  # joy
        2: "positive",  # love
        3: "negative",  # anger
        4: "negative",  # fear
        5: "neutral",   # surprise
    }
    y_true_3class = [emotion_to_sentiment[y] for y in y_true]

    acc = accuracy_score(y_true_3class, vader_labels)
    f1 = f1_score(y_true_3class, vader_labels, average="weighted")

    return {
        "model": "VADER (lexicon)",
        "task": "3-class sentiment",
        "accuracy": acc,
        "f1_weighted": f1,
        "train_time_sec": 0.0,
        "inference_time_sec": inference_time,
        "n_params": "~7,500 rules",
        "requires_gpu": False,
    }


def evaluate_tfidf():
    """Train and evaluate TF-IDF + LogReg on emotion test set."""
    dataset = load_dataset("dair-ai/emotion")

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_test  = vectorizer.transform(dataset["test"]["text"])
    y_train = np.array(dataset["train"]["label"])
    y_test  = np.array(dataset["test"]["label"])

    lr = LogisticRegression(C=10.0, max_iter=1000, random_state=42)

    t_start = time.time()
    lr.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_start = time.time()
    y_pred = lr.predict(X_test)
    inference_time = time.time() - t_start

    return {
        "model": "TF-IDF + LogReg",
        "task": "6-class emotion",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "n_params": f"~{X_train.shape[1] * 6:,} weights",
        "requires_gpu": False,
    }


def evaluate_tfidf_svm():
    """Train and evaluate TF-IDF + Linear SVM on emotion test set."""
    dataset = load_dataset("dair-ai/emotion")

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_test  = vectorizer.transform(dataset["test"]["text"])
    y_train = np.array(dataset["train"]["label"])
    y_test  = np.array(dataset["test"]["label"])

    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)

    t_start = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_start = time.time()
    y_pred = svm.predict(X_test)
    inference_time = time.time() - t_start

    return {
        "model": "TF-IDF + SVM",
        "task": "6-class emotion",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "n_params": f"~{X_train.shape[1] * 6:,} weights",
        "requires_gpu": False,
    }


def evaluate_distilbert():
    """Evaluate DistilBERT on emotion test set using saved model."""
    model_dir = os.path.join(SCRIPT_DIR, "distilbert_emotion_model", "final")

    if not os.path.exists(model_dir):
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dataset = load_dataset("dair-ai/emotion")
    texts = dataset["test"]["text"]
    y_test = np.array(dataset["test"]["label"])

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    batch_size = 32

    t_start = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    inference_time = time.time() - t_start
    y_pred = np.array(all_preds)

    return {
        "model": "DistilBERT (fine-tuned)",
        "task": "6-class emotion",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "train_time_sec": 319.0,  # from rigorous 3-seed results (319.0s ± 2.5s)
        "inference_time_sec": inference_time,
        "n_params": "~66M",
        "requires_gpu": True,
    }


def evaluate_bert():
    """Evaluate BERT on emotion test set using saved model."""
    model_dir = os.path.join(SCRIPT_DIR, "bert_emotion_model", "final")

    if not os.path.exists(model_dir):
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dataset = load_dataset("dair-ai/emotion")
    texts = dataset["test"]["text"]
    y_test = np.array(dataset["test"]["label"])

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    batch_size = 32

    t_start = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    inference_time = time.time() - t_start
    y_pred = np.array(all_preds)

    return {
        "model": "BERT (fine-tuned)",
        "task": "6-class emotion",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "train_time_sec": 622.0,  # from rigorous 3-seed results (621.8s ± 5.3s)
        "inference_time_sec": inference_time,
        "n_params": "~110M",
        "requires_gpu": True,
    }


def plot_comparison_bar(results, save_path):
    """Grouped bar chart comparing F1 and accuracy."""
    models = [r["model"] for r in results]
    accs   = [r["accuracy"] for r in results]
    f1s    = [r["f1_weighted"] for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",
                   color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, f1s, width, label="F1 (weighted)",
                   color="#e74c3c", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison — Emotion Classification",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_tradeoff(results, save_path):
    """Scatter: F1 vs inference time (speed-accuracy tradeoff)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]
    for i, r in enumerate(results):
        ax.scatter(r["inference_time_sec"], r["f1_weighted"],
                   s=200, c=colors[i % len(colors)], edgecolors="black", zorder=5)
        ax.annotate(r["model"],
                    (r["inference_time_sec"], r["f1_weighted"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight="bold")

    ax.set_xlabel("Inference Time (seconds, test set)")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Speed-Accuracy Tradeoff", fontsize=13, fontweight="bold")
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def compare_reddit_predictions(log):
    """Compare emotion distributions on Reddit data across methods."""
    log(f"\n{'=' * 60}")
    log("  REDDIT PREDICTIONS — Cross-Method Comparison")
    log(f"{'=' * 60}")

    dfs = {}

    # Load VADER
    if os.path.exists(VADER_COMMENTS):
        dfs["VADER"] = pd.read_csv(VADER_COMMENTS)

    # Load BERT
    if os.path.exists(BERT_COMMENTS):
        dfs["BERT"] = pd.read_csv(BERT_COMMENTS)

    # Load DistilBERT
    if os.path.exists(DISTILBERT_COMMENTS):
        dfs["DistilBERT"] = pd.read_csv(DISTILBERT_COMMENTS)

    # Load TF-IDF
    if os.path.exists(TFIDF_COMMENTS):
        dfs["TF-IDF"] = pd.read_csv(TFIDF_COMMENTS)

    if not dfs:
        log("\n  No Reddit scored files found. Run the pipeline first.")
        return

    # Emotion distribution per method
    for name, df in dfs.items():
        if name == "VADER":
            col = "vader_label"
            if col not in df.columns:
                continue
            log(f"\n  {name} — Sentiment Distribution:")
            dist = df[col].value_counts()
        elif name == "BERT":
            col = "bert_emotion"
            if col not in df.columns:
                continue
            log(f"\n  {name} — Emotion Distribution:")
            dist = df[col].value_counts()
        elif name == "DistilBERT":
            col = "distilbert_emotion"
            if col not in df.columns:
                continue
            log(f"\n  {name} — Emotion Distribution:")
            dist = df[col].value_counts()
        else:  # TF-IDF
            col = "tfidf_emotion"
            if col not in df.columns:
                continue
            log(f"\n  {name} — Emotion Distribution:")
            dist = df[col].value_counts()

        for label, count in dist.items():
            pct = 100 * count / len(df)
            log(f"    {label:<12} {count:>6,}  ({pct:.1f}%)")

    # If both BERT and TF-IDF exist, compare agreement on Reddit data
    if "BERT" in dfs and "TF-IDF" in dfs:
        df_b = dfs["BERT"]
        df_t = dfs["TF-IDF"]

        if "comment_id" in df_b.columns and "comment_id" in df_t.columns:
            merged = df_b[["comment_id", "bert_emotion"]].merge(
                df_t[["comment_id", "tfidf_emotion"]], on="comment_id"
            )
            agree = (merged["bert_emotion"] == merged["tfidf_emotion"]).mean()
            log(f"\n  BERT vs TF-IDF agreement on Reddit: {100*agree:.1f}%")

            # Per-emotion agreement
            log(f"  Per-emotion agreement:")
            for emotion in LABEL_NAMES:
                bert_mask = merged["bert_emotion"] == emotion
                if bert_mask.sum() > 0:
                    emo_agree = (merged.loc[bert_mask, "tfidf_emotion"] == emotion).mean()
                    log(f"    {emotion:<12} {100*emo_agree:.1f}% (n={bert_mask.sum():,})")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  MODEL COMPARISON — VADER vs TF-IDF vs BERT")
    log("=" * 60)

    # ── Evaluate all models on emotion test set ──────────────────────────
    log("\nEvaluating models on Saravia emotion test set...")

    results = []

    log("\n  [1/5] VADER (lexicon-based)...")
    vader_res = evaluate_vader_on_emotion_dataset()
    results.append(vader_res)
    log(f"    Accuracy={vader_res['accuracy']:.4f}, F1={vader_res['f1_weighted']:.4f}")
    log(f"    (3-class: pos/neg/neutral — VADER cannot do 6-class)")

    log("\n  [2/5] TF-IDF + Logistic Regression...")
    tfidf_res = evaluate_tfidf()
    results.append(tfidf_res)
    log(f"    Accuracy={tfidf_res['accuracy']:.4f}, F1={tfidf_res['f1_weighted']:.4f}")

    log("\n  [3/5] TF-IDF + Linear SVM...")
    svm_res = evaluate_tfidf_svm()
    results.append(svm_res)
    log(f"    Accuracy={svm_res['accuracy']:.4f}, F1={svm_res['f1_weighted']:.4f}")

    log("\n  [4/5] DistilBERT (fine-tuned)...")
    distilbert_res = evaluate_distilbert()
    if distilbert_res:
        results.append(distilbert_res)
        log(f"    Accuracy={distilbert_res['accuracy']:.4f}, F1={distilbert_res['f1_weighted']:.4f}")
    else:
        log("    DistilBERT model not found — skipping. Run 05b_distilbert_train.py first.")

    log("\n  [5/5] BERT (fine-tuned)...")
    bert_res = evaluate_bert()
    if bert_res:
        results.append(bert_res)
        log(f"    Accuracy={bert_res['accuracy']:.4f}, F1={bert_res['f1_weighted']:.4f}")
    else:
        log("    BERT model not found — skipping. Run 04_bert_train.py first.")

    # ── Summary table ────────────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log("  COMPARISON TABLE")
    log(f"{'=' * 60}")

    header = (f"  {'Model':<22} {'Task':<16} {'Acc':>7} {'F1':>7} "
              f"{'Train(s)':>9} {'Infer(s)':>9} {'GPU?':>5}")
    log(f"\n{header}")
    log("  " + "-" * (len(header) - 2))

    for r in results:
        line = (f"  {r['model']:<22} {r['task']:<16} "
                f"{r['accuracy']:>7.4f} {r['f1_weighted']:>7.4f} "
                f"{r['train_time_sec']:>9.2f} {r['inference_time_sec']:>9.2f} "
                f"{'Yes' if r['requires_gpu'] else 'No':>5}")
        log(line)

    # ── Speed comparisons ────────────────────────────────────────────────
    if len(results) >= 2:
        log(f"\n  Speed Analysis:")
        if bert_res:
            speedup = bert_res["inference_time_sec"] / max(tfidf_res["inference_time_sec"], 0.001)
            log(f"    TF-IDF is {speedup:.0f}x faster than BERT at inference")

        if bert_res and distilbert_res:
            db_speedup = bert_res["inference_time_sec"] / max(distilbert_res["inference_time_sec"], 0.001)
            log(f"    DistilBERT is {db_speedup:.1f}x faster than BERT at inference")
            f1_drop = bert_res["f1_weighted"] - distilbert_res["f1_weighted"]
            log(f"    DistilBERT F1 drop vs BERT: {f1_drop:.4f} "
                f"({'negligible' if abs(f1_drop) < 0.01 else 'modest' if abs(f1_drop) < 0.03 else 'significant'})")

        vader_res_time = results[0]["inference_time_sec"]
        tfidf_res_time = results[1]["inference_time_sec"]
        if tfidf_res_time > 0:
            log(f"    VADER inference: {vader_res_time:.2f}s, "
                f"TF-IDF: {tfidf_res_time:.4f}s")

    # ── Practical recommendations ────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log("  PRACTICAL RECOMMENDATIONS")
    log(f"{'=' * 60}")

    log("""
  1. VADER is best for:
     - Quick sentiment polarity (pos/neg/neutral)
     - No training data available
     - Real-time processing of very large datasets
     - Social media text with slang, emojis, exclamation marks

  2. TF-IDF + LogReg is best for:
     - Multi-class classification without GPU
     - Fast training and inference
     - When labeled data is available
     - Baseline that's hard to beat for the compute cost

  3. DistilBERT is the OVERALL WINNER:
     - HIGHEST F1 of all models (0.9332 vs BERT's 0.9282)
     - 40% smaller than BERT (66M vs 110M parameters)
     - ~1.4x faster inference, ~50% faster training (319s vs 622s)
     - Best accuracy-speed tradeoff — recommended for production

  4. BERT is best for:
     - When context matters (sarcasm, mixed emotions)
     - Research settings where having the largest model is preferred
     - Fine-grained emotion detection (love vs joy, anger vs fear)
     - Note: BERT did NOT beat DistilBERT on this task (0.9282 vs 0.9332)

  For this mental health analysis:
  - DistilBERT is the top performer: highest F1, lower cost than BERT
  - BERT is a close second but does not justify its extra compute
  - TF-IDF validates that deep learning gains justify the complexity
  - VADER offers a complementary sentiment signal
  - The 5-method comparison strengthens conclusions

  LIMITATIONS:
  - All model-estimated emotions on Reddit are predictions, not ground truth
  - Models trained on Saravia tweets; Reddit domain may differ (domain mismatch)
  - VADER's 3-class task is not directly comparable to 6-class emotion
  - Subreddit sample sizes are imbalanced (worldnews dominates)
  - Reddit users are not representative of the general population (selection bias)""")

    log("\n" + "=" * 60)

    # ── Charts ───────────────────────────────────────────────────────────
    log("\nGenerating comparison charts...")

    plot_comparison_bar(results,
                        os.path.join(CHARTS_DIR, "model_comparison_bar.png"))

    if len(results) >= 2:
        plot_tradeoff(results,
                      os.path.join(CHARTS_DIR, "model_comparison_tradeoff.png"))

    # ── Reddit predictions comparison ────────────────────────────────────
    compare_reddit_predictions(log)

    # Save
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
