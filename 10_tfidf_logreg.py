"""
TF-IDF + Logistic Regression — Emotion Classification
DSS5104 — Mental Health Analysis Project

Classical ML baseline using TF-IDF features + Logistic Regression.
Tunes C hyperparameter, trains with 3 seeds, classifies Reddit comments.

Output: reddit_comments_tfidf.csv, tfidf_logreg_results.txt
"""

import os
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
COMMENTS_CLEAN = os.path.join(SCRIPT_DIR, "reddit_comments_clean.csv")
COMMENTS_OUT   = os.path.join(SCRIPT_DIR, "reddit_comments_tfidf.csv")
RESULTS_FILE   = os.path.join(SCRIPT_DIR, "tfidf_logreg_results.txt")

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SEEDS = [42, 123, 456]


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  TF-IDF + LOGISTIC REGRESSION")
    log("=" * 60)

    # Load dataset
    log("\nLoading emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")

    # Vectorize
    log("\nStep 1: TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)

    X_train = vectorizer.fit_transform(dataset["train"]["text"])
    X_val   = vectorizer.transform(dataset["validation"]["text"])
    X_test  = vectorizer.transform(dataset["test"]["text"])
    y_train = np.array(dataset["train"]["label"])
    y_val   = np.array(dataset["validation"]["label"])
    y_test  = np.array(dataset["test"]["label"])

    log(f"  Vocabulary: {len(vectorizer.vocabulary_):,} features")

    # Tune C
    log("\nStep 2: Hyperparameter tuning (C)...")
    best_c = None
    best_f1 = 0
    for c_val in [0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        val_f1 = f1_score(y_val, lr.predict(X_val), average="weighted")
        log(f"  C={c_val}: val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c_val
    log(f"  -> Best C={best_c} (F1={best_f1:.4f})")

    # Train with 3 seeds
    log(f"\nStep 3: Training with {len(SEEDS)} seeds (C={best_c})...")
    all_results = []
    for seed in SEEDS:
        lr = LogisticRegression(C=best_c, max_iter=1000, random_state=seed)
        t0 = time.time()
        lr.fit(X_train, y_train)
        train_time = time.time() - t0
        t0 = time.time()
        y_pred = lr.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        log(f"  Seed {seed}: acc={acc:.4f}, F1={f1:.4f}, "
            f"train={train_time:.2f}s, infer={infer_time:.4f}s")
        all_results.append({"seed": seed, "accuracy": acc, "f1": f1,
                            "train_time": train_time, "infer_time": infer_time,
                            "y_pred": y_pred})

    # Summary
    log("\n" + "=" * 60)
    log("  RESULTS (mean +/- std)")
    log("=" * 60)
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    log(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    log(f"  F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    last = all_results[-1]
    log(f"\n  Classification Report (seed={last['seed']}):")
    log(classification_report(y_test, last["y_pred"],
                              target_names=LABEL_NAMES, digits=4))

    # Classify Reddit
    log("\nStep 4: Classifying Reddit comments...")
    df = pd.read_csv(COMMENTS_CLEAN)
    log(f"  {len(df):,} comments loaded")

    texts = df["body"].fillna("").astype(str).tolist()
    X_reddit = vectorizer.transform(texts)

    best_lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    best_lr.fit(X_train, y_train)
    lr_preds = best_lr.predict(X_reddit)
    lr_probs = best_lr.predict_proba(X_reddit)

    df["tfidf_emotion"] = [LABEL_NAMES[p] for p in lr_preds]
    df["tfidf_confidence"] = lr_probs.max(axis=1)
    for i, name in enumerate(LABEL_NAMES):
        df[f"tfidf_{name}"] = lr_probs[:, i]

    df.to_csv(COMMENTS_OUT, index=False)
    log(f"  Saved -> {COMMENTS_OUT}")

    log(f"\n  Reddit emotion distribution (LogReg):")
    dist = df["tfidf_emotion"].value_counts()
    for emotion, count in dist.items():
        log(f"    {emotion:<10} {count:>6,}  ({100*count/len(df):.1f}%)")

    log("\n" + "=" * 60)

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
