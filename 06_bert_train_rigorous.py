"""
BERT Rigorous Training — Multiple Seeds + Hyperparameter Tuning
DSS5104 — Mental Health Analysis Project

Extends the base BERT training (04_bert_train.py) with:
  - 3 random seeds for statistical rigor (mean +/- std)
  - Learning rate tuning (2e-5, 3e-5, 5e-5)
  - Training and inference timing
  - Side-by-side comparison table

This does NOT retrain the production model — it evaluates BERT's
sensitivity to hyperparameters and random initialization.

Output: bert_rigorous_results.txt

Install: pip install transformers datasets torch scikit-learn accelerate
"""

import os
import time
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "bert-base-cased"
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "bert_rigorous_runs")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "bert_rigorous_results.txt")

LABEL_NAMES    = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SEEDS          = [42, 123, 456]
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
EPOCHS         = 3
BATCH_SIZE     = 16
MAX_SEQ_LENGTH = 128


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 60)
    log("  BERT RIGOROUS TRAINING — Seeds + Hyperparameter Tuning")
    log("=" * 60)

    # ── Step 1: Load and tokenize ────────────────────────────────────────
    log("\nStep 1: Loading and tokenizing emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    log(f"  Train: {len(tokenized['train']):,}, Val: {len(tokenized['validation']):,}, "
        f"Test: {len(tokenized['test']):,}")

    # ── Step 2: Hyperparameter tuning on validation set ──────────────────
    log("\nStep 2: Learning rate tuning (on validation set, seed=42)...")

    best_lr = None
    best_val_f1 = 0

    for lr in LEARNING_RATES:
        log(f"\n  LR={lr}:")
        run_dir = os.path.join(OUTPUT_DIR, f"tune_lr{lr}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_NAMES)
        )

        args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=32,
            learning_rate=lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=200,
            report_to="none",
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        t_start = time.time()
        trainer.train()
        train_time = time.time() - t_start

        val_results = trainer.evaluate(tokenized["validation"])
        val_f1 = val_results["eval_f1"]
        val_acc = val_results["eval_accuracy"]

        log(f"    Val acc={val_acc:.4f}, F1={val_f1:.4f}, time={train_time:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_lr = lr

    log(f"\n  -> Best LR: {best_lr} (val F1={best_val_f1:.4f})")

    # ── Step 3: Train with 3 seeds using best LR ────────────────────────
    log(f"\nStep 3: Training with {len(SEEDS)} seeds (LR={best_lr})...")

    all_results = []

    for seed in SEEDS:
        log(f"\n  Seed {seed}:")
        run_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_NAMES)
        )

        args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=32,
            learning_rate=best_lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=200,
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        )

        # Train
        t_start = time.time()
        trainer.train()
        train_time = time.time() - t_start

        # Evaluate on test set
        t_start = time.time()
        test_results = trainer.evaluate(tokenized["test"])
        inference_time = time.time() - t_start

        test_acc = test_results["eval_accuracy"]
        test_f1 = test_results["eval_f1"]

        log(f"    acc={test_acc:.4f}, F1={test_f1:.4f}, "
            f"train={train_time:.1f}s, infer={inference_time:.2f}s")

        # Detailed predictions for classification report
        predictions = trainer.predict(tokenized["test"])
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        all_results.append({
            "seed": seed,
            "accuracy": test_acc,
            "f1_weighted": test_f1,
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "y_pred": preds,
            "y_true": labels,
        })

    # ── Step 4: Summary ──────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  RESULTS: BERT (mean +/- std across 3 seeds)")
    log("=" * 60)

    accs   = [r["accuracy"] for r in all_results]
    f1s    = [r["f1_weighted"] for r in all_results]
    trains = [r["train_time_sec"] for r in all_results]
    infers = [r["inference_time_sec"] for r in all_results]

    log(f"\n  Best learning rate: {best_lr}")
    log(f"  Accuracy:       {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    log(f"  F1 (weighted):  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    log(f"  Train time:     {np.mean(trains):.1f}s +/- {np.std(trains):.1f}s")
    log(f"  Inference time: {np.mean(infers):.2f}s +/- {np.std(infers):.2f}s")

    # Classification report from last seed
    last = all_results[-1]
    log(f"\n  Classification Report (seed={last['seed']}):")
    log(classification_report(
        last["y_true"], last["y_pred"],
        target_names=LABEL_NAMES, digits=4
    ))

    log("=" * 60)

    # Save
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
