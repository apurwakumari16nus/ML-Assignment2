"""
DistilBERT Fine-Tuning for 6-Category Emotion Classification
DSS5104 — Mental Health Analysis Project

Fine-tunes distilbert-base-cased on the Saravia emotion dataset.
DistilBERT is 40% smaller (66M vs 110M params) and ~60% faster than BERT.

Includes learning rate tuning and 3-seed training for statistical rigor.
Saves the best model for Reddit classification (script 09).

Output: distilbert_emotion_model/final/, distilbert_results.txt
"""

import os
import json
import time
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME      = "distilbert-base-cased"
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "distilbert_emotion_model")
FINAL_DIR       = os.path.join(OUTPUT_DIR, "final")
RIGOROUS_DIR    = os.path.join(SCRIPT_DIR, "distilbert_rigorous_runs")
RESULTS_FILE    = os.path.join(SCRIPT_DIR, "distilbert_results.txt")

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
    log("  DISTILBERT — Training + Statistical Rigor")
    log("=" * 60)
    log(f"  Model: {MODEL_NAME}")
    log(f"  DistilBERT is 40% smaller, ~60% faster than BERT-base")

    # Step 1: Load and tokenize
    log("\nStep 1: Loading and tokenizing emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    log(f"  Train: {len(tokenized['train']):,}, Val: {len(tokenized['validation']):,}, "
        f"Test: {len(tokenized['test']):,}")

    # Step 2: Learning rate tuning
    log("\nStep 2: Learning rate tuning (on validation set, seed=42)...")

    best_lr = None
    best_val_f1 = 0

    for lr in LEARNING_RATES:
        log(f"\n  LR={lr}:")
        run_dir = os.path.join(RIGOROUS_DIR, f"tune_lr{lr}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_NAMES))

        args = TrainingArguments(
            output_dir=run_dir, num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=32,
            learning_rate=lr, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="no",
            logging_steps=200, report_to="none", seed=42,
        )

        trainer = Trainer(model=model, args=args,
                          train_dataset=tokenized["train"],
                          eval_dataset=tokenized["validation"],
                          compute_metrics=compute_metrics)

        t_start = time.time()
        trainer.train()
        train_time = time.time() - t_start

        val_results = trainer.evaluate(tokenized["validation"])
        val_f1 = val_results["eval_f1"]
        log(f"    Val acc={val_results['eval_accuracy']:.4f}, F1={val_f1:.4f}, time={train_time:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_lr = lr

    log(f"\n  -> Best LR: {best_lr} (val F1={best_val_f1:.4f})")

    # Step 3: Train with 3 seeds
    log(f"\nStep 3: Training with {len(SEEDS)} seeds (LR={best_lr})...")

    all_results = []
    best_test_f1 = 0
    best_trainer = None

    for seed in SEEDS:
        log(f"\n  Seed {seed}:")
        run_dir = os.path.join(RIGOROUS_DIR, f"seed_{seed}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_NAMES))

        args = TrainingArguments(
            output_dir=run_dir, num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=32,
            learning_rate=best_lr, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="no",
            logging_steps=200, report_to="none", seed=seed,
        )

        trainer = Trainer(model=model, args=args,
                          train_dataset=tokenized["train"],
                          eval_dataset=tokenized["validation"],
                          compute_metrics=compute_metrics)

        t_start = time.time()
        trainer.train()
        train_time = time.time() - t_start

        t_start = time.time()
        test_results = trainer.evaluate(tokenized["test"])
        inference_time = time.time() - t_start

        test_acc = test_results["eval_accuracy"]
        test_f1 = test_results["eval_f1"]

        log(f"    acc={test_acc:.4f}, F1={test_f1:.4f}, "
            f"train={train_time:.1f}s, infer={inference_time:.2f}s")

        predictions = trainer.predict(tokenized["test"])
        preds = np.argmax(predictions.predictions, axis=1)

        all_results.append({
            "seed": seed, "accuracy": test_acc, "f1_weighted": test_f1,
            "train_time_sec": train_time, "inference_time_sec": inference_time,
            "y_pred": preds, "y_true": predictions.label_ids,
        })

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_trainer = trainer

    # Step 4: Summary
    log("\n" + "=" * 60)
    log("  RESULTS: DistilBERT (mean +/- std across 3 seeds)")
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

    last = all_results[-1]
    log(f"\n  Classification Report (seed={last['seed']}):")
    log(classification_report(last["y_true"], last["y_pred"],
                              target_names=LABEL_NAMES, digits=4))

    # Step 5: Save best model
    log("\nStep 5: Saving best model...")
    os.makedirs(FINAL_DIR, exist_ok=True)
    best_trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    label_map = {i: name for i, name in enumerate(LABEL_NAMES)}
    with open(os.path.join(FINAL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    log(f"  Saved -> {FINAL_DIR}")

    log("\n" + "=" * 60)
    log("  TRAINING COMPLETE")
    log(f"  Next step: python 09_distilbert_classify.py")
    log("=" * 60)

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
