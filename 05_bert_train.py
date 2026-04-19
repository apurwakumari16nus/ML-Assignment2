"""
BERT Fine-Tuning for 6-Category Emotion Classification
DSS5104 — Mental Health Analysis Project (Step 4a)

Fine-tunes bert-base-cased on the Saravia et al. (2018) emotion dataset
to classify text into 6 categories: sadness, joy, love, anger, fear, surprise.

This matches the methodology from the reference paper (Wang et al., 2025).

The fine-tuned model is saved locally and used by bert_classify.py to
score the Reddit comments.

Install:
  pip install transformers datasets torch scikit-learn

On Apple Silicon Mac, PyTorch will use MPS (Metal) for GPU acceleration.
On CPU-only machines, training will be slower but still works.
"""

import os
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
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "bert_emotion_model")
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Training hyperparameters (tuned for a good balance of speed and accuracy)
EPOCHS         = 3
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
MAX_SEQ_LENGTH = 128


def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 for evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    print("=" * 60)
    print("  BERT Fine-Tuning — 6 Emotion Categories")
    print("=" * 60)
    print(f"  Base model:  {MODEL_NAME}")
    print(f"  Categories:  {LABEL_NAMES}")
    print(f"  Output dir:  {OUTPUT_DIR}\n")

    # ── Step 1: Load the emotion dataset ─────────────────────────────────
    print("Step 1/4 — Loading emotion dataset (Saravia et al. 2018)...")
    dataset = load_dataset("dair-ai/emotion")

    print(f"  Train:      {len(dataset['train']):,} samples")
    print(f"  Validation: {len(dataset['validation']):,} samples")
    print(f"  Test:       {len(dataset['test']):,} samples")

    # Show label distribution
    from collections import Counter
    label_dist = Counter(dataset["train"]["label"])
    print("\n  Label distribution (train):")
    for label_id, count in sorted(label_dist.items()):
        print(f"    {LABEL_NAMES[label_id]:<10} {count:>6,}")

    # ── Step 2: Tokenize ─────────────────────────────────────────────────
    print("\nStep 2/4 — Tokenizing...")
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

    print("  Done.")

    # ── Step 3: Train ────────────────────────────────────────────────────
    print(f"\nStep 3/4 — Training ({EPOCHS} epochs, batch size {BATCH_SIZE})...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        report_to="none",  # disable wandb/mlflow
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Step 4: Evaluate on test set ─────────────────────────────────────
    print("\nStep 4/4 — Evaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print(f"  Test accuracy:     {test_results['eval_accuracy']:.4f}")
    print(f"  Test F1 (weighted): {test_results['eval_f1']:.4f}")

    # Detailed classification report
    predictions = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print("\n  Classification Report:")
    print(classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        digits=4,
    ))

    # ── Save model and tokenizer ─────────────────────────────────────────
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved -> {final_dir}")

    # Save label mapping for the classify script
    import json
    label_map = {i: name for i, name in enumerate(LABEL_NAMES)}
    with open(os.path.join(final_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Next step: python bert_classify.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
