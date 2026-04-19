"""
Quick Comparison: Your Fine-Tuned BERT vs HuggingFace Pre-Trained Emotion Model
DSS5104 — Mental Health Analysis Project

Compares predictions on a sample of Reddit comments:
  - YOUR model: bert_emotion_model/final/ (fine-tuned on Saravia, 6 classes)
  - HF model: j-hartmann/emotion-english-distilroberta-base (pre-trained, 7 classes)

The HF model has 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise
Your model has 6 classes: sadness, joy, love, anger, fear, surprise

We compare on the 5 overlapping emotions (anger, fear, joy, sadness, surprise).

Usage: python check_pretrained_vs_finetuned.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
YOUR_MODEL_DIR = os.path.join(SCRIPT_DIR, "bert_emotion_model", "final")
BERT_COMMENTS  = os.path.join(SCRIPT_DIR, "reddit_comments_bert.csv")

HF_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Overlapping emotions between the two models
OVERLAP_EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]

SAMPLE_SIZE = 1000  # compare on 1000 random comments (fast enough)


def load_your_model():
    """Load your fine-tuned BERT."""
    tok = AutoTokenizer.from_pretrained(YOUR_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(YOUR_MODEL_DIR)

    with open(os.path.join(YOUR_MODEL_DIR, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    model.to(device)
    model.eval()
    return tok, model, label_map, device


def predict_your_model(texts, tok, model, label_map, device, batch_size=32):
    """Predict emotions using your fine-tuned model."""
    all_preds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend([label_map[p] for p in preds])
    return all_preds


def main():
    print("=" * 60)
    print("  YOUR FINE-TUNED BERT vs HUGGINGFACE PRE-TRAINED MODEL")
    print("=" * 60)

    # Load Reddit comments
    if not os.path.exists(BERT_COMMENTS):
        print(f"ERROR: {BERT_COMMENTS} not found. Run 08_bert_classify.py first.")
        return

    df = pd.read_csv(BERT_COMMENTS)
    print(f"\n  Total Reddit comments: {len(df):,}")

    # Sample
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    texts = sample["body"].fillna("").astype(str).tolist()
    your_preds_from_csv = sample["bert_emotion"].tolist()
    print(f"  Sampled {len(texts)} for comparison")

    # Load your model and predict fresh
    print("\n  Loading your fine-tuned BERT...")
    tok, model, label_map, device = load_your_model()
    your_preds = predict_your_model(texts, tok, model, label_map, device)
    print(f"  Your model predictions done.")

    # Load HuggingFace pre-trained model
    print(f"\n  Loading HuggingFace model: {HF_MODEL_NAME}")
    print("  (This will download ~1GB on first run)")
    hf_pipe = pipeline("text-classification", model=HF_MODEL_NAME,
                       top_k=1, truncation=True, max_length=128,
                       device=device)

    print("  Running HF model predictions...")
    hf_results = hf_pipe(texts, batch_size=32)
    hf_preds = [r[0]["label"] for r in hf_results]
    print(f"  HF model predictions done.")

    # ── Compare ──
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    # Overall distribution comparison
    print(f"\n  Emotion Distribution (on {len(texts)} comments):\n")
    print(f"  {'Emotion':<12} {'Your BERT':>12} {'HF Model':>12}")
    print(f"  {'─' * 36}")

    your_dist = pd.Series(your_preds).value_counts(normalize=True) * 100
    hf_dist = pd.Series(hf_preds).value_counts(normalize=True) * 100

    all_emotions = sorted(set(list(your_dist.index) + list(hf_dist.index)))
    for emo in all_emotions:
        y_pct = your_dist.get(emo, 0)
        h_pct = hf_dist.get(emo, 0)
        print(f"  {emo:<12} {y_pct:>11.1f}% {h_pct:>11.1f}%")

    # Agreement on overlapping emotions
    agree_count = 0
    overlap_count = 0
    for yp, hp in zip(your_preds, hf_preds):
        if yp in OVERLAP_EMOTIONS and hp in OVERLAP_EMOTIONS:
            overlap_count += 1
            if yp == hp:
                agree_count += 1

    if overlap_count > 0:
        agree_pct = 100 * agree_count / overlap_count
        print(f"\n  Agreement on overlapping emotions: {agree_pct:.1f}% "
              f"({agree_count}/{overlap_count})")

    total_agree = sum(1 for y, h in zip(your_preds, hf_preds) if y == h)
    print(f"  Exact match (all classes): {100*total_agree/len(texts):.1f}% "
          f"({total_agree}/{len(texts)})")

    # Per-emotion agreement
    print(f"\n  Per-Emotion Agreement (where your model predicts X,")
    print(f"  does HF model also predict X?):\n")
    print(f"  {'Your Pred':<12} {'Count':>6} {'HF Agrees':>10} {'Agree %':>8}")
    print(f"  {'─' * 38}")

    for emo in sorted(set(your_preds)):
        mask = [i for i, y in enumerate(your_preds) if y == emo]
        if len(mask) == 0:
            continue
        hf_same = sum(1 for i in mask if hf_preds[i] == emo)
        print(f"  {emo:<12} {len(mask):>6} {hf_same:>10} {100*hf_same/len(mask):>7.1f}%")

    # Show 10 examples where they disagree
    print(f"\n{'─' * 60}")
    print("  SAMPLE DISAGREEMENTS (first 10):")
    print(f"{'─' * 60}")

    shown = 0
    for i in range(len(texts)):
        if your_preds[i] != hf_preds[i] and shown < 10:
            text_short = texts[i][:80]
            print(f"\n  [{shown+1}] \"{text_short}...\"")
            print(f"       Your BERT: {your_preds[i]:<12}  HF Model: {hf_preds[i]}")
            shown += 1

    # Distress ratio comparison
    print(f"\n{'=' * 60}")
    print("  DISTRESS RATIO COMPARISON")
    print(f"{'=' * 60}")

    distress_yours = sum(1 for p in your_preds if p in ["sadness", "fear", "anger"])
    distress_hf = sum(1 for p in hf_preds if p in ["sadness", "fear", "anger"])
    # HF model has "disgust" which is also negative
    distress_hf_expanded = sum(1 for p in hf_preds
                                if p in ["sadness", "fear", "anger", "disgust"])

    print(f"\n  Your BERT distress (sad+fear+anger):  {100*distress_yours/len(texts):.1f}%")
    print(f"  HF model distress (sad+fear+anger):   {100*distress_hf/len(texts):.1f}%")
    print(f"  HF model distress (+disgust):         {100*distress_hf_expanded/len(texts):.1f}%")

    print(f"\n{'=' * 60}")
    print("  INTERPRETATION")
    print(f"{'=' * 60}")
    print(f"""
  The HF model (j-hartmann) was trained on 6 different datasets
  and has 7 classes (adds 'disgust' and 'neutral', drops 'love').

  Key differences to expect:
  - HF has 'neutral' class — many of your 'joy' or 'anger' may
    map to 'neutral' in HF (Reddit comments are often matter-of-fact)
  - HF has 'disgust' — some of your 'anger' may split into 'disgust'
  - Your model has 'love' — HF doesn't, so these will mismatch

  High agreement = your fine-tuned model produces similar results
  to an independent, well-validated model. This is good validation.

  Low agreement = the models see Reddit text differently, likely
  due to different training data and class definitions.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
