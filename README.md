# 🧠 Emotion Classifier — BERT vs DistilBERT vs RoBERTa

A deep learning project that fine-tunes and compares three state-of-the-art transformer models for **text-based emotion classification**, built as part of a Data Science portfolio at the Universitat Politècnica de València (UPV).

---

## 📌 Problem Statement

Understanding human emotions from text is one of the most valuable challenges in Natural Language Processing. Applications range from **mental health monitoring** and therapeutic support tools, to customer sentiment analysis and human-computer interaction.

This project fine-tunes three transformer models to classify text into six emotions — **sadness, joy, love, anger, fear and surprise** — and systematically compares their performance, efficiency, and real-world trade-offs.

---

## 🎯 Motivation

Emotion detection from text has direct applications in high-impact fields:

- **Psychology & mental health** — automatic screening tools that detect emotional distress in patient-written text, supporting early intervention
- **Therapeutic chatbots** — conversational agents that adapt their responses based on the user's detected emotional state
- **Social media analysis** — identifying emotional trends at scale across large populations
- **Human-computer interaction** — building systems that respond empathetically to user input

---

## 📂 Dataset

**[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** — a dataset of English Twitter messages labeled with six emotions.

| Split | Samples |
|-------|---------|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

**Classes:** sadness · joy · love · anger · fear · surprise

> ⚠️ Note: the dataset is imbalanced — *surprise* has ~572 samples vs ~5,765 for *sadness*. This affects per-class F1 scores, particularly for underrepresented emotions.

---

## 🤖 Models

Three pre-trained transformer models were fine-tuned for sequence classification:

| Model | Parameters | Size | Architecture |
|-------|-----------|------|-------------|
| `bert-base-uncased` | 110M | 438 MB | 12-layer Transformer encoder |
| `distilbert-base-uncased` | 67M | 268 MB | Distilled BERT, 40% smaller |
| `roberta-base` | 125M | 501 MB | Robustly optimized BERT |

---

## ⚙️ Training Setup

All models were fine-tuned using the Hugging Face `Trainer` API with the following configuration:
```python
TrainingArguments(
    num_train_epochs        = 10,       # ceiling — early stopping cuts this short
    per_device_train_batch_size = 32,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,     # L2 regularization
    warmup_ratio            = 0.1,      # LR warmup for first 10% of steps
    max_grad_norm           = 1.0,      # gradient clipping
    eval_strategy           = "epoch",
    metric_for_best_model   = "f1",
    fp16                    = True,     # mixed precision on GPU
)
```

**Early stopping** was applied with `patience=2` and `threshold=0.001`, monitoring validation F1. Models were trained on Google Colab with a T4 GPU.

---

## 📊 Results

| Model | Accuracy | F1 Weighted | Epochs trained |
|-------|----------|-------------|----------------|
| BERT | — | — | — |
| DistilBERT | — | — | — |
| RoBERTa | — | — | — |

> Results will be populated after training. All results will be stored in a JSON file with the most relevant metrics.

---

## 🗂️ Project Structure
```
emotion_classifier.ipynb     # Training pipeline (run on Google Colab)
emotion_dashboard.py         # Streamlit dashboard
requirements.txt
metrics/
  results.json               # All metrics exported from training
saved_models/
  bert-base-uncased/
  distilbert-base-uncased/
  roberta-base/
```

--

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `PyTorch` | Deep learning framework, GPU training |
| `Transformers` (HuggingFace) | Pre-trained models, Trainer API, tokenizers |
| `Datasets` (HuggingFace) | Dataset loading and preprocessing |
| `Evaluate` (HuggingFace) | Metric computation (accuracy, F1) |

---

## 👤 Author

**Kento Kamakura Gimeno** — Data Science & Industrial Management   