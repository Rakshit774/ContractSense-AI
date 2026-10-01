# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ContractSenseAI  ·  Week 4  ·  DistilBERT Fine-tuning             ║
# ║  Run AFTER Week3 notebook. Needs: data/processed_dataset.csv        ║
# ║  Training time: ~25–45 min on CPU · ~5 min on GPU                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Imports                                                   │
# └─────────────────────────────────────────────────────────────────────┘
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update({"figure.dpi": 130})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
print(f"   PyTorch version : {torch.__version__}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — Load processed dataset                                   │
# └─────────────────────────────────────────────────────────────────────┘
df = pd.read_csv("../data/processed_dataset.csv")
print(f"Loaded {len(df):,} rows")

# Label encode: RED=0, YELLOW=1, GREEN=2  (consistent ordering)
LABEL_ORDER = ["RED", "YELLOW", "GREEN"]
label2id    = {l: i for i, l in enumerate(LABEL_ORDER)}
id2label    = {i: l for l, i in label2id.items()}

df["label_id"] = df["risk_level"].map(label2id)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_test  = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train : {len(df_train):,}  |  Test : {len(df_test):,}")
print(f"label2id: {label2id}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Tokenizer and PyTorch Dataset class                      │
# └─────────────────────────────────────────────────────────────────────┘
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128    # Legal clauses avg 254 chars ≈ 60 tokens; 128 is safe

print(f"Loading tokenizer: {MODEL_NAME} ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
print("✅ Tokenizer loaded")


class ClauseDataset(Dataset):
    """
    PyTorch Dataset that tokenizes clause text on the fly.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(),
            "attention_mask" : enc["attention_mask"].squeeze(),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


BATCH_SIZE = 16

train_dataset = ClauseDataset(df_train["clause_text"].values,
                               df_train["label_id"].values,
                               tokenizer, MAX_LEN)
test_dataset  = ClauseDataset(df_test["clause_text"].values,
                               df_test["label_id"].values,
                               tokenizer, MAX_LEN)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                           shuffle=False)

print(f"Train batches : {len(train_loader)}")
print(f"Test  batches : {len(test_loader)}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Load DistilBERT model                                    │
# └─────────────────────────────────────────────────────────────────────┘
print(f"Loading {MODEL_NAME} for sequence classification...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = len(LABEL_ORDER),
    id2label   = id2label,
    label2id   = label2id,
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model loaded")
print(f"   Total params     : {total_params:,}")
print(f"   Trainable params : {train_params:,}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — Training loop                                            │
# └─────────────────────────────────────────────────────────────────────┘
EPOCHS    = 3
LR        = 2e-5
WARMUP    = 0.1    # 10% of steps for warmup

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = int(total_steps * WARMUP),
    num_training_steps = total_steps,
)

# Class weights to handle imbalance (same idea as baseline)
class_counts = df_train["label_id"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
class_weights = class_weights / class_weights.sum() * len(LABEL_ORDER)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

history = {"train_loss": [], "val_loss": [], "val_f1": []}

print(f"Starting training: {EPOCHS} epochs · LR={LR} · batch={BATCH_SIZE}")
print("─" * 55)

for epoch in range(EPOCHS):
    # ── Training phase ──────────────────────────────────────────────
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # ── Validation phase ─────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, total_val_loss = [], [], 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_loader)
    val_f1       = f1_score(all_labels, all_preds, average="macro")
    val_acc      = accuracy_score(all_labels, all_preds)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_f1"].append(val_f1)

    print(f"\n{'━'*55}")
    print(f"  Epoch {epoch+1}/{EPOCHS} complete")
    print(f"  Train loss : {avg_train_loss:.4f}")
    print(f"  Val   loss : {avg_val_loss:.4f}")
    print(f"  Val   F1   : {val_f1:.4f}")
    print(f"  Val   Acc  : {val_acc*100:.1f}%")
    print(f"{'━'*55}\n")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Final evaluation                                         │
# └─────────────────────────────────────────────────────────────────────┘
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

y_pred_bert = [id2label[p] for p in all_preds]
y_true_bert = [id2label[l] for l in all_labels]

acc_bert = accuracy_score(y_true_bert, y_pred_bert)
f1_bert  = f1_score(y_true_bert, y_pred_bert, average="macro")
LABELS   = ["RED", "YELLOW", "GREEN"]

print("=" * 55)
print("  DistilBERT Fine-tuned Results")
print("=" * 55)
print(f"  Accuracy  : {acc_bert:.4f}  ({acc_bert*100:.1f}%)")
print(f"  Macro F1  : {f1_bert:.4f}")
print("=" * 55)
print()
print(classification_report(y_true_bert, y_pred_bert, target_names=LABELS))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Training curve                                           │
# └─────────────────────────────────────────────────────────────────────┘
epochs_range = range(1, EPOCHS + 1)
fig, axes    = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("DistilBERT Training History", fontsize=13, fontweight="bold")

axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train loss",
             color="#378ADD", linewidth=2)
axes[0].plot(epochs_range, history["val_loss"],   "s--", label="Val loss",
             color="#E05A5A", linewidth=2)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_range, history["val_f1"], "^-",
             color="#1D9E75", linewidth=2, markersize=8)
axes[1].set_title("Validation Macro F1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("../outputs/06_distilbert_training_curve.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/06_distilbert_training_curve.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 8 — Confusion matrix for DistilBERT                          │
# └─────────────────────────────────────────────────────────────────────┘
RISK_COLORS = {"RED": "#E05A5A", "YELLOW": "#F0B429", "GREEN": "#1D9E75"}
cm = confusion_matrix(y_true_bert, y_pred_bert, labels=LABELS)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS,
            linewidths=0.5, ax=ax)
ax.set_title("DistilBERT Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("../outputs/07_confusion_matrix_distilbert.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/07_confusion_matrix_distilbert.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 9 — Model comparison table (Baseline vs DistilBERT)         │
# │  Load baseline metrics saved from Week 3 notebook                  │
# └─────────────────────────────────────────────────────────────────────┘
# Baseline metrics (paste from Week 3 output if needed)
baseline_acc = None
baseline_f1  = None

try:
    baseline_model = joblib.load("../models/baseline_lr_pipeline.pkl")
    df_test_raw    = pd.read_csv("../data/processed_dataset.csv")
    df_t           = df_test_raw[df_test_raw["split"] == "test"]
    y_base         = baseline_model.predict(df_t["clause_text"].values)
    baseline_acc   = accuracy_score(df_t["risk_level"].values, y_base)
    baseline_f1    = f1_score(df_t["risk_level"].values, y_base, average="macro")
except Exception as e:
    print(f"Note: Baseline model not found ({e}) — run Week3 first")
    baseline_acc, baseline_f1 = 0.0, 0.0

comparison = pd.DataFrame({
    "Model"          : ["TF-IDF + LogReg (Baseline)", "DistilBERT (Fine-tuned)"],
    "Test Accuracy"  : [f"{baseline_acc*100:.1f}%", f"{acc_bert*100:.1f}%"],
    "Macro F1"       : [f"{baseline_f1:.4f}", f"{f1_bert:.4f}"],
    "Training Time"  : ["< 10 sec (CPU)", "~30 min (CPU) / 5 min (GPU)"],
    "Model Size"     : ["~5 MB", "~250 MB"],
    "Inference Speed": ["< 1ms / clause", "~15ms / clause"],
})

print("\n" + "═"*70)
print("  MODEL COMPARISON")
print("═"*70)
print(comparison.to_string(index=False))
print("═"*70)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 10 — Save DistilBERT model and tokenizer                     │
# └─────────────────────────────────────────────────────────────────────┘
SAVE_DIR = "../models/distilbert_risk_classifier"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save label mapping
with open(f"{SAVE_DIR}/label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print(f"✅ DistilBERT model saved → {SAVE_DIR}/")
print(f"   Files: config.json, pytorch_model.bin, tokenizer files, label_map.json")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 11 — Quick inference function (reuse in backend later)       │
# └─────────────────────────────────────────────────────────────────────┘
def predict_risk(clause_text: str,
                 model=model,
                 tokenizer=tokenizer,
                 device=device) -> dict:
    """
    Predict risk level for a single clause string.
    Returns: {"risk": "RED"|"YELLOW"|"GREEN", "confidence": 0.0–1.0}
    """
    model.eval()
    enc = tokenizer(
        clause_text,
        max_length     = 128,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
        ).logits

    probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id    = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    return {
        "risk"       : id2label[pred_id],
        "confidence" : round(confidence, 4),
        "all_scores" : {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


# Test it
test_clauses = [
    "Contractor shall not be liable for any indirect or consequential damages.",
    "This agreement is governed by the laws of the State of California.",
    "Employee agrees not to work for any competitor for 24 months after leaving.",
]

print("\nINFERENCE TEST")
print("━" * 60)
color_map = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}
for c in test_clauses:
    result = predict_risk(c)
    print(f"Clause : {c[:80]}...")
    print(f"Result : {color_map[result['risk']]} {result['risk']}  "
          f"(confidence {result['confidence']*100:.1f}%)")
    print(f"Scores : {result['all_scores']}\n")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FINAL SUMMARY CARD                                                │
# └─────────────────────────────────────────────────────────────────────┘
print()
print("╔══════════════════════════════════════════════════════╗")
print("║         DISTILBERT MODEL SUMMARY CARD                ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Base model     distilbert-base-uncased              ║")
print(f"║  Fine-tuned on  ContractSenseAI CUAD dataset         ║")
print(f"║  Epochs         {EPOCHS}                                     ║")
print(f"║  Learning rate  {LR}                               ║")
print(f"║  Max seq len    {MAX_LEN} tokens                          ║")
print(f"║  Test Accuracy  {acc_bert*100:>5.1f}%                              ║")
print(f"║  Macro F1       {f1_bert:.4f}                             ║")
print(f"║  Saved to       models/distilbert_risk_classifier/   ║")
print("╚══════════════════════════════════════════════════════╝")
