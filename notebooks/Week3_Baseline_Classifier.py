# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ContractSenseAI  ·  Week 3  ·  Baseline Clause Risk Classifier     ║
# ║  Run AFTER Week1_2 notebook. Needs: data/processed_dataset.csv      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Imports                                                   │
# └─────────────────────────────────────────────────────────────────────┘
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       f1_score, accuracy_score)
from sklearn.preprocessing     import LabelEncoder

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})
print("✅ All imports successful")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — Load processed dataset                                   │
# └─────────────────────────────────────────────────────────────────────┘
df = pd.read_csv("../data/processed_dataset.csv")
print(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}")
df.head(3)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Text pre-processing                                      │
# └─────────────────────────────────────────────────────────────────────┘
def clean_text(text: str) -> str:
    """
    Lightweight cleaning — keeps legal language intact.
    We don't aggressively strip words because legal text is
    domain-specific and even common words carry meaning.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    text = re.sub(r"[^\w\s\.\,\;\:\-\(\)]", " ", text)  # keep legal punctuation
    return text.lower()

df["clean_text"] = df["clause_text"].apply(clean_text)
print("Sample cleaned text:")
print(df["clean_text"].values[0][:300])


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Train / test split (use the original dataset splits)     │
# └─────────────────────────────────────────────────────────────────────┘
df_train = df[df["split"] == "train"].copy()
df_test  = df[df["split"] == "test"].copy()

print(f"Train samples : {len(df_train):,}")
print(f"Test  samples : {len(df_test):,}")
print(f"\nTrain risk distribution:")
print(df_train["risk_level"].value_counts().to_string())
print(f"\nTest risk distribution:")
print(df_test["risk_level"].value_counts().to_string())

X_train = df_train["clean_text"].astype(str).values.tolist()
y_train = df_train["risk_level"].astype(str).values.tolist()
X_test  = df_test["clean_text"].astype(str).values.tolist()
y_test  = df_test["risk_level"].astype(str).values.tolist()

LABELS        = ["RED", "YELLOW", "GREEN"]
RISK_COLORS   = {"RED": "#E05A5A", "YELLOW": "#F0B429", "GREEN": "#1D9E75"}


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — Build and train TF-IDF + Logistic Regression baseline   │
# └─────────────────────────────────────────────────────────────────────┘
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features   = 15_000,
        ngram_range    = (1, 2),     # unigrams + bigrams
        sublinear_tf   = True,       # log-scale TF
        min_df         = 2,          # ignore very rare terms
        strip_accents  = "unicode",
    )),
    ("clf", LogisticRegression(
        max_iter       = 1000,
        class_weight   = "balanced", # handles class imbalance automatically
        C              = 1.0,
        solver         = "lbfgs",
        random_state   = 42,
    )),
])

print("Training TF-IDF + Logistic Regression...")
lr_pipeline.fit(X_train, y_train)
print("✅ Training complete")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Evaluate on test set                                     │
# └─────────────────────────────────────────────────────────────────────┘
y_pred_lr = lr_pipeline.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr  = f1_score(y_test, y_pred_lr, average="macro")

print("=" * 55)
print(f"  TF-IDF + Logistic Regression Results")
print("=" * 55)
print(f"  Accuracy      : {acc_lr:.4f}  ({acc_lr*100:.1f}%)")
print(f"  Macro F1      : {f1_lr:.4f}")
print("=" * 55)
print()
print(classification_report(y_test, y_pred_lr, target_names=LABELS))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Confusion matrix (visual your guide will love)           │
# └─────────────────────────────────────────────────────────────────────┘
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Percentage
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix (% per true class)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(f"../outputs/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved → outputs/{filename}")

plot_confusion_matrix(y_test, y_pred_lr, LABELS,
                      "Baseline · TF-IDF + Logistic Regression",
                      "04_confusion_matrix_baseline.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 8 — Cross-validation (5-fold) for more reliable F1 estimate  │
# └─────────────────────────────────────────────────────────────────────┘
print("Running 5-fold cross-validation on full dataset...")

X_all = df["clean_text"].astype(str).values.tolist()
y_all = df["risk_level"].astype(str).values.tolist()

cv_scores = cross_val_score(lr_pipeline, X_all, y_all,
                            cv=StratifiedKFold(n_splits=5, shuffle=True,
                                               random_state=42),
                            scoring="f1_macro", n_jobs=-1)

print(f"\n5-Fold CV Macro F1 scores: {cv_scores.round(4)}")
print(f"Mean  ± Std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 9 — Top predictive words per risk class (interpretability)   │
# └─────────────────────────────────────────────────────────────────────┘
tfidf    = lr_pipeline.named_steps["tfidf"]
clf      = lr_pipeline.named_steps["clf"]
feature_names = np.array(tfidf.get_feature_names_out())

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Top 15 TF-IDF Features per Risk Class",
             fontsize=13, fontweight="bold")

for i, (class_name, ax) in enumerate(zip(clf.classes_, axes)):
    coef      = clf.coef_[i]
    top_idx   = np.argsort(coef)[-15:]
    top_words = feature_names[top_idx]
    top_vals  = coef[top_idx]
    color     = RISK_COLORS.get(class_name, "#888")

    ax.barh(top_words, top_vals, color=color, edgecolor="white")
    ax.set_title(f"{class_name} risk", fontweight="bold", color=color)
    ax.set_xlabel("Coefficient weight")

plt.tight_layout()
plt.savefig("../outputs/05_top_features_per_class.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/05_top_features_per_class.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 10 — Quick test: classify a custom clause                    │
# └─────────────────────────────────────────────────────────────────────┘
test_clauses = [
    "The liability of either party shall not exceed the total fees "
    "paid in the twelve months preceding the claim.",

    "This agreement shall be governed by the laws of the State of Delaware.",

    "Employee agrees not to engage in any competitive business "
    "within a 50-mile radius for a period of two years after termination.",

    "Seller shall not be liable for any indirect, incidental, special, "
    "or consequential damages regardless of the form of action.",
]

print("━" * 60)
print("QUICK INFERENCE TEST")
print("━" * 60)
for clause in test_clauses:
    pred  = lr_pipeline.predict([clause])[0]
    proba = lr_pipeline.predict_proba([clause])[0]
    conf  = max(proba) * 100
    color_map = {"RED":"🔴", "YELLOW":"🟡", "GREEN":"🟢"}
    print(f"\nClause : {clause[:90]}...")
    print(f"Risk   : {color_map[pred]} {pred}  (confidence {conf:.1f}%)")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 11 — Save baseline model                                     │
# └─────────────────────────────────────────────────────────────────────┘
joblib.dump(lr_pipeline, "../models/baseline_lr_pipeline.pkl")
print("✅ Baseline model saved → models/baseline_lr_pipeline.pkl")

# Summary card for your guide presentation
print()
print("╔══════════════════════════════════════════════════════╗")
print("║           BASELINE MODEL SUMMARY CARD                ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Model          TF-IDF + Logistic Regression         ║")
print(f"║  Features       TF-IDF 15K unigrams + bigrams        ║")
print(f"║  Classes        RED / YELLOW / GREEN                  ║")
print(f"║  Test Accuracy  {acc_lr*100:>5.1f}%                              ║")
print(f"║  Macro F1       {f1_lr:.4f}                             ║")
print(f"║  CV F1 (5-fold) {cv_scores.mean():.4f} ± {cv_scores.std():.4f}              ║")
print(f"║  Training time  < 10 seconds (CPU)                   ║")
print("╚══════════════════════════════════════════════════════╝")
