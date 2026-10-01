
# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Imports                                                   │
# └─────────────────────────────────────────────────────────────────────┘
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})
print("✅ All imports successful")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — Parser: converts CUAD JSON → flat DataFrame              │
# └─────────────────────────────────────────────────────────────────────┘
def extract_clause_type(question: str) -> str:
    """Pull the clause name from inside the first quoted string."""
    match = re.search(r'"([^"]+)"', question)
    return match.group(1) if match else "Unknown"


def parse_cuad_json(filepath: str, split_name: str = "") -> pd.DataFrame:
    """
    Parses a CUAD-format JSON file.

    Structure:
        { "version": ...,
          "data": [
            { "title": "ContractName",
              "paragraphs": [
                { "context": "full contract text...",
                  "qas": [
                    { "id": "...",
                      "question": "Highlight ... related to \"ClauseType\" ...",
                      "answers": [{"text": "...", "answer_start": 42}],
                      "is_impossible": "True" | "False"
                    }, ...   # 41 questions per contract
                  ]
                }
              ]
            }, ...
          ]
        }

    Returns one row per PRESENT clause (is_impossible == False).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for contract in raw["data"]:
        title = contract["title"]
        for para in contract["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                clause_type  = extract_clause_type(qa["question"])
                is_impossible = str(qa["is_impossible"]).strip().lower() == "true"
                has_answer    = len(qa["answers"]) > 0

                # Only keep clauses that ARE present in the contract
                if not is_impossible and has_answer:
                    records.append({
                        "contract_title" : title,
                        "clause_type"    : clause_type,
                        "clause_text"    : qa["answers"][0]["text"].strip(),
                        "answer_start"   : qa["answers"][0]["answer_start"],
                        "context_snippet": context[:300],   # first 300 chars of contract
                        "split"          : split_name,
                    })

    df = pd.DataFrame(records)
    print(f"✅ Parsed {filepath}")
    print(f"   Contracts : {len(raw['data']):,}")
    print(f"   QA pairs  : {len(raw['data']) * 41:,}")
    print(f"   Clauses   : {len(df):,}  (is_impossible=False)")
    return df


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Load the dataset                                         │
# │  Put your JSON files in  ContractSenseAI/data/                     │
# └─────────────────────────────────────────────────────────────────────┘
TEST_JSON  = "../data/test.json"
TRAIN_JSON = "../data/train_separate_questions.json"

df_test  = parse_cuad_json(TEST_JSON,  split_name="test")
df_train = parse_cuad_json(TRAIN_JSON, split_name="train")

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"\n📊 Combined dataset: {len(df):,} clause samples")
df.head(3)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Basic data quality checks                                │
# └─────────────────────────────────────────────────────────────────────┘
print("═" * 55)
print(f"Shape            : {df.shape}")
print(f"Null values      :\n{df.isnull().sum()}")
print(f"\nDuplicate rows   : {df.duplicated(subset=['clause_text','clause_type']).sum()}")

# Add text length column (useful for analysis)
df["text_len"] = df["clause_text"].str.len()

print(f"\nClause text length stats:")
print(df["text_len"].describe().round(1).to_string())

# Drop duplicates and empty texts
before = len(df)
df = df.drop_duplicates(subset=["clause_text", "clause_type"])
df = df[df["clause_text"].str.len() > 5]
print(f"\n✅ After cleaning: {len(df):,} rows  (dropped {before - len(df)})")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — Clause type distribution                                 │
# └─────────────────────────────────────────────────────────────────────┘
clause_counts = df["clause_type"].value_counts()
n_classes     = len(clause_counts)

print(f"Unique clause types : {n_classes}")
print(f"\nTop 10:\n{clause_counts.head(10).to_string()}")
print(f"\nBottom 5:\n{clause_counts.tail(5).to_string()}")

# Plot
fig, ax = plt.subplots(figsize=(14, 9))
palette = sns.color_palette("RdYlGn", n_classes)[::-1]
clause_counts.plot(kind="barh", ax=ax, color=palette,
                   edgecolor="white", linewidth=0.4)
ax.set_xlabel("Number of clause samples", fontsize=11)
ax.set_title(f"Clause Type Distribution  |  {n_classes} types  ·  "
             f"{len(df):,} total samples", fontsize=13, fontweight="bold", pad=14)
for i, v in enumerate(clause_counts.values):
    ax.text(v + clause_counts.max() * 0.005, i,
            f" {v:,}", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig("../outputs/01_clause_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/01_clause_distribution.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Sample clause texts per type (sanity check)              │
# └─────────────────────────────────────────────────────────────────────┘
print("SAMPLE CLAUSE TEXT PER TYPE (first 220 chars)\n")
for clause_type in clause_counts.index[:12]:
    sample = df[df["clause_type"] == clause_type]["clause_text"].values[0]
    print(f"▶  [{clause_type}]")
    print(f"   {sample[:220]}{'...' if len(sample)>220 else ''}")
    print()


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Risk label mapping (YOUR design decision)                │
# │                                                                     │
# │  RED    = Heavily favours the other party OR creates significant   │
# │           financial / legal risk for the signing party             │
# │  YELLOW = Needs careful review, common but negotiable              │
# │  GREEN  = Standard administrative / low-risk clause                │
# └─────────────────────────────────────────────────────────────────────┘
RISK_MAP = {
    # ── RED: High Risk ────────────────────────────────────────────────
    "Uncapped Liability"                 : "RED",
    "Unlimited/All-You-Can-Eat-License"  : "RED",
    "Liquidated Damages"                 : "RED",
    "Non-Compete"                        : "RED",
    "No-Solicit Of Employees"            : "RED",
    "No-Solicit Of Customers"            : "RED",
    "Exclusivity"                        : "RED",
    "Anti-Assignment"                    : "RED",
    "Ip Ownership Assignment"            : "RED",
    "Change Of Control"                  : "RED",
    "Revenue/Profit Sharing"             : "RED",

    # ── YELLOW: Medium Risk ───────────────────────────────────────────
    "Cap On Liability"                   : "YELLOW",
    "Termination For Convenience"        : "YELLOW",
    "Audit Rights"                       : "YELLOW",
    "Most Favored Nation"                : "YELLOW",
    "Non-Disparagement"                  : "YELLOW",
    "Warranty Duration"                  : "YELLOW",
    "Insurance"                          : "YELLOW",
    "Post-Termination Services"          : "YELLOW",
    "Minimum Commitment"                 : "YELLOW",
    "Volume Restriction"                 : "YELLOW",
    "Price Restrictions"                 : "YELLOW",
    "Renewal Term"                       : "YELLOW",
    "Notice Period To Terminate Renewal" : "YELLOW",
    "Rofr/Rofo/Rofn"                     : "YELLOW",
    "Competitive Restriction Exception"  : "YELLOW",
    "Irrevocable Or Perpetual License"   : "YELLOW",
    "Non-Transferable License"           : "YELLOW",
    "Affiliate License-Licensee"         : "YELLOW",
    "Affiliate License-Licensor"         : "YELLOW",
    "Joint Ip Ownership"                 : "YELLOW",
    "Source Code Escrow"                 : "YELLOW",
    "Covenant Not To Sue"                : "YELLOW",
    "Third Party Beneficiary"            : "YELLOW",

    # ── GREEN: Low Risk ───────────────────────────────────────────────
    "Document Name"                      : "GREEN",
    "Parties"                            : "GREEN",
    "Agreement Date"                     : "GREEN",
    "Effective Date"                     : "GREEN",
    "Expiration Date"                    : "GREEN",
    "Governing Law"                      : "GREEN",
    "License Grant"                      : "GREEN",
}

df["risk_level"] = df["clause_type"].map(RISK_MAP)

# Handle anything not in the map
unmapped = df[df["risk_level"].isnull()]["clause_type"].unique()
if len(unmapped) > 0:
    print(f"⚠️  Unmapped types (defaulting to YELLOW): {list(unmapped)}")
    df["risk_level"] = df["risk_level"].fillna("YELLOW")

print("Risk level counts:")
print(df["risk_level"].value_counts().to_string())


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 8 — Risk distribution charts                                 │
# └─────────────────────────────────────────────────────────────────────┘
RISK_COLORS = {"RED": "#E05A5A", "YELLOW": "#F0B429", "GREEN": "#1D9E75"}
risk_counts  = df["risk_level"].value_counts().reindex(["RED","YELLOW","GREEN"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("ContractSenseAI — Risk Level Analysis",
             fontsize=14, fontweight="bold", y=1.01)

# 1. Bar chart
axes[0].bar(risk_counts.index, risk_counts.values,
            color=[RISK_COLORS[r] for r in risk_counts.index],
            edgecolor="white", linewidth=0.8, width=0.5)
axes[0].set_title("Risk Level Count", fontweight="bold")
axes[0].set_ylabel("Number of clauses")
for i, v in enumerate(risk_counts.values):
    axes[0].text(i, v + risk_counts.max()*0.01, f"{v:,}",
                 ha="center", fontweight="bold", fontsize=11)

# 2. Pie chart
axes[1].pie(risk_counts.values,
            labels=risk_counts.index,
            colors=[RISK_COLORS[r] for r in risk_counts.index],
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 12})
axes[1].set_title("Risk Level Share", fontweight="bold")

# 3. Clause type breakdown coloured by risk
ct_risk = df.groupby(["clause_type","risk_level"]).size().unstack(fill_value=0)
ct_risk = ct_risk.reindex(columns=["RED","YELLOW","GREEN"])
ct_risk["total"] = ct_risk.sum(axis=1)
ct_risk = ct_risk.sort_values("total", ascending=True).drop(columns="total")
ct_risk.plot(kind="barh", stacked=True, ax=axes[2],
             color=[RISK_COLORS[c] for c in ct_risk.columns],
             edgecolor="white", linewidth=0.3)
axes[2].set_title("Risk Breakdown per Clause Type", fontweight="bold")
axes[2].set_xlabel("Samples")
axes[2].legend(title="Risk", loc="lower right")

plt.tight_layout()
plt.savefig("../outputs/02_risk_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/02_risk_distribution.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 9 — Text length analysis per risk level                      │
# └─────────────────────────────────────────────────────────────────────┘
fig, ax = plt.subplots(figsize=(9, 4))
for risk in ["RED", "YELLOW", "GREEN"]:
    subset = df[df["risk_level"] == risk]["text_len"]
    subset.plot(kind="kde", ax=ax, color=RISK_COLORS[risk],
                label=f"{risk} (μ={int(subset.mean())})", linewidth=2)
ax.set_xlabel("Clause text length (characters)", fontsize=11)
ax.set_title("Clause Length Distribution by Risk Level",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../outputs/03_text_length_by_risk.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → outputs/03_text_length_by_risk.png")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CELL 10 — Save processed dataset                                  │
# └─────────────────────────────────────────────────────────────────────┘
SAVE_COLS = ["contract_title", "clause_type", "clause_text",
             "risk_level", "text_len", "split"]

df[SAVE_COLS].to_csv("../data/processed_dataset.csv", index=False)

print(f"✅ Saved → data/processed_dataset.csv")
print(f"   Total rows : {len(df):,}")
print(f"   Train rows : {len(df[df['split']=='train']):,}")
print(f"   Test rows  : {len(df[df['split']=='test']):,}")
print(f"\nFinal shape: {df[SAVE_COLS].shape}")
df[SAVE_COLS].sample(5)
