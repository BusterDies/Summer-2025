#!/usr/bin/env python3
"""
plot_roc.py – Generate multi-label ROC curves for the dementia-symptom model
----------------------------------------------------------------------------

Assumes:
  • X.npy, Y.npy  – feature matrix and label matrix
  • scaler.joblib – StandardScaler fitted on the train split
  • clf_*.joblib  – trained One-Vs-Rest logistic-regression classifier

Produces:
  roc_multi.png   – ROC curves (per class + macro average)
"""

from pathlib import Path
import joblib, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# --- Paths -----------------------------------------------------------------
BASE = Path("/home/pachethridge47/Summer-2025/datasets")
X      = np.load(BASE / "X.npy")
Y      = np.load(BASE / "Y.npy")          # multilabel (n_samples × 4)

SCALER = BASE / "scaler.joblib"
MODEL  = BASE / "clf_20250716_114031.joblib"   # <- adjust if your filename differs

LABELS = [
    "mood_dysregulation",
    "problems_communicating",
    "confusion_disorientation",
    "memory_loss",
]

# --- Train/test split identical to training --------------------------------
_, X_test, _, Y_test = train_test_split(
    X, Y, test_size=0.20, stratify=Y, random_state=42
)

# --- Load scaler + model, transform test features --------------------------
scaler = joblib.load(SCALER)
clf    = joblib.load(MODEL)

X_test_std = scaler.transform(X_test)
y_score    = clf.decision_function(X_test_std)   # shape = (n_samples, 4)

# --- Compute ROC curve and AUC for each class ------------------------------
fpr, tpr, roc_auc = {}, {}, {}
for i, lab in enumerate(LABELS):
    fpr[lab], tpr[lab], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[lab] = auc(fpr[lab], tpr[lab])

# --- Macro-average ROC ------------------------------------------------------
# 1) Aggregate all FPR points
all_fpr = np.unique(np.concatenate([fpr[lab] for lab in LABELS]))
# 2) Compute mean TPR
mean_tpr = np.zeros_like(all_fpr)
for lab in LABELS:
    mean_tpr += np.interp(all_fpr, fpr[lab], tpr[lab])
mean_tpr /= len(LABELS)
roc_auc["macro"] = auc(all_fpr, mean_tpr)

# --- Plot ------------------------------------------------------------------
plt.figure(figsize=(7, 6))
# Per-class curves
for lab, color in zip(LABELS, ["C0", "C1", "C2", "C3"]):
    plt.plot(fpr[lab], tpr[lab], color=color, lw=2,
             label=f"{lab.replace('_', ' ').title()} (AUC = {roc_auc[lab]:.2f})")

# Macro-average curve
plt.plot(all_fpr, mean_tpr, color="black", lw=3, linestyle="--",
         label=f"Macro average (AUC = {roc_auc['macro']:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)   # reference line
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves – One-Vs-Rest Logistic Regression")
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig("roc_multi.png", dpi=300)
print("✓ ROC figure saved as roc_multi.png")
