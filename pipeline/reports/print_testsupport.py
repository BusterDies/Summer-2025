#!/usr/bin/env python3
"""
print_test_support.py — show class counts in the 20 % test fold
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

BASE = Path("/home/pachethridge47/Summer-2025/datasets")
X = np.load(BASE / "X.npy")
Y = np.load(BASE / "Y.npy")

# ➊  Use the SAME order you passed to MultiLabelBinarizer in build_dataset.py
LABELS = [
    "mood_dysregulation",
    "problems_communicating",
    "confusion_disorientation",
    "memory_loss",
]

_, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.20, stratify=Y, random_state=42)

print("Support counts in 20 % test fold")
for i, lab in enumerate(LABELS):
    print(f"{lab:24s} {int(Y_test[:, i].sum()):3}")
