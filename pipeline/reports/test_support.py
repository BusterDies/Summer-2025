# test_support.py  – counts per class in the test fold
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

LABELS = [
    "mood_dysregulation",
    "problems_communicating",
    "confusion_disorientation",
    "memory_loss",
]

BASE = Path("/home/pachethridge47/Summer-2025/datasets")
X = np.load(BASE / "X.npy")
Y = np.load(BASE / "Y.npy")

_, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.20, stratify=Y, random_state=42)

print(f"Test-fold windows = {X_test.shape[0]}")
for i, lab in enumerate(LABELS):
    print(f"{lab:24s} {Y_test[:,i].sum():3.0f}")
