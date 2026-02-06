#!/usr/bin/env python
"""
build_dataset_v2.py – create normalised feature matrix X and multilabel target Y
from your Label-Studio exports, expanded into 2-s windows with 1-s hop.
If you pass --train it also fits & saves a baseline One-vs-Rest Logistic-Regression model
using GroupShuffleSplit (whole clips kept together).
Supports --history N: concatenate features from the last N windows as input context.
"""
from __future__ import annotations
from pathlib import Path
import argparse, datetime as dt, joblib, numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict, deque   # NEW

# ───────────────────────── paths ──────────────────────────
ROOT      = Path("/home/pachethridge47/Summer-2025")
ANN_DIR   = ROOT / "annotations_movies"
DATA_DIR  = ROOT / "processeddata/task-1"
OUT_DIR   = ROOT / "datasets_movies_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(str(ROOT / "pipeline"))
from ls_loader_v2 import iter_samples   # ← sliding windows loader

# ──────────────────── label vocabulary ───────────────────
SYMPTOMS = [
    "mood_dysregulation",
    "problems_communicating",
    "confusion_disorientation",
    "memory_loss",
]
mlb = MultiLabelBinarizer(classes=SYMPTOMS).fit([SYMPTOMS])

# ───────────────── text embedder  ─────────────────────────
from sentence_transformers import SentenceTransformer
TXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TXT_DIM   = TXT_MODEL.get_sentence_embedding_dimension()

def embed(txt: str) -> np.ndarray:
    if not txt.strip():
        return np.zeros(TXT_DIM, dtype=np.float32)
    return TXT_MODEL.encode(txt, show_progress_bar=False)

# ───────────────── helper: safe mean ─────────────────────
au_missing, prosody_missing = 0, 0  # counters

def safe_mean(arr: np.ndarray, dim: int, counter_name: str) -> np.ndarray:
    global au_missing, prosody_missing
    if arr.size == 0:
        if counter_name == "au":
            au_missing += 1
        elif counter_name == "prosody":
            prosody_missing += 1
        return np.zeros(dim, dtype=np.float32)
    return arr.mean(axis=0)

expected_au_dim = 8       # AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26
expected_prosody_dim = 88 # eGeMAPS LLD features

# ───────────────── build X / Y / groups ──────────────────
X_rows, Y_rows, groups = [], [], []

# NEW: add per-video buffers for history
feat_buf = defaultdict(lambda: deque(maxlen=0))   # will reset later

# ─────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="fit & save baseline OvR-LogReg after building X/Y")
    ap.add_argument("--history", type=int, default=0,
                    help="number of previous windows to concatenate as context")  # NEW
    args = ap.parse_args()

    # reset buffer with correct maxlen
    feat_buf = defaultdict(lambda: deque(maxlen=args.history))

    for samp in tqdm(iter_samples(ANN_DIR, DATA_DIR), desc="windows"):
        au_vec  = safe_mean(samp["au"], expected_au_dim, "au")
        pro_vec = safe_mean(samp["prosody"], expected_prosody_dim, "prosody")
        txt_vec = embed(samp["text"])
        x_now   = np.hstack([au_vec, pro_vec, txt_vec]).astype(np.float32)

        vid = str(samp["video"])
        past = list(feat_buf[vid])

        # pad with zeros if not enough history yet
        if len(past) < args.history:
            pad = [np.zeros_like(x_now) for _ in range(args.history - len(past))]
            past = pad + past

        # concatenate past + current
        x_ctx = np.hstack(past + [x_now]).astype(np.float32)

        X_rows.append(x_ctx)
        Y_rows.append(mlb.transform([samp["labels"]])[0])
        groups.append(vid)

        # update buffer
        feat_buf[vid].append(x_now)

    X = np.vstack(X_rows).astype(np.float32)
    Y = np.vstack(Y_rows).astype(np.float32)
    groups = np.array(groups)
    print("raw shapes:", X.shape, Y.shape)

    # ───────────────── normalise & persist ───────────────────
    scaler = StandardScaler().fit(X)
    Xn     = scaler.transform(X)
    print("µ σ →", scaler.mean_[:5].round(3), scaler.scale_[:5].round(3))

    np.save(OUT_DIR / "X.npy", Xn)
    np.save(OUT_DIR / "Y.npy", Y)
    np.save(OUT_DIR / "groups.npy", groups)
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    print("✓ saved X.npy, Y.npy, groups.npy & scaler.joblib →", OUT_DIR)

    # ─────────────────── training ───────────────────────────
    def train_baseline(Xn: np.ndarray, Y: np.ndarray, groups: np.ndarray):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
        train_idx, test_idx = next(gss.split(Xn, Y, groups=groups))

        Xtr, Xte = Xn[train_idx], Xn[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000,
                               solver="liblinear",
                               class_weight="balanced",
                               n_jobs=-1))
        clf.fit(Xtr, Ytr)

        # generate classification report
        report = classification_report(
            Yte, clf.predict(Xte), target_names=SYMPTOMS, zero_division=0)
        print(report)

        # save model + indices + report
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = OUT_DIR / f"clf_{stamp}.joblib"
        joblib.dump(clf, model_path)
        np.save(OUT_DIR / f"train_idx_{stamp}.npy", train_idx)
        np.save(OUT_DIR / f"test_idx_{stamp}.npy", test_idx)

        report_path = OUT_DIR / f"classification_report_{stamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print("✓ baseline model saved →", model_path)
        print("✓ train/test indices saved → "
              f"train_idx_{stamp}.npy, test_idx_{stamp}.npy")
        print("✓ classification report saved →", report_path)

        # report missing data stats
        total = len(groups)
        print(f"\nMissing-data summary:")
        print(f"  AU windows with no data       : {au_missing} / {total}")
        print(f"  Prosody windows with no data  : {prosody_missing} / {total}")

    if args.train:
        train_baseline(Xn, Y, groups)
