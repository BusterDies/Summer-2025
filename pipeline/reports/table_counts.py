#!/usr/bin/env python3
"""
table_counts.py – totals and percentages for the full labelled set
"""

from pathlib import Path
from collections import Counter          # ← fixed
import json, math

LABELS = [
    "problems_communicating",
    "memory_loss",
    "confusion_disorientation",
    "mood_dysregulation",
]

def hms_to_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def table_counts(json_path, win=2.0, hop=1.0):
    counts = Counter()
    total_windows = 0
    data = json.loads(Path(json_path).read_text())

    for task in data:
        for ann in task["annotations"]:
            for res in ann["result"]:
                labs = res["value"].get("timeserieslabels", [])
                t0   = hms_to_sec(res["value"]["start"])
                t1   = hms_to_sec(res["value"]["end"])
                dur  = max(0, t1 - t0)
                if dur == 0:
                    continue
                n_win = 1 if dur <= win else 1 + math.floor((dur - win)/hop)
                total_windows += n_win
                for lb in labs:
                    counts[lb] += n_win

    print(f"Total labelled windows = {total_windows:,}\n")
    for lab in LABELS:
        pct = counts[lab] / total_windows * 100 if total_windows else 0
        print(f"{lab:24s} {counts[lab]:5}   {pct:4.0f}%")

if __name__ == "__main__":
    table_counts(
        "/home/pachethridge47/Summer-2025/annotations/"
        "project-6-at-2025-07-16-16-20-981c26d5.json"
    )
