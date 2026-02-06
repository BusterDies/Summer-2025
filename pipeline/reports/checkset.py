#!/usr/bin/env python3
"""
count_windows.py  –  Stats on labelled 2-s windows
--------------------------------------------------
Sliding-window rule : 2.0-s window, 1.0-s hop (50 % overlap)

X / Y files for the current dataset build:
    /home/pachethridge47/Summer-2025/datasets/X.npy
    /home/pachethridge47/Summer-2025/datasets/Y.npy
"""

import sys, json, math
from pathlib import Path
from collections import defaultdict

SYMPTOMS = [
    "problems_communicating",
    "memory_loss",
    "confusion_disorientation",
    "mood_dysregulation",
]

def hms_to_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def main(js_path: Path, win=2.0, hop=1.0):
    data = json.loads(js_path.read_text())
    per_class = defaultdict(int)
    total_win = 0
    total_sec = 0
    clip_ids  = set()

    for task in data:
        clip_ids.add(task.get("file_upload") or task["data"].get("video"))
        for ann in task["annotations"]:
            for res in ann["result"]:
                labels = res["value"].get("timeserieslabels", [])
                t0 = hms_to_sec(res["value"]["start"])
                t1 = hms_to_sec(res["value"]["end"])
                dur = max(0.0, t1 - t0)
                if dur == 0:
                    continue
                n_win = 1 if dur <= win else 1 + math.floor((dur - win) / hop)
                total_win += n_win
                total_sec += dur
                for lab in labels:
                    per_class[lab] += n_win

    print("==== Dementia-Symptom Window Stats ====")
    print(f"Source clips        : {len(clip_ids)}")
    mm, ss = divmod(total_sec, 60)
    print(f"Annotated duration  : {int(mm):02d}:{ss:05.2f} (mm:ss)")
    print(f"Window length       : {win:.1f} s  (hop {hop:.1f} s)")
    print(f"Total windows       : {total_win:,}")
    print("\nPer-class counts")
    for lab in SYMPTOMS:
        print(f"  {lab:24s} {per_class[lab]:6}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python count_windows.py export.json")
    main(Path(sys.argv[1]))
