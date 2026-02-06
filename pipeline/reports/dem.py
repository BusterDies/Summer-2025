#!/usr/bin/env python3
"""
total_windows_all.py – count every 2-s × 50 %-overlap window
                       in the *entire* 5-h corpus, whether labelled or not.
"""

from pathlib import Path
import json, math

WIN = 2.0    # window length  (s)
HOP = 1.0    # stride         (s)   50 % overlap

EXPORT = Path(
    "/home/pachethridge47/Summer-2025/annotations/"
    "project-6-at-2025-07-16-16-20-981c26d5.json"
)

def hms_to_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def main():
    data = json.loads(EXPORT.read_text())
    total = 0
    for task in data:
        # ① try duration field stored during import
        dur = task["data"].get("duration")
        # ② if missing, fall back to last end-time in annotations
        if dur is None:
            ends = [
                hms_to_sec(res["value"]["end"])
                for ann in task["annotations"]
                for res in ann["result"]
            ]
            dur = max(ends) if ends else 0
        total += 1 if dur <= WIN else 1 + math.floor((dur - WIN) / HOP)

    print(f"All possible 2-s windows (5-h corpus) : {total:,}")

if __name__ == "__main__":
    main()
