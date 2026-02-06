# ls_loader_v2.py – sliding-window version
from __future__ import annotations
import json, re, argparse, itertools, pprint
from pathlib import Path
from typing import Iterable, Dict, Any
import pandas as pd
import numpy as np

# edit these if needed
ANN_DIR   = Path("/home/pachethridge47/Summer-2025/annotations_movies")
CSV_ROOT  = Path("/home/pachethridge47/Summer-2025/processeddata")

# valid dementia symptom labels
VALID_LABELS = {
    "mood_dysregulation",
    "problems_communicating",
    "confusion_disorientation",
    "memory_loss"
}

# window parameters
WIN = 3.0   # seconds
HOP = 3.0   # seconds

def _t2s(hms: str) -> float:
    """HH:MM:SS.sss → seconds"""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def _slice_csv(csv_path: Path, t0: float, t1: float) -> pd.DataFrame:
    """Return slice of CSV rows between t0 and t1 (seconds)."""
    df = pd.read_csv(csv_path)
    df["__sec__"] = pd.to_timedelta(df["time"]).dt.total_seconds()
    return df.loc[(df["__sec__"] >= t0) & (df["__sec__"] < t1)].drop(columns=["time", "__sec__"])

def _resolve(url: str) -> Path:
    """Convert LS /data/local-files/?d=... URL to real path under CSV_ROOT."""
    return Path(re.sub(r"^/data/local-files/\?d=", str(CSV_ROOT) + "/", url))

def iter_samples(
    ann_dir: Path | str = ANN_DIR,
    csv_root: Path | str = CSV_ROOT,
) -> Iterable[Dict[str, Any]]:
    """
    Iterate over Label Studio export files and yield sliding-window samples.
    Each sample corresponds to a 2-s window (1-s hop) within an annotated region.
    """
    ann_dir  = Path(ann_dir)
    csv_root = Path(csv_root)

    for export_file in ann_dir.glob("*.json"):
        data = json.loads(export_file.read_text())

        for task in data:
            paths = {k: _resolve(v) for k, v in task["data"].items()}
            ann   = task["annotations"][0]["result"]
            video = paths["video"]
            transcript = json.loads(paths["transcript_json"].read_text())

            for region in ann:
                if region["from_name"] != "symptom":
                    continue

                val    = region["value"]
                T0, T1 = _t2s(val["start"]), _t2s(val["end"])
                labels = [lbl for lbl in val.get("timeserieslabels", []) if lbl in VALID_LABELS]
                if not labels:
                    continue

                # slide 2-s windows with 1-s hop
                cur = T0
                while cur + WIN <= T1:
                    au_df  = _slice_csv(paths["au_csv"],      cur, cur + WIN)
                    pr_df  = _slice_csv(paths["prosody_csv"], cur, cur + WIN)

                    txt = " ".join(
                        seg["text"]
                        for seg in transcript
                        if not (seg["end"] < cur or seg["start"] > cur + WIN)
                    )

                    yield dict(
                        video=str(video),
                        labels=labels,
                        t0=cur,
                        t1=cur + WIN,
                        au=au_df.to_numpy(),
                        prosody=pr_df.to_numpy(),
                        text=txt,
                    )
                    cur += HOP

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="show first N samples")
    args = ap.parse_args()

    for sample in itertools.islice(iter_samples(), args.n):
        pprint.pprint(
            {
                k: (v.shape if isinstance(v, np.ndarray) else v[:60] + "…")
                if k in {"au", "prosody", "text"} else v
                for k, v in sample.items()
            }
        )
