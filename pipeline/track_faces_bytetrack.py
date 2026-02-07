#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
if not hasattr(np, "float"):
    np.float = float
from yolox.tracker.byte_tracker import BYTETracker


class BTArgs:
    # Reasonable defaults for face tracking; we can tune later
    track_thresh = 0.4
    track_buffer = 60
    match_thresh = 0.8
    mot20 = False


def compute_iou(a, b):
    # a,b: [x1,y1,x2,y2]
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH

    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (areaA + areaB - inter + 1e-6))


def main():
    ap = argparse.ArgumentParser(description="Assign ByteTrack track_id to per-frame face detections CSV.")
    ap.add_argument("--in_csv", required=True, help="Input faces.csv (frame_idx,time_sec,x1,y1,x2,y2,det_score)")
    ap.add_argument("--out_csv", required=True, help="Output faces_tracked.csv with track_id")
    ap.add_argument("--fps", type=float, default=30.0, help="Video FPS used by ByteTrack")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = {"frame_idx", "time_sec", "x1", "y1", "x2", "y2", "det_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.in_csv}: {sorted(missing)}")

    tracker = BYTETracker(BTArgs(), frame_rate=args.fps)

    out_rows = []

    for frame_idx, g in df.groupby("frame_idx", sort=True):
        dets = g[["x1", "y1", "x2", "y2", "det_score"]].to_numpy(dtype=np.float32)

        # ByteTrack update returns track objects with tlbr + track_id
        online_targets = tracker.update(dets, (1, 1), (1, 1))  # img sizes not needed for tlbr IoU match here

        tracks = []
        for t in online_targets:
            tlbr = np.array(t.tlbr, dtype=np.float32)  # [x1,y1,x2,y2]
            tracks.append((tlbr, int(t.track_id)))

        # Assign each original detection a track_id by best IoU with current tracks
        for _, row in g.iterrows():
            bb = np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32)

            best_iou = 0.0
            best_id = -1
            for tlbr, tid in tracks:
                v = compute_iou(bb, tlbr)
                if v > best_iou:
                    best_iou = v
                    best_id = tid

            out = row.to_dict()
            out["track_id"] = int(best_id)
            out["track_iou"] = float(best_iou)
            out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)

    # Helpful stats
    valid = out_df[out_df["track_id"] != -1]
    print(f"Saved tracked detections -> {args.out_csv}")
    print(f"Rows: {len(out_df)} | Valid track rows: {len(valid)}")
    if len(valid):
        print("Unique track_ids:", valid["track_id"].nunique())
        print("Top track_id counts:")
        print(valid["track_id"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

