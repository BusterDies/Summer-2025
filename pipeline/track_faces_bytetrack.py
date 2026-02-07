#!/usr/bin/env python3
"""
track_faces_bytetrack.py

Input:
  faces.csv from extract_faces_insightface.py with columns:
    frame_idx,time_sec,x1,y1,x2,y2,det_score

Output:
  faces_tracked.csv with added columns:
    track_id,track_iou

Important:
  This ByteTrack implementation requires non-None img_info/img_size in tracker.update().

Prints:
  TRACKED_CSV_PATH:<path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
if not hasattr(np, "float"):
    np.float = float
    
import pandas as pd

from yolox.tracker.byte_tracker import BYTETracker


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b = [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class _Args:
    # ByteTrack expects an args-like object
    def __init__(self, track_thresh=0.25, match_thresh=0.8, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = False


def _get_hw_from_video(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video to read H/W: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid frame size from video: (H,W)=({h},{w}) for {video_path}")
    return h, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True, help="Video used to get frame H/W (required by ByteTrack)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--track-thresh", type=float, default=0.25)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--min-iou-assign", type=float, default=0.30, help="IoU threshold to assign a det to a track")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    required = {"frame_idx", "time_sec", "x1", "y1", "x2", "y2", "det_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {args.in_csv}: {sorted(missing)}")

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["det_score"] = df["det_score"].astype(float)

    H, W = _get_hw_from_video(args.video)

    tracker = BYTETracker(
        _Args(track_thresh=args.track_thresh, match_thresh=args.match_thresh, track_buffer=args.track_buffer),
        frame_rate=args.fps,
    )

    track_id_col = np.full(len(df), -1, dtype=int)
    track_iou_col = np.full(len(df), np.nan, dtype=float)

    for frame_idx, g in df.groupby("frame_idx", sort=True):
        dets = g[["x1", "y1", "x2", "y2", "det_score"]].to_numpy(dtype=np.float32)
        if dets.shape[0] == 0:
            continue

        # Key fix: img_info and img_size must be non-None in this ByteTrack version
        tracks = tracker.update(dets, img_info=(H, W), img_size=(H, W))

        track_boxes = []
        track_ids = []
        for t in tracks:
            tlbr = np.asarray(t.tlbr, dtype=np.float32)  # [x1,y1,x2,y2]
            track_boxes.append(tlbr)
            track_ids.append(int(t.track_id))

        if not track_boxes:
            continue

        track_boxes = np.stack(track_boxes, axis=0)

        for idx, det_row in g.iterrows():
            det_box = det_row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
            ious = np.array([iou_xyxy(det_box, tb) for tb in track_boxes], dtype=np.float32)
            best = int(np.argmax(ious))
            best_iou = float(ious[best])

            if best_iou >= args.min_iou_assign:
                track_id_col[idx] = track_ids[best]
                track_iou_col[idx] = best_iou

    out = df.copy()
    out["track_id"] = track_id_col
    out["track_iou"] = track_iou_col
    out.to_csv(args.out_csv, index=False)

    print(f"TRACKED_CSV_PATH:{args.out_csv}")


if __name__ == "__main__":
    main()
