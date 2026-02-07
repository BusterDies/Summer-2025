#!/usr/bin/env python3
"""
extract_faces_insightface.py

Runs InsightFace on a video and outputs:
  - <prefix>_faces.csv : detections per frame (multi-face)
  - <prefix>_faces_emb.npy : 512-d embeddings per detection (same row order as csv)

CSV columns:
  frame_idx,time_sec,x1,y1,x2,y2,det_score

Prints:
  FACES_CSV_PATH:<path>
  FACES_EMB_PATH:<path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from insightface.app import FaceAnalysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="Path to input video (mp4 recommended)")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--prefix", type=str, required=True, help="Prefix for output files")
    p.add_argument("--ctx-id", type=int, default=0, help="InsightFace ctx_id: 0=GPU, -1=CPU")
    p.add_argument("--det-size", type=int, nargs=2, default=(640, 640), help="Detector size, e.g. 640 640")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    faces_csv = args.out / f"{args.prefix}_faces.csv"
    faces_emb = args.out / f"{args.prefix}_faces_emb.npy"

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=args.ctx_id, det_size=tuple(args.det_size))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    rows = []
    embs = []

    frame_idx = -1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        time_sec = frame_idx / fps
        faces = app.get(frame_bgr)

        for f in faces:
            # bbox: [x1,y1,x2,y2]
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            score = float(getattr(f, "det_score", 0.0))
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": float(time_sec),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "det_score": score,
                }
            )
            # embedding is always present for FaceAnalysis; ensure float32
            emb = np.asarray(f.embedding, dtype=np.float32)
            embs.append(emb)

    cap.release()

    if not rows:
        raise SystemExit("No faces detected. Nothing written.")

    df = pd.DataFrame(rows)
    df.to_csv(faces_csv, index=False)

    emb_arr = np.vstack(embs).astype(np.float32)
    np.save(faces_emb, emb_arr)

    print(f"FACES_CSV_PATH:{faces_csv}")
    print(f"FACES_EMB_PATH:{faces_emb}")


if __name__ == "__main__":
    main()
