#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def extract_faces(video_path: str, out_dir: str, det_size: int = 640, every_n: int = 1, ctx_id: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- init InsightFace ---
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    csv_path = out_dir / "faces.csv"
    emb_path = out_dir / "faces_emb.npy"

    rows = []
    embs = []

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if every_n > 1 and (frame_idx % every_n != 0):
            continue

        t_sec = frame_idx / fps
        faces = app.get(frame)

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            score = float(f.det_score)

            emb = np.asarray(f.embedding, dtype=np.float32)
            embs.append(emb)

            rows.append({
                "frame_idx":frame_idx,
                "time_sec": f"{t_sec:.6f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_score": f"{score:.6f}",
            })

    cap.release()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "time_sec", "x1", "y1", "x2", "y2", "det_score"])
        writer.writeheader()
        writer.writerows(rows)



        if embs:
            np.save(emb_path, np.stack(embs, axis=0))
        else:
            np.save(emb_path, np.zeros((0,512), dtype=np.float32))
        
        print(f"Saved {len(rows)} detections -> {csv_path}")
        print(f"Saved embeddings -> {emb_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--det-size",type=int, default=640)
    p.add_argument("--every-n", type=int, default=1)
    p.add_argument("--ctx-id",type=int, default=0, help="0=GPU, -1=CPU")
    args = p.parse_args()

    extract_faces(args.video, args.out, args.det_size, args.every_n, args.ctx_id)


if __name__ == "__main__":
    main()