#!/usr/bin/env python3
import argparse
import os
import cv2
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--max_frames", type=int, default=600)
    ap.add_argument("--every_n", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.tracks_csv)
    for col in ["frame_idx","x1","y1","x2","y2","track_id"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {args.tracks_csv}")

    # group detections by frame
    by_frame = {int(k): g for k, g in df.groupby("frame_idx")}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out_video, fourcc, fps/args.every_n, (w, h))

    f = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % args.every_n != 0:
            f += 1
            continue

        if f in by_frame:
            g = by_frame[f]
            for _, r in g.iterrows():
                x1,y1,x2,y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                tid = int(r.track_id)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        out.write(frame)
        written += 1
        f += 1
        if written >= args.max_frames:
            break

    cap.release()
    out.release()
    print("Wrote:", args.out_video)

if __name__ == "__main__":
    main()
