#!/usr/bin/env python3
"""
extract_au10.py

Two modes:

1) Legacy single-face mode (default)
   - Transcodes IN_DIR/<video> -> OUT_DIR/<stem>.mp4 (30fps)
   - Uses OpenFace RetinaFace detector to pick *one* face per frame
   - Writes <stem>_au.csv

2) Tracked multi-face mode (recommended)
   - Requires --faces-tracked-csv produced by InsightFace + ByteTrack
   - Crops *every* tracked face per frame and predicts AUs per (frame, track_id)
   - Writes <stem>_au_tracked.csv containing track_id + metadata + AU columns

Designed to be called from run_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import torch

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor


# ───────── USER CONFIG ────────────────────────────────────────────────
IN_DIR = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR_DEF = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
WEIGHTS_DIR = Path("/home/pachethridge47/weights")
TARGET_FPS = 30
FFMPEG = "ffmpeg"
CSV_SUFFIX = "_au.csv"
CSV_SUFFIX_TRACKED = "_au_tracked.csv"
FACE_JPG = "_face.jpg"  # legacy snapshot suffix
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
# ─────────────────────────────────────────────────────────────────────


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def unique_path_pair(out_dir: Path, stem: str, suffix: str) -> tuple[Path, Path]:
    """Return unique (mp4_path, csv_path) without overwriting."""
    idx = 0
    while True:
        suf = f"_{idx}" if idx else ""
        mp4 = out_dir / f"{stem}{suf}.mp4"
        csv = out_dir / f"{stem}{suf}{suffix}"
        if not mp4.exists() and not csv.exists():
            return mp4, csv
        idx += 1


def next_json_path(out_dir: Path) -> Path:
    nums = [
        int(m.group(1))
        for f in out_dir.glob("task-*.json")
        if (m := re.fullmatch(r"task-(\d+)\.json", f.name))
    ]
    return out_dir / f"task-{(max(nums) + 1) if nums else 1}.json"


def transcode(src: Path, dst: Path):
    encoders = subprocess.check_output([FFMPEG, "-v", "quiet", "-encoders"], text=True)
    enc = "libopenh264" if " libopenh264 " in encoders else "mpeg4"
    tmp = dst.with_suffix(".tmp.mp4")
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={TARGET_FPS}",
            "-c:v",
            enc,
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ]
    )
    tmp.rename(dst)
    print(f"    ✓ transcoded with {enc}")


def fmt_time(t: float) -> str:
    us = int(round(t * 1000)) * 1000
    h, us = divmod(us, 3_600_000_000)
    m, us = divmod(us, 60_000_000)
    s, us = divmod(us, 1_000_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{us:06d}"


# ───────── models ─────────────────────────────────────────────────────
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT:
# FaceDetector (RetinaFace) is ONLY needed for legacy single-face mode.
# In tracked mode, we use InsightFace/ByteTrack boxes and do not want OpenFace
# trying to load './weights/mobilenetV1X0.25_pretrain.tar'.
FD = None  # lazy init

PREDICTOR = MultitaskPredictor(WEIGHTS_DIR / "MTL_backbone.pth", device=DEV)

def _get_fd():
    global FD
    if FD is None:
        # If you ever run legacy mode, OpenFace's RetinaFace code tries to load
        # './weights/mobilenetV1X0.25_pretrain.tar' using the current working directory.
        # We temporarily chdir to the parent of WEIGHTS_DIR so ./weights exists.
        cwd = os.getcwd()
        try:
            os.chdir(str(WEIGHTS_DIR.parent))
            FD = FaceDetector(WEIGHTS_DIR / "Alignment_RetinaFace.pth", device=DEV)
        finally:
            os.chdir(cwd)
    return FD


# ---------- legacy two-stage face detector (single face) ---------------
UPSCALE = 2


def detect_face_single(bgr_frame):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, bgr_frame)
        crop, _ = _get_fd().get_face(tmp.name)
    os.remove(tmp.name)
    if crop is not None and crop.size:
        return crop

    big = cv2.resize(bgr_frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, big)
        crop, _ = _get_fd().get_face(tmp.name)
    os.remove(tmp.name)
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    crop = cv2.resize(
        crop,
        (max(1, w // UPSCALE), max(1, h // UPSCALE)),
        interpolation=cv2.INTER_AREA,
    )
    return crop


@torch.inference_mode()
def extract_single_face(mp4: Path, csv_out: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(mp4))
    rows = []
    frame_idx = -1
    face_saved = False
    snapshot = out_dir / f"{mp4.stem}{FACE_JPG}"

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        crop = detect_face_single(bgr)
        if crop is None:
            continue

        if not face_saved:
            cv2.imwrite(str(snapshot), crop)
            print(f"    ✓ face snapshot → {snapshot.name}")
            face_saved = True

        _, _, au = PREDICTOR.predict(crop)
        rows.append(
            {
                "sec": frame_idx / TARGET_FPS,
                **dict(zip(AU_NAMES, au.squeeze().tolist())),
            }
        )

    cap.release()
    if not rows:
        sys.exit("⚠ No faces detected – CSV not written")

    df = pd.DataFrame(rows)
    df["time"] = df["sec"].apply(fmt_time)
    df = df[["time"] + AU_NAMES]
    df.to_csv(csv_out, index=False, float_format="%.16f")
    print(f"    ✓ wrote {len(df):,} rows")


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(0, min(w, x2))
    y2c = max(0, min(h, y2))
    if x2c <= x1c:
        x2c = min(w, x1c + 1)
    if y2c <= y1c:
        y2c = min(h, y1c + 1)
    return x1c, y1c, x2c, y2c


@torch.inference_mode()
def extract_tracked_faces(
    mp4: Path,
    faces_tracked_csv: Path,
    csv_out: Path,
    top_k: int | None,
    min_det_score: float,
    min_box: int,
):
    """Predict AUs per (frame_idx, track_id) using tracked bboxes."""

    df = pd.read_csv(faces_tracked_csv)
    required = {"frame_idx", "time_sec", "x1", "y1", "x2", "y2", "det_score", "track_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {faces_tracked_csv}: {sorted(missing)}")

    # Normalize types
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["time_sec"] = df["time_sec"].astype(float)
    df["det_score"] = df["det_score"].astype(float)

    # Group detections per frame for fast lookup
    grouped = {k: g for k, g in df.groupby("frame_idx", sort=True)}

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {mp4}")

    rows: list[dict] = []
    frame_idx = -1

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        g = grouped.get(frame_idx)
        if g is None or len(g) == 0:
            continue

        # Optional filtering: keep top-K by det_score
        if top_k is not None and top_k > 0 and len(g) > top_k:
            g = g.sort_values("det_score", ascending=False).head(top_k)
        else:
            g = g.sort_values("det_score", ascending=False)

        H, W = bgr.shape[:2]

        for _, det in g.iterrows():
            score = float(det.det_score)
            if score < min_det_score:
                continue

            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, W, H)
            bw, bh = x2 - x1, y2 - y1
            if bw < min_box or bh < min_box:
                continue

            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            au_success = 1
            try:
                _, _, au = PREDICTOR.predict(crop)
                au_vals = au.squeeze().tolist()
            except Exception:
                au_success = 0
                au_vals = [float("nan")] * len(AU_NAMES)

            t_sec = float(det.time_sec)
            rows.append(
                {
                    "time": fmt_time(t_sec),
                    "frame_idx": int(frame_idx),
                    "time_sec": t_sec,
                    "track_id": int(det.track_id),
                    "det_score": score,
                    "track_iou": float(det.track_iou) if "track_iou" in det else float("nan"),
                    "bbox_w": int(bw),
                    "bbox_h": int(bh),
                    "au_success": int(au_success),
                    **dict(zip(AU_NAMES, au_vals)),
                }
            )

    cap.release()

    if not rows:
        sys.exit("⚠ No tracked face crops produced – CSV not written")

    out_df = pd.DataFrame(rows)
    meta_cols = [
        "time",
        "frame_idx",
        "time_sec",
        "track_id",
        "det_score",
        "track_iou",
        "bbox_w",
        "bbox_h",
        "au_success",
    ]
    out_df = out_df[meta_cols + AU_NAMES]
    out_df.to_csv(csv_out, index=False, float_format="%.16f")
    print(f"    ✓ wrote {len(out_df):,} rows")


def write_json(out_dir: Path, mp4: Path, csv: Path):
    base = f"/data/local-files/?d={out_dir.name}"
    payload = [{"data": {"video": f"{base}/{mp4.name}", "au_csv": f"{base}/{csv.name}"}}]
    next_json_path(out_dir).write_text(json.dumps(payload, indent=2))
    print("LS task JSON → written")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("video", help="video filename inside IN_DIR (legacy), still required")
    cli.add_argument("--out-dir", type=Path, default=OUT_DIR_DEF)
    cli.add_argument("--skip-json", action="store_true")

    cli.add_argument("--input-mp4", type=Path, default=None, help="Use already-transcoded mp4 and skip transcoding")
    cli.add_argument("--faces-tracked-csv", type=Path, default=None, help="Enables multi-face tracked AU extraction")
    cli.add_argument("--out-csv", type=Path, default=None, help="Explicit output CSV path")
    cli.add_argument("--top-k", type=int, default=None, help="Run AUs only for top-K faces per frame by det_score")
    cli.add_argument("--min-det-score", type=float, default=0.20)
    cli.add_argument("--min-box", type=int, default=32)

    args = cli.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    src = IN_DIR / args.video
    if args.input_mp4 is None:
        if not src.exists():
            sys.exit(f"✖  {src} not found")
        mp4_out, csv_default = unique_path_pair(out_dir, src.stem, CSV_SUFFIX)
        print(f"Transcoding   → {mp4_out.name}")
        transcode(src, mp4_out)
    else:
        mp4_out = args.input_mp4
        if not mp4_out.exists():
            sys.exit(f"✖  {mp4_out} not found")
        _, csv_default = unique_path_pair(
            out_dir,
            mp4_out.stem,
            CSV_SUFFIX_TRACKED if args.faces_tracked_csv else CSV_SUFFIX,
        )

    csv_out = args.out_csv or csv_default

    print(f"VIDEO_PATH:{mp4_out}")

    if args.faces_tracked_csv:
        print(f"Extracting AU (tracked) → {csv_out.name}")
        extract_tracked_faces(
            mp4_out,
            args.faces_tracked_csv,
            csv_out,
            top_k=args.top_k,
            min_det_score=args.min_det_score,
            min_box=args.min_box,
        )
    else:
        print(f"Extracting AU (single) → {csv_out.name}")
        extract_single_face(mp4_out, csv_out, out_dir)

    print(f"CSV_PATH:{csv_out}")

    if not args.skip_json:
        write_json(out_dir, mp4_out, csv_out)


if __name__ == "__main__":
    main()
