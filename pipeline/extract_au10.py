#!/usr/bin/env python3
"""
extract_au10.py  –  AU-9 logic + tiny hooks for run_pipeline.py
(added: single-frame face snapshot for visual QA)
"""
from __future__ import annotations
import argparse, cv2, json, os, re, subprocess, sys, tempfile, torch, pandas as pd
from pathlib import Path
from openface.face_detection  import FaceDetector
from openface.multitask_model import MultitaskPredictor

# ───────── USER CONFIG (unchanged) ───────────────────────────────────────
IN_DIR      = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR_DEF = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
WEIGHTS_DIR = Path("/home/pachethridge47/weights")
TARGET_FPS  = 30
FFMPEG      = "ffmpeg"
CSV_SUFFIX  = "_au.csv"
FACE_JPG    = "_face.jpg"          # ← NEW: snapshot suffix
AU_NAMES    = ["AU1","AU2","AU4","AU6","AU9","AU12","AU25","AU26"]
# ─────────────────────────────────────────────────────────────────────────

# ---------- CLI (unchanged) ---------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("video", help="video file inside IN_DIR")
cli.add_argument("--out-dir", type=Path, default=OUT_DIR_DEF,
                 help="folder for mp4/csv/json (default: processeddata/task-1)")
cli.add_argument("--skip-json", action="store_true",
                 help="suppress the LS-JSON file (pipeline mode)")
args = cli.parse_args()
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ------------------------------------------------------
def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def unique_video_csv(stem: str) -> tuple[Path, Path]:
    idx = 0
    while True:
        suf = f"_{idx}" if idx else ""
        mp4 = OUT_DIR / f"{stem}{suf}.mp4"
        csv = OUT_DIR / f"{stem}{suf}{CSV_SUFFIX}"
        if not mp4.exists() and not csv.exists():
            return mp4, csv
        idx += 1

def next_json_path() -> Path:
    nums = [int(m.group(1))
            for f in OUT_DIR.glob("task-*.json")
            if (m := re.fullmatch(r"task-(\d+)\.json", f.name))]
    return OUT_DIR / f"task-{(max(nums)+1) if nums else 1}.json"

def transcode(src: Path, dst: Path):
    encoders = subprocess.check_output([FFMPEG, "-v", "quiet", "-encoders"], text=True)
    enc = "libopenh264" if " libopenh264 " in encoders else "mpeg4"
    tmp = dst.with_suffix(".tmp.mp4")
    run([FFMPEG, "-y", "-i", str(src),
         "-vf", f"fps={TARGET_FPS}",
         "-c:v", enc, "-pix_fmt", "yuv420p", str(tmp)])
    tmp.rename(dst)
    print(f"    ✓ transcoded with {enc}")

def fmt_time(t: float) -> str:
    us = int(round(t * 1000)) * 1000
    h, us = divmod(us, 3_600_000_000)
    m, us = divmod(us,     60_000_000)
    s, us = divmod(us,      1_000_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{us:06d}"

# ---------- models -------------------------------------------------------
DEV       = "cuda" if torch.cuda.is_available() else "cpu"
FD        = FaceDetector(WEIGHTS_DIR / "Alignment_RetinaFace.pth", device=DEV)
PREDICTOR = MultitaskPredictor(WEIGHTS_DIR / "MTL_backbone.pth",   device=DEV)

# ---------- two-stage face detector --------------------------------------
UPSCALE = 2      # enlarge frame this many times for the fallback pass

def detect_face(bgr_frame):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, bgr_frame)
        crop, _ = FD.get_face(tmp.name)
    os.remove(tmp.name)
    if crop is not None and crop.size:
        return crop
    big = cv2.resize(bgr_frame, None, fx=UPSCALE, fy=UPSCALE,
                     interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, big)
        crop, _ = FD.get_face(tmp.name)
    os.remove(tmp.name)
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (max(1, w // UPSCALE), max(1, h // UPSCALE)),
                      interpolation=cv2.INTER_AREA)
    return crop

# ---------- AU extraction ------------------------------------------------
@torch.inference_mode()
def extract(mp4: Path, csv_out: Path):
    cap, rows, frame = cv2.VideoCapture(str(mp4)), [], 0
    face_saved = False                                        # ← NEW flag
    snapshot   = OUT_DIR / f"{mp4.stem}{FACE_JPG}"            # jpg path
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame += 1
        crop = detect_face(bgr)
        if crop is None:
            continue
        # save the first successfully detected face
        if not face_saved:
            cv2.imwrite(str(snapshot), crop)
            print(f"    ✓ face snapshot → {snapshot.name}")
            face_saved = True

        _, _, au = PREDICTOR.predict(crop)
        rows.append({"sec": (frame-1)/TARGET_FPS,
                     **dict(zip(AU_NAMES, au.squeeze().tolist()))})
    cap.release()
    if not rows:
        sys.exit("⚠ No faces detected – CSV not written")
    df = pd.DataFrame(rows)
    df["time"] = df["sec"].apply(fmt_time)
    df = df[["time"] + AU_NAMES]
    df.to_csv(csv_out, index=False, float_format="%.16f")
    print(f"    ✓ wrote {len(df):,} rows")

def write_json(mp4: Path, csv: Path):
    base = f"/data/local-files/?d={OUT_DIR.name}"
    payload = [{
        "data": {
            "video":  f"{base}/{mp4.name}",
            "au_csv": f"{base}/{csv.name}"
        }
    }]
    next_json_path().write_text(json.dumps(payload, indent=2))
    print("LS task JSON → written")

# ---------- main ---------------------------------------------------------
def main():
    src = IN_DIR / args.video
    if not src.exists():
        sys.exit(f"✖  {src} not found")

    mp4, csv = unique_video_csv(src.stem)

    print(f"Transcoding   → {mp4.name}")
    transcode(src, mp4)
    print(f"VIDEO_PATH:{mp4}")

    print(f"Extracting AU → {csv.name}")
    extract(mp4, csv)
    print(f"CSV_PATH:{csv}")

    if not args.skip_json:
        write_json(mp4, csv)

if __name__ == "__main__":
    main()
