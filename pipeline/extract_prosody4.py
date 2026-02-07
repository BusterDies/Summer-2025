#!/usr/bin/env python3
# ---------------------------------------------
# extract_prosody4.py — vocal prosody extractor
# ---------------------------------------------
"""
Prosody extraction using openSMILE eGeMAPSv02 LLD.

Modes:
1) Legacy (default):
   python extract_prosody4.py Anita.mp4
   -> reads IN_DIR/Anita.mp4

2) Pipeline-aligned:
   python extract_prosody4.py Anita.mp4 --input-mp4 /path/to/Anita_6.mp4
   -> reads the transcoded mp4 and names output using that mp4 stem (Anita_6_prosody.csv)

Prints:
  VIDEO_PATH:<mp4 used>
  CSV_PATH:<prosody csv path>
"""

from __future__ import annotations

import argparse, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd


# ───────── USER CONFIG ────────────────────────────────────────────────
IN_DIR  = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR_DEF = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")

FRAME_HZ = 30
HOP_SEC = 1 / FRAME_HZ
FRAME_SIZE = 0.025
CSV_SUFFIX = "_prosody.csv"
FFMPEG = "ffmpeg"
# ─────────────────────────────────────────────────────────────────────


def wav_from_mp4(mp4: Path) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        [FFMPEG, "-y", "-i", str(mp4), "-ac", "1", "-ar", "16000", "-vn", str(tmp)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp


def fmt_ts(frame: int) -> str:
    us = frame * 1_000_000 // FRAME_HZ
    h, rem = divmod(us, 3_600_000_000)
    m, rem = divmod(rem, 60_000_000)
    s, us = divmod(rem, 1_000_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{us:06d}"


def idx_to_sec(idx):
    if isinstance(idx, pd.MultiIndex) and "end" in idx.names:
        idx = idx.get_level_values("end")
    elif isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(-1)

    if isinstance(idx, pd.TimedeltaIndex):
        sec = idx.total_seconds()
    else:
        sec = pd.to_numeric(idx, errors="coerce")
        if sec.isna().all():
            sec = pd.to_timedelta(idx, errors="coerce").dt.total_seconds()

    return (sec + FRAME_SIZE).to_numpy(float)


def unique_csv(stem: str, out_dir: Path) -> Path:
    for i in range(999):
        p = out_dir / f"{stem}{'' if i == 0 else f'_{i}'}{CSV_SUFFIX}"
        if not p.exists():
            return p
    raise RuntimeError("too many duplicates")


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("video", type=str, help="Original filename (used for legacy mode + metadata)")
    cli.add_argument("--skip-json", action="store_true")
    cli.add_argument("--out-dir", type=Path, default=OUT_DIR_DEF)

    # NEW: pipeline-aligned mode
    cli.add_argument("--input-mp4", type=Path, default=None, help="Use this mp4 (e.g., transcoded 30fps mp4)")
    cli.add_argument("--prefix", type=str, default=None, help="Override output prefix (default: mp4 stem)")

    args = cli.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose source mp4
    if args.input_mp4 is not None:
        mp4 = args.input_mp4
        if not mp4.exists():
            sys.exit(f"❌ {mp4} not found")
    else:
        mp4 = IN_DIR / args.video
        if not mp4.exists():
            sys.exit(f"❌ {mp4} not found")

    prefix = args.prefix or mp4.stem
    csv_path = unique_csv(prefix, out_dir)

    # openSMILE config
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        sampling_rate=16000,
    )

    print(f"VIDEO_PATH:{mp4}")

    wav = wav_from_mp4(mp4)
    try:
        df = smile.process_file(str(wav))
        sec = idx_to_sec(df.index)
        mask = ~np.isnan(sec)
        df = df.iloc[mask]
        sec = sec[mask]

        frame = np.floor(sec / HOP_SEC + 1e-9).astype(int)

        df.insert(0, "frame", frame)
        df = df.sort_values("frame").drop_duplicates("frame")

        # zero-frame align
        df["frame"] -= int(df["frame"].iloc[0])

        df.insert(0, "time", df["frame"].apply(fmt_ts))
        df.drop(columns="frame", inplace=True)

        df.to_csv(csv_path, index=False, float_format="%.16f")
    finally:
        wav.unlink(missing_ok=True)

    print(f"CSV_PATH:{csv_path}")

    if not args.skip_json:
        base = f"/data/local-files/?d={out_dir.name}"
        j = out_dir / "task-unified-prosody.json"
        # Keep the original IN_DIR video name in JSON for LS, but attach prosody from our chosen mp4 stem
        src_for_ls = (IN_DIR / args.video) if (IN_DIR / args.video).exists() else mp4
        j.write_text(
            json.dumps(
                [{"data": {"video": f"{base}/{Path(src_for_ls).name}", "prosody_csv": f"{base}/{csv_path.name}"}}],
                indent=2,
            )
        )
        print(f"LS-JSON  : {j}")


if __name__ == "__main__":
    main()
