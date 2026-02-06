# ---------------------------------------------
# extract_prosody4.py — vocal prosody extractor
# ---------------------------------------------
"""Adds --skip-json and --out-dir, plus stdout path hints."""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, opensmile, pandas as pd

_cli2 = argparse.ArgumentParser()
_cli2.add_argument("video", type=str)
_cli2.add_argument("--skip-json", action="store_true")
_cli2.add_argument("--out-dir", type=Path, default=Path("/home/pachethridge47/Summer-2025/processeddata/task-1"))
args2 = _cli2.parse_args()

IN_DIR  = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR = args2.out_dir; OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_HZ = 30; HOP_SEC = 1/FRAME_HZ; FRAME_SIZE = 0.025; CSV_SUFFIX = "_prosody.csv"
FFMPEG = "ffmpeg"

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                        sampling_rate=16000)

def wav_from_mp4(mp4):
    tmp = Path(tempfile.mktemp(suffix=".wav"));
    subprocess.run([FFMPEG,"-y","-i",str(mp4),"-ac","1","-ar","16000","-vn",str(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
    return tmp

def fmt_ts(frame):
    us = frame*1_000_000//FRAME_HZ; h, rem = divmod(us,3_600_000_000); m, rem = divmod(rem,60_000_000); s, us = divmod(rem,1_000_000); return f"{h:02d}:{m:02d}:{s:02d}.{us:06d}"

def idx_to_sec(idx):
    if isinstance(idx, pd.MultiIndex) and "end" in idx.names: idx = idx.get_level_values("end")
    elif isinstance(idx, pd.MultiIndex): idx = idx.get_level_values(-1)
    if isinstance(idx, pd.TimedeltaIndex): sec = idx.total_seconds()
    else:
        sec = pd.to_numeric(idx, errors="coerce");
        if sec.isna().all(): sec = pd.to_timedelta(idx, errors="coerce").dt.total_seconds()
    return (sec + FRAME_SIZE).to_numpy(float)

def unique_csv(stem):
    for i in range(999):
        p = OUT_DIR / f"{stem}{'' if i==0 else f'_{i}'}{CSV_SUFFIX}"
        if not p.exists(): return p
    raise RuntimeError("too many duplicates")

src = IN_DIR / args2.video
if not src.exists(): sys.exit(f"❌ {src} not found")

csv_path = unique_csv(src.stem)
wav = wav_from_mp4(src)
try:
    df = smile.process_file(str(wav)); sec = idx_to_sec(df.index); mask = ~np.isnan(sec); df = df.iloc[mask]; sec = sec[mask]; frame = np.floor(sec/HOP_SEC+1e-9).astype(int); df.insert(0,"frame",frame); df = df.sort_values("frame").drop_duplicates("frame"); df["frame"] -= df["frame"].iloc[0]; df.insert(0,"time",df["frame"].apply(fmt_ts)); df.drop(columns="frame", inplace=True); df.to_csv(csv_path, index=False, float_format="%.16f")
finally:
    wav.unlink(missing_ok=True)
print(f"CSV_PATH:{csv_path}")

if not args2.skip_json:
    base = f"/data/local-files/?d={OUT_DIR.name}"
    j = OUT_DIR / "task-unified-prosody.json"
    j.write_text(json.dumps([{"data":{"video":f"{base}/{src.name}","prosody_csv":f"{base}/{csv_path.name}"}}], indent=2))
    print(f"LS‑JSON  : {j}")