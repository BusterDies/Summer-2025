# ---------------------------------------------
# whisperxdi3.py — diarized transcript extractor
# ---------------------------------------------
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import torch, whisperx

_cli3 = argparse.ArgumentParser()
_cli3.add_argument("video", type=str)
_cli3.add_argument("--out-dir", type=Path, default=Path("/home/pachethridge47/Summer-2025/processeddata/task-1"))
_cli3.add_argument("--cpu-diar", action="store_true")
args3 = _cli3.parse_args()

IN_DIR  = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR = args3.out_dir; OUT_DIR.mkdir(parents=True, exist_ok=True)

audio_path = IN_DIR / args3.video
if not audio_path.exists(): sys.exit("video not found")

stem = Path(args3.video).stem
out_json = OUT_DIR / f"{stem}_transcript.json"

def normalize(obj, lang_hint="en"):
    if isinstance(obj, dict): segs, lang = obj.get("segments",[]), obj.get("language", lang_hint)
    elif isinstance(obj, list): segs, lang = obj, lang_hint
    else: raise TypeError
    segs = [s if isinstance(s, dict) else {"text": s} for s in segs]
    return {"segments": segs, "language": lang}

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = whisperx.load_model("large-v3", device=device)
res    = normalize(model.transcribe(str(audio_path)))
align_m, meta = whisperx.load_align_model(res["language"], device=device)
res["segments"] = whisperx.align(res["segments"], align_m, meta, str(audio_path), device=device)
from whisperx.diarize import DiarizationPipeline
pipe = DiarizationPipeline(device="cpu" if args3.cpu_diar else device, use_auth_token=os.getenv("HF_TOKEN") or True)
dia_seg = pipe(str(audio_path), min_speakers=1, max_speakers=4)
spk_out = whisperx.assign_word_speakers(dia_seg, res["segments"])
json.dump(spk_out["segments"], out_json.open("w"), indent=2)
print(f"TRANSCRIPT_PATH:{out_json}")
