#!/usr/bin/env python3
# ---------------------------------------------
# whisperxdi3.py — diarized transcript extractor
# ---------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import whisperx


# ───────── USER CONFIG ────────────────────────────────────────────────
IN_DIR = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR_DEF = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
# ─────────────────────────────────────────────────────────────────────


def unique_json(out_dir: Path, stem: str) -> Path:
    """Avoid overwriting existing transcripts."""
    for i in range(999):
        p = out_dir / f"{stem}{'' if i == 0 else f'_{i}'}_transcript.json"
        if not p.exists():
            return p
    raise RuntimeError("too many duplicates")


def normalize(obj, lang_hint: str = "en") -> dict:
    if isinstance(obj, dict):
        segs, lang = obj.get("segments", []), obj.get("language", lang_hint)
    elif isinstance(obj, list):
        segs, lang = obj, lang_hint
    else:
        raise TypeError(f"Unexpected transcription type: {type(obj)}")

    segs = [s if isinstance(s, dict) else {"text": s} for s in segs]
    return {"segments": segs, "language": lang}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("video", type=str, help="Original filename inside IN_DIR (legacy mode + metadata)")
    cli.add_argument("--out-dir", type=Path, default=OUT_DIR_DEF)
    cli.add_argument("--cpu-diar", action="store_true")

    # Pipeline-aligned mode
    cli.add_argument("--input-mp4", type=Path, default=None, help="Use this mp4 for audio (e.g., transcoded 30fps)")
    cli.add_argument("--prefix", type=str, default=None, help="Override output prefix (default: input mp4 stem)")
    args = cli.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose audio source + naming stem
    if args.input_mp4 is not None:
        audio_path = args.input_mp4
        if not audio_path.exists():
            sys.exit(f"❌ {audio_path} not found")
        base_stem = args.prefix or audio_path.stem
    else:
        audio_path = IN_DIR / args.video
        if not audio_path.exists():
            sys.exit(f"❌ {audio_path} not found")
        base_stem = args.prefix or Path(args.video).stem

    out_json = unique_json(out_dir, base_stem)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcribe
    model = whisperx.load_model("large-v3", device=device)
    res = normalize(model.transcribe(str(audio_path)))

    # Align
    align_m, meta = whisperx.load_align_model(res["language"], device=device)
    res["segments"] = whisperx.align(res["segments"], align_m, meta, str(audio_path), device=device)

    # Diarize
    from whisperx.diarize import DiarizationPipeline

    pipe = DiarizationPipeline(
        device="cpu" if args.cpu_diar else device,
        use_auth_token=os.getenv("HF_TOKEN") or True,
    )
    dia_seg = pipe(str(audio_path), min_speakers=1, max_speakers=4)

    # Assign speakers
    spk_out = whisperx.assign_word_speakers(dia_seg, res["segments"])

    # Write segments list (same format you were using)
    with out_json.open("w") as f:
        json.dump(spk_out["segments"], f, indent=2)

    print(f"TRANSCRIPT_PATH:{out_json}")


if __name__ == "__main__":
    main()
