#!/usr/bin/env python3
"""
run_pipeline.py – master launcher

Usage:
  python run_pipeline.py <video.mp4> [--cpu-diar] [--ctx-id 0] [--top-k 2]

Pipeline order:
  0) transcode to stable 30fps mp4 (once)
  1) InsightFace detections  -> <stem>_faces.csv, <stem>_faces_emb.npy
  2) ByteTrack tracking      -> <stem>_faces_tracked.csv
  3) AU extraction (tracked) -> <stem>_au_tracked.csv
  4) Prosody                 -> <stem>_prosody.csv
  5) WhisperX diarization     -> <stem>_transcript.json
  6) Label Studio task JSON   -> task-<n>.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IN_DIR = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 30
FFMPEG = "ffmpeg"


def _run(cmd: list[str], label: str) -> str:
    print(f"\n── {label} ───────────────────────────────────────────────")
    print("$", " ".join(cmd), "\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"{ts} | {line}", end="")
            lines.append(line)
    except KeyboardInterrupt:
        proc.kill()
        raise
    proc.wait()

    if proc.returncode:
        tail = "".join(lines[-40:])
        print(
            textwrap.dedent(
                f"""
                ── {label} FAILED (rc={proc.returncode}) ─────────────────────
                {tail}
                """
            ),
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return "".join(lines)


def _next_json() -> Path:
    nums = [
        int(m.group(1))
        for f in OUT_DIR.glob("task-*.json")
        if (m := re.fullmatch(r"task-(\d+)\.json", f.name))
    ]
    return OUT_DIR / f"task-{(max(nums) + 1) if nums else 1}.json"


def _unique_mp4(stem: str) -> Path:
    idx = 0
    while True:
        suf = f"_{idx}" if idx else ""
        mp4 = OUT_DIR / f"{stem}{suf}.mp4"
        if not mp4.exists():
            return mp4
        idx += 1


def _transcode(src: Path, dst: Path) -> None:
    encoders = subprocess.check_output([FFMPEG, "-v", "quiet", "-encoders"], text=True)
    enc = "libopenh264" if " libopenh264 " in encoders else "mpeg4"
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
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
    _run(cmd, "ffmpeg transcode")
    tmp.rename(dst)
    print(f"\n✓ transcoded → {dst.name} ({enc}, {TARGET_FPS} fps)")


def _must(pattern: str, text: str, label: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"Could not parse {label} from output using {pattern!r}")
    return m.group(1).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="MP4 inside IN_DIR")
    p.add_argument("--cpu-diar", action="store_true")
    p.add_argument("--ctx-id", type=int, default=0, help="InsightFace ctx_id: 0=GPU, -1=CPU")
    p.add_argument("--top-k", type=int, default=None, help="Limit AU extraction to top-K faces per frame by det_score")
    p.add_argument("--min-det-score", type=float, default=0.20)
    p.add_argument("--min-box", type=int, default=32)
    args = p.parse_args()

    src = IN_DIR / args.video
    if not src.exists():
        sys.exit(f"✖ {src} not found")

    INSIGHT_PY = SCRIPT_DIR / "extract_faces_insightface.py"
    TRACK_PY = SCRIPT_DIR / "track_faces_bytetrack.py"
    AU_PY = SCRIPT_DIR / "extract_au10.py"
    PROS_PY = SCRIPT_DIR / "extract_prosody4.py"
    WHIS_PY = SCRIPT_DIR / "whisperxdi3.py"

    # 0) transcode ONCE (authoritative mp4 for all downstream frame_idx)
    mp4_out = _unique_mp4(src.stem)
    _transcode(src, mp4_out)

    # 1) InsightFace detections
    ins_log = _run(
        [
            "python",
            str(INSIGHT_PY),
            "--video",
            str(mp4_out),
            "--out",
            str(OUT_DIR),
            "--prefix",
            mp4_out.stem,
            "--ctx-id",
            str(args.ctx_id),
        ],
        "InsightFace detections",
    )
    faces_csv = _must(r"FACES_CSV_PATH:(.*)", ins_log, "faces.csv path")
    faces_emb = _must(r"FACES_EMB_PATH:(.*)", ins_log, "faces_emb.npy path")

    # 2) ByteTrack
    faces_tracked = str(OUT_DIR / f"{mp4_out.stem}_faces_tracked.csv")
    bt_log = _run(
        [
            "python",
            str(TRACK_PY),
            "--in_csv",
            faces_csv,
            "--out_csv",
            faces_tracked,
            "--video",
            str(mp4_out),
            "--fps",
            str(TARGET_FPS),
        ],
        "ByteTrack tracking",
    )

    faces_tracked = _must(r"TRACKED_CSV_PATH:(.*)", bt_log, "faces_tracked.csv path")

    # 3) AU extraction (tracked)
    au_tracked = str(OUT_DIR / f"{mp4_out.stem}_au_tracked.csv")
    au_cmd = [
        "python",
        str(AU_PY),
        args.video,
        "--skip-json",
        "--out-dir",
        str(OUT_DIR),
        "--input-mp4",
        str(mp4_out),
        "--faces-tracked-csv",
        faces_tracked,
        "--out-csv",
        au_tracked,
        "--min-det-score",
        str(args.min_det_score),
        "--min-box",
        str(args.min_box),
    ]
    if args.top_k is not None:
        au_cmd += ["--top-k", str(args.top_k)]
    au_log = _run(au_cmd, "AU extractor (tracked)")
    au_csv = _must(r"CSV_PATH:(.*)", au_log, "AU csv path")

    # 4) prosody (uses the original file name in IN_DIR, consistent with your current script)
    pros_log = _run(
        [
            "python", str(PROS_PY),
            args.video,                # keep this for metadata/LS naming
            "--skip-json",
            "--out-dir", str(OUT_DIR),
            "--input-mp4", str(mp4_out)  # <<< ADD THIS
        ],
        "prosody extractor",
    )

    pros_csv = _must(r"CSV_PATH:(.*)", pros_log, "prosody csv path")

    # 5) WhisperX diarization
    wh_cmd = [
        "python", str(WHIS_PY),
        args.video,
        "--out-dir", str(OUT_DIR),
        "--input-mp4", str(mp4_out),
        "--prefix", mp4_out.stem,
    ]
    if args.cpu_diar:
        wh_cmd.append("--cpu-diar")
    wh_log = _run(wh_cmd, "WhisperX diarizer")
    trans_json = _must(r"TRANSCRIPT_PATH:(.*)", wh_log, "transcript json path")

    # 6) build Label Studio task JSON (rich pointers for debugging)
    base = f"/data/local-files/?d={OUT_DIR.name}"
    payload = [
        {
            "data": {
                "video": f"{base}/{Path(mp4_out).name}",
                "faces_csv": f"{base}/{Path(faces_csv).name}",
                "faces_emb": f"{base}/{Path(faces_emb).name}",
                "faces_tracked_csv": f"{base}/{Path(faces_tracked).name}",
                "au_tracked_csv": f"{base}/{Path(au_csv).name}",
                "prosody_csv": f"{base}/{Path(pros_csv).name}",
                "transcript_json": f"{base}/{Path(trans_json).name}",
            }
        }
    ]

    json_path = _next_json()
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ LS task JSON written → {json_path.name}")


if __name__ == "__main__":
    main()
