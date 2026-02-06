# ------------------------------------------------------------
# run_pipeline.py – absolute-path master launcher (streaming)
# ------------------------------------------------------------
"""
python run_pipeline.py <video.mp4> [--cpu-diar]
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, textwrap
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IN_DIR     = Path("/home/pachethridge47/Summer-2025/DementiaBank/GR")
OUT_DIR    = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# ── live runner ─────────────────────────────────────────────────────────
def _run(cmd: list[str], label: str) -> str:
    print(f"\n── {label} ───────────────────────────────────────────────")
    print("$", " ".join(cmd), "\n", flush=True)

    proc   = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    try:
        for line in proc.stdout:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"{ts} | {line}", end="")
            lines.append(line)
    except KeyboardInterrupt:
        proc.kill(); raise
    proc.wait()

    if proc.returncode:
        tail = ''.join(lines[-20:])
        print(textwrap.dedent(f"""
        ── {label} FAILED (rc={proc.returncode}) ─────────────────────
        {tail}
        """), file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return ''.join(lines)

# ── task-number helper ──────────────────────────────────────────────────
def _next_json() -> Path:
    nums = [int(m.group(1)) for f in OUT_DIR.glob("task-*.json")
            if (m := re.fullmatch(r"task-(\d+)\.json", f.name))]
    return OUT_DIR / f"task-{(max(nums)+1) if nums else 1}.json"

# ── CLI -----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("video", help="MP4 inside IN_DIR")
p.add_argument("--cpu-diar", action="store_true")
args = p.parse_args()

src = IN_DIR / args.video
if not src.exists():
    sys.exit(f"✖ {src} not found")

AU_PY   = SCRIPT_DIR / "extract_au10.py"
PROS_PY = SCRIPT_DIR / "extract_prosody4.py"
WHIS_PY = SCRIPT_DIR / "whisperxdi3.py"

# ── 1) AU ───────────────────────────────────────────────────────────────
au_log = _run(["python", str(AU_PY), args.video,
               "--skip-json", "--out-dir", str(OUT_DIR)], "AU extractor")
au_csv  = re.search(r"CSV_PATH:(.*)",  au_log).group(1).strip()
mp4_out = re.search(r"VIDEO_PATH:(.*)", au_log).group(1).strip()

# ── 2) prosody ──────────────────────────────────────────────────────────
pros_log = _run(["python", str(PROS_PY), args.video,
                 "--skip-json", "--out-dir", str(OUT_DIR)],
                "prosody extractor")
pros_csv = re.search(r"CSV_PATH:(.*)", pros_log).group(1).strip()

# ── 3) WhisperX ─────────────────────────────────────────────────────────
wh_cmd = ["python", str(WHIS_PY), args.video,
          "--out-dir", str(OUT_DIR)]
if args.cpu_diar:
    wh_cmd.append("--cpu-diar")
wh_log = _run(wh_cmd, "WhisperX diarizer")
trans_json = re.search(r"TRANSCRIPT_PATH:(.*)", wh_log).group(1).strip()

# ── 4) build LS task JSON ───────────────────────────────────────────────
base = f"/data/local-files/?d={OUT_DIR.name}"
payload = [{
    "data": {
        "video":           f"{base}/{Path(mp4_out).name}",
        "au_csv":          f"{base}/{Path(au_csv).name}",
        "prosody_csv":     f"{base}/{Path(pros_csv).name}",
        "transcript_json": f"{base}/{Path(trans_json).name}"
    }
}]

json_path = _next_json()                       # ← compute once
json_path.write_text(json.dumps(payload, indent=2))
print(f"\n✓ LS task JSON written → {json_path.name}")
