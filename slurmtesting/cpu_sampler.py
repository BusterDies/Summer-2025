#!/usr/bin/env python3
"""
cpu_sampler.py  –  auto-estimate SLURM wall-time from a 20-second probe run

Usage
-----
python cpu_sampler.py <full_path_to_video> [--cores N] [--safety X.Y]

• A 20-s clip is cut with ffmpeg.
• run_pipeline.py is executed on that clip.
• Per-second CPU usage is sampled until three consecutive ~0 % readings.
• Average core usage (robust-mean) and clip runtime => speed factor Fs.
• Full-video seconds × Fs × (Cavg/cores) × safety → wall-time estimate.
• “slurmtask#.sh” is dropped in /slurmtesting with the proper SBATCH line.
• Only the probe’s outputs (mp4 / CSVs / face.jpg / transcript / task-N.json)
  are deleted; everything else in processeddata/task-1 stays untouched.
"""
from __future__ import annotations
import argparse, json, shutil, sys, time, re, statistics, subprocess
from math import ceil
from pathlib import Path
import psutil

# ───────── CONFIG PATHS ────────────────────────────────────────────────
PIPELINE       = "/home/pachethridge47/Summer-2025/pipeline/run_pipeline.py"
SLURM_DIR      = Path("/home/pachethridge47/Summer-2025/slurmtesting")
TASK_DIR       = Path("/home/pachethridge47/Summer-2025/processeddata/task-1")
SAMPLE_CLIP    = SLURM_DIR / "sample20s.mp4"
SAMPLE_SECONDS = 20.0

# ───────── HELPERS ─────────────────────────────────────────────────────
def trim_video(src: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-t", str(SAMPLE_SECONDS), "-c", "copy", str(SAMPLE_CLIP)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def launch_pipeline() -> subprocess.Popen:
    return subprocess.Popen(["python", PIPELINE, str(SAMPLE_CLIP)])

def sample_cores(pid: int, interval=1.0, zero_cut=0.01, max_zero=3):
    root = psutil.Process(pid)
    zeros, cores = 0, []
    start = time.time()

    while True:
        procs = [root] + root.children(recursive=True)
        for p in procs:
            try: p.cpu_percent(None)
            except psutil.Error: pass            # prime counters

        time.sleep(interval)

        total = sum(
            p.cpu_percent(None) for p in procs
            if p.is_running() and not p.status() == psutil.STATUS_ZOMBIE)

        cur = total / 100.0
        cores.append(cur)
        print(f"{time.strftime('%H:%M:%S')}  {total:6.1f}%  {cur:5.2f} cores")

        zeros = zeros + 1 if cur < zero_cut else 0
        if zeros >= max_zero: break

    elapsed = time.time() - start
    return cores, elapsed

def robust_mean(xs: list[float]) -> float:
    if not xs: return 1.0
    med = statistics.median(xs)
    keep = [x for x in xs if x >= med]
    return statistics.mean(keep) if keep else statistics.mean(xs)

def video_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=True)
    return float(out.stdout.strip())

def next_slurm_id() -> int:
    ids = [int(m.group(1))
           for f in SLURM_DIR.glob("slurmtask*.sh")
           if (m := re.fullmatch(r"slurmtask(\d+)", f.stem))]
    return max(ids, default=0) + 1

def write_slurm(video: Path, runtime_s: int, cores: int) -> Path:
    h, rem = divmod(runtime_s, 3600)
    m, s   = divmod(rem, 60)
    script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -n {cores}
#SBATCH -t {h}:{m:02}:{s:02}

python {PIPELINE} {video}
"""
    out = SLURM_DIR / f"slurmtask{next_slurm_id()}.sh"
    out.write_text(script)
    return out

def cleanup_created(before: set[Path], stem: str):
    after   = set(TASK_DIR.iterdir())
    created = after - before
    task_re = re.compile(r"task-\d+\.json$")
    for p in created:
        if p.name.startswith(stem) or task_re.fullmatch(p.name):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)
            print(f"🗑  removed {p.name}")

# ───────── MAIN ────────────────────────────────────────────────────────
def main():
    cli = argparse.ArgumentParser(description="Estimate SLURM wall-time from 20-s probe")
    cli.add_argument("video", help="Full path to the source video")
    cli.add_argument("--cores",   type=int,   default=32,  help="Cores to request (default 32)")
    cli.add_argument("--safety",  type=float, default=1.5, help="Safety multiplier (default 1.5)")
    args = cli.parse_args()

    src = Path(args.video).resolve()
    if not src.exists():
        sys.exit(f"✖ {src} not found")

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_DIR.mkdir(parents=True, exist_ok=True)

    before = set(TASK_DIR.iterdir())          # snapshot pre-run
    trim_video(src)

    proc = launch_pipeline()
    cores, tp = sample_cores(proc.pid)
    proc.wait()                               # block until pipeline exits

    cavg  = robust_mean(cores)
    fs    = tp / SAMPLE_SECONDS               # compute seconds-per-sec @ cavg
    mult  = fs * (cavg / args.cores)          # adjust to requested cores
    full  = video_seconds(src)
    est   = ceil(full * mult * args.safety)

    print("\n── SAMPLE STATS ──────────────────────────────────────────")
    print(f"clip runtime           : {tp:.1f}s  (20 s video)")
    print(f"avg cores (robust)     : {cavg:.2f}")
    print(f"speed factor Fs        : {fs:.3f}  (sec CPU / sec video)")
    print(f"multiplier (Fs*C/N)    : {mult:.3f}")
    print(f"full video length      : {full:.1f}s")
    print("────────────────────────────────────────────────────────────")

    slurm = write_slurm(src, est, args.cores)
    cleanup_created(before, SAMPLE_CLIP.stem)

    print(f"✓ SLURM file written → {slurm}")
    print(f"✓ Estimate wall-time  → {est//3600}h{(est%3600)//60:02}m")

if __name__ == "__main__":
    main()
