#!/bin/bash
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -n 140
#SBATCH -t 18:00:00

python /home/pachethridge47/Summer-2025/pipeline/run_pipeline.py /home/pachethridge47/Summer-2025/DementiaBank/GR/TheFather.mp4
