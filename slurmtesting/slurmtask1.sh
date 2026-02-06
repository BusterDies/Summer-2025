#!/bin/bash
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -n 32
#SBATCH -t 0:01:10

python /home/pachethridge47/Summer-2025/pipeline/run_pipeline.py /home/pachethridge47/Summer-2025/DementiaBank/GR/Anita.mp4
