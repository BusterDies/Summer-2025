#!/bin/bash
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -n 80
#SBATCH -t 0:07:07

python /home/pachethridge47/Summer-2025/pipeline/run_pipeline.py /home/pachethridge47/Summer-2025/DementiaBank/GR/Anita.mp4
