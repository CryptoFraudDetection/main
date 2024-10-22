#!/bin/bash
#SBATCH -p performance
#SBATCH -t 0-01:00:00
#SBATCH --gpus 1
#SBATCH -J dummy_model
#SBATCH -o dummy_out.log
#SBATCH -e dummy_err.log
.venv/bin/python scripts/dummy.py
