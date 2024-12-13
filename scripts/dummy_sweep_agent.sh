#!/bin/bash
#SBATCH --job-name=dummy_model          # Job name
#SBATCH --error=logs/dummy_err_%j.log   # File to write standard error (%j expands to job ID)
#SBATCH --output=logs/dummy_out_%j.log  # File to write standard output (%j expands to job ID)
#SBATCH --partition=performance         # Partition to submit the job to
#SBATCH --time=1-00:00:00               # Maximum runtime (d-hh:mm:ss)
#SBATCH --cpus-per-task=12              # Number of CPU cores per task
#SBATCH --mem=16GB                      # Memory required per node
#SBATCH --gpus=1                        # Number of GPUs required
#SBATCH --ntasks=1                      # Number of tasks (processes)

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment (.venv) not found."
    exit 1
fi

# Check if sweep ID is provided as an argument
if [ -z "$1" ]; then
    echo "No sweep ID provided. This script expects a sweep ID."
    exit 1
else
    echo "Sweep ID provided: $1. Running with sweep ID."
    python scripts/dummy.py --sweep_id "$1"
fi
