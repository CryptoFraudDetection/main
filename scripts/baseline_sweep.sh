#!/bin/bash
#SBATCH --job-name=baseline           # Job name
#SBATCH --error=logs/baseline_%j.err  # Error log file (%j = job ID)
#SBATCH --output=logs/baseline_%j.out # Output log file
#SBATCH --partition=performance       # Partition/queue name
#SBATCH --time=1-00:00:00             # Maximum runtime (2 days)
#SBATCH --cpus-per-task=12            # CPUs per task
#SBATCH --mem=32GB                    # Memory per node
#SBATCH --gpus=1                      # Number of GPUs
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment (.venv) not found."
    exit 1
fi

# Install package in editable mode
pip install -e .

# Run sweep agent
if [ -z "$1" ]; then
    echo "No sweep ID provided. Starting new sweep..."
    python scripts/baseline_sweep.py
else
    echo "Running agent for sweep: $1"
    python scripts/baseline_sweep.py --sweep_id "$1"
fi