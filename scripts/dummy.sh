#!/bin/bash
#SBATCH -p performance
#SBATCH -t 0-01:00:00
#SBATCH --gpus 1
#SBATCH -J dummy_model
#SBATCH -o logs/dummy_out_%j.log
#SBATCH -e logs/dummy_err_%j.log

# activate the virtual environment
source .venv/bin/activate

# Check if sweep ID is provided as an argument
if [ -z "$1" ]; then
    # No sweep ID passed, run the initial command
    python scripts/dummy.py
else
    # Sweep ID passed, add agent to existing sweep
    python scripts/dummy.py --sweep_id "$1"
fi
