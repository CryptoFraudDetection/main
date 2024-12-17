#!/bin/bash
#SBATCH --job-name=sentiment                # Job name
#SBATCH --error=logs/sentiment_err_%j.log   # File to write standard error (%j expands to job ID)
#SBATCH --output=logs/sentiment_out_%j.log  # File to write standard output (%j expands to job ID)
#SBATCH --partition=performance             # Partition to submit the job to
#SBATCH --time=1-00:00:00                   # Maximum runtime (d-hh:mm:ss)
#SBATCH --cpus-per-task=12                  # Number of CPU cores per task
#SBATCH --mem=16GB                          # Memory required per node
#SBATCH --gpus=1                            # Number of GPUs required
#SBATCH --ntasks=1                          # Number of tasks (processes)

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment (.venv) not found."
    exit 1
fi

echo "generating sentiment scores for reddit and twitter"
python scripts/hf_sentiment.py
