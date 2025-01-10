"""Hyperparameter optimization for baseline LSTM model using W&B sweeps."""

import argparse

import wandb
from CryptoFraudDetection.models.baseline import train_model

# Sweep configuration aligned with baseline model parameters
sweep_config = {
    "method": "random",
    "metric": {
        "name": "val_accuracy",  # Match the metric name from baseline model
        "goal": "maximize",
    },
    "parameters": {
        # All parameters from baseline model's default config
        "lr": {"values": [5e-3]},
        "hidden_size": {"values": [32]},
        "num_layers": {"values": [2]},
        "dropout": {"values": [0.0]},
        "batch_size": {"values": [64]},
        "weight_decay": {"values": [0.0]},
        # Fixed parameters
        "epochs": {"value": 1000},
        "patience": {"value": 100},
        "n_cutoff_points": {"value": 1},
        "n_groups_cutoff_points": {"value": 1},
        "threshold": {"value": 0.5},
    },
}

ENTITY = "nod0ndel"
PROJECT = "crypto-fraud-detection-baseline"


def main():
    """Initialize or join a W&B sweep for the baseline model."""
    parser = argparse.ArgumentParser(
        description="W&B sweep for baseline model",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        help="Existing sweep ID to join. If not provided, creates new sweep.",
    )
    args = parser.parse_args()

    if args.sweep_id:
        print(f"Using provided sweep_id: {args.sweep_id}")
        sweep_id = args.sweep_id
    else:
        print("Initializing new sweep...")
        sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
        print(f"\nNew sweep initialized: {PROJECT}/{sweep_id}\n")
        print(
            f"Run more agents with:\npython baseline_sweep.py --sweep_id {ENTITY}/{PROJECT}/{sweep_id}\n",
        )

    # Start agent with configured project name
    wandb.agent(sweep_id, function=lambda: train_model(project=PROJECT))


if __name__ == "__main__":
    main()
