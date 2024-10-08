import argparse
import wandb
from CryptoFraudDetection.models.dummy import train_model

sweep_config = {
    "method": "random",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.0001,
            "max": 0.01
        },
        "epochs": {"value": 1000},
        "input_dim": {"value": 10},
        "batch_size": {"values": [16, 32, 64, 128]}
    }
}
entity = 'nod0ndel'
project = "dummy-model-sweep"

def main():
    parser = argparse.ArgumentParser(description="WandB agent script")
    parser.add_argument("--sweep_id", type=str, help="The ID of the WandB sweep to run. If not provided, a new sweep will be initialized.")
    args = parser.parse_args()

    if args.sweep_id:
        print(f"Using provided sweep_id: {args.sweep_id}")
        sweep_id = args.sweep_id
    else:
        print("No sweep_id provided. Initializing a new sweep...")
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
        print(f"\n\nNew sweep initialized. Sweep ID: {project}/{sweep_id}\n\n")
        print(f"Run more agents with:\n\npython models/dummy.py --sweep_id {entity}/{project}/{sweep_id}\n\n")

    wandb.agent(sweep_id, function=train_model)

if __name__ == "__main__":
    main()
