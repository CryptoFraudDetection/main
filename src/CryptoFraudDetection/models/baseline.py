from pathlib import Path

import numpy as np
import torch
import torchmetrics
import tqdm
from torch import nn, optim
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torchmetrics import classification

import wandb
from CryptoFraudDetection.utils import data_pipeline, enums, logger

torch.manual_seed(42)
np.random.seed(42)

_LOGGER = logger.Logger(
    name=__name__,
    level=enums.LoggerMode.INFO,
    log_dir="../logs",
)


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Classification head: we take the last hidden state and feed it into a linear layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return self.sigmoid(logits).squeeze(-1)


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metric_functions: dict,
) -> tuple[float, dict]:
    """Train model for one epoch following efficient metric computation pattern.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run computations on
        metric_functions: Dictionary of metric functions

    Returns:
        Tuple of (epoch_loss, metric_values)

    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    # Reset all metrics at start of epoch
    for metric_func in metric_functions.values():
        metric_func.reset()

    for x_batch, y_batch in tqdm.tqdm(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        n_samples += y_batch.size(0)

        # Update metrics
        for metric_func in metric_functions.values():
            metric_func.update(outputs, y_batch)

    # Compute epoch metrics
    loss = running_loss / n_samples
    metric_values = {
        name: metric_func.compute().item()
        for name, metric_func in metric_functions.items()
    }

    return loss, metric_values


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_functions: dict,
) -> tuple[float, dict]:
    """Validate model for one epoch using efficient metric computation."""
    model.eval()
    running_loss = 0.0
    n_samples = 0

    # Reset all metrics at start of validation
    for metric_func in metric_functions.values():
        metric_func.reset()

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        running_loss += loss.item()
        n_samples += y_batch.size(0)

        for metric_func in metric_functions.values():
            metric_func.update(outputs, y_batch)

    loss = running_loss / n_samples
    metric_values = {
        name: metric_func.compute().item()
        for name, metric_func in metric_functions.items()
    }

    return loss, metric_values


type Batch = tuple[torch.Tensor, torch.Tensor]
type Sample = Batch


def collate_fn(batch: list[Sample]) -> Batch:
    """Efficiently collate samples into batched tensors.

    Pads feature sequences to the length of the longest sequence in the batch
    and stacks the labels directly.

    Args:
        batch: List of (features, label) tuples where:
            - features: Tensor of shape (seq_len, input_size)
            - label: Scalar tensor

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Padded features tensor of shape (batch_size, max_seq_len, input_size)
            - Stacked labels tensor of shape (batch_size,)

    """
    features, labels = zip(*batch, strict=True)
    x = rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    y = torch.stack(labels)
    return x, y


class MeanPrediction(torchmetrics.Metric):
    """Calculate the mean value of the predictions (y_pred)."""

    def __init__(self, dist_sync_on_step: bool = False):
        """Initialize Class.

        Args:
            dist_sync_on_step: if True, synchronizes metric state across processes at each `forward()` before returning the value at the step.

        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Add states to track the sum of predictions and the count of predictions
        # "sum" and "count" will be reduced across multiple processes if distributed training is used.
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update the states with the latest batch of predictions.

        Args:
            y_pred: Model predictions for the batch (can be logits, probabilities, or any numeric values).
            y_true: Ground truth values (not used in this metric, but must be present for consistency with torchmetrics API).

        """
        # Ensure y_pred is a float tensor
        y_pred = y_pred.float()

        # Accumulate the sum of all predictions in this batch
        self.sum += y_pred.sum()

        # Accumulate the number of predictions in this batch
        self.count += y_pred.numel()

    def compute(self) -> torch.Tensor:
        """Compute and return the mean of all predictions seen so far."""
        # Handle the case of zero predictions to avoid division by zero
        if self.count == 0:
            return torch.tensor(0.0, device=self.sum.device)

        return self.sum / self.count


def get_metric_objects(
    device: torch.device,
    prefix: str = "",
    threshold: float = 0.5,
    val_metrics: list[str] | None = None,
) -> dict[str, torchmetrics.Metric]:
    """Initialize binary classification metrics.

    Args:
        device: The device to move metrics to.
        prefix: Prefix for metric names. Defaults to "".
        threshold: Threshold for converting probabilities to class labels. Defaults to 0.5.
        val_metrics: List of metric names to include. If None, defaults to ["val_accuracy", "val_mean_prediction"].

    Returns:
        dict[str, torchmetrics.Metric]: Dictionary mapping metric names to metric objects.

    """
    if val_metrics is None:
        val_metrics = [f"{prefix}_accuracy", f"{prefix}_mean_prediction"]

    # Initialize all possible metrics
    all_metrics = {
        f"{prefix}_accuracy": classification.BinaryAccuracy(
            threshold=threshold,
        ).to(device),
        f"{prefix}_precision": classification.BinaryPrecision(
            threshold=threshold,
        ).to(device),
        f"{prefix}_recall": classification.BinaryRecall(
            threshold=threshold,
        ).to(device),
        f"{prefix}_f1": classification.BinaryF1Score(threshold=threshold).to(
            device,
        ),
        f"{prefix}_mean_prediction": MeanPrediction().to(device),
    }

    # Filter metrics based on val_metrics list
    if prefix == "val":
        return {
            metric_name: metric_obj
            for metric_name, metric_obj in all_metrics.items()
            if metric_name in val_metrics
        }
    return all_metrics


def _train_fold(
    train_dataset: data_pipeline.CryptoDataSet,
    val_dataset: data_pipeline.CryptoDataSet,
    config: dict,
    logger_: logger.Logger,
    project: str,
) -> None:
    """Train model with hyperparameter tuning and log to Weights & Biases.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Hyperparameters and training configurations
        logger_: Logger instance
        project: Name of the W&B project

    """
    with wandb.init(project=project, config=config):
        config = wandb.config
        run_dir = Path(wandb.run.dir)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # We must determine input_size from the shape of the merged_features
        # Grab a single sample from the train_dataset to see the dimensionality
        sample_x, _ = train_dataset[0]
        input_size = sample_x.shape[1]  # shape is (seq_len, input_size)

        model = _LSTMClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

        best_val_loss, best_val_accuracy = float("inf"), 0.0
        no_improvement_epochs = 0
        train_metric_functions = get_metric_objects(
            device,
            "train",
            config["threshold"],
        )
        val_metric_functions = get_metric_objects(
            device,
            "val",
            config["threshold"],
        )
        for _ in range(config.epochs):
            train_loss, train_metrics = _train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train_metric_functions,
            )
            val_loss, val_metrics = _validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                val_metric_functions,
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **train_metrics,
                    **val_metrics,
                },
            )

            # Save best model (optional)
            if (val_loss < best_val_loss) and (
                val_metrics["val_accuracy"] > best_val_accuracy
            ):
                best_val_loss = val_loss
                best_val_accuracy = val_metrics["val_accuracy"]
                no_improvement_epochs = 0

                # Convert all config values to strings
                config_str = "_".join(
                    [f"{key}_{value}" for key, value in config.items()],
                )
                model_path = run_dir / f"model_{config_str}.pt"
                torch.save(model.state_dict(), model_path)
                logger_.info(f"Model saved to {model_path}")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= config["patience"]:
                    logger_.info("Early stopping triggered")
                    break


def train_model(
    config: dict | None = None,
    project: str = "crypto-fraud-detection-baseline",
) -> None:
    """Train model with Leave-One-Out Cross Validation.

    Args:
        config: Model and training configuration. If None, uses default config.
        project: Name of the W&B project for experiment tracking.

    """
    if config is None:
        config = {
            "epochs": 10,
            "batch_size": 8,
            "lr": 0.001,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "n_cutoff_points": 100,
            "n_groups_cutoff_points": 10,
            "threshold": 0.5,
            "patience": 20,
            "weight_decay": 0.0,
        }

    # Read data from data pipeline
    crypto_data = data_pipeline.CryptoData(_LOGGER, Path("data"))
    train_df, _ = crypto_data.load_data()
    train_coins = train_df["coin"].unique()
    _LOGGER.info(f"Starting LOOCV for {len(train_coins)} coins.")

    for i, coin in enumerate(train_coins):
        config["val_coin"] = coin
        train_df, val_df = crypto_data.train_val_split(coin)

        train_dataset = data_pipeline.CryptoDataSet(
            df=train_df,
            logger_=_LOGGER,
            n_cutoff_points=config["n_cutoff_points"],
            n_groups_cutoff_points=config["n_groups_cutoff_points"],
            n_time_steps=config.get("n_time_steps"),
        )
        val_dataset = data_pipeline.CryptoDataSet(
            df=val_df,
            logger_=_LOGGER,
            n_cutoff_points=config["n_cutoff_points"],
            n_groups_cutoff_points=config["n_groups_cutoff_points"],
            n_time_steps=config.get("n_time_steps"),
        )

        _train_fold(train_dataset, val_dataset, config, _LOGGER, project)
        _LOGGER.info(f"Fold {i+1}/{len(train_coins)} with coin {coin} done.")

    _LOGGER.info("All folds are done.")


if __name__ == "__main__":
    train_model()
