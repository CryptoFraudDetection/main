from pathlib import Path

import numpy as np
import pandas as pd
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
            if metric_name
            in [f"{prefix}_accuracy", f"{prefix}_mean_prediction"]
        }
    return all_metrics


def _train_fold(
    train_dataset: data_pipeline.CryptoDataSet,
    val_dataset: data_pipeline.CryptoDataSet,
    wandb_config: dict,
    logger_: logger.Logger,
    device: torch.device,
    train_metric_functions: dict[str, torchmetrics.Metric],
    val_metric_functions: dict[str, torchmetrics.Metric],
    fold_info: dict,
) -> None:
    """Train model with hyperparameter tuning and log to Weights & Biases.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        wandb_config: Hyperparameters and training configurations
        logger_: Logger instance
        device: Device to run computations on
        train_metric_functions: Dictionary of training metric functions
        val_metric_functions: Dictionary of validation metric functions
        fold_info: Dictionary containing fold metadata (coin, fold_number)

    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=wandb_config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=wandb_config.batch_size,
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
        hidden_size=wandb_config.hidden_size,
        num_layers=wandb_config.num_layers,
        dropout=wandb_config.dropout,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=wandb_config.lr,
        weight_decay=wandb_config.weight_decay,
    )

    stats = []
    for epoch in range(wandb_config.epochs):
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
        epoch_stats = {
            "epoch": epoch,
            "fold": fold_info["fold_number"],
            "coin": fold_info["coin"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **train_metrics,
            **val_metrics,
        }
        stats.append(epoch_stats)

    return stats


def _aggregate_metrics(
    all_fold_stats: list[dict],
    epoch: int,
    coin_info: dict,
) -> dict:
    """Aggregate metrics from all folds for a single epoch.

    Args:
        all_fold_stats: List of dictionaries containing fold statistics
        epoch: Current epoch number
        coin_info: Dictionary mapping coins to their scam status

    Returns:
        Dictionary containing aggregated metrics for the epoch

    """
    # Filter stats for current epoch
    epoch_stats = [
        stat
        for stats in all_fold_stats
        for stat in stats
        if stat["epoch"] == epoch
    ]

    # Collect metrics per fold
    fold_metrics = {}
    for stat in epoch_stats:
        coin = stat["coin"]
        prefix = f"fold_{coin}"
        # Exclude metadata fields
        metrics = {
            f"{prefix}/{k}": v
            for k, v in stat.items()
            if k not in ["epoch", "fold", "coin"]
        }
        fold_metrics.update(metrics)

    # Calculate mean metrics
    metric_keys = [
        k for k in epoch_stats[0].keys() if k not in ["epoch", "fold", "coin"]
    ]
    mean_metrics = {}

    # Overall means
    for key in metric_keys:
        values = [stat[key] for stat in epoch_stats]
        mean_metrics[f"mean_{key}"] = np.mean(values)

    # Scam/non-scam means
    scam_stats = [
        stat for stat in epoch_stats if bool(coin_info.get(stat["coin"]))
    ]
    non_scam_stats = [
        stat for stat in epoch_stats if not bool(coin_info.get(stat["coin"]))
    ]

    for key in metric_keys:
        if scam_stats:
            mean_metrics[f"mean_{key}_scam"] = np.mean(
                [stat[key] for stat in scam_stats],
            )
        if non_scam_stats:
            mean_metrics[f"mean_{key}_non_scam"] = np.mean(
                [stat[key] for stat in non_scam_stats],
            )

    return {**fold_metrics, **mean_metrics}


def train_model(
    wandb_config: wandb.Config,
    wandb_project: str,
    dataset_config: dict,
    metric_config: dict,
    overfit_test: bool = False,
) -> None:
    """Train model with Leave-One-Out Cross Validation.

    Args:
        config: Model and training configuration.
        project: Name of the W&B project for experiment tracking.
        dataset_config: config for dataset creation.

    """
    run = wandb.init(project=wandb_project, config=wandb_config)
    wandb_config = wandb.config

    # Read data from data pipeline
    crypto_data = data_pipeline.CryptoData(_LOGGER, Path("data"))
    train_df, _ = crypto_data.load_data()
    train_coins = train_df["coin"].unique()
    _LOGGER.info(f"Starting LOOCV for {len(train_coins)} coins.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_metric_functions = get_metric_objects(
        device,
        "train",
        metric_config["threshold"],
    )
    val_metric_functions = get_metric_objects(
        device,
        "val",
        metric_config["threshold"],
    )

    all_fold_stats = []
    coin_info = {c["symbol"]: c for c in crypto_data.coin_info}
    for i, coin in enumerate(train_coins):
        train_df, val_df = crypto_data.train_val_split(coin)

        if overfit_test:
            dataset_config["n_time_steps"] = 5
        train_dataset = data_pipeline.CryptoDataSet(
            df=train_df,
            logger_=_LOGGER,
            **dataset_config,
        )
        val_dataset = data_pipeline.CryptoDataSet(
            df=val_df,
            logger_=_LOGGER,
            **dataset_config,
        )

        fold_info = {
            "fold_number": i + 1,
            "coin": coin,
            "is_scam": bool(coin_info.get(coin)),
        }

        fold_stats = _train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            wandb_config=wandb_config,
            logger_=_LOGGER,
            device=device,
            train_metric_functions=train_metric_functions,
            val_metric_functions=val_metric_functions,
            fold_info=fold_info,
        )
        all_fold_stats.append(fold_stats)

        best_val_accurcay = max(stat["val_accuracy"] for stat in fold_stats)
        _LOGGER.info(
            f"Fold {i+1}/{len(train_coins)} coin={coin} done. Val acc={best_val_accurcay:.3f}",
        )

    for epoch in range(wandb_config.epochs):
        metrics = _aggregate_metrics(all_fold_stats, epoch, coin_info)
        wandb.log(metrics, step=epoch)

    wandb.finish()


if __name__ == "__main__":
    train_model()
