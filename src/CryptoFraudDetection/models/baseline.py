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


type EpochStats = dict[str, float | str]


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metric_functions: dict,
) -> EpochStats:
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
    epoch_stats: EpochStats = {
        name: metric_func.compute().item()
        for name, metric_func in metric_functions.items()
    }
    epoch_stats["loss"] = loss

    return epoch_stats


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metric_functions: dict,
) -> EpochStats:
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
    epoch_stats: EpochStats = {
        name: metric_func.compute().item()
        for name, metric_func in metric_functions.items()
    }
    epoch_stats["loss"] = loss

    return epoch_stats


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


type FoldStats = list[EpochStats]


def _train_fold(
    train_dataset: data_pipeline.CryptoDataSet,
    val_dataset: data_pipeline.CryptoDataSet,
    wandb_config: dict,
    device: torch.device,
    train_metric_functions: dict[str, torchmetrics.Metric],
    val_metric_functions: dict[str, torchmetrics.Metric],
    fold_info: dict,
) -> FoldStats:
    """Train model with hyperparameter tuning and log to Weights & Biases.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        wandb_config: Hyperparameters and training configurations
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
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_metric_functions,
        )
        val_metrics = _validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            val_metric_functions,
        )
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        epoch_stats = {
            "epoch": epoch,
            "fold": fold_info["fold_number"],
            "coin": fold_info["coin"],
            **train_metrics,
            **val_metrics,
        }
        stats.append(epoch_stats)

    return stats


def _aggregate_metrics(
    all_fold_stats: list[dict],
    epoch: int,
    coin_info: list[dict],
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
        epoch_stats
        for fold_stats in all_fold_stats
        for epoch_stats in fold_stats
        if epoch_stats["epoch"] == epoch
    ]

    # Collect metrics per fold
    fold_metrics = {}
    for epoch_stat in epoch_stats:
        coin = epoch_stat["coin"]
        prefix = f"fold_{coin}"
        # Prefix metrics with fold name
        metrics = {
            f"{prefix}/{k}": v
            for k, v in epoch_stat.items()
            # Exclude metadata fields
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
    fraud_list = {c["symbol"]: c["fraud"] for c in coin_info.values()}
    scam_stats = [stat for stat in epoch_stats if fraud_list[stat["coin"]]]
    non_scam_stats = [
        stat for stat in epoch_stats if not fraud_list[stat["coin"]]
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


def _extract_single_coin_info(coin_info: list[dict], coin: str) -> dict:
    coin_info_ = [c for c in coin_info if c["symbol"] == coin]

    if len(coin_info_) == 0:
        _LOGGER.error(f"Coin {coin} not found in coin_info.")
    elif len(coin_info_) > 1:
        _LOGGER.error(f"Multiple entries found for coin {coin}.")

    return coin_info[0]


type CVStats = dict[FoldStats]


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
    # Initialize W&B run
    run = wandb.init(project=wandb_project, config=wandb_config)
    wandb_config = wandb.config

    # Read data from data pipeline
    crypto_data = data_pipeline.CryptoData(_LOGGER, Path("data"))
    train_df, _ = crypto_data.load_data()
    train_coins = train_df["coin"].unique()
    _LOGGER.info(f"Starting LOOCV for {len(train_coins)} coins.")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Metric functions
    train_metric_functions = {
        "accuracy": classification.BinaryAccuracy(
            threshold=metric_config["threshold"],
        ).to(device),
        "mean_prediction": MeanPrediction().to(device),
    }
    val_metric_functions = {
        "accuracy": classification.BinaryAccuracy(
            threshold=metric_config["threshold"],
        ).to(device),
        "mean_prediction": MeanPrediction().to(device),
    }

    # LOOCV
    all_fold_stats: CVStats = {}
    for i, coin in enumerate(train_coins):
        # Data Loader
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

        # Train Fold
        coin_info = _extract_single_coin_info(crypto_data.coin_info, coin)
        fold_info = {
            "fold_number": i + 1,
            "coin": coin,
            "is_scam": coin_info["fraud"],
        }
        all_fold_stats[coin] = _train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            wandb_config=wandb_config,
            device=device,
            train_metric_functions=train_metric_functions,
            val_metric_functions=val_metric_functions,
            fold_info=fold_info,
        )

    # Aggregate metrics and log to W&B
    for epoch in range(wandb_config.epochs):
        metrics = _aggregate_metrics(
            all_fold_stats,
            epoch,
            crypto_data.coin_info,
        )
        wandb.log(metrics, step=epoch)

    wandb.finish()


if __name__ == "__main__":
    train_model()
