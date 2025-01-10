from pathlib import Path

import numpy as np
import torch
import torchmetrics.classification
import tqdm
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb
from CryptoFraudDetection.utils import data_pipeline, enums, logger

torch.manual_seed(42)
np.random.seed(42)

LOGGER = logger.Logger(
    name=__name__,
    level=enums.LoggerMode.INFO,
    log_dir="../logs",
)


# -------------------------------------------------------------------
# Basline LSTM Classification Model
# -------------------------------------------------------------------
class LSTMClassifier(nn.Module):
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
        # Take the last time-step output: shape (batch_size, hidden_size * num_directions)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        probs = self.sigmoid(logits)
        return probs


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    metric_functions: dict,
) -> float:
    for metric_func in metric_functions.values():
        metric_func.reset()

    model.train()
    running_loss = 0.0

    for x_batch, y_batch in tqdm.tqdm(train_loader):
        x_batch = x_batch.to(device)
        # y_batch shape: (batch_size, 1)
        y_batch = y_batch[:, -1].unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)

        rounded_outputs = torch.round(outputs)
        for metric_func in metric_functions.values():
            metric_func(rounded_outputs, y_batch)

    epoch_loss = running_loss / len(train_loader.dataset)

    metric_values = {}
    for metric, metric_func in metric_functions.items():
        metric_values[metric] = metric_func.compute().item()

    return epoch_loss, metric_values


@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    metrics_functions: dict,
) -> tuple[float, float, float, float, float]:
    """Validate the model for one epoch.

    Args:
        model (nn.Module): The model being evaluated.
        val_loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device for computation.

    Returns:
        loss, accuracy, precision, recall, f1

    """
    for metric_func in metrics_functions.values():
        metric_func.reset()

    model.eval()
    running_loss = 0.0

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch[:, -1].unsqueeze(1).to(device)

        outputs = model(x_batch)
        rounded_outputs = torch.round(outputs)
        loss = criterion(rounded_outputs, y_batch)
        running_loss += loss.item() * x_batch.size(0)
        for metric_func in metrics_functions.values():
            metric_func(rounded_outputs, y_batch)

    epoch_loss = running_loss / len(val_loader.dataset)
    metric_values = {}
    for metric, metric_func in metrics_functions.items():
        metric_values[metric] = metric_func.compute().item()
    return epoch_loss, metric_values


type Batch = tuple[torch.Tensor, torch.Tensor]
type Sample = Batch


def collate_fn(batch: list[Sample]) -> Batch:
    """Collate function to prepare a batch of data for DataLoader.

    Pads sequences in x and y tensors to the length of the longest
    sequence in the batch and adds a batch dimension.

    Args:
        batch: A list of tuples where each tuple contains:
            - x: Tensor of shape (seq_len, input_size)
            - y: Tensor of shape (seq_len,)

    Returns:
        Batch: A tuple containing:
            - x_padded: Padded x tensor of shape (batch_size, max_seq_len, input_size)
            - y_padded: Padded y tensor of shape (batch_size, max_seq_len)

    """
    x, y = zip(*batch, strict=True)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0.0)
    return x_padded, y_padded


# Initialize metrics with correct task and averaging
def get_metric_functions(
    device: torch.DeviceObjType, prefix: str = "",
) -> dict[str, torchmetrics.Metric]:
    return {
        f"{prefix}accuracy": torchmetrics.classification.BinaryAccuracy().to(
            device,
        ),
        f"{prefix}precision": torchmetrics.classification.BinaryPrecision().to(
            device,
        ),
        f"{prefix}recall": torchmetrics.classification.BinaryRecall().to(
            device,
        ),
        f"{prefix}f1": torchmetrics.classification.BinaryF1Score().to(device),
    }


# -------------------------------------------------------------------
# Train one Fold Function
# -------------------------------------------------------------------
def train_fold(
    train_dataset: data_pipeline.CryptoDataSet,
    val_dataset: data_pipeline.CryptoDataSet,
    config: dict,
    logger_: logger.Logger,
) -> None:
    """Read data and train the model with hyperparameter tuning.

    This function will be invoked by `wandb.agent(...)` for each set of
    hyperparameters. It performs training and evaluation, logging metrics to
    Weights & Biases.

    Args:
        config (Optional[dict]): A dictionary containing hyperparameters and training configurations.
            Keys may include:
            - 'val_coin' (str): Coin which is left out for this fold.
            - 'epochs' (int): Number of training epochs.
            - 'batch_size' (int): Batch size for training.
            - 'lr' (float): Learning rate for the optimizer.
            - 'hidden_size' (int): Number of hidden units in the LSTM.
            - 'num_layers' (int): Number of LSTM layers.
            - 'dropout' (float): Dropout rate for regularization.
            - 'n_cutoff_points' (int): Number of cutoff points for sequence segmentation.
            - 'n_groups_cutoff_points' (int): Number of groups for cutoff points.

    """
    with wandb.init(config=config):
        config = wandb.config
        run_dir = Path(wandb.run.dir)

        # ---------------------------
        # 1. Prepare Data
        # ---------------------------
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

        # ---------------------------
        # 2. Define Model, Loss, Optimizer
        # ---------------------------
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # ---------------------------
        # 3. Training Loop
        # ---------------------------
        best_val_loss, best_val_f1 = float("inf"), 0.0
        patience = 5
        no_improvement_epochs = 0
        train_metric_functions = get_metric_functions(device, "train")
        val_metric_functions = get_metric_functions(device, "val")
        for epoch in range(config.epochs):
            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train_metric_functions,
            )
            val_loss, val_metrics = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                val_metric_functions,
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "val_coin": config["val_coin"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **train_metrics,
                    **val_metrics,
                },
            )

            # Save best model (optional)
            if (val_loss < best_val_loss) and (
                val_metrics["val_f1"] > best_val_f1
            ):
                best_val_loss = val_loss
                best_val_f1 = val_metrics["val_f1"]
                no_improvement_epochs = 0

                # Convert all config values to strings and join them with underscores
                config_str = "_".join(
                    [f"{key}_{value}" for key, value in config.items()],
                )
                model_path = run_dir / f"model_{config_str}.pt"
                torch.save(model.state_dict(), model_path)
                logger_.info(f"Model saved to {model_path}")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger_.info("Early stopping triggered")
                    break

            logger_.debug(
                f"Epoch {epoch+1}/{config.epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}",
            )


def train_model(config: dict | None = None) -> None:
    if config is None:
        config = {
            "epochs": 10,
            "batch_size": 8,
            "lr": 0.001,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.2,
            "n_cutoff_points": 10,
            "n_groups_cutoff_points": 1,
        }

    # Read data from data pipeline
    crypto_data = data_pipeline.CryptoData(LOGGER, Path("data"))
    train_df, _ = crypto_data.load_data()
    train_coins = train_df["coin"].unique()
    LOGGER.info(f"Starting LOOCV for {len(train_coins)} coins.")
    for i, coin in enumerate(train_coins):
        config["val_coin"] = coin
        train_df, val_df = crypto_data.train_val_split(coin)

        train_dataset = data_pipeline.CryptoDataSet(
            df=train_df,
            logger_=LOGGER,
            n_cutoff_points=config["n_cutoff_points"],
            n_groups_cutoff_points=config["n_groups_cutoff_points"],
        )
        val_dataset = data_pipeline.CryptoDataSet(
            df=val_df,
            logger_=LOGGER,
            n_cutoff_points=config["n_cutoff_points"],
            n_groups_cutoff_points=config["n_groups_cutoff_points"],
        )

        train_fold(train_dataset, val_dataset, config, LOGGER)
        LOGGER.info(f"Fold {i+1}/{len(train_coins)} with coin {coin} done.")
    LOGGER.info("All folds are done.")


if __name__ == "__main__":
    train_model()
