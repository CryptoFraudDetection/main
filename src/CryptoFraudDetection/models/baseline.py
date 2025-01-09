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
) -> float:
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        # x_batch shape: (batch_size, seq_len, input_size)
        # y_batch shape: (batch_size, seq_len) -> but we only use y of last step or we might do a single label
        # NOTE: In your dataset, y_batch is shape (batch_size, seq_len).
        #       If you are doing a single-label classification (fraud or not), you could:
        #       - Take the *last* label of the sequence
        #       - Or do some other labeling strategy
        #
        # For simplicity: let's take y_batch[:, -1] as the label that indicates if the coin eventually is fraud or not
        # (In reality, you may define your labeling differently.)
        x_batch = x_batch.to(device)
        # Here, we assume the fraud label is *constant* over the entire sequence,
        # so just take the last label for the entire sequence.
        # If your dataset has a single y per sequence, you can do y_batch = y_batch[:, 0]
        y_batch = (
            y_batch[:, -1].unsqueeze(1).to(device)
        )  # shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
    accuracy_metric,
    precision_metric,
    recall_metric,
    f1_metric,
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
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    model.eval()
    running_loss = 0.0

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch[:, -1].unsqueeze(1).to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        running_loss += loss.item() * x_batch.size(0)
        accuracy_metric(outputs, y_batch)
        precision_metric(outputs, y_batch)
        recall_metric(outputs, y_batch)
        f1_metric(outputs, y_batch)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = accuracy_metric.compute().item()
    epoch_precision = precision_metric.compute().item()
    epoch_recall = recall_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()
    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        accuracy_metric = torchmetrics.classification.Accuracy(
            task="binary",
        ).to(device)
        precision_metric = torchmetrics.classification.Precision(
            task="binary",
        ).to(device)
        recall_metric = torchmetrics.classification.Recall(task="binary").to(
            device,
        )
        f1_metric = torchmetrics.classification.F1Score(task="binary").to(
            device,
        )
        for epoch in range(config.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = (
                validate_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    accuracy_metric,
                    precision_metric,
                    recall_metric,
                    f1_metric,
                )
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "val_coin": config["val_coin"],
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                },
            )

            # Save best model (optional)
            if (val_loss < best_val_loss) and (val_f1 > best_val_f1):
                best_val_loss = val_loss
                best_val_f1 = val_f1
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
            "batch_size": 1,
            "lr": 0.001,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "n_cutoff_points": 100,
            "n_groups_cutoff_points": 10,
        }

    # Read data from data pipeline
    crypto_data = data_pipeline.CryptoData(LOGGER)
    train_df, _ = crypto_data.load_data()

    for coin in tqdm(crypto_data.coin_info):
        train_df, val_df = crypto_data.train_val_split(coin["symbol"])

        train_dataset = data_pipeline.CryptoDataSet(
            df=train_df,
            logger_=logger,
            n_cutoff_points=config.n_cutoff_points,
            n_groups_cutoff_points=config.n_groups_cutoff_points,
        )
        val_dataset = data_pipeline.CryptoDataSet(
            df=val_df,
            logger_=logger,
            n_cutoff_points=config.n_cutoff_points,
            n_groups_cutoff_points=config.n_groups_cutoff_points,
        )

        train_fold(train_dataset, val_dataset, config, LOGGER)
        LOGGER.info(f"Fold with coin {coin["symbol"]} done.")
    LOGGER.info("All folds are done.")
