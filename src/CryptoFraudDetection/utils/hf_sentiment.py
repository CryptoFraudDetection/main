import random

import numpy as np
import torch
import transformers
from tqdm import tqdm

from torch.utils.data import DataLoader
from CryptoFraudDetection.utils import logger
from CryptoFraudDetection.utils.enums import LoggerMode

class Scoring:
    """Performs sentiment classification using a specified model.

    Instance Variables:
        model: The transformer model used for scoring text.
        tokenizer: The tokenizer used for processing text.
        text: A list of strings to be classified.
        sentiment_scores: A list to store sentiment scores.
        _logger: Logger instance for logging operations.
    """

    def __init__(
        self,
        logger_: logger.Logger,
        model_name: str = "siebert/sentiment-roberta-large-english",
        batch_size: int = 128,
    ):
        self._logger = logger_
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU).")
            self.batch_size = batch_size
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Metal Performance Shaders (MPS) on macOS.")
            self.batch_size = batch_size
        else:
            self.device = torch.device("cpu")
            print("Using CPU. Consider using a GPU for faster performance.")
            self.batch_size = batch_size // 4

        # Load model and tokenizer
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.sentiment_scores = []

    def _tokenize_batch(self, batch):
        """Tokenize a batch of text and move it to the device."""
        tokenized_inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in tokenized_inputs.items()}

    def score(self, text: list[str] | str) -> float | list:
        """Scores the given text for sentiment classification.

        Args:
            text: A string or list of strings to classify sentiment.

        Returns:
            A sentiment score (float in [0, 1]) or a list of sentiment scores.
        """
        if text is None:
            raise ValueError("Text must be provided for scoring.")

        # Handle single string or list of strings
        if isinstance(text, str):
            text = [text]
            single_input = True
        elif isinstance(text, list):
            single_input = False
        else:
            raise TypeError("Text must be a string or a list of strings.")

        self.sentiment_scores = []   # Reset sentiment scores

        # Batch processing
        data_loader = DataLoader(
            text,
            batch_size=self.batch_size,
            collate_fn=self._tokenize_batch,
            drop_last=False,
        )

        for batch_inputs in tqdm(data_loader, desc="Scoring batches"):
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            positive_probs = probs[:, 1].tolist()  # Extract positive class probabilities
            self.sentiment_scores.extend(positive_probs)

        return self.sentiment_scores[0] if single_input else self.sentiment_scores


def score(
    logger_: logger.Logger,
    text: list[str] | str,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> list | float:
    """Scores text for sentiment classification using a specified model.

    Args:
        logger_: Logger instance for logging operations.
        text: A string or list of strings to generate sentiment scores.
        model_name: The model name to use for generating sentiment scores. Defaults to the Scoring's default.
        batch_size: Number of texts to process in a single batch.

    Returns:
        A sentiment score float or a list of sentiment scores for the input text(s).
    """
    parameters = {}
    if model_name:
        parameters["model_name"] = model_name
    if batch_size:
        parameters["batch_size"] = batch_size
    scorer = Scoring(logger_, **parameters)
    return scorer.score(text)

