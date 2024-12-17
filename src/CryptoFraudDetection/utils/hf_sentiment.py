import torch
import transformers
from tqdm import tqdm

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
    ):
        self._logger = logger_
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.text = []
        self.sentiment_scores = []

    def score(self, text: list[str] | str) -> float | list:
        """Scores the given text for sentiment classification.

        Args:
            text: A string or list of strings to classify sentiment.

        Returns:
            A sentiment score (float in [0, 1]) or a list of sentiment scores.
        """
        text = text if text is not None else self.text
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

        self.sentiment_scores = []  # Reset sentiment scores
        for i, t in enumerate(tqdm(text)):
            self._logger.debug(f"Scoring text {i + 1}/{len(text)}: {t}")
            inputs = self.tokenizer(
                t, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits  # Model output logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Extract the positive class probability
            score = probs[0, 1].item()
            self.sentiment_scores.append(score)

        return self.sentiment_scores[0] if single_input else self.sentiment_scores


def score(
    logger_: logger.Logger,
    text: list[str] | str,
    model_name: str | None = None,
) -> list | float:
    """Scores text for sentiment classification using a specified model.

    Args:
        logger_: Logger instance for logging operations.
        text: A string or list of strings to generate sentiment scores.
        model_name: The model name to use for generating sentiment scores. Defaults to the Scoring's default.

    Returns:
        A sentiment score float or a list of sentiment scores for the input text(s).
    """
    parameters = {}
    if model_name:
        parameters["model_name"] = model_name
    scorer = Scoring(logger_, **parameters)
    return scorer.score(text)
