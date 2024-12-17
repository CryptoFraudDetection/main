import torch
import transformers
from tqdm import tqdm

from CryptoFraudDetection.utils import logger
from CryptoFraudDetection.utils.enums import LoggerMode


class Scoring:
    """Embeds text using a specified model.

    Instance Variables:
        model: The transformer model used for scoring text.
        tokenizer: The tokenizer used for processing text.
        text: A list of strings to be embedded.
        sentiment_scores: A list to store sentiment scores.
        _logger: Logger instance for logging operations.
    """

    def __init__(
        self,
        logger_: logger.Logger,
        model_name: str = "ccdv/lsg-distilbert-base-uncased-4096",
        text: list[str] | str | None = None,
    ):
        self._logger = logger_
        self.text = text

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, pool_with_global=True,)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.sentiment_scores = []

    def score(self, text: list[str] | str | None = None) -> float:
        """Score the given text.

        Args:
            text: A string, list of strings, or None. Uses the instance's text if None.

        Returns:
            The generated sentiment score based on the text.
        """
        text = text if text is not None else self.text
        match text:
            case None:
                return None
            case str():
                text = [text]
                original_input_type = str
            case list():
                original_input_type = list
            case _:
                self._logger.error("Invalid data type for text.")
                return None

        self._logger.debug(
            f"Embedding text. Total number of inputs: {len(text)}"
        )

        self.sentiment_scores = []  # replace scores if they exist
        for i, text in enumerate(tqdm(text)):
            self._logger.debug(f"Scoring text {i}/{len(text)}: {text}")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, num_classes)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            self.sentiment_scores.append(probs)


        if original_input_type == str:
            return self.sentiment_scores[0]
        return self.sentiment_scores


def score(
    logger_: logger.Logger,
    text: list[str] | str,
    model_name: str | None = None,
    model_architecture: str | None = None,
) -> list:
    """Embeds text using a specified model.

    Args:
        logger_: Logger instance for logging operations.
        text: A string or list of strings to generate sentiment scores.
        model_name: The model name to use for generating sentiment scores. Defaults to the Scoring's default.
        model_architecture: The architecture of the model to be used for scoring the text. Defaults to the Scoring's default.

    Returns:
        A sentiment score float or a list of sentiment scores for the input text(s).
    """
    parameters = {}
    if model_name:
        parameters["model_name"] = model_name
    if model_architecture:
        parameters["model_architecture"] = model_architecture

    scorer = Scoring(logger_, text=text, **parameters)
    return scorer.sentiment_scores


if __name__ == "__main__":
    LOGGER = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")
    scorer = Scoring(LOGGER)