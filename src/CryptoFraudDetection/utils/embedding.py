import transformers
from numpy import linalg

import CryptoFraudDetection.utils.logger as logger


class Embedder:
    """Embeds text using a specified model.

    Instance Variables:
        model: The transformer model used for generating embeddings.
        text: A list of strings to be embedded.
        embeddings: A list to store generated embeddings.
        _logger: Logger instance for logging operations.
    """

    def __init__(
        self,
        logger_: logger.Logger,
        model_name: str = "jinaai/jina-embeddings-v2-small-en",
        text: list[str] | str | None = None,
    ):
        self.model = transformers.AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
        match text:
            case None:
                self.text: list = []
            case str():
                self.text = [text]
            case list():
                self.text = text
        self.embeddings: list = []
        self._logger: logger.Logger = logger_

    def embed(self, text: list[str] | str | None = None) -> list:
        """Generates embeddings for the given text.

        Args:
            text: A string, list of strings, or None. Uses the instance's text if None.

        Returns:
            A list of embeddings for the input text.
        """
        text_list = text if text is not None else self.text
        match text_list:
            case None:
                return None
            case str():
                text_list = [text_list]
            case list():
                pass
            case _:
                self._logger.error("Invalid data type for text.")
                return None

        self._logger.debug(
            f"Embedding text. Total number of inputs: {len(text_list)}"
        )

        for i, text in enumerate(text_list):
            self._logger.debug(f"Embedding text {i}/{len(text_list)}: {text}")
            embedding = self.model.encode(text)
            self.embeddings.append(embedding)

        if len(self.embeddings) == 1:
            return self.embeddings[0]
        return self.embeddings


def embed(
    logger_: logger.Logger,
    text: list[str] | str,
    model_name: str | None = None
) -> list:
    """Embeds text using a specified model.

    Args:
        logger_: Logger instance for logging operations.
        text: A string or list of strings to embed.
        model_name: The model name to use for embedding. Defaults to the Embedder's default.

    Returns:
        A list of embeddings for the input text.
    """
    if model_name:
        embedder = Embedder(logger_, model_name, text=text)
    else:
        embedder = Embedder(logger_, text=text)
    return embedder.embed()


def cos_sim(a, b):
    """Calculates the cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity as a float.
    """
    return (a @ b.T) / (linalg.norm(a) * linalg.norm(b))
