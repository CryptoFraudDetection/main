import CryptoFraudDetection.utils.logger as logger

import transformers
from numpy import linalg

class Embedder:
    def __init__(self, logger_: logger.Logger, model_name: str = "jinaai/jina-embeddings-v2-small-en", text: list[str] | str | None = None):
        self.model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True)
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
        text_ = text if text is not None else self.text
        match text_:
            case None:
                return None
            case str():
                text_ = [text_]
            case list():
                pass
            case _:
                self._logger.error("Invalid data type for text.")
                return None
        self._logger.debug(f"Embedding text: {text_}")
        self.embeddings = self.model.encode(text_)  # TODO: data type?
        return self.embeddings


def embed(logger_: logger.Logger, text: list[str] | str, model_name: str | None = None) -> list:
    if model_name is not None:
        embedder = Embedder(logger_, model_name, test=text)
    else:
        embedder = Embedder(logger_, text=text)
    return embedder.embed()

def cos_sim(a, b):
    return (a @ b.T) / (linalg.norm(a)*linalg.norm(b))
