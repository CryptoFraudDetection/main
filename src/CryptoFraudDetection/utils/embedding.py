from numpy import linalg
import torch
import transformers

import CryptoFraudDetection.utils.logger as logger
class Embedder:
    def __init__(self, logger_: logger.Logger, model_name: str = "jinaai/jina-embeddings-v2-small-en", text: list[str] | str | None = None):
        self._logger: logger.Logger = logger_
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self._logger.info("Using CUDA (NVIDIA GPU).")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self._logger.info("Using Metal Performance Shaders (MPS) on macOS.")
        else:
            self.device = torch.device("cpu")
            self._logger.info("Using CPU. Consider using a GPU for faster performance.")
        self.model = transformers.AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        match text:
            case None:
                self.text: list = []
            case str():
                self.text = [text]
            case list():
                self.text = text
        self.embeddings: list = []

    def embed(self, text: list[str] | str | None = None, max_tokens: int = 8192) -> list:
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

        embeddings = []
        for t in text_:
            # Tokenize and check if truncation is needed
            encoded = self.tokenizer(t, add_special_tokens=True)
            if len(encoded["input_ids"]) > max_tokens:
                self._logger.warning(
                    f"Input text exceeds {max_tokens} tokens and will be truncated. Original length: {len(encoded['input_ids'])} tokens. Text: {t[:100]}..."
                )
            
            # Truncate to max_tokens
            truncated_input_ids = encoded["input_ids"][:max_tokens]
            truncated_attention_mask = [1] * len(truncated_input_ids)

            # Convert truncated inputs to tensors and move to GPU
            tokenized_inputs = {
                "input_ids": torch.tensor([truncated_input_ids]).to(self.device),
                "attention_mask": torch.tensor([truncated_attention_mask]).to(self.device),
            }

            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(embedding.cpu().detach().numpy())

        return embeddings



def embed(logger_: logger.Logger, text: list[str] | str, model_name: str | None = None) -> list:
    if model_name is not None:
        embedder = Embedder(logger_, model_name, test=text)
    else:
        embedder = Embedder(logger_, text=text)
    return embedder.embed()

def cos_sim(a, b):
    return (a @ b.T) / (linalg.norm(a)*linalg.norm(b))
