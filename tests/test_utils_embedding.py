"""
This module contains the tests for the utils.embedding module.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.utils import embedding
from CryptoFraudDetection.utils.enums import LoggerMode

LOGGER = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")

def test_initialization():
    """Test the initialization of the RedditScraper class"""
    embedding_ = embedding.Embedder(logger_=LOGGER)
    assert embedding_ is not None

def test_embed():
    """Test the embed method of the Embedding class"""
    text = "This is a test sentence."
    
    embedder = embedding.Embedder(logger_=LOGGER)
    embedded_text = embedder.embed(text)
    
    assert embedded_text is not None
    assert len(embedded_text) == 1
    assert len(embedded_text[0]) > 0

def test_embed_multiple():
    """Test the embed_multiple method of the Embedding class"""
    text = ["This is a test sentence.", "This is another test sentence."]
    
    embedder = embedding.Embedder(logger_=LOGGER)
    embedded_text = embedder.embed(text)
    
    assert embedded_text is not None
    assert len(embedded_text) == 2
    for x in embedded_text:
        assert len(x) > 0

def test_on_the_fly_embedding():
    """Test on the fly embedding of text."""
    text = ["This is a test sentence.", "This is another test sentence."]
    
    embedded_text = embedding.embed(logger_=LOGGER, text=text)
    
    assert embedded_text is not None
    assert len(embedded_text) == 2
    for x in embedded_text:
        assert len(x) > 0

def test_cos_sim():
    """Test the cos_sim method of the Embedding class"""
    text = ["Hi, how are you?", "Hello, how are you?"]
    
    embedded_text = embedding.embed(logger_=LOGGER, text=text)
    cos_sim = embedding.cos_sim(embedded_text[0], embedded_text[1])
    LOGGER.debug(f"Cosine similarity of {text}: {cos_sim}")
    
    assert cos_sim is not None
    assert cos_sim > 0.9

def test_long_text_embedding():
    """Test embedding of a long text."""
    sentence = "This is a test sentence."
    
    text = [" ".join([sentence] * n) for n in (1_000, 1_000_000)]
    LOGGER.debug(f"Embedding long text. Text lengths: {[len(t) for t in text]}")
    # prevent long debug output
    info_logger = logger.Logger(name=__name__, level=LoggerMode.INFO, log_dir="../logs")
    embedded_text = embedding.embed(logger_=info_logger, text=text)
    
    assert embedded_text is not None
    assert len(embedded_text) == 2
    for x in embedded_text:
        assert len(x) > 0

if __name__ == '__main__':
    test_initialization()
    test_embed()
    test_embed_multiple()
    test_cos_sim()
    test_on_the_fly_embedding()
    test_long_text_embedding()