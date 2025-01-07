"""
This module contains the tests for the utils.hf_sentiment module.

hf_sentiment is a module to generate sentiment scores using a Hugging Face model.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.utils import hf_sentiment
from CryptoFraudDetection.utils.enums import LoggerMode

LOGGER = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")

def test_initialization():
    """Test the instantiation of the Scoring class"""
    sentiment_classifier = hf_sentiment.Scoring(logger_=LOGGER)
    assert sentiment_classifier is not None
    assert sentiment_classifier.model is not None, "Model should be loaded during initialization."
    assert sentiment_classifier.tokenizer is not None, "Tokenizer should be loaded during initialization."

def test_score():
    """Test the score method of the Scoring class"""
    text = "This is a test sentence."
    
    sentiment_classifier = hf_sentiment.Scoring(logger_=LOGGER)
    sentiment_score = sentiment_classifier.score(text)
    
    assert sentiment_score is not None
    assert isinstance(sentiment_score, float)
    assert 1 >= sentiment_score >= 0

def test_score_multiple():
    """Test the score method of the Scoring class with multiple sentences"""
    text = ["I hate this shit so much!", "I love this show so much!"]
    
    sentiment_classifier = hf_sentiment.Scoring(logger_=LOGGER)
    sentiment_scores = sentiment_classifier.score(text)
    
    assert sentiment_scores is not None
    assert isinstance(sentiment_scores, list)
    assert len(sentiment_scores) == 2
    for x in sentiment_scores:
        assert isinstance(x, float)
        assert 1 >= x >= 0
    assert sentiment_scores[0] < 0.1
    assert sentiment_scores[1] > 0.9

def test_on_the_fly_scoring():
    """Test on the fly scoring of text."""
    text = ["I hate this shit so much!", "I love this show so much!"]
    
    sentiment_scores = hf_sentiment.score(logger_=LOGGER, text=text)
    
    assert sentiment_scores is not None
    assert isinstance(sentiment_scores, list)
    assert len(sentiment_scores) == 2
    for x in sentiment_scores:
        assert isinstance(x, float)
        assert 1 >= x >= 0
    assert sentiment_scores[0] < 0.1
    assert sentiment_scores[1] > 0.9

def test_long_text_scoring():
    """Test scoring of long texts."""
    text = [
        " ".join(["I hate this shit so much!"] * 1_000),
        " ".join(["I love this show so much!"] * 10_000)
    ]

    LOGGER.debug(f"Scoring long text. Text lengths: {[len(t) for t in text]}")
    # prevent long debug output
    info_logger = logger.Logger(name=__name__, level=LoggerMode.INFO, log_dir="../logs")
    sentiment_scores = hf_sentiment.score(logger_=info_logger, text=text)
    
    assert sentiment_scores is not None
    assert isinstance(sentiment_scores, list)
    assert len(sentiment_scores) == 2
    for x in sentiment_scores:
        assert isinstance(x, float)
        assert 1 >= x >= 0
    assert sentiment_scores[0] < 0.1
    assert sentiment_scores[1] > 0.9
    

if __name__ == '__main__':
    test_initialization()
    test_score()
    test_score_multiple()
    test_on_the_fly_scoring()
    test_long_text_scoring()