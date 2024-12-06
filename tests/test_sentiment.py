"""
This module contains the tests for the scraper.comparitech module.
"""
import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.utils import sentiment
from CryptoFraudDetection.utils.enums import LoggerMode

logger_ = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")

def test_get_client():
    """
    Test if getting the client works.
    """
    client = sentiment.get_client()
    assert client is not None
    logger_.debug(f"Client: {client}")
    
def test_get_chat_bot():
    """
    Test if getting the chatbot works.
    """
    chat_bot = sentiment.get_chat_bot()
    assert chat_bot is not None
    logger_.debug(f"ChatBot: {chat_bot}")
    
def test_create_response():
    """
    Test if creating a response works.
    """
    chat_bot = sentiment.get_chat_bot()
    response = chat_bot("Hello")
    assert response is not None
    logger_.debug(f"Response: {response}")
    
def test_sentiment():
    """
    Test if sentiment analysis works.
    """
    texts = ["Hello", "Goodbye"]
    scores = sentiment.sentiment(texts)
    assert scores is not None
    assert len(scores) == len(texts)
    assert all([score is not None for score in scores])
    logger_.debug(f"Scores: {scores}")
    
if __name__ == '__main__':
    test_get_client()
    test_get_chat_bot()
    test_create_response()
    test_sentiment()
