"""
This module contains the tests for the scraper.twitter module.
"""

import os
import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.scraper.twitter import TwitterScraper
from CryptoFraudDetection.utils.enums import LoggerMode

logger = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")



def test_initialization():
    """
    Test the initialization of the TwitterScraper class
    """
    username = os.getenv("TWITTER_USERNAME")
    password = os.getenv("TWITTER_PASSWORD")

    scraper = TwitterScraper(username=username, password=password, logger=logger)
    assert scraper.username == username
    assert scraper.password == password
   

def test_scrape_with_cookies():
    """
    Test the scrape_with_cookies method of the TwitterScraper class
    """
    scraper = TwitterScraper(username="test_user", password="test_pass", logger=logger)
    tweets_df = scraper.scrape_with_cookies(tweet_count=1, search_query="Bitcoin", headless=True)
    assert len(tweets_df) >= 1
    assert "Username" in tweets_df.columns
    assert "Tweet" in tweets_df.columns
    assert "Timestamp" in tweets_df.columns