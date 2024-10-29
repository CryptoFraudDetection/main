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
    Test the initialization of the TwitterScraper class with and without credentials.
    """
    # Test with credentials
    username = "test_user"
    password = "test_pass"
    scraper_with_credentials = TwitterScraper(
        username=username, password=password, logger=logger
    )
    assert scraper_with_credentials.username == username
    assert scraper_with_credentials.password == password

    # Test without credentials
    scraper_without_credentials = TwitterScraper(logger=logger)
    assert scraper_without_credentials.username is None
    assert scraper_without_credentials.password is None


def test_scrape_with_cookies():
    """
    Test the scrape_with_cookies method of the TwitterScraper class
    """
    scraper = TwitterScraper(username="test_user", password="test_pass", logger=logger)
    tweets_data = scraper.scrape_with_cookies(
        tweet_count=1, search_query="Bitcoin", headless=True
    )

    # Verify that there is at least one tweet in the data
    assert len(tweets_data["Username"]) >= 1
    assert len(tweets_data["Tweet"]) >= 1

    # Check for the required keys in the dictionary
    assert "Username" in tweets_data
    assert "Tweet" in tweets_data
    assert "Timestamp" in tweets_data
    assert "Likes" in tweets_data
    assert "Impressions" in tweets_data
