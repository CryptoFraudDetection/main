"""This module contains the tests for the scraper.twitter module."""

import pytest

from CryptoFraudDetection.scraper.twitter import TwitterScraper
from CryptoFraudDetection.utils.enums import LoggerMode
from CryptoFraudDetection.utils.logger import Logger

logger_ = Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="/logs")


def test_initialization() -> None:
    """Test the initialization of the TwitterScraper class with and without credentials."""
    # Test with credentials
    username = "test_user"
    password = "test_pass"
    cookies_file_path = "test_cookies.json"
    scraper_with_credentials = TwitterScraper(
        logger=logger_,
        username=username,
        password=password,
        cookies_file_path=cookies_file_path,
    )
    assert scraper_with_credentials.username == username
    assert scraper_with_credentials.password == password
    assert scraper_with_credentials.cookies_file_path == cookies_file_path

    # Test without credentials
    scraper_without_credentials = TwitterScraper(logger=logger_)
    assert scraper_without_credentials.username == ""
    assert scraper_without_credentials.password == ""
    assert scraper_without_credentials.cookies_file_path == ""


@pytest.mark.xfail(
    reason="Often fails due to being detected as a bot by Twitter"
)
def test_scrape_with_cookies() -> None:
    """Test the scrape_with_cookies method of the TwitterScraper class."""
    scraper = TwitterScraper(
        username="test_user",
        password="test_pass",
        cookies_file_path="data/raw/cookies/x.json",
        logger=logger_,
    )
    tweets_data = scraper.scrape_with_cookies(
        tweet_count=1,
        search_query="Bitcoin",
        headless=True,
    )

    # Verify that there is at least one tweet in the data
    assert len(tweets_data["Username"]) > 0
    assert len(tweets_data["Tweet"]) > 0

    # Check for the required keys in the dictionary
    assert "Username" in tweets_data
    assert "Tweet" in tweets_data
    assert "Timestamp" in tweets_data
    assert "Likes" in tweets_data
    assert "Impressions" in tweets_data
    assert "Comments" in tweets_data
    assert "Reposts" in tweets_data
    assert "Bookmarks" in tweets_data
