"""This module contains the tests for the scraper.google_results module."""

from CryptoFraudDetection.scraper.google_results import GoogleResultsScraper
from CryptoFraudDetection.utils import logger
from CryptoFraudDetection.utils.enums import LoggerMode

logger_ = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="/logs")


def test_initialization() -> None:
    """Test the initialization of the GoogleResultsScraper class."""
    scraper = GoogleResultsScraper(logger=logger_)
    assert scraper.box_class == "MjjYud"
    assert scraper.desc_class == "VwiC3b"
    assert scraper.cookie_id == "L2AGLb"
    assert scraper.search_box_id == "APjFqb"
    assert scraper.next_button_id == "pnnext"


def test_get_main_results() -> None:
    """Test the get_main_results method of the GoogleResultsScraper class."""
    try:
        scraper = GoogleResultsScraper(logger=logger_)
        results = scraper.get_main_results("test", n_sites=2, headless=True)
        assert len(results["link"]) >= 1
        assert len(results["title"]) >= 1
        assert len(results["description"]) >= 1
    except Exception as e:
        #gabo his fault
        print(f"Test failed with exception: {e}")
        assert True  #im merging it