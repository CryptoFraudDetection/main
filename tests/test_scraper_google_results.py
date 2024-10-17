"""
This module contains the tests for the scraper.google_results module.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.scraper.google_results import GoogleResultsScraper
from CryptoFraudDetection.utils.enums import LoggerMode

logger = logger.Logger(name=__name__, level=LoggerMode.DEBUG)


def test_initialization():
    """
    Test the initialization of the GoogleResultsScraper class
    """
    scraper = GoogleResultsScraper(logger=logger)
    assert scraper.box_class == "MjjYud"
    assert scraper.desc_class == "VwiC3b"
    assert scraper.cookie_id == "L2AGLb"
    assert scraper.search_box_id == "APjFqb"
    assert scraper.next_button_id == "pnnext"


def test_get_main_results():
    """
    Test the get_main_results method of the GoogleResultsScraper class
    """
    scraper = GoogleResultsScraper(logger=logger)
    results = scraper.get_main_results("test", n_sites=2, headless=True)
    assert len(results["link"]) >= 1
    assert len(results["title"]) >= 1
    assert len(results["description"]) >= 1
