"""
This module contains the tests for the scraper.comparitech module.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.scraper.comparitech import ComparitechScraper
from CryptoFraudDetection.utils.enums import LoggerMode

logger = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")


def test_initialization():
    """
    Test the initialization of the ComparitechScraper class
    """
    scraper = ComparitechScraper(logger=logger)
    assert scraper.base_url == "https://datawrapper.dwcdn.net/9nRA9/107/"
    assert scraper.page_load_timeout == 30
    assert scraper.element_wait_timeout == 10


def test_get_main_results():
    """
    Test the get_data method of the ComparitechScraper class
    """
    scraper = ComparitechScraper(logger=logger)
    results = scraper.get_data(test_run=True)
    assert results
    for column in list(results.keys()):
        assert results[column]
        assert len(results[column]) > 0
