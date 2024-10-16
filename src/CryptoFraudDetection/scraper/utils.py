"""
Miscellaneous utility functions for the scraper module.
"""

from selenium import webdriver


def get_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Return a Selenium Chrome WebDriver object.

    Args:
        headless (bool): Whether to run the browser in headless mode.

    Returns:
        WebDriver: A Selenium WebDriver object.
    """
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    return webdriver.Firefox(options=options)
