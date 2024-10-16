from selenium import webdriver

"""
Miscellaneous utility functions for the scraper module.
"""


def get_driver(headless=False):
    """
    Return a Selenium Chrome WebDriver object.

    Returns:
        WebDriver: A Selenium WebDriver object.
    """
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    return webdriver.Firefox(options=options)
