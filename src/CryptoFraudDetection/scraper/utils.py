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
    options.add_argument("--no-sandbox")  # Evtl entfernen, ist nicht sicher
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Firefox(options=options)
