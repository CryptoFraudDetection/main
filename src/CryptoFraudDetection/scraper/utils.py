"""
Miscellaneous utility functions for the scraper module.
"""

from selenium import webdriver


def get_driver(
    headless: bool = False, proxy_address: str | None = None
) -> webdriver.Firefox:
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
    if proxy_address:
        options.add_argument(f"--proxy-server={proxy_address}")

    return webdriver.Firefox(options=options)
