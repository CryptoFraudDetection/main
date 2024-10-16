"""
File: google_results.py

Description:
- This file contains the GoogleResultsScraper class, used to scrape search results from Google.
  It leverages Selenium for browser automation and BeautifulSoup for parsing HTML content.
"""

import time
from collections import defaultdict
from typing import Dict, List

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

import CryptoFraudDetection.scraper.utils as utils
from CryptoFraudDetection.utils.exceptions import (
    DetectedBotException,
    InvalidParameterException,
)
from CryptoFraudDetection.utils.logger import Logger


class GoogleResultsScraper:
    """
    A scraper class that interacts with Google Search to extract search result links, titles, and descriptions using Selenium.

    Attributes:
        logger (Logger): Logger instance used for logging.
        box_class (str): CSS class name for search result boxes.
        desc_class (str): CSS class name for search result descriptions.
        cookie_id (str): ID for Google's cookie acceptance button.
        search_box_id (str): ID for Google's search box.
        next_button_id (str): ID for Google's 'Next' button to go to the next page of results.

    Methods:
        get_main_results(query: str, n_sites: int = 5, headless: bool = False) -> Dict[str, List[str]]:
            Scrapes Google search results for a given query.
    """

    def __init__(
        self,
        logger: Logger,
        box_class: str = "MjjYud",
        desc_class: str = "VwiC3b",
        cookie_id: str = "L2AGLb",
        search_box_id: str = "APjFqb",
        next_button_id: str = "pnnext",
    ) -> None:
        """
        Initializes the GoogleResultsScraper class.

        Args:
            logger (Logger): The logger instance to use for logging.
            box_class (str): CSS class name for the search result container. Defaults to "MjjYud".
            desc_class (str): CSS class name for the description. Defaults to "VwiC3b".
            cookie_id (str): ID for the cookie acceptance button. Defaults to "L2AGLb".
            search_box_id (str): ID for the search input box. Defaults to "APjFqb".
            next_button_id (str): ID for the next page button. Defaults to "pnnext".
        """
        self.logger = logger
        self.box_class = box_class
        self.desc_class = desc_class
        self.cookie_id = cookie_id
        self.search_box_id = search_box_id
        self.next_button_id = next_button_id

    def get_main_results(
        self,
        query: str,
        n_sites: int = 5,
        delay_between_pages: float = 1.0,
        headless: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Scrapes Google search results for the given query using Selenium.

        Args:
            query (str): The search query to be performed on Google.
            n_sites (int): The number of Google search result pages to scrape (default is 5).
            delay_between_pages (float): Delay (in seconds) between page navigations. Default is 1 second.
            headless (bool): Whether to run the browser in headless mode (default is False).

        Returns:
            Dict[str, List[str]]: A dictionary containing the links, titles, and descriptions of the search results.

        Raises:
            ValueError: If the number of sites is less than 1.
            NoSuchElementException: If a required HTML element is not found.
            Exception: If Google detects the scraper as a bot or if no results are found.
        """
        # Validate the number of sites to scrape
        if n_sites < 1:
            self.logger.handle_exception(
                InvalidParameterException, "Number of sites must be at least 1"
            )

        self.logger.info(f"Starting Google search for query: {query}")

        driver = utils.get_driver(headless=headless)
        driver.get("https://www.google.com")

        # Accept Google's cookies
        try:
            driver.find_element(By.ID, self.cookie_id).click()
            self.logger.info("Accepted Google's cookies.")
        except NoSuchElementException:
            self.logger.warning("Cookie acceptance button not found.")

        time.sleep(delay_between_pages)

        # Enter search query in the search box and submit
        try:
            search_box = driver.find_element(By.ID, self.search_box_id)
            search_box.send_keys(query)
            search_box.submit()
            self.logger.info("Search query submitted successfully.")
        except NoSuchElementException:
            self.logger.handle_exception(
                NoSuchElementException,
                "Could not find the search box element.",
            )

        # Dictionary to store the scraped results
        results: Dict[str, List[str]] = defaultdict(list)

        # Loop through the specified number of Google search result pages
        for i in range(n_sites):
            time.sleep(delay_between_pages)

            # Find all result boxes based on the box_class
            try:
                result_boxes = driver.find_elements(By.CLASS_NAME, self.box_class)
            except NoSuchElementException:
                self.logger.handle_exception(
                    NoSuchElementException, "Could not find any result boxes."
                )

            # Check if Google has detected the scraper as a bot
            if len(result_boxes) == 0:
                self.logger.handle_exception(
                    DetectedBotException,
                    "No results found. Possibly blocked by Google.",
                )

            # Extract link, title, and description from each result box
            for box in result_boxes:
                try:
                    link = box.find_element(By.TAG_NAME, "a").get_attribute("href")
                    title = box.find_element(By.TAG_NAME, "h3").text
                    desc = box.find_element(By.CLASS_NAME, self.desc_class).text

                    results["link"].append(link)
                    results["title"].append(title)
                    results["description"].append(desc)
                except NoSuchElementException:
                    self.logger.warning("Missing elements in result box.")
                    continue

            # Navigate to the next page of results
            if i != n_sites - 1:
                try:
                    next_button = driver.find_element(By.ID, self.next_button_id)
                    next_button.click()
                    self.logger.info(f"Navigating to page {i + 2} of results.")
                except NoSuchElementException:
                    self.logger.warning("Next page button not found.")
                    break

        driver.quit()
        self.logger.info(
            f"Scraped {len(results['link'])} results from Google using the query: {query}. Finished."
        )
        return results
