"""
File: google_results.py

Description:
- This file contains the GoogleResultsScraper class, used to scrape search results from Google.
"""

import time
import hashlib
from collections import defaultdict
from typing import Dict, List
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By

from CryptoFraudDetection.scraper import utils
from CryptoFraudDetection.utils.exceptions import (
    DetectedBotException,
    InvalidParameterException,
)
from CryptoFraudDetection.utils.logger import Logger


class GoogleResultsScraper:
    """
    A scraper class that interacts with Google Search to extract search result links,
    titles, and descriptions using Selenium.

    Attributes:
        logger (Logger): Logger instance used for logging.
        box_class (str): CSS class name for search result boxes.
        desc_class (str): CSS class name for search result descriptions.
        cookie_id (str): ID for Google's cookie acceptance button.
        search_box_id (str): ID for Google's search box.
        next_button_id (str): ID for Google's 'Next' button to go to the next page of results.

    Methods:
        get_main_results: Scrapes Google search results for the given query.
        _validate_input: Validates the number of search result pages to scrape.
        _perform_search: Opens Google, accepts cookies, and submits the search query.
        _accept_cookies: Accepts Google's cookies if the prompt appears.
        _submit_search_query: Finds the search box, enters the query, and submits it.
        _scrape_multiple_pages: Loops through multiple result pages and extracts search results.
        _get_result_boxes: Fetches search result boxes from the current Google result page.
        _extract_results: Extracts links, titles, and descriptions from result boxes.
        _click_next_page: Clicks the 'Next' button to go to the next result page.
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
            query (str): The search query to be performed on Google.
        """
        self.logger = logger
        self.box_class = box_class
        self.desc_class = desc_class
        self.cookie_id = cookie_id
        self.search_box_id = search_box_id
        self.next_button_id = next_button_id
        self.query = ""

    def get_main_results(
        self,
        query: str,
        n_sites: int = 5,
        delay_between_pages: float = 1.0,
        headless: bool = False,
        proxy_address: str | None = None,
    ) -> Dict[str, List[str]]:
        """
        Function to scrape Google search results for the given query.

        Args:
            query (str): The search query to be performed on Google.
            n_sites (int): The number of Google search result pages to scrape (default is 5).
            delay_between_pages (float): Delay (in seconds) between page navigations.
                Default is 1 second.
            headless (bool): Whether to run the browser in headless mode (default is False).
            proxy_address (str): The proxy address to use for the browser (default is None).

        Returns:
            Dict[str, List[str]]: A dictionary containing the links, titles,
            and descriptions of the search results.
        """
        self.query = query
        self._validate_input(n_sites)
        driver = utils.get_driver(headless=headless, proxy_address=proxy_address)

        try:
            self._perform_search(driver, self.query, delay_between_pages)
            results = self._scrape_multiple_pages(driver, n_sites, delay_between_pages)
        finally:
            driver.quit()

        return results

    def _validate_input(self, n_sites: int) -> None:
        """
        Validates the number of search result pages to scrape.

        Args:
            n_sites (int): The number of Google search result pages to scrape.

        Raises:
            InvalidParameterException: If the number of sites is less than 1.
        """
        if n_sites < 1:
            self.logger.handle_exception(
                InvalidParameterException, "Number of sites must be at least 1"
            )

    def _perform_search(
        self, driver: webdriver.Firefox, query: str, delay: float
    ) -> None:
        """
        Opens Google, accepts cookies, and submits the search query.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.
            query (str): The search query to be performed on Google.
            delay (float): The delay between actions (in seconds).
        """
        driver.get("https://www.google.com")
        self._accept_cookies(driver)
        time.sleep(delay)
        self._submit_search_query(driver, query)

    def _accept_cookies(self, driver: webdriver.Firefox) -> None:
        """
        Accepts Google's cookies if the prompt appears.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.
        """
        try:
            driver.find_element(By.ID, self.cookie_id).click()
            self.logger.info("Accepted Google's cookies.")
        except NoSuchElementException:
            self.logger.warning("Cookie acceptance button not found.")

    def _submit_search_query(self, driver: webdriver.Firefox, query: str) -> None:
        """
        Finds the search box, enters the query, and submits it.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.
            query (str): The search query to be submitted.

        Raises:
            NoSuchElementException: If the search box is not found on the page.
        """
        try:
            search_box = driver.find_element(By.ID, self.search_box_id)
            search_box.send_keys(query)
            search_box.submit()
            self.logger.info("Search query submitted successfully.")
        except NoSuchElementException:
            self.logger.handle_exception(
                NoSuchElementException, "Could not find the search box element."
            )

    def _scrape_multiple_pages(
        self,
        driver: webdriver.Firefox,
        n_sites: int,
        delay: float,
    ) -> Dict[str, List[str]]:
        """
        Loops through multiple result pages and extracts search results.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.
            n_sites (int): The number of Google search result pages to scrape.
            delay (float): Delay (in seconds) between page navigations.

        Returns:
            Dict[str, List[str]]: A dictionary containing the links, titles,
            and descriptions of the search results.
        """
        results: Dict[str, List[str]] = defaultdict(list)

        for i in range(n_sites):
            time.sleep(delay)
            result_boxes = self._get_result_boxes(driver)
            if not result_boxes:
                self.logger.handle_exception(
                    DetectedBotException,
                    "No results found. Possibly blocked by Google.",
                )

            self._extract_results(result_boxes, results)

            if i != n_sites - 1 and not self._click_next_page(driver):
                break

        return results

    def _get_result_boxes(self, driver: webdriver.Firefox) -> List[WebElement]:
        """
        Fetches search result boxes from the current Google result page.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.

        Returns:
            List[WebElement]: A list of WebElement objects representing search result boxes.

        Raises:
            NoSuchElementException: If no result boxes are found on the page.
        """
        try:
            return driver.find_elements(By.CLASS_NAME, self.box_class)
        except NoSuchElementException:
            self.logger.handle_exception(
                NoSuchElementException, "Could not find any result boxes."
            )
            return []

    def _extract_results(
        self, result_boxes: List[WebElement], results: Dict[str, List[str]]
    ) -> None:
        """
        Extracts links, titles, and descriptions from result boxes.

        Args:
            result_boxes (List[WebElement]): A list of WebElement objects
                representing search result boxes.
            results (Dict[str, List[str]]): A dictionary to store extracted
                links, titles, and descriptions.
        """
        for box in result_boxes:
            try:
                link = str(box.find_element(By.TAG_NAME, "a").get_attribute("href"))
                title = str(box.find_element(By.TAG_NAME, "h3").text)
                desc = str(box.find_element(By.CLASS_NAME, self.desc_class).text)

                results["link"].append(link)
                results["title"].append(title)
                results["description"].append(desc)
                results["query"].append(self.query)

                id_pre_hash = link + title
                results["id"].append(hashlib.md5(id_pre_hash.encode()).hexdigest())
            except NoSuchElementException:
                self.logger.debug("Missing elements in result box. Skipping")
                continue

    def _click_next_page(self, driver: webdriver.Firefox) -> bool:
        """
        Clicks the 'Next' button to go to the next result page.

        Args:
            driver (WebDriver): The Selenium WebDriver instance.

        Returns:
            bool: True if the 'Next' button was clicked successfully, False if not.

        Raises:
            NoSuchElementException: If the next page button is not found on the page.
        """
        try:
            next_button = driver.find_element(By.ID, self.next_button_id)
            next_button.click()
            self.logger.info("Navigated to the next page of results.")
            return True
        except NoSuchElementException:
            self.logger.warning("Next page button not found.")
            return False
