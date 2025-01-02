"""
File: comparitech.py.

Description:
    A web scraper for extracting cryptocurrency scam data from Comparitech's Crypto Scam List.
    Uses Selenium WebDriver to navigate and extract tabular data from the datawrapper interface.
    Implements specific error handling for each potential failure point.
"""

from collections import defaultdict

from selenium.common.exceptions import (
    JavascriptException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from CryptoFraudDetection.scraper import utils
from CryptoFraudDetection.utils.logger import Logger


class ComparitechScraper:
    """
    A scraper for extracting cryptocurrency scam data from Comparitech.

    This class handles the web scraping of tabular data from Comparitech's Crypto Scam List,
    including pagination and data extraction with specific error handling for each operation.

    Attributes:
        logger (Logger): Logger instance for tracking scraping progress and errors.
        base_url (str): Base URL for the Comparitech Crypto Scam List.
        page_load_timeout (int): Maximum time in seconds to wait for a page to load.
        element_wait_timeout (int): Maximum time in seconds to wait for an element to appear.

    Example:
        >>> logger = Logger("crypto_scraper")
        >>> scraper = ComparitechScraper(logger)
        >>> data = scraper.get_data(headless=True)

    """

    def __init__(
        self,
        logger: Logger,
        base_url: str = "https://datawrapper.dwcdn.net/9nRA9/107/",
        page_load_timeout: int = 30,
        element_wait_timeout: int = 10,
    ) -> None:
        """
        Initialize the ComparitechScraper.

        Args:
            logger: Logger instance for tracking scraping progress and errors
            base_url: Base URL for the Comparitech Crypto Scam List
            page_load_timeout: Maximum time in seconds to wait for page loads
            element_wait_timeout: Maximum time in seconds to wait for elements to appear

        """
        self.logger = logger
        self.base_url = base_url
        self.page_load_timeout = page_load_timeout
        self.element_wait_timeout = element_wait_timeout

    def get_data(
        self,
        headless: bool = True,
        test_run: bool = False,
    ) -> dict[str, list[str]]:
        """
        Execute the scraping process and return collected data.

        Args:
            headless: Whether to run browser in headless mode. Defaults to True.
            test_run: Whether to limit scraping to a small number of pages. Defaults to False.

        Returns:
            Dictionary mapping column names to lists of values. Each key represents
            a column header, and its value is a list of strings containing the
            data from that column.

        Raises:
            WebDriverException: If there are issues with browser initialization or control
            TimeoutException: If the page fails to load within the specified timeout

        """
        driver = None
        data: dict[str, list[str]] = {}
        try:
            # Initialize the WebDriver with appropriate settings
            self.logger.info("Initializing web driver")
            driver = utils.get_driver(headless)
            driver.set_page_load_timeout(self.page_load_timeout)

            # Execute the main scraping process
            data = self._perform_scrape(driver, test_run)

        except WebDriverException as e:
            self.logger.handle_exception(
                WebDriverException,
                f"Failed to initialize or use WebDriver: {e!s}",
            )

        # Close the WebDriver instance after scraping
        finally:
            if driver:
                try:
                    self.logger.debug("Closing web driver")
                    driver.quit()
                except WebDriverException as e:
                    self.logger.handle_exception(
                        WebDriverException,
                        f"Error closing WebDriver: {e!s}",
                        "warning",
                    )

        return data

    def _perform_scrape(
        self,
        driver: WebDriver,
        test_run: bool,
    ) -> dict[str, list[str]]:
        """
        Perform the main scraping operation.

        Navigates through all pages of the table, collecting data from each page
        until no more pages are available.

        Args:
            driver: Selenium WebDriver instance to use for web interactions
            test_run: Whether to limit scraping to a small number of pages

        Returns:
            Dictionary containing the scraped data, mapping column names to lists
            of values from all pages combined

        Raises:
            TimeoutException: If page loading times out
            WebDriverException: For other WebDriver related errors
            StaleElementReferenceException: If page elements become stale during scraping

        """
        try:
            # Navigate to the target URL
            self.logger.info(f"Navigating to {self.base_url}")
            driver.get(self.base_url)

        except TimeoutException as e:
            self.logger.handle_exception(
                TimeoutException,
                f"Failed to load the initial page - timeout: {e!s}",
            )

        except WebDriverException as e:
            self.logger.handle_exception(
                WebDriverException,
                f"Failed to navigate to URL: {e!s}",
            )

        # Initialize result storage and page counter
        results: dict[str, list[str]] = defaultdict(list)
        page_num = 0

        while True:
            page_num += 1

            self.logger.info(f"Scraping page {page_num}")

            # Extract data from current page
            page_results = self._extract_results(driver)

            # Break if no data found on current page
            if not page_results:
                self.logger.info(f"No data found on page {page_num}")
                break

            # Merge current page results into overall results
            for key, value in page_results.items():
                results[key].extend(value)

            # Try to navigate to next page, break if no more pages
            if not self._click_next_page(driver):
                break

            # Break early if test run
            if test_run and page_num == 3:
                break

        self.logger.info(f"Completed scraping {page_num} pages")
        return dict(results)

    def _extract_results(self, driver: WebDriver) -> dict[str, list[str]]:
        """
        Extract data from the current page's table.

        Locates the table on the current page and extracts all cell values,
        organizing them by column headers.

        Args:
            driver: Selenium WebDriver instance pointing to the page with the table

        Returns:
            Dictionary mapping column names to lists of values for the current page.
            Returns empty dictionary if no data is found or on error.

        Raises:
            TimeoutException: If table elements don't load within timeout
            StaleElementReferenceException: If elements become stale during processing

        """
        results: dict[str, list[str]] = defaultdict(list)

        try:
            # Wait for table to become available in the DOM
            self.logger.debug("Waiting for table to load")
            table = WebDriverWait(driver, self.element_wait_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "table")),
            )

            # Extract and validate table headers
            columns = table.find_elements(By.TAG_NAME, "th")
            if not columns:
                self.logger.info("No table headers found")
                return results

            # Extract and validate table rows
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")
            if not rows:
                self.logger.info("No table rows found")
                return results

            # Process each row and extract cell data
            for row in rows:
                th_cells = row.find_elements(By.TAG_NAME, "th")
                td_cells = row.find_elements(By.TAG_NAME, "td")
                cells = th_cells + td_cells

                # Map cell data to corresponding column
                for i, cell in enumerate(cells):
                    if i < len(columns):  # Ensure column index exists
                        cell_text = cell.text.strip()
                        results[columns[i].text].append(cell_text)

        except TimeoutException as e:
            self.logger.handle_exception(
                TimeoutException,
                f"Timeout waiting for table to load: {e!s}",
            )

        except StaleElementReferenceException as e:
            self.logger.handle_exception(
                StaleElementReferenceException,
                f"Elements became stale during extraction: {e!s}",
            )

        return results

    def _click_next_page(self, driver: WebDriver) -> bool:
        """
        Attempt to navigate to the next page of results.

        Uses JavaScript to click the next page button, as this is more reliable
        than selenium's click method for this particular interface.

        Args:
            driver: Selenium WebDriver instance on the current page

        Returns:
            True if successfully navigated to next page, False if at last page
            or if navigation fails

        """
        try:
            # Attempt to click the next page button using JavaScript
            self.logger.debug("Attempting to click next page button")
            driver.execute_script(
                "document.querySelector('button.next').click()",
            )
            return True

        except JavascriptException:
            self.logger.debug("Reached last page or next button not found")
            return False
