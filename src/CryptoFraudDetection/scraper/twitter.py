"""
File: twitter.py

Description:
- This file is used to scrape data from the website Twitter (X).
"""

import os
import time
import json
import random
from typing import List, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import (
    WebDriverException,
    NoSuchElementException,
    TimeoutException,
    ElementNotInteractableException,
    JavascriptException,
)

import CryptoFraudDetection.scraper.utils as utils
from CryptoFraudDetection.utils.logger import Logger
from CryptoFraudDetection.utils.exceptions import (
    AuthenticationError,
)


class TwitterScraper:
    """
    A scraper class for Twitter (X) that can be used to scrape tweets
    and other data from the platform.

    Attributes:
        username (str): The Twitter username.
        password (str): The Twitter password.
        logger (Logger): Logger for logging messages.
        cookies_loaded (bool): Whether cookies have been loaded for the current session.
        cookies_file_path (str): Path to the file containing cookies.
    """

    def __init__(
        self,
        logger: Logger,
        username: str = "",
        password: str = "",
        cookies_file_path: str = "",
    ) -> None:
        """
        Initializes the TwitterScraper class with optional login credentials and a logger.

        Args:
            username (str, optional): The Twitter username.
            password (str, optional): The Twitter password.
            logger (Logger, optional): Logger for logging messages.
        """
        self.username = username
        self.password = password
        self.logger = logger
        self.cookies_loaded = False
        self.cookies_file_path = cookies_file_path

    def login_save_cookies(
        self,
        path: str,
        headless: bool = False,
    ) -> None:
        """
        Logs into Twitter and saves cookies for later use.

        Args:
            path (str): Path to the file where cookies should be saved.
            headless (bool): Whether to run the scraper in headless mode.
        """
        driver = utils.get_driver(headless=headless)

        try:
            self.navigate_to_login_page(driver)
            self.enter_credentials(driver)
            self.save_cookies(driver, path)

        finally:
            driver.quit()

    def navigate_to_login_page(
        self,
        driver: webdriver.Firefox,
    ) -> None:
        """
        Navigates to the Twitter login page.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
        """
        try:
            driver.get("https://www.x.com")
            wait = WebDriverWait(driver, 10)
            self.random_sleep(
                interval_1=(6, 11), probability_interval_1=1, probability_interval_2=0.0
            )
            login_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[@href='/login']"))
            )

            try:
                driver.execute_script("arguments[0].click();", login_button)
            except JavascriptException as e:
                self.logger.handle_exception(
                    JavascriptException,
                    f"Cannot click login button: {e}",
                )

        except NoSuchElementException as e:
            self.logger.handle_exception(
                NoSuchElementException,
                f"Login button not found: {e}",
            )

    def enter_credentials(
        self,
        driver: webdriver.Firefox,
    ) -> None:
        """
        Enters login credentials on the Twitter login page.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
        """
        wait = WebDriverWait(driver, 10)

        try:
            # Enter username
            wait.until(EC.presence_of_element_located((By.NAME, "text")))
            username_field = driver.find_element(By.NAME, "text")
            self.random_sleep(
                interval_1=(2, 4), probability_interval_1=1, probability_interval_2=0.0
            )
            username_field.send_keys(self.username)
            username_field.send_keys(Keys.RETURN)
        except NoSuchElementException as e:
            self.logger.handle_exception(
                NoSuchElementException,
                f"Username field not found: {e}",
            )

        try:
            # Enter password
            wait.until(EC.presence_of_element_located((By.NAME, "password")))
            password_field = driver.find_element(By.NAME, "password")
            self.random_sleep(
                interval_1=(2, 4), probability_interval_1=1, probability_interval_2=0.0
            )
            password_field.send_keys(self.password)
            password_field.send_keys(Keys.RETURN)
        except NoSuchElementException as e:
            self.logger.handle_exception(
                NoSuchElementException,
                f"Password field not found: {e}",
            )

        self.random_sleep(
            interval_1=(4, 6), probability_interval_1=1, probability_interval_2=0.0
        )

    def save_cookies(
        self,
        driver: webdriver.Firefox,
        path: str,
    ) -> None:
        """
        Saves cookies after login for future sessions.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
            path (str): Path to the file where cookies should be saved.
        """
        cookies = driver.get_cookies()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(cookies, file)
        self.logger.info("Cookies saved.")

    def random_sleep(
        self,
        interval_1=(3, 8),
        interval_2=(13, 20),
        interval_3=(25, 35),
        probability_interval_1=0.89,
        probability_interval_2=0.1,
    ):
        """
        Pauses execution for a random time based on specified intervals and probabilities.

        Args:
            interval_1 (tuple): The first interval (min, max), default is (3, 8).
            interval_2 (tuple): The second interval (min, max), default is (15, 20).
            interval_3 (tuple): The third interval (min, max), default is (30, 35).
            probability_interval_1 (float): Probability of choosing interval_1 (between 0 and 1).
            probability_interval_2 (float): Probability of choosing interval_2 (between 0 and 1).

        Probability for interval_3 is implicitly
            1 - (probability_interval_1 + probability_interval_2).
        """
        rand_val = random.random()

        if rand_val < probability_interval_1:
            sleep_time = random.uniform(*interval_1)
        elif rand_val < probability_interval_1 + probability_interval_2:
            sleep_time = random.uniform(*interval_2)
        else:
            sleep_time = random.uniform(*interval_3)

        time.sleep(sleep_time)

    def scrape_with_cookies(
        self,
        tweet_count: int = 1,
        search_query: str = "Bitcoin",
        headless: bool = True,
    ) -> dict[str, list[str]]:
        """
        Scrapes tweets using saved cookies and returns the result in a DataFrame.

        Args:
            tweet_count (int): Number of tweets to scrape.
            search_query (str): The search query for tweets.
            headless (bool): Whether to run the scraper in headless mode.

        Returns:
            pd.DataFrame: DataFrame containing tweet data.
        """
        driver = utils.get_driver(headless=headless)

        try:
            self.load_cookies(driver)
            self.navigate_to_explore(driver)
            self.perform_search(driver, search_query)
            return self.scrape_tweets(driver, tweet_count)

        finally:
            driver.quit()

    def load_cookies(
        self,
        driver: webdriver.Firefox,
    ) -> None:
        """
        Loads cookies from a file and refreshes the page. If the file is not found or an error
        occurs, cookies are loaded from an environment variable.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
        """
        driver.get("https://www.x.com")

        # Check if the cookie file exists
        if os.path.exists(self.cookies_file_path):
            with open(self.cookies_file_path, "r", encoding="utf-8") as file:
                try:
                    cookies = json.load(file)
                    self.logger.info(
                        f"Cookies successfully loaded from {self.cookies_file_path}"
                    )
                except json.JSONDecodeError:
                    self.logger.handle_exception(
                        ValueError, f"Invalid JSON format in {self.cookies_file_path}."
                    )
                    return
        elif cookie_file_content := os.getenv("COOKIE_FILE_CONTENT_X"):
            try:
                cookies = json.loads(cookie_file_content)
                self.logger.info(
                    "Cookies successfully loaded from environment variable."
                )
            except json.JSONDecodeError as e:
                self.logger.handle_exception(
                    ValueError,
                    f"Invalid JSON format in COOKIE_FILE_CONTENT_X environment variable. {e}",
                )
                return
        else:
            self.logger.handle_exception(
                FileNotFoundError,
                f"Neither cookie file '{self.cookies_file_path}' nor 'COOKIE_FILE_CONTENT_X' environment variable could provide cookies.",
            )
            return

        for cookie in cookies:
            driver.add_cookie(cookie)

        driver.refresh()
        self.cookies_loaded = True

        self.random_sleep(
            interval_1=(6, 11), probability_interval_1=1, probability_interval_2=0.0
        )

    def _check_authentication(self) -> None:
        """
        Checks if authentication details are provided. Raises an AuthenticationError if neither
        cookies nor username/password are available for scraping.
        """
        if not (self.cookies_loaded or (self.username and self.password)):
            raise AuthenticationError()

    def navigate_to_explore(
        self,
        driver: webdriver.Firefox,
    ) -> None:
        """
        Navigates to the Twitter explore page and handles any popups.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
        """
        driver.get("https://www.x.com/explore")
        self.random_sleep(
            interval_1=(6, 11), probability_interval_1=1, probability_interval_2=0.0
        )
        self.logger.info(
            f"Page title after loading cookies and navigating to Explore: {driver.title}"
        )

        try:
            close_button_wait = WebDriverWait(driver, 10)
            close_button = close_button_wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(@aria-label, 'Close')]")
                )
            )
            driver.execute_script("arguments[0].click();", close_button)
            self.logger.info("Close button clicked successfully.")

        except (NoSuchElementException, TimeoutException) as e:
            self.logger.warning(
                f"Close button not found or not clickable within the timeout period. {e}"
            )

        except WebDriverException as e:
            self.logger.warning(
                f"WebDriverException encountered when trying to click the close button. {e}"
            )

    def perform_search(
        self,
        driver: webdriver.Firefox,
        search_query: str,
    ) -> None:
        """
        Performs a search for the given query.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
            search_query (str): The search query to look for.
        """
        try:
            search_bar = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//input[@aria-label='Search query']")
                )
            )
            search_bar.clear()
            search_bar.send_keys(search_query)
            search_bar.send_keys(Keys.RETURN)
            self.logger.info(f"Searched for: {search_query}")
            self.random_sleep(
                interval_1=(6, 11),
                probability_interval_1=0.95,
                probability_interval_2=0.05,
            )
        except NoSuchElementException:
            self.logger.error("Search bar not found on the page.")
        except TimeoutException:
            self.logger.error(
                "Search bar did not become clickable within the timeout period."
            )
        except ElementNotInteractableException:
            self.logger.error("Search bar was found but could not be interacted with.")

    def scrape_tweets(
        self,
        driver: webdriver.Firefox,
        tweet_count: int,
    ) -> dict[str, list[str]]:
        """
        Scrapes the specified number of tweets and returns a dictionary of tweet data.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.
            tweet_count (int): Number of tweets to scrape.

        Returns:
            dict: Dictionary containing tweet data with keys "Username", "Tweet", "Timestamp",
                  "Likes", and "Impressions". Each key maps to a list of scraped values for that
                  attribute.
        """
        tweets_scraped = 0
        tweet_data = []

        self._check_authentication()

        while tweets_scraped < tweet_count:
            tweets = self.get_tweets(driver)
            for tweet in tweets:
                try:
                    username, content, timestamp, likes, impressions = (
                        self.extract_tweet_details(tweet)
                    )
                    tweet_data.append(
                        [username, content, timestamp, likes, impressions]
                    )
                    tweets_scraped += 1
                    self.logger.info(
                        f"Scraped tweet {tweets_scraped}/{tweet_count}: {username} - {content[:50]}"
                    )

                except (NoSuchElementException, TimeoutException) as e:
                    self.logger.warning(
                        f"Could not find an element in tweet details: {e}"
                    )
                except AttributeError as e:
                    self.logger.warning(
                        f"Attribute missing in tweet details extraction: {e}"
                    )
            self.random_sleep(
                interval_1=(3, 7),
                probability_interval_1=0.85,
                probability_interval_2=0.1,
            )

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.random_sleep(
                interval_1=(2, 6), probability_interval_1=1, probability_interval_2=0.0
            )

        self.logger.info("Tweets scraped into dictionary.")
        return {
            "Username": [row[0] for row in tweet_data],
            "Tweet": [row[1] for row in tweet_data],
            "Timestamp": [row[2] for row in tweet_data],
            "Likes": [row[3] for row in tweet_data],
            "Impressions": [row[4] for row in tweet_data],
        }

    def get_tweets(
        self,
        driver: webdriver.Firefox,
    ) -> List[WebElement]:
        """
        Retrieves the list of tweets from the current page.

        Args:
            driver (webdriver.Firefox): Selenium WebDriver instance.

        Returns:
            List[WebElement]: List of tweet elements.
        """
        try:
            return WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, "//article[@data-testid='tweet']")
                )
            )
        except TimeoutException as e:
            self.logger.warning(f"Timed out while waiting for tweet elements: {e}")
        except NoSuchElementException as e:
            self.logger.warning(f"No tweet elements found on the page: {e}")

        return []

    def extract_tweet_details(
        self,
        tweet: WebElement,
    ) -> Tuple[str, str, str, str, str]:
        """
        Extracts details from a tweet such as username, content, timestamp, likes, and impressions.

        Args:
            tweet (WebElement): The tweet element.

        Returns:
            Tuple[str, str, str, str, str]: Extracted tweet details
                (username, content, timestamp, likes, impressions).
        """
        # Setze Standardwerte für alle Details
        username = "Unknown"
        content = ""
        timestamp = "N/A"
        likes = "0"
        impressions = "N/A"

        try:
            username = tweet.find_element(
                By.XPATH, ".//span[contains(text(), '@')]"
            ).text
        except NoSuchElementException:
            self.logger.info("Username element not found; using default 'Unknown'.")

        try:
            content = tweet.find_element(
                By.XPATH, ".//div[@data-testid='tweetText']"
            ).text
        except NoSuchElementException:
            self.logger.info("Content element not found; using empty default.")

        try:
            timestamp_element = tweet.find_element(By.XPATH, ".//time")
            if timestamp_ := timestamp_element.get_attribute("datetime"):
                timestamp = timestamp_
        except NoSuchElementException:
            self.logger.info("Timestamp element not found; using default 'N/A'.")

        # Extract likes
        try:
            likes = (
                tweet.find_element(
                    By.XPATH, ".//div[@data-testid='like']//span"
                ).get_attribute("innerHTML")
                or "0"
            )
        except NoSuchElementException:
            self.logger.info("Likes element not found; defaulting to 0.")

        # Extract impressions
        try:
            impressions = tweet.find_element(
                By.XPATH, ".//div[@data-testid='view']"
            ).text
            if impressions == "":
                impressions = (
                    tweet.find_element(
                        By.XPATH, ".//div[@data-testid='view']//span"
                    ).get_attribute("innerHTML")
                    or "N/A"
                )
        except NoSuchElementException:
            self.logger.info("Impressions element not found; defaulting to N/A.")

        return username, content, timestamp, likes, impressions
