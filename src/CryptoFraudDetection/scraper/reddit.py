"""
File: reddit.py

Description:
- This file is used to scrape data from the website Reddit.

Authors:
- Aaron Brülisauer me@nodon.io
"""

import logging
import pandas as pd
import pickle
import random
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class RedditScraper:
    def __init__(self, logger: logging.Logger, base_url:str, subreddit:str, search_query:str, limit:int=5, wait_range:tuple[int,int] = (2,5), cookies_file:str="cookies/reddit-cookies.pkl"):
        self._logger:logging.Logger = logger
        self._base_url:str = base_url
        self._subreddit:str = subreddit
        self._search_query:str = search_query
        self._limit:int = limit
        self._wait_range:tuple[int,int] = wait_range
        self._cookies_file:str = cookies_file
        self.driver: webdriver.Firefox = webdriver.Firefox()
        self.post_data: list[dict] = []

    def _load_cookies(self) -> None:
        """Load cookies to avoid login on each run."""
        self.driver.get(self._base_url)  # Open a blank page to set cookies
        try:
            with open(self._cookies_file, "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)
            self._logger.info("Cookies loaded successfully.")
        except FileNotFoundError:
            self._logger.info("No cookies file found. Proceeding without loading cookies.")
            
    def _wait(self):
        """Wait for a random time between the specified range."""
        time.sleep(random.uniform(*self._wait_range))

    
    def _wait_for_element(self, locator, timeout=10):
        """Wait for an element to be present."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return element
        except TimeoutException:
            self._logger.error(f"Timeout waiting for element with locator: {locator}")
            return None


    def search_posts(self) -> None:
        """Search for posts in a specific subreddit."""
        url = f"{self._base_url}/{self._subreddit}/search?q={self._search_query}&restrict_sr=on&sort=new&t=all&limit={self._limit}&after=id___"
        self.driver.get(url)   # TODO: error handling (differentiate between exceptions and maybe try again? sometimes dying might be ok)
        
        # Wait for search results to load
        search_results_loaded = self._wait_for_element(
            (By.XPATH, '//div[contains(@class, "search-result-link")]'), timeout=10
        )
        if not search_results_loaded:
            self._logger.error("Search results did not load.")
            return
        
        # Extract list of posts
        search_results = self.driver.find_elements(
            By.XPATH, '//div[contains(@class, "search-result-link")]'
        )
        
        # Process each post
        # TODO: explain more and why?
        for result in search_results:
            post = self._extract_post_data(result)
            self.post_data.append(post)

    def _extract_post_data(self, result):
        # TODO result bad name
        """Extract individual post data."""
        try:
            # TODO: as loop
            post = {
                "id": result.get_attribute("data-fullname"),
                # TODO: looking for same object two times
                "title": result.find_element(By.XPATH, './/a[contains(@class, "search-title")]').text,
                "url": result.find_element(By.XPATH, './/a[contains(@class, "search-title")]').get_attribute("href"),
                "score": result.find_element(By.XPATH, './/span[contains(@class, "search-score")]').text,
                "comment": result.find_element(By.XPATH, './/a[contains(@class, "search-comments")]').text,
                "date": result.find_element(By.XPATH, ".//time[@datetime]").get_attribute("datetime"),
                "author": result.find_element(By.XPATH, './/a[contains(@class, "author")]').text,
                "author_url": result.find_element(By.XPATH, './/a[contains(@class, "author")]').get_attribute("href"),
            }
            return post
        except Exception as e:
            self._logger.error(f"Error extracting post details: {e}")
            return {}


    def _extract_comments(self, comment_div) -> dict:
        return {}
    
    def _extract_post_text(self):
        """Wait for, locate and extract the post text"""
        post_text_locator = (
            By.XPATH,
            '//div[contains(@class, "entry")]//div[contains(@class, "md")]',
        )
        post_text_div = self._wait_for_element(
            post_text_locator, timeout=10
        )
        return post_text_div.text


    def scrape_post_content(self, url) -> dict:
        """Load content from an individual post URL."""
        self._wait()
        
        try:
            self.driver.get(url)
        except Exception as e:
            self._logger.error(f"Error loading post URL {url}: {e}")
        
        return {
            "id": '',
            "author": '',
            "text": self._extract_post_text(),
            'children': [],
            'date': '',
            'score': 0,
        }

    def _save_cookies(self):
        """Save cookies for future sessions."""
        with open(self._cookies_file, "wb") as file:
            pickle.dump(self.driver.get_cookies(), file)
            self._logger.info("Cookies saved successfully.")

    def quit(self):
        """Quit the WebDriver session."""
        self.driver.quit()

    def scrape(self):
        """Scrape for posts and there content."""
        try:
            self._load_cookies()
            self.search_posts()
            for post in self.post_data:
                self.scrape_post_content(post)
            self._save_cookies()
        finally:
            self.quit()
        return self.post_data

    def to_dataframe(self):
        """Convert scraped data to a pandas DataFrame."""
        return pd.DataFrame(self.post_data)
