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
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


class RedditScraper:
    def __init__(
        self,
        logger: logging.Logger,
        base_url: str = "https://old.reddit.com",
        wait_range: tuple[int, int] = (2, 5),
        cookies_file: str = "cookies/reddit-cookies.pkl",
        headless: bool = True,
    ):
        self._logger: logging.Logger = logger
        self._base_url: str = base_url
        self._wait_range: tuple[int, int] = wait_range
        self._cookies_file: str = cookies_file
        self.post_data: list[dict] = []
        self.driver = None
        self.headless = headless

    def start_driver(self):
        """Start the WebDriver session if not already started."""
        # TODO: headless does not work, why?
        if self.driver is None:
            options = webdriver.FirefoxOptions()
            options.headless = self.headless
            self.driver = webdriver.Firefox(options=options)
            self._logger.info("WebDriver session started.")

    def quit(self):
        """Quit the WebDriver session if it's running."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            self._logger.info("WebDriver session quit.")


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
            self._logger.info(
                "No cookies file found. Proceeding without loading cookies."
            )

    def _save_cookies(self):
        """Save cookies for future sessions."""
        with open(self._cookies_file, "wb") as file:
            pickle.dump(self.driver.get_cookies(), file)
            self._logger.info("Cookies saved successfully.")


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

    def _extract_post_metadata(self, post_div):
        """Extract individual post metadata from the list of posts."""
        try:
            xpaths = {
                "title": './/a[contains(@class, "search-title")]',
                "score": './/span[contains(@class, "search-score")]',
                "comments": './/a[contains(@class, "search-comments")]',
                "date": ".//time[@datetime]",
                "author": './/a[contains(@class, "author")]',
            }
            elements = {
                name: post_div.find_element(By.XPATH, xpath)
                for name, xpath in xpaths.items()
            }
            return {
                "id": post_div.get_attribute("data-fullname"),
                "title": elements["title"].text,
                "url": elements["title"].get_attribute("href"),
                "score": elements["score"].text,
                "comment": elements["comments"].text,
                "date": elements["date"].get_attribute("datetime"),
                "author": elements["author"].text,
                "author_url": elements["author"].get_attribute("href"),
            }
        except Exception as e:
            self._logger.error(f"Error extracting post details: {e}")
            return {}

    def get_post_list(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 10000,
        start_date: str = None,  # yyymmdd
        after_post_id: str = None,
        max_num_posts_per_search=100,
    ) -> None:
        """Search for posts in a specific subreddit."""
        for num_processed_posts in range(0, limit, max_num_posts_per_search):
            search_limit = min(max_num_posts_per_search, limit - num_processed_posts)
            url = (
                self._base_url
                + "/"
                + subreddit
                + "/search?q="
                + search_query
                + "&restrict_sr=on"
                + "&sort=new"
                + "&t=all"
                + f"&limit={search_limit}"
            )
            # TODO: start from date x
            # TODO: end at date x
            if after_post_id:
                url += f"&after={after_post_id}"
            # TODO: error handling (differentiate between exceptions and maybe try again? sometimes dying might be ok)
            self._wait()
            self._logger.info(f"Loading URL: {url}")
            self.driver.get(url)

            # Wait for search results to load
            self._wait_for_element(
                (By.XPATH, '//div[contains(@class, "search-result-link")]')
            )

            # Extract list of posts
            search_results = self.driver.find_elements(
                By.XPATH, '//div[contains(@class, "search-result-link")]'
            )

            # Extract post metadata from the html elements
            for result in search_results:
                post = self._extract_post_metadata(result)
                post['subreddit'] = subreddit
                post['search_query'] = search_query
                self.post_data.append(post)
        
            # Get the ID of the oldest post to use in the next search
            after_post_id = search_results[-1].get_attribute("data-fullname")
            
            # Break if oldest post is older than start_date
            if start_date:
                oldest_post_date = pd.to_datetime(self.post_data[-1]['date']).tz_localize(None)
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date).tz_localize(None)
                if oldest_post_date < start_date:
                    break

        return self.post_data



    def _extract_comments(self, comments_divs) -> dict:
        childs_xpath = (
            './div[contains(@class, "child")]'
            "/div"
            '/div[contains(@class, "comment")]'
        )
        text_xpath = (
            './div[contains(@class, "entry")]'
            '/form[contains(@class, "usertext")]'
            '/div[contains(@class, "usertext-body")]'
            '/div[contains(@class, "md")]'
        )
        author_xpath = (
            './div[contains(@class, "entry")]'
            '/p[contains(@class, "tagline")]'
            '/a[contains(@class, "author")]'
        )
        comments = []
        for comment_div in comments_divs:
            try:
                text = comment_div.find_element(By.XPATH, text_xpath).text.strip()
                if text:
                    author = comment_div.find_element(By.XPATH, author_xpath).text
                    comment = {"text": text, "author": author}
                    childern_divs = comment_div.find_elements(By.XPATH, childs_xpath)
                    if len(childern_divs) > 0:
                        children = self._extract_comments(childern_divs)
                        if children:
                            comment["children"] = children
                    comments.append(comment)
            except NoSuchElementException as e:
                pass

        return comments

    def _extract_all_comments(self):
        comments_xpath = (
            '//div[contains(@class, "commentarea")]'
            '/div[contains(@class, "nestedlisting")]'
            '/div[contains(@class, "comment")]'
        )
        self._wait_for_element((By.XPATH, comments_xpath))
        comments_divs = self.driver.find_elements(By.XPATH, comments_xpath)
        return self._extract_comments(comments_divs)

    def scrape_post_content(self, post: dict) -> dict:
        """Load content from an individual post URL."""
        self._wait()
        
        try:
            self.driver.get(post["url"])
        except Exception as e:
            self._logger.error(f"Error loading post URL {post['url']}: {e}")

        post_text_xpath = (
            '//div[contains(@class, "entry")]//div[contains(@class, "md")]'
        )
        post_text_div = self._wait_for_element((By.XPATH, post_text_xpath))

        post["text"] = post_text_div.text
        post["children"] = self._extract_all_comments()
        
        return post
    
    def get_posts_without_content(self):
        """Get posts without text content."""
        return [post for post in self.post_data if post.get("text") is None]
    
    def scrape_all_post_contents(self):
        """Scrape the content of posts in self.post_data."""
        for post in self.get_posts_without_content():
            self.scrape_post_content(post)
        return len(self.get_posts_without_content())

    def to_dataframe(self):
        """Convert scraped data to a pandas DataFrame."""
        return pd.DataFrame(self.post_data)


def scrape_reddit(logger:logging.Logger,subreddit:str,query:str,limit:int=3, max_num_posts_per_search:int=2):
    """
    scrape Reddit
    """
    scraper = RedditScraper(logger)
    scraper.start_driver()
    scraper.get_post_list(subreddit, query, limit=limit, max_num_posts_per_search=max_num_posts_per_search)
    scraper.scrape_all_post_contents()
    scraper.quit()
    df = scraper.to_dataframe()
    return df