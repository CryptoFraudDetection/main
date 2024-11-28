"""
File: reddit.py

Description:
- This file is used to scrape data from the website Reddit.

Authors:
- Aaron Brülisauer me@nodon.io
- Florian Baumgartner florian.baumgartner1@students.fhnw.ch
"""

import time
import random

import pandas as pd
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from CryptoFraudDetection.scraper import utils
from CryptoFraudDetection.utils.logger import Logger
from CryptoFraudDetection.utils import exceptions


class RedditScraper:
    def __init__(
        self,
        logger: Logger,
        base_url: str = "https://old.reddit.com",
        wait_range: tuple[int, int] = (1, 3),
        blocked_wait_times: list[int] = [5, 10, 15],
        headless: bool = True,
        max_search_limit: int = 100,
    ):
        self._logger: Logger = logger
        self._base_url: str = base_url
        self._wait_range: tuple[int, int] = wait_range
        self._blocked_wait_times: list[int] = blocked_wait_times
        self.headless: bool = headless
        self._max_search_limit: int = max_search_limit

        self.driver = None
        self.post_data: list[dict] = []

    def _get_next_proxy(self, link="https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=csv&timeout=2000"):
        proxy_list = pd.read_csv(link)
        proxy_list = proxy_list.sample(1)
        return proxy_list.iloc[0]
    
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
            self._logger.debug(f"Timeout waiting for element with locator: {locator}")
            return None
        except Exception as e:
            raise ValueError(f"Unexpected error while waiting for element: {e}")

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
            elements = {}
            for name, xpath in xpaths.items():
                try:
                    self._wait_for_element((By.XPATH, xpath))
                    elements[name] = post_div.find_element(By.XPATH, xpath)
                except NoSuchElementException:
                    pass
            post_metadata = {
                "id": post_div.get_attribute("data-fullname"),
                "title": elements["title"].text,
                "url": elements["title"].get_attribute("href"),
                "num_comment": elements["comments"].text.split()[0],
                "date": elements["date"].get_attribute("datetime"),
                "author": elements["author"].text,
                "author_url": elements["author"].get_attribute("href"),
            }
            if "score" in elements:
                post_metadata["score"] = (elements["score"].text.split()[0],)
            return post_metadata
        except Exception as e:
            self._logger.error(f"Error extracting post details: {e}")
            return {}

    def _extract_comments_rec(self, comments_divs) -> dict:
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
                        children = self._extract_comments_rec(childern_divs)
                        if children:
                            comment["children"] = children
                    comments.append(comment)
            except NoSuchElementException as e:
                pass

        return comments

    def _extract_comments(self):
        comments_xpath = (
            '//div[contains(@class, "commentarea")]'
            '/div[contains(@class, "nestedlisting")]'
            '/div[contains(@class, "comment")]'
        )

        # Return if there are no comments
        no_comment_xpath = (
            comments_xpath[0] + '/div[contains(@class, "panestack-title")]'
        )
        try:
            no_comment_div = self.driver.find_element(By.XPATH, no_comment_xpath)
            if no_comment_div.text == "no comments (yet)":
                return []
        except:
            pass

        # Extract comments
        self._wait_for_element((By.XPATH, comments_xpath))
        comments_divs = self.driver.find_elements(By.XPATH, comments_xpath)
        return self._extract_comments_rec(comments_divs)

    def _check_if_blocked(self) -> bool:
        post = self._wait_for_element((By.XPATH, "/html/body"))
        if not post:
            raise ValueError("Can't evaluate if blocked because post is None")
        try:
            body_pre = post.find_element(By.XPATH, "/html/body/pre")
            if body_pre.text == "":
                raise exceptions.DetectedBotException("Detected as a bot")
            else:
                return  # Probably not blocked
        except NoSuchElementException:
            return  # Probably not blocked

    def _load_page(self, url: str, xpath: str):
        """Load a page and check if we are blocked."""
        self._wait()
        for i in range(len(self._blocked_wait_times)):
            try:
                self._logger.debug(f"Loading URL: {url}")
                self.driver.get(url)
            except Exception as e:
                # TODO: retry?
                self._logger.error(f"Error loading URL {url}: {e}")

            try:
                self._check_if_blocked()
                break  # Success
            except exceptions.DetectedBotException:
                wait_time = self._blocked_wait_times[i]
                self._logger.warning(
                    f"Waiting for {wait_time} because we got blocked on post {url}"
                )
                time.sleep(wait_time)

        # Locate the element
        element = self._wait_for_element((By.XPATH, xpath))

        if not element:
            self._logger.warning(f"Element {xpath} not found in post URL: {url}")
            return None

        return element

    def scrape_post_list(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 100,
        after_post_id: str = None,
    ) -> list:
        """Search for posts in a specific subreddit, max 100"""
        if limit > self._max_search_limit:
            raise ValueError(f"Limit must be less than {self._max_search_limit}")
        url = (
            self._base_url
            + "/"
            + subreddit
            + "/search?q="
            + search_query
            + "&restrict_sr=on"
            + "&sort=new"
            + "&t=all"
            + f"&limit={limit}"
        )
        if after_post_id:
            url += f"&after={after_post_id}"
        # TODO: error handling (differentiate between exceptions and maybe try again? sometimes dying might be ok)
        search_result_listing = self._load_page(
            url, '//div[contains(@class, "search-result-listing")]'
        )
        if search_result_listing:
            # Check if there are no search results
            footer = search_result_listing.find_element(By.XPATH, ".//footer")
            if "there doesn't seem to be anything here" in footer.text:
                return []

            # Wait for search results to load
            self._wait_for_element(
                (By.XPATH, '//div[contains(@class, "search-result-link")]')
            )
            search_results: list | None = self.driver.find_elements(
                By.XPATH, '//div[contains(@class, "search-result-link")]'
            )
            if search_results:
                return search_results

        raise ValueError(
            f"Error while getting search results\nsubreddit: {subreddit}\nsearch_query: {search_query}\nlimit: {limit}\nafter postid: {after_post_id}\nurl: {url}"
        )

    def scrape_multipage_post_list(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 500,
        start_date: str = None,
        after_post_id: str = None,
        retry: int = 3,
    ) -> None:
        """Search for posts in a specific subreddit on multiple sites to get more then 100."""
        for num_processed_posts in range(0, limit, self._max_search_limit):
            search_limit = min(self._max_search_limit, limit - num_processed_posts)

            for _ in range(retry):
                # Load one page of search results
                search_results = self.scrape_post_list(
                    subreddit, search_query, search_limit, after_post_id
                )
                if search_results:
                    break  # Success

            # Break if there are no more search results
            if isinstance(search_results, list) and len(search_results) == 0:
                break
            
            # Catch if no search_results (unexpected)
            if not search_results:
                raise ValueError(
                    f"Error while getting search results\nsubreddit: {subreddit}\nsearch_query: {search_query}\nlimit: {limit}\nafter postid: {after_post_id}"
                )

            # Extract post metadata from the html elements
            for result in search_results:
                post = self._extract_post_metadata(result)
                post["subreddit"] = subreddit
                post["search_query"] = search_query
                self.post_data.append(post)

            # Get the ID of the oldest post to use in the next search
            after_post_id = search_results[-1].get_attribute("data-fullname")

            # Break if oldest post is older than start_date
            if start_date:
                # Convert to date
                oldest_post_date = pd.to_datetime(
                    self.post_data[-1]["date"]
                ).tz_localize(None)
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date).tz_localize(None)

                if oldest_post_date < start_date:
                    break

        return self.post_data

    def scrape_post_content(self, post: dict) -> dict:
        """Load content from an individual post URL."""
        # Locate the post entry
        post_entry_xpath = '//div[contains(@class, "entry")]'
        post_entry = self._load_page(post["url"], post_entry_xpath)

        if post_entry is None:
            return None  # Might be retried if mutiple runs are done

        # Extract post text
        post_text_xpath = './/div[contains(@class, "md")]'
        try:
            post_text_div = post_entry.find_element(By.XPATH, post_text_xpath)
            post["text"] = post_text_div.text
        except NoSuchElementException:
            # It is possible that the post has no text content
            post["text"] = ""
        except Exception as e:
            raise ValueError(f"Unexpected error while extracting post text: {e}")

        # Extract comments
        post["children"] = self._extract_comments()

        return post

    def get_posts_without_content(self):
        """Get posts without text content."""
        return [post for post in self.post_data if post.get("text") is None]

    def scrape_all_missing_post_contents(self, retry: int = 3):
        """Scrape the content of posts (without content) in self.post_data."""
        for _ in range(retry):
            for post in self.get_posts_without_content():
                self.scrape_post_content(post)
            num_posts_without_content = len(self.get_posts_without_content())
            if num_posts_without_content == 0:
                return  # Success

        raise f"{num_posts_without_content} are still missing after {retry} tries."

    def start_driver(self):
        """Start the WebDriver session if not already started."""
        while True:
            try:
                proxy = self._get_next_proxy()
                self.driver = utils.get_driver(
                    self.headless, proxy.protocol, f"{proxy.ip}:{proxy.port}"
                )
                self._logger.debug("WebDriver session started.")
                return  # Success
            except exceptions.ProxyNotWorking:
                self._logger.info("Proxy not working, trying next one.")

    def quit(self):
        """Quit the WebDriver session if it's running."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            self._logger.debug("WebDriver session quit.")

    def to_dataframe(self):
        """Convert scraped data to a pandas DataFrame."""
        return pd.DataFrame(self.post_data)

    def scrape(self, *args, **kwargs):
        """
        scrape Reddit
        """
        self.start_driver()
        self.scrape_multipage_post_list(*args, **kwargs)
        self.scrape_all_missing_post_contents()
        self.quit()
        return self.to_dataframe()
