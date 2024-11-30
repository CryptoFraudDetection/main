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
    """
    A class to scrape Reddit posts using Selenium.

    Attributes:
            driver: Selenium WebDriver instance.
            post_data (list[dict]): List to store scraped post information.
    """
    def __init__(
        self,
        logger: Logger,
        base_url: str = "https://old.reddit.com",
        wait_range: tuple[int, int] = (0.25, 0.75),
        headless: bool = True,
        max_search_limit: int = 100,
        page_retry: int = 1_000,
        scrape_post_list_retry: int = 10,
        scrape_missing_posts_retry: int = 10,
    ):
        """
        Initialize the RedditScraper with the given configuration parameters.

        Args:
            logger (Logger): Logger instance for logging messages and errors.
            base_url (str, optional): The base URL of Reddit to scrape from.
                Defaults to "https://old.reddit.com".
            wait_range (tuple[int, int], optional): Tuple specifying the minimum
                and maximum wait times between actions to mimic human behavior.
                Defaults to (0.25, 0.75).
            headless (bool, optional): If True, runs the browser in headless mode.
                Defaults to True.
            max_search_limit (int, optional): Maximum number of posts to retrieve.
                Defaults to 100.
            page_retry (int, optional): Number of times to retry loading a page
                upon failure. Defaults to 1,000.
            scrape_post_list_retry (int, optional): Number of retries for scraping
                the post list. Defaults to 10.
            scrape_missing_posts_retry (int, optional): Number of retries for
                scraping missing posts. Defaults to 10.
        """
        self._logger: Logger = logger
        self._base_url: str = base_url
        self._wait_range: tuple[int, int] = wait_range
        self.headless: bool = headless
        self._max_search_limit: int = max_search_limit
        self.page_retry: int = page_retry
        self.scrape_post_list_retry: int = scrape_post_list_retry
        self.scrape_missing_posts_retry: int = scrape_missing_posts_retry

        self.driver = None # Will hold the Selenium WebDriver instance
        self.post_data: list[dict] = []

    def _get_next_proxy(self, link="https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=csv&timeout=2000"):
        """
        Fetch a new proxy from the provided proxy list URL.

        Args:
            link (str, optional): The URL to fetch the proxy list from.
                Defaults to a ProxyScrape API endpoint.

        Returns:
            str: A randomly selected proxy from the list.
        """
        # Read the proxy list from the given URL and take one random proxy
        proxy_list = pd.read_csv(link)
        proxy_list = proxy_list.sample(1)
        return proxy_list.iloc[0]
    
    def _wait(self):
        """
        Pause execution for a random duration within the specified wait range.

        This method introduces a randomized delay between actions to mimic human behavior
        and avoid detection by anti-bot measures.
        """

        time.sleep(random.uniform(*self._wait_range))

    def _wait_for_element(self, locator, timeout=10):
        """
        Wait until the specified element is present.

        Args:
            locator (tuple): A tuple of (By, value) used to locate the element.
            timeout (int, optional): Maximum time to wait for the element, in seconds.
                Defaults to 10.

        Returns:
            WebElement or None: The WebElement if found within the timeout period,
                otherwise None.

        Raises:
            ValueError: If an unexpected error occurs while waiting for the element.
    """
        try:
            # Log a debug message if the element was not found in time
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return element
        except TimeoutException:
            self._logger.debug(f"Timeout waiting for element with locator: {locator}")
            return None
        except Exception as e:
            # Raise a ValueError for any other exceptions
            raise ValueError(f"Unexpected error while waiting for element: {e}")

    def _extract_post_metadata(self, post_div):
        """
        Extract metadata from an individual Reddit post element.

        Args:
            post_div (WebElement): The WebElement representing a single Reddit post.

        Returns:
            dict: A dictionary containing the extracted post metadata. Returns an empty
                dictionary if extraction fails.
    """
        try:
            # Define XPath expressions to locate the required elements within the post
            xpaths = {
                "title": './/a[contains(@class, "search-title")]',
                "score": './/span[contains(@class, "search-score")]',
                "comments": './/a[contains(@class, "search-comments")]',
                "date": ".//time[@datetime]",
                "author": './/a[contains(@class, "author")]',
            }
            elements = {}
            # Iterate over each XPath to find and store the corresponding elements
            for name, xpath in xpaths.items():
                try:
                    # Wait for the element
                    self._wait_for_element((By.XPATH, xpath))
                    # Find the element within the post_div
                    elements[name] = post_div.find_element(By.XPATH, xpath)

                except NoSuchElementException:
                    pass

            #  Construct the metadata dictionary
            post_metadata = {
                "id": post_div.get_attribute("data-fullname"),
                "title": elements["title"].text,
                "url": elements["title"].get_attribute("href"),
                "num_comment": elements["comments"].text.split()[0],
                "date": elements["date"].get_attribute("datetime"),
                "author": elements["author"].text,
                "author_url": elements["author"].get_attribute("href"),
            }
            # Include the post score if available
            if "score" in elements:
                post_metadata["score"] = (elements["score"].text.split()[0],)
            return post_metadata

        except Exception as e:
            # Log any exceptions that occur
            self._logger.error(f"Error extracting post details: {e}")
            return {}

    def _extract_comments_rec(self, comments_divs) -> dict:
        """
        Recursively extract comments and their replies from the given comment div elements.

        Args:
            comments_divs (List[WebElement]): A list of WebElements representing comment divs.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing comment data, including text, author, and any nested replies.

        Note:
            This method navigates the nested structure of Reddit comments by recursively processing child comment divs.
        """

        # Define XPath expressions to locate elements within a comment
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
        # Iterate over each comment div
        for comment_div in comments_divs:
            try:
                text = comment_div.find_element(By.XPATH, text_xpath).text.strip()
                if text:
                    # Extract the comment text
                    author = comment_div.find_element(By.XPATH, author_xpath).text
                    # Create a dictionary to store comment data
                    comment = {"text": text, "author": author}
                    # Find child comments (replies)
                    childern_divs = comment_div.find_elements(By.XPATH, childs_xpath)
                    if len(childern_divs) > 0:
                        # Recursively extract child comments
                        children = self._extract_comments_rec(childern_divs)
                        if children:
                            comment["children"] = children
                    # Add the comment to the list
                    comments.append(comment)
            except NoSuchElementException as e:
                # Skip if any expected element is not found
                pass

        return comments

    def _extract_comments(self):
        """
        Extract all comments from the current Reddit post page.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing comment data.

        Notes:
            - Checks if there are comments on the page; returns an empty list if none are found.
            - Uses a recursive helper method to extract comments and their nested replies.
        """
        # Define the XPath to locate the top-level comment divs
        comments_xpath = (
            '//div[contains(@class, "commentarea")]'
            '/div[contains(@class, "nestedlisting")]'
            '/div[contains(@class, "comment")]'
        )

        # Define the XPath to check if there are no comments
        no_comment_xpath = (
            comments_xpath[0] + '/div[contains(@class, "panestack-title")]'
        )
        try:
            # Try to find the 'no comments' indicator
            no_comment_div = self.driver.find_element(By.XPATH, no_comment_xpath)
            if no_comment_div.text == "no comments (yet)":
                return []
        except:
            # If the 'no comments' indicator is not found, proceed to extract comments
            pass

        # Wait until the comments are present
        self._wait_for_element((By.XPATH, comments_xpath))
        comments_divs = self.driver.find_elements(By.XPATH, comments_xpath)
        # Recursively extract comments and their replies
        return self._extract_comments_rec(comments_divs)

    def _check_if_blocked(self) -> bool:
        """
        Check if the scraper has been blocked or detected as a bot.

        Raises:
            ValueError: If the page body cannot be accessed.
            DetectedBotException: If the scraper is detected as a bot.
        """
        # Wait for the <body> element to be present
        post = self._wait_for_element((By.XPATH, "/html/body"))
        if not post:
            # Can't determine blocking status without the body element
            raise ValueError("Can't evaluate if blocked because post is None")
        try:
            # Try to find a <pre> element within the body, which may indicate a block
            body_pre = post.find_element(By.XPATH, "/html/body/pre")
            if body_pre.text == "":
                # If the <pre> element is empty, assume bot detection and raise an exception
                raise exceptions.DetectedBotException("Detected as a bot")
            else:
                return  # Probably not blocked
        except NoSuchElementException:
            return  # Probably not blocked

    def _load_page(self, url: str, xpath: str):
        """
        Load a webpage and check if the scraper has been blocked.

        Args:
            url (str): The URL of the page to load.
            xpath (str): The XPath of the element to wait for after loading the page.

        Returns:
            WebElement or None: The WebElement found using the provided XPath if successful,
                otherwise None.

        Raises:
            DetectedBotException: If the scraper is detected as a bot.
            ValueError: If unable to determine blocking status due to missing elements.

        Notes:
            - This method attempts to load the page multiple times, as specified by
              `self.page_retry`.
            - If blocked or an error occurs, it switches the proxy and retries.
        """
        # Wait for a random duration to mimic human behavior
        self._wait()
        for _ in range(self.page_retry):
            try:
                self._logger.debug(f"Loading URL: {url}")
                self.driver.get(url)
                self._check_if_blocked()
                element = self._wait_for_element((By.XPATH, xpath))
                if element:
                    return element  # Success

            except exceptions.DetectedBotException:
                # Log a warning if detected as a bot and prepare to switch proxy
                self._logger.warning(f"Switching proxy. Got blocked on post {url}")
            except Exception as e:
                pass  # Retry the loading process

            # Switch proxy due to an error or being blocked
            self._logger.warning(f"Switching proxy. Unexpected error while loading page {url}")
            # Restart the driver to use a new proxy
            self.start_driver()

    def scrape_post_list(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 100,
        after_post_id: str = None,
    ) -> list:
        """
        Search for posts in a specific subreddit using a search query.

        Args:
            subreddit (str): The name of the subreddit to search.
            search_query (str): The search query string.
            limit (int, optional): Maximum number of posts to retrieve (up to 100). Defaults to 100.
            after_post_id (str, optional): The ID of the post after which to continue the search
                (used for pagination). Defaults to None.

        Returns:
            List[WebElement]: A list of WebElements representing the search result posts.
                Returns an empty list if no results are found or an error occurs.

        Notes:
            - Reddit limits the number of posts that can be retrieved in a single search to 100.
            - This method constructs the search URL, loads the page, and extracts the search results.
        """
        # Check if the requested limit exceeds the maximum allowed limit
        if limit > self._max_search_limit:
            self._logger.warning(f"Limit for loading a single page of search results must be less than {self._max_search_limit}")

        # Construct the search URL with query parameters
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
        # Add the 'after' parameter for pagination if an after_post_id is provided
        if after_post_id:
            url += f"&after={after_post_id}"
        #  Load the search results page and wait for the search result listing to be present
        search_result_listing = self._load_page(
            url, '//div[contains(@class, "search-result-listing")]'
        )

        if search_result_listing:
            # Check if there are no search results
            try:
                footer = search_result_listing.find_element(By.XPATH, ".//footer")
                if "there doesn't seem to be anything here" in footer.text:
                    return []  # No search results
            except:
                pass  # Continue

            # Wait for the individual search result links to load
            self._wait_for_element(
                (By.XPATH, '//div[contains(@class, "search-result-link")]')
            )
            # Find all the search result elements
            search_results: list | None = self.driver.find_elements(
                By.XPATH, '//div[contains(@class, "search-result-link")]'
            )
            if search_results:
                return search_results  # Success

        self._logger.warning(f"Error while getting search results\nsubreddit: {subreddit}\nsearch_query: {search_query}\nlimit: {limit}\nafter postid: {after_post_id}\nurl: {url}")

    def scrape_multipage_post_list(
        self,
        subreddit: str,
        search_query: str,
        limit: int = 500,
        start_date: str = None,
        after_post_id: str = None,
    ) -> None:
        """
        Search for posts in a specific subreddit across multiple pages to retrieve more than 100 posts.

        Args:
            subreddit (str): The name of the subreddit to search within.
            search_query (str): The search query string.
            limit (int, optional): The total number of posts to retrieve. Defaults to 500 but the limit will be max 300
            start_date (str or pd.Timestamp, optional): The earliest date of posts to retrieve.
                Posts older than this date will not be included. Defaults to None.
            after_post_id (str, optional): The ID of the post after which to continue the search
                (used for pagination). Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing post metadata.

        Notes:
            - Reddit limits the number of posts that can be retrieved per page to 100 and up to 300 overall.
              This method paginates through the search results to retrieve up to the specified limit.
            - If a `start_date` is provided, the search will stop when it encounters posts older than this date.
            - The method handles retries and proxy switching in case of failures or detection as a bot.
        """
        # Loop through the pages to collect posts until the limit is reached
        for num_processed_posts in range(0, limit, self._max_search_limit):
            search_limit = min(self._max_search_limit, limit - num_processed_posts)

            search_results = None
            # Retry fetching the post list if necessary
            for i in range(self.scrape_post_list_retry):
                if i > 0:
                    self._logger.info(f"Retry {i+1} for getting post list for subreddit {subreddit} and search query {search_query}")
                try:
                    # Load one page of search results
                    search_results = self.scrape_post_list(
                        subreddit, search_query, search_limit, after_post_id
                    )
                    if search_results is not None:
                        break  # Successfully retrieved search results
                except:
                    self.start_driver()  # Switch proxy and retry

            # Break if there are no more search results
            if isinstance(search_results, list) and len(search_results) == 0:
                break
            
            # TODO: warn but do not raise 
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
                # Convert the date strings to datetime objects
                oldest_post_date = pd.to_datetime(
                    self.post_data[-1]["date"]
                ).tz_localize(None)
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date).tz_localize(None)

                if oldest_post_date < start_date:
                    break

        return self.post_data

    def scrape_post_content(self, post: dict) -> dict:
        """
        Load and extract content from an individual Reddit post URL.

        Args:
            post (dict): A dictionary containing at least the 'url' key of the post to scrape.

        Returns:
            dict: The updated post dictionary with additional keys for 'text' and 'children' (comments).
                Returns None if the post content could not be loaded.

        Raises:
            ValueError: If an unexpected error occurs while extracting the post text.

        Notes:
            - This method loads the post page, extracts the post text, and retrieves the comments.
            - If the post has no text content, the 'text' key will be an empty string.
            - If the post page could not be loaded, the method returns None to allow for potential retries.
        """
        # Define the XPath to locate the main post entry on the page
        post_entry_xpath = '//div[contains(@class, "entry")]'
        # Load the post page and wait for the main entry element to be present
        post_entry = self._load_page(post["url"], post_entry_xpath)

        if post_entry is None:
            return None  # Might be retried if mutiple runs are done

        # Define the XPath to locate the post text content within the entry
        post_text_xpath = './/div[contains(@class, "md")]'
        try:
            # Find the div containing the post text
            post_text_div = post_entry.find_element(By.XPATH, post_text_xpath)
            # Extract and store the post text
            post["text"] = post_text_div.text
        except NoSuchElementException:
            # It is possible that the post has no text content, so set 'text' to an empty string
            post["text"] = ""
        except Exception as e:
            raise ValueError(f"Unexpected error while extracting post text: {e}")

        # Extract comments
        post["children"] = self._extract_comments()

        return post

    def get_posts_without_content(self):
        """
        Get posts without text content.

        Returns:
        List[Dict[str, Any]]: A list of post dictionaries where the 'text' key is missing or None.
        """

        return [post for post in self.post_data if post.get("text") is None]

    def scrape_all_missing_post_contents(self):
        """
        Scrape the content of posts in `self.post_data` that are missing text content.

        This method attempts to scrape the post content for all posts in `self.post_data`
        where the 'text' key is missing or None. It retries the scraping process up to
        `self.scrape_missing_posts_retry` times for posts that still lack content after each attempt.

        Returns:
            None

        Notes:
            - Logs informational messages during retries and a warning if posts remain without content.
            - Useful for ensuring that all collected posts have their content and comments scraped.
        """
        for i in range(self.scrape_missing_posts_retry):
            if i > 0:
                # Get the number of posts still missing
                num_posts_without_content = len(self.get_posts_without_content())
                self._logger.info(f"Retry {i+1} for getting post content. {num_posts_without_content} posts are missing.")
            # Attempt to scrape content for each post without text
            for post in self.get_posts_without_content():
                self.scrape_post_content(post)
            # Check if all posts now have content
            num_posts_without_content = len(self.get_posts_without_content())
            if num_posts_without_content == 0:
                return  # Success

            # If not, the loop will retry scraping missing posts

        # Log a warning if posts are still missing content after all retries
        self._logger.warning(f"{num_posts_without_content} are still missing after {self.scrape_missing_posts_retry} tries.")

    def start_driver(self):
        """
        Start the WebDriver session with a new proxy.

        This method initializes a new Selenium WebDriver session using a proxy obtained from `_get_next_proxy()`.
        If a driver session is already running, it quits that session before starting a new one.
        It continues to attempt to start a driver with new proxies until successful.

        Raises:
            exceptions.ProxyNotWorking: If the proxy is not working.

        Notes:
            - This method loops indefinitely until a working proxy is found.
            - If a proxy is not working, it logs the information and tries the next one.
        """
        while True:
            try:
                if self.driver is not None:
                    self.quit()
                proxy = self._get_next_proxy()
                self.driver = utils.get_driver(
                    self.headless, proxy.protocol, f"{proxy.ip}:{proxy.port}"
                )
                self._logger.debug("WebDriver session started.")
                return  # Success
            except exceptions.ProxyNotWorking:
                self._logger.info("Proxy not working, trying next one.")

    def quit(self):
        """
        Quit the WebDriver session if it's running.

        This method checks if the Selenium WebDriver session is active. If so, it quits the driver and
        sets the `driver` attribute to `None`.
        """
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            self._logger.debug("WebDriver session quit.")

    def to_dataframe(self):
        """
        Convert the scraped data to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the scraped post data.

        Notes:
            - The DataFrame is constructed from the `self.post_data` list of dictionaries.
            - Useful for data analysis and storage after scraping is complete.
        """
        return pd.DataFrame(self.post_data)

    def scrape(self, *args, **kwargs):
        """
        Perform the entire scraping process.

        This method orchestrates the scraping workflow, including starting the driver, scraping multiple
        pages of post listings, scraping missing post contents, and quitting the driver. It finally
        returns the scraped data as a pandas DataFrame.

        Args:
            *args: Variable length argument list to pass to `scrape_multipage_post_list`.
            **kwargs: Arbitrary keyword arguments to pass to `scrape_multipage_post_list`.

        Returns:
            pd.DataFrame: A DataFrame containing the scraped post data.

        Notes:
            - If no data is scraped, the returned DataFrame may be empty.
            - The method ensures that the WebDriver session is properly started and terminated.
        """
        self.start_driver()
        self.scrape_multipage_post_list(*args, **kwargs)
        self.scrape_all_missing_post_contents()
        self.quit()
        # TODO: if no data return None
        # TODO: return self.post_data ?
        # return self.to_dataframe()
