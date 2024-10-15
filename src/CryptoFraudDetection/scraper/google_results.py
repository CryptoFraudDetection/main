"""
File: google_results.py

Description:
- This file is used to scrape data from Google Results.

Authors:
- 
"""

import time
import contextlib
from collections import defaultdict

import CryptoFraudDetection.scraper.utils as utils
from CryptoFraudDetection.utils.exceptions import *

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from tqdm import tqdm

BOX_CLASS = "MjjYud"  # kann sich ändern, offload oder xpath lösung ohne klasse
DESC_CLASS = "VwiC3b"


class GoogleResultsScraper:
    def __init__(self):
        pass

    def get_main_results(self, query, n_sites=5, headless=False):
        """
        Get the results from Google Search
        """
        # Parameter, returns

        # Add comments to better navigate in the code structure

        if n_sites < 1:
            raise InvalidParameterException("Number of sites must be at least 1")

        driver = utils.get_driver(headless=headless)
        driver.get("https://www.google.com")
        driver.find_element(
            by=By.XPATH, value='//*[@id="L2AGLb"]'
        ).click()  # Class into var

        time.sleep(0.5)
        search_box = driver.find_element(
            by=By.XPATH, value='//*[@id="APjFqb"]'
        )  # Class into var
        search_box.send_keys(query)
        search_box.submit()

        ## More Error Handling

        time.sleep(1)
        results = defaultdict(list)
        for i in tqdm(range(n_sites), desc="Scraping Google", unit="site"):
            soup = BeautifulSoup(driver.page_source, "html.parser")
            boxes = soup.find_all(
                "div", class_=BOX_CLASS
            )  # überdenken, ob bs entfernen, da alles mit selenium gemacht werden kann

            if len(boxes) == 0:
                raise DetectedBotException("Google detected agent as bot.")

            for result in boxes:
                with contextlib.suppress(NoSuchElementException):
                    link = result.find("a")["href"]
                    title = result.find("h3").get_text()  # if None could throw except
                    desc = result.find("div", class_=DESC_CLASS).get_text()

                    results["link"].append(link)
                    results["title"].append(title)
                    results["description"].append(desc)

            if i != n_sites - 1:
                try:
                    next_button = driver.find_element(
                        by=By.XPATH, value='//*[@id="pnnext"]'  # Class into var
                    )
                    next_button.click()

                    # time.sleep(0.5)
                except Exception as e:  # genauere Exception
                    print(e)

        driver.quit()
        return results
