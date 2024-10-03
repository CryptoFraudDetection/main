import CryptoFraudDetection.scraper.utils as utils
from selenium.webdriver.common.by import By
import time

"""
File: google_results.py

Description:
- This file is used to scrape data from Google Results.

Authors:
- 
"""


class GoogleResultsScraper:
    def __init__(self):
        pass

    def get_main_results(self, query, headless=False):
        """
        Get the results from Google Search
        """
        driver = utils.get_driver(headless=headless)
        driver.get("https://www.google.com")

        time.sleep(0.5)
        driver.find_element(by=By.XPATH, value='//*[@id="L2AGLb"]').click()

        time.sleep(0.5)
        search_box = driver.find_element(by=By.XPATH, value='//*[@id="APjFqb"]')
        search_box.send_keys(query)
        search_box.submit()

        time.sleep(3)
        # find all h3 elements in an a element
        results = driver.find_elements(by=By.XPATH, value="//a/h3")
        results = [result.text for result in results if result.text != ""]

        driver.quit()
        return results
