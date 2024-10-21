"""
File: twitter.py

Description:
- This file is used to scrape data from the website Twitter (X).

Authors:
- 
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import CryptoFraudDetection.scraper.utils as utils

import time
import re
import json, os
import csv
import random

class TwitterScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password


    def login_save_cookies(self):
        driver = utils.get_driver(headless=False)

        try:
            self.navigate_to_login_page(driver)
            self.enter_credentials(driver)
            self.save_cookies(driver)

        finally:
            driver.quit()


    def navigate_to_login_page(self, driver):
        """Navigate to the login page of the website."""
        driver.get("https://www.x.com")
        wait = WebDriverWait(driver, 10)
        time.sleep(10)
        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@href='/login']")))
        driver.execute_script("arguments[0].click();", login_button)


    def enter_credentials(self, driver):
        """Enter the username and password on the login page."""
        wait = WebDriverWait(driver, 10)
        
        # Enter username
        wait.until(EC.presence_of_element_located((By.NAME, "text")))
        username_field = driver.find_element(By.NAME, "text")
        username_field.send_keys(self.username)
        username_field.send_keys(Keys.RETURN)
        
        # Enter password
        wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_field = driver.find_element(By.NAME, "password")
        password_field.send_keys(self.password)
        password_field.send_keys(Keys.RETURN)
        
        time.sleep(5)


    def save_cookies(self, driver):
        """Save the cookies after login."""
        cookies = driver.get_cookies()
        with open('../data/cookies_x_1_0.json', 'w') as file:
            json.dump(cookies, file)
        print("Cookies saved.")


    def scrape_with_cookies(self, tweet_count=1, search_query="Bitcoin", headless=False):
        driver = utils.get_driver(headless=headless)

        try:
            self.load_cookies(driver)
            self.navigate_to_explore(driver)
            self.perform_search(driver, search_query)
            self.scrape_tweets(driver, tweet_count, search_query)

        finally:
            driver.quit()


    def load_cookies(self, driver):
        """Load cookies from a file and refresh the page."""
        driver.get("https://www.x.com")
        with open('../data/cookies_x_1_0.json', 'r') as file:
            cookies = json.load(file)

        for cookie in cookies:
            driver.add_cookie(cookie)
        
        driver.refresh()
        time.sleep(10)


    def navigate_to_explore(self, driver):
        """Navigate to the explore page and handle any popups."""
        driver.get("https://www.x.com/explore")
        time.sleep(10)
        print("Page title after loading cookies and navigating to Explore:", driver.title)

        try:
            close_button_wait = WebDriverWait(driver, 5)
            close_button = close_button_wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'Close')]"))
            )
            driver.execute_script("arguments[0].click();", close_button)
            print("Close button clicked successfully.")
        except Exception as e:
            print("Could not find or click the close button:", e)


    def perform_search(self, driver, search_query):
        """Perform a search for the given query."""
        try:
            search_bar = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@aria-label='Search query']"))
            )
            search_bar.clear()  # Clear the search bar before typing
            search_bar.send_keys(search_query)
            search_bar.send_keys(Keys.RETURN)  # Press Enter key to submit the search
            print(f"Searched for: {search_query}")
            time.sleep(5)
        except Exception as e:
            print("Could not find or enter into the search bar:", e)


    def scrape_tweets(self, driver, tweet_count, search_query):
        """Scrape the tweets from the page and save them to a CSV file."""
        tweets_scraped = 0
        sanitized_query = re.sub(r'[^\w\s]', '', search_query)  # Entfernt Sonderzeichen
        sanitized_query = re.sub(r'\s+', '_', sanitized_query)  # Ersetzt Leerzeichen durch Unterstriche
        file_name = f"../data/{sanitized_query}_tweets.csv"

        with open(file_name, mode="w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Username", "Tweet", "Timestamp", "Likes", "Impressions"])

            while tweets_scraped < tweet_count:
                tweets = self.get_tweets(driver)
                for tweet in tweets:
                    try:
                        username, content, timestamp, likes, impressions = self.extract_tweet_details(tweet)
                        writer.writerow([username, content, timestamp, likes, impressions])
                        tweets_scraped += 1
                        print(f"Scraped tweet {tweets_scraped}/{tweet_count}: {username} - {content[:50]}")

                        if tweets_scraped >= tweet_count:
                            break

                        sleep_time = random.uniform(2, 5)
                        print(f"Sleeping for {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)

                    except Exception as e:
                        print(f"Could not extract tweet details: {e}")

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)

        print("Tweets saved to CSV.")


    def get_tweets(self, driver):
        """Retrieve the list of tweets from the current page."""
        try:
            return WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//article[@data-testid='tweet']"))
            )
        except Exception as e:
            print("Could not find or scrape tweets:", e)
            return []


    def extract_tweet_details(self, tweet):
        """Extract details such as username, tweet content, timestamp, likes, and impressions from a tweet element."""
        try:
            username = tweet.find_element(By.XPATH, ".//span[contains(text(), '@')]").text
            content = tweet.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text
            timestamp_element = tweet.find_element(By.XPATH, ".//time")
            timestamp = timestamp_element.get_attribute("datetime")

            try:
                likes = tweet.find_element(By.XPATH, ".//div[@data-testid='like']//span").get_attribute("innerHTML")
            except Exception:
                likes = "0"

            try:
                impressions = tweet.find_element(By.XPATH, ".//div[@data-testid='view']").text
                if impressions == "":
                    impressions = tweet.find_element(By.XPATH, ".//div[@data-testid='view']//span").get_attribute("innerHTML")
            except Exception:
                impressions = "N/A"

            return username, content, timestamp, likes, impressions
        except Exception as e:
            print(f"Could not extract tweet details: {e}")
            return None, None, None, "0", "N/A"