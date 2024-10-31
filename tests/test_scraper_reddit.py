"""
This module contains the tests for the scraper.comparitech module.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.scraper.reddit import RedditScraper
from CryptoFraudDetection.utils.enums import LoggerMode

logger_ = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")


def test_initialization():
    """
    Test the initialization of the RedditScraper class
    """
    RedditScraper(logger_, '', '', '')


def test_scrape_post_content():
    """
    Test the scrape_post_content method of the RedditScraper class
    """
    scraper = RedditScraper(logger_, '', '', '')
    url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
    post = scraper.scrape_post_content(url)

    assert post.get('text', None) not in [None, '']
    #assert isinstance(post.get('score', None), int)
    
    # # Check if the first comment has the required keys
    # for key in ['id', 'author', 'text', 'score', 'date', 'children']:
    #     assert post['children'][0].get(key, None)
    # # Check if the first child of the first comment has the required keys
    # for key in ['id', 'author', 'text', 'score', 'date', 'children']:
    #     assert post['children'][0]['children'][0].get(key, None)
    scraper.quit()

def test_scrape_post_id():
    """
    Test the scrape_post_id method of the RedditScraper class
    """
    scraper = RedditScraper(logger_, '', '', '')
    url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
    id = scraper.scrape_post_content(url)
    
    assert id.get('id', None) not in [None,'']

def test_scrape_post_author():
    """
    Test the scrape_post_author method of the RedditScraper class
    """
    scraper = RedditScraper(logger_, '', '', '')
    url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
    author = scraper.scrape_post_content(url)
    
    assert author.get('author', None) not in [None,'']

def test_scrape_post_date():
    """
    Test the scrape_post_date method of the RedditScraper class
    """
    scraper = RedditScraper(logger_, '', '', '')
    url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
    date = scraper.scrape_post_content(url)
    
    assert date.get('date', None) not in [None,'']
    
