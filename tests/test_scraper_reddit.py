"""
This module contains the tests for the scraper.comparitech module.
"""

import CryptoFraudDetection.utils.logger as logger
from CryptoFraudDetection.scraper.reddit import RedditScraper, scrape_reddit
from CryptoFraudDetection.utils.enums import LoggerMode

logger_ = logger.Logger(name=__name__, level=LoggerMode.DEBUG, log_dir="../logs")


def test_initialization():
    """
    Test the initialization of the RedditScraper class
    """
    try:
        scraper = RedditScraper(logger_)
        scraper.start_driver()
    finally:
        scraper.quit()

def test__extract_all_comments():
    """
    Test the _extract_comments method of the RedditScraper class
    """
    try:
        scraper = RedditScraper(logger_)
        scraper.start_driver()
        url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
        scraper.scrape_post_content({'url': url})
        comments = scraper._extract_all_comments()
        assert isinstance(comments, list)
        assert len(comments) > 0
        for comment in comments:
            assert isinstance(comment, dict)
            assert comment.get('text', None) not in [None, '']
            assert comment.get('author', None) not in [None, '']
            if comment.get('children', None) is not None:
                assert isinstance(comment['children'], list)
    finally:
        scraper.quit()

def test_scrape_post_content():
    """
    Test the scrape_post_content method of the RedditScraper class
    """
    try:
        scraper = RedditScraper(logger_)
        scraper.start_driver()
        url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
        post = scraper.scrape_post_content({'url': url})

        assert post.get('text', None) not in [None, '']
        assert post.get('children', None) is not None
    finally:
        scraper.quit()

def test_get_multipage_post_list():
    """
    Test the get_multipage_post_list method of the RedditScraper class
    """
    try:
        scraper = RedditScraper(logger_, max_search_limit=2)
        scraper.start_driver()
        posts = scraper.get_multipage_post_list('r/CryptoCurrency', 'Terra Luna', limit=3)
        assert isinstance(posts, list)
        assert len(posts) > 0
    finally:
        scraper.quit()

def test_get_multipage_post_list_with_start_date():
    try:
        scraper = RedditScraper(logger_, max_search_limit=2)
        scraper.start_driver()
        posts = scraper.get_multipage_post_list('r/CryptoCurrency', 'Terra Luna', limit=100, start_date='2024-07-10', after_post_id='t3_1fh7myu')
        assert isinstance(posts, list)
        assert len(posts) == 4
    finally:
        scraper.quit()


def test_scrape_reddit():
    """
    Test the RedditScraper class
    """
    df = scrape_reddit(logger_,'r/CryptoCurrency', 'Terra Luna', limit=101)
    assert df is not None
    assert len(df) == 101
    
if __name__ == '__main__':
    # test_initialization()
    # test__extract_all_comments()
    # test_scrape_post_content()
    # test_get_multipage_post_list()
    # test_get_multipage_post_list_with_start_date()
    test_scrape_reddit()
