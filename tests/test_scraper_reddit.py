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
    scraper = RedditScraper(logger_, headless=False)
    scraper.start_driver()
    scraper.quit()


def test__extract_all_comments():
    """
    Test the _extract_comments method of the RedditScraper class
    """
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
    scraper.quit()

def test_scrape_post_content():
    """
    Test the scrape_post_content method of the RedditScraper class
    """
    scraper = RedditScraper(logger_)
    scraper.start_driver()
    url = 'https://old.reddit.com/r/Fire/comments/11lfj4e/lost_400000_my_whole_net_worth_to_the_terra_luna/'
    post = scraper.scrape_post_content({'url': url})

    assert post.get('text', None) not in [None, '']
    assert post.get('children', None) is not None
    scraper.quit()
    
def test_get_post_list():
    """
    Test the get_post_list method of the RedditScraper class
    """
    scraper = RedditScraper(logger_)
    scraper.start_driver()
    posts = scraper.get_post_list('r/CryptoCurrency', 'Terra Luna', limit=3, max_num_posts_per_search=2)
    assert isinstance(posts, list)
    assert len(posts) > 0
    scraper.quit()
    
def test_reddit_scraper():
    """
    Test the RedditScraper class
    """
    scraper = RedditScraper(logger_)
    scraper.start_driver()
    scraper.get_post_list('r/CryptoCurrency', 'Terra Luna', limit=3, max_num_posts_per_search=2)
    scraper.scrape_all_post_contents()
    scraper.quit()
    df = scraper.to_dataframe()
    assert df is not None
    assert len(df) > 0
    print(df)
    
    
if __name__ == '__main__':
    # test_scrape_post_content()
    #test_get_post_list()
    test_reddit_scraper()
