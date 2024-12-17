from pathlib import Path

import pandas as pd

from CryptoFraudDetection.utils import enums
from CryptoFraudDetection.utils import hf_sentiment
from CryptoFraudDetection.utils import logger

LOGGER = logger.Logger(name=__name__, level=enums.LoggerMode.INFO, log_dir="../logs")

reddit_parquet_path = "../data/processed/reddit_embedded.parquet"
sentiment_reddit_parquet_path = "../data/processed/reddit_sentiment.parquet"
twitter_parquet_path = "../data/processed/x_embedded.parquet"
sentiment_twitter_parquet_path = "../data/processed/x_sentiment.parquet"

if not Path(sentiment_reddit_parquet_path).exists():
    df = pd.read_parquet(reddit_parquet_path)
    text = df["body"].tolist()

    sentiment_scores = hf_sentiment.score(logger_=LOGGER, text=text)
    df["sentiment_score"] = sentiment_scores
    df.to_parquet(sentiment_reddit_parquet_path)

if not Path(sentiment_twitter_parquet_path).exists():
    df = pd.read_parquet(twitter_parquet_path)
    text = df["tweet"].tolist()

    sentiment_scores = hf_sentiment.score(logger_=LOGGER, text=text)
    df["sentiment_score"] = sentiment_scores
    df.to_parquet(sentiment_twitter_parquet_path)
