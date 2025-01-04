"""Data pipeline for the CryptoFraudDetection project."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from CryptoFraudDetection.utils import logger


def _to_utc(x: pd.Series) -> pd.Series:
    if pd.to_datetime(x).tzinfo is None:
        return pd.to_datetime(x).tz_localize("UTC")
    return pd.to_datetime(x).tz_convert("UTC")


def read_data(
    logger_: logger.Logger,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # find data dir
    data_dir = Path("data")
    if not data_dir.is_dir():
        data_dir = Path("../data")
        if not data_dir.is_dir():
            logger_.error(
                "data directory not found.\n"
                f"current directory: {os.getcwd()}",
            )

    # load coin info
    with data_dir.joinpath("raw/coins.json").open() as f:
        coin_info = json.load(f)

    # load price data
    csv_name_overwrite = {
        "lunc": "luna",
        "ftt": "ftx",
        "bf": "bitforex",
        "teddy v2": "teddydoge",
    }
    price_dfs = []
    for coin in coin_info:
        symbol = coin["symbol"].lower()
        coin["csv_name"] = data_dir.joinpath(
            "raw",
            "coin_price_data",
            csv_name_overwrite.get(symbol, symbol) + ".csv",
        )
        single_coin_price_df = pd.read_csv(coin["csv_name"])
        single_coin_price_df["coin"] = coin["symbol"]
        price_dfs.append(single_coin_price_df)
    price_df = pd.concat(price_dfs)
    price_df = price_df.rename(
        {
            "time": "datetime",
        },
        axis="columns",
    )
    price_df["datetime"] = price_df["datetime"].apply(_to_utc)

    # add scam status to price data
    coin_scam_status = {coin["symbol"]: coin["fraud"] for coin in coin_info}
    price_df["fraud"] = price_df["coin"].map(coin_scam_status)

    twitter_df = pd.read_parquet(
        data_dir.joinpath("processed/x_posts_embeddings.parquet"),
    )
    twitter_df = twitter_df.rename(
        {
            "timestamp": "datetime",
            "searchkeyword": "coin",
            "likes": "score",
            "comments": "n_comments",
        },
        axis="columns",
    )
    twitter_df["datetime"] = twitter_df["datetime"].apply(_to_utc)

    reddit_df = pd.read_parquet(
        data_dir.joinpath("processed/reddit_embedded.parquet"),
    )
    reddit_df = reddit_df.rename(
        {
            "created": "datetime",
            "search_query": "coin",
            "num_comments": "n_comments",
            "embedded_text": "embedding",
        },
        axis="columns",
    )
    reddit_df["datetime"] = reddit_df["datetime"].apply(_to_utc)

    return coin_info, price_df, twitter_df, reddit_df


def validate_price_data(price_df: pd.DataFrame, logger_: logger.Logger):
    required_columns = ["datetime", "open", "high", "low", "close", "volume"]

    for column in required_columns:
        if column not in price_df.columns:
            logger_.error(f"missing column {column} in price data")

    if price_df[required_columns].isna().sum().sum() != 0:
        logger_.error("missing values in price data")


def group_scocial_media_df(
    social_media_df: pd.DataFrame,
    datetime_index: pd.DatetimeIndex,
    coin_mapping: dict[str, str],
    logger_: logger.Logger,
    date_column: str = "datetime",
    score_column: str = "score",
    n_comments_column: str = "n_comments",
    embedding_column: str = "embedding",
    coin_column: str = "coin",
) -> pd.DataFrame:
    """Group a social media DataFrame by coin and aligns entries to the given time index.

    Args:
        social_media_df (pd.DataFrame): The social media data to group.
        datetime_index (pd.DatetimeIndex): The time index to align the data to.
        date_column (str): The column containing datetime information.
        score_column (str): The column containing scores to sum.
        n_comments_column (str): The column containing the number of comments to sum.
        embedding_column (str): The column containing embeddings to average.
        logger_ (Any): Logger instance for reporting issues.

    Returns:
        pd.DataFrame: A DataFrame grouped and aligned to the time index, with scores,
            number of comments summed, and embeddings averaged.

    """
    social_media_df = social_media_df.sort_values(date_column)

    # replace coin names with mapping
    social_media_df[coin_column] = social_media_df[coin_column].map(
        coin_mapping,
    )
    if social_media_df[coin_column].isna().sum() != 0:
        logger_.error("failed to convert coin names to symbols")

    # Group by coin and align to time index
    grouped_data = []

    for coin, group in social_media_df.groupby(coin_column):
        group["interval_index"] = datetime_index.get_indexer(
            group[date_column], method="pad",
        )

        # Aggregate data within each time interval
        agg = (
            group.groupby("interval_index")
            .agg(
                score = pd.NamedAgg(score_column, "sum"),
                n_comments = pd.NamedAgg(n_comments_column, "sum"),
                embedding = pd.NamedAgg(
                    embedding_column,
                    lambda x: np.mean(np.stack(x), axis=0),
                ),
                count = pd.NamedAgg(column=embedding_column, aggfunc="count"),
            )
            .reset_index()
        )

        # Add back the time index and coin information
        agg["datetime"] = datetime_index[agg["interval_index"]]
        agg["coin"] = coin
        agg = agg.drop(["interval_index"], axis=1)
        grouped_data.append(agg)

    return  pd.concat(grouped_data, ignore_index=True)


def _rename_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(
        columns={
            col: f"{prefix}_{col}"
            for col in df.columns
            if col not in ["datetime", "coin"]
        },
    )


def merge_dfs(
    price_df: pd.DataFrame, twitter_df: pd.DataFrame, reddit_df: pd.DataFrame,
) -> pd.DataFrame:
    twitter_df_ = _rename_columns(twitter_df, "twitter")
    reddit_df_ = _rename_columns(reddit_df, "reddit")

    merged_df = pd.merge(
        price_df, twitter_df_, how="left", on=["datetime", "coin"],
    )
    return pd.merge(merged_df, reddit_df_, how="left", on=["datetime", "coin"])
