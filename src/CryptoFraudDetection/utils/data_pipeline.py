"""Data pipeline for the CryptoFraudDetection project."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from CryptoFraudDetection.utils import logger

random.seed(42)
np.random.seed(42)


def _rename_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(
        columns={
            col: f"{prefix}_{col}"
            for col in df.columns
            if col not in ["datetime", "coin"]
        },
    )


class CryptoData:
    def __init__(
        self,
        data_dir: Path,
        logger_: logger.Logger,
        coin_info_file_path: Path = Path("raw/coins.json"),
        price_dir_path: Path = Path("raw/coin_price_data"),
        twitter_parquet_path: Path = Path(
            "processed/x_posts_embeddings.parquet",
        ),
        reddit_parquet_path: Path = Path("processed/reddit_embedded.parquet"),
    ):
        self._logger = logger_
        self._data_dir = data_dir

        self._coin_info_file_path = coin_info_file_path
        self._price_dir_path = price_dir_path
        self._twitter_parquet_path = twitter_parquet_path
        self._reddit_parquet_path = reddit_parquet_path

        self.coin_info = None
        self._twitter_df = None
        self._reddit_df = None
        self.merged_df = None
        self.train_df = None
        self.test_df = None

    def _validate_price_data(self):
        required_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in required_columns:
            if column not in self._price_df.columns:
                self._logger.error(f"missing column {column} in price data")

        if self._price_df[required_columns].isna().sum().sum() != 0:
            self._logger.error("missing values in price data")

    def load_data(self) -> None:
        # load coin info
        with self._data_dir.joinpath(self._coin_info_file_path).open() as f:
            self.coin_info = json.load(f)

        # load price data
        csv_name_overwrite = {
            "lunc": "luna",
            "ftt": "ftx",
            "bf": "bitforex",
            "teddy v2": "teddydoge",
        }
        price_dfs = []
        for coin in self.coin_info:
            symbol = coin["symbol"].lower()
            coin["csv_name"] = self._data_dir.joinpath(
                self._price_dir_path,
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
        price_df["datetime"] = pd.to_datetime(price_df["datetime"], utc=True)
        # add scam status to price data
        coin_scam_status = {
            coin["symbol"]: coin["fraud"] for coin in self.coin_info
        }
        price_df["fraud"] = price_df["coin"].map(coin_scam_status)
        self._price_df = price_df

        # Load Twitter data
        twitter_df = pd.read_parquet(
            self._data_dir.joinpath(self._twitter_parquet_path),
        )
        twitter_column_map = {
            "timestamp": "datetime",
            "searchkeyword": "coin",
            "likes": "score",
            "comments": "n_comments",
        }
        twitter_df = twitter_df.rename(twitter_column_map, axis="columns")
        twitter_df["datetime"] = pd.to_datetime(
            twitter_df["datetime"],
            utc=True,
        )
        twitter_df["embedding"] = twitter_df["embedding"]
        self._twitter_df = twitter_df

        # Load Reddit data
        reddit_df = pd.read_parquet(
            self._data_dir.joinpath(self._reddit_parquet_path),
        )
        reddit_column_map = {
            "created": "datetime",
            "search_query": "coin",
            "num_comments": "n_comments",
            "embedded_text": "embedding",
        }
        reddit_df = reddit_df.rename(reddit_column_map, axis="columns")
        reddit_df["datetime"] = pd.to_datetime(reddit_df["datetime"], utc=True)
        reddit_df["embedding"] = reddit_df["embedding"]
        self._reddit_df = reddit_df
        self._validate_price_data()

    def _group_social_media_df(
        self,
        social_media_df: pd.DataFrame,
        datetime_index: pd.DatetimeIndex,
        coin_mapping: dict[str, str],
    ) -> pd.DataFrame:
        date_column = "datetime"
        score_column = "score"
        n_comments_column = "n_comments"
        embedding_column = "embedding"
        coin_column = "coin"
        
        social_media_df = social_media_df.sort_values(date_column)

        # replace coin names with mapping
        social_media_df[coin_column] = social_media_df[coin_column].map(
            coin_mapping,
        )
        cols = [coin_column, date_column]
        if social_media_df[cols].isna().sum().sum() != 0:
            self._logger.error(f"This columns can not cointain NAs: {cols}")

        # Group by coin and align to time index
        grouped_data = []

        for coin, group in social_media_df.groupby(coin_column):
            group["interval_index"] = datetime_index.get_indexer(
                group[date_column],
                method="pad",
            )

            # Aggregate data within each time interval
            agg = (
                group.groupby("interval_index")
                .agg(
                    score=pd.NamedAgg(score_column, "sum"),
                    n_comments=pd.NamedAgg(n_comments_column, "sum"),
                    embedding=pd.NamedAgg(
                        embedding_column,
                        lambda x: np.mean(np.stack(x), axis=0),
                    ),
                    count=pd.NamedAgg(
                        column=embedding_column,
                        aggfunc="count",
                    ),
                )
                .reset_index()
            )

            # Add back the time index and coin information
            agg["datetime"] = datetime_index[agg["interval_index"]]
            agg["coin"] = coin
            agg = agg.drop(["interval_index"], axis=1)
            grouped_data.append(agg)

        return pd.concat(grouped_data, ignore_index=True)

    def _group_social_media_dfs(self) -> None:
        coin_symbol_map = {
            coin["name"]: coin["symbol"] for coin in self.coin_info
        }
        datetime_index = pd.DatetimeIndex(
            sorted(self._price_df["datetime"].unique()),
        )

        self._grouped_twitter_df = self._group_social_media_df(
            self._twitter_df,
            datetime_index,
            coin_symbol_map,
        )
        self._grouped_reddit_df = self._group_social_media_df(
            self._reddit_df,
            datetime_index,
            coin_symbol_map,
        )

    def _fill_na(self):
        def fill_na_zero_embedding(
            x: list | None,
            empty_embedding: list,
        ) -> list:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return empty_embedding
            return x

        for col_name in ["twitter_embedding", "reddit_embedding"]:
            col = self.merged_df[col_name].dropna()
            embedding_lengths = set(map(len, col))
            if len(embedding_lengths) != 1:
                self._logger.error(
                    f"Embeddings in embedding column {col_name} do not all have the same length!",
                )
            empty_embedding = [0] * next(iter(embedding_lengths))
            self.merged_df[col_name] = self.merged_df[col_name].apply(
                fill_na_zero_embedding,
                args=(empty_embedding,),
            )
        self.merged_df = self.merged_df.fillna(0)

    def _merge_dfs(self) -> None:
        twitter_df = _rename_columns(self._grouped_twitter_df, "twitter")
        reddit_df = _rename_columns(self._grouped_reddit_df, "reddit")

        merged_df = pd.merge(
            self._price_df,
            twitter_df,
            how="left",
            on=["datetime", "coin"],
        )
        self.merged_df = pd.merge(
            merged_df,
            reddit_df,
            how="left",
            on=["datetime", "coin"],
        )
        self._fill_na()

    def _train_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_coins = [
            coin["symbol"] for coin in self.coin_info if coin["test"]
        ]
        self._logger.info(
            f"train:test number of coins: {len(self.coin_info) - len(test_coins)}:{len(test_coins)}",
        )
        self.train_df = self.merged_df[
            ~self.merged_df["coin"].isin(test_coins)
        ]
        self.test_df = self.merged_df[self.merged_df["coin"].isin(test_coins)]
        return self.train_df, self.test_df

    def preprocess(self) -> None:
        self._group_social_media_dfs()
        self._merge_dfs()
        return self._train_test_split()

    def save_data(
        self,
        train_parquet_path: Path,
        test_parquet_path: Path,
    ) -> None:
        self.train_df.to_parquet(train_parquet_path)
        self.test_df.to_parquet(test_parquet_path)


class CryptoDataSet(data.Dataset):
    """Crypto Data Set with price, twitter and reddit features"""

    def __init__(
        self,
        df: pd.DataFrame,
        logger_: logger.Logger,
        n_cutoff_points: int = 100,
        n_groups_cutoff_points: int = 10,
        embedding_columns=("twitter_embedding", "reddit_embedding"),
    ):
        self._logger = logger_
        self.df = df.copy()
        self._n_cutoff_points = n_cutoff_points
        self._n_groups_cutoff_points = n_groups_cutoff_points
        self._embedding_columns = embedding_columns

        self._merge_features()
        self._cutoff_points_all_coins()

    def _random_cutoff_points(
        self,
        df: pd.DataFrame,
        datetime_column: str = "datetime",
    ) -> list[pd.Timestamp]:
        if datetime_column not in df.columns:
            self._logger.error(
                f"Missing required column '{datetime_column}' in DataFrame.",
            )

        df = df.sort_values(datetime_column)
        unique_times = pd.to_datetime(df[datetime_column].unique())
        time_steps = len(unique_times)

        points_per_split = (
            self._n_cutoff_points // self._n_groups_cutoff_points
        )
        boundaries = np.linspace(
            0,
            time_steps,
            num=self._n_groups_cutoff_points + 1,
            dtype=int,
        )

        all_points = []
        for i in range(self._n_groups_cutoff_points):
            low, high = boundaries[i], boundaries[i + 1]
            size = (
                points_per_split
                if i < self._n_groups_cutoff_points - 1
                else (points_per_split - 1)
            )

            num_available = max(0, high - low)
            if num_available < size:
                self._logger.error(
                    f"Not enough points in range {low}-{high} to sample {size}.",
                )

            sampled_indices = np.random.choice(
                range(low, high),
                size=size,
                replace=False,
            )
            all_points.extend(sampled_indices)

        # Convert indices to datetime values
        all_points = sorted(unique_times[all_points])

        # Append the final cutoff datetime
        all_points.append(unique_times[-1])
        return all_points

    def _cutoff_points_all_coins(self) -> None:
        coins = self.df["coin"].unique()
        cutoff_points = {
            coin: self._random_cutoff_points(self.df[self.df["coin"] == coin])
            for coin in coins
        }
        self.cutoff_points = pd.DataFrame(
            [
                (coin, date)
                for coin, dates in cutoff_points.items()
                for date in dates
            ],
            columns=["coin", "cutoff_point"],
        )

    def _merge_features(self) -> None:
        def merge_features_row(row: pd.Series) -> np.ndarray:
            features = []
            for element in row:
                if isinstance(element, (list, np.ndarray)):
                    features.extend(element)
                else:
                    features.append(element)
            return np.array(features, np.float32)

        feature_columns = [
            col
            for col in self.df.columns
            if col not in ["coin", "datetime", "fraud"]
        ]
        self.df["merged_features"] = self.df[feature_columns].apply(
            merge_features_row,
            axis=1,
        )
        # Merged feature np.ndarrays must be of the same length!
        if len(set(map(len, self.df["merged_features"]))) > 1:
            self._logger.error(
                "Inconsistent feature lengths in 'merged_features'",
            )

    def __len__(self) -> int:
        # Return size of dataset
        return self.cutoff_points.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the coin and cutoff point for the given index
        coin_cutoff_point_combination = self.cutoff_points.iloc[index]
        coin = coin_cutoff_point_combination["coin"]
        cutoff_point = coin_cutoff_point_combination["cutoff_point"]

        # Get coin data before cut-off point
        filtered_df = self.df[
            (self.df["coin"] == coin) & (self.df["datetime"] < cutoff_point)
        ]
        if filtered_df.empty:
            self._logger.error(
                f"No data matched the filter coin: '{coin}' & date < '{cutoff_point}'",
            )

        # Get features 2d-array
        x = np.array(filtered_df["merged_features"].tolist(), dtype=np.float32)
        # Get target variable 1d-array
        y = filtered_df["fraud"].to_numpy(dtype=np.float32)

        # Convert to PyTorch tensors
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
