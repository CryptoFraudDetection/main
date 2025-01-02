"""
File: data_insertion.py.

Description:
    This file is used to insert data into Elasticsearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from elasticsearch.helpers import BulkIndexError, bulk

from CryptoFraudDetection.elasticsearch.elastic_client import (
    get_elasticsearch_client,
)

if TYPE_CHECKING:
    from CryptoFraudDetection.utils.logger import Logger

es = get_elasticsearch_client()


def insert_dataframe(
    logger: Logger,
    index: str,
    df: pd.DataFrame,
) -> tuple[int, int | list[dict[str, list]]]:
    """
    Insert a pandas DataFrame into Elasticsearch.

    Attributes:
        logger (Logger): The logger object.
        index (str): The name of the Elasticsearch index to insert into.
        df (pd.DataFrame): The DataFrame to insert.

    Returns:
        response (tuple[int, int | list[dict[str, list]]]): The response from the Elasticsearch bulk insertion.

    """
    if "id" in df.columns:
        data = [
            {
                "_index": index,
                "_id": record["id"],
                "_op_type": "create",
                "_source": record,
            }
            for record in df.to_dict(orient="records")
        ]
    else:
        data = [
            {
                "_index": index,
                "_source": record,
            }
            for record in df.to_dict(orient="records")
        ]

    try:
        logger.info(f"Inserting {len(data)} records into {index}...")
        response = bulk(client=es, actions=data, raise_on_error=False)
    except BulkIndexError as e:
        logger.handle_exception(
            BulkIndexError,
            f"Skipped some documents:\n{e}",
        )
    return response


def insert_dict(
    logger: Logger,
    index: str,
    data_dict: dict[str, list[str]],
) -> tuple[int, int | list[dict[str, Any]]]:
    """
    Insert a list of dictionaries into Elasticsearch.

    Attributes:
        logger (Logger): The logger object.
        index (str): The name of the Elasticsearch index to insert into.
        data_dict (dict[str, list[str]]): The dictionary to insert.

    Returns:
        response (tuple[int, int | list[dict[str, Any]]): The response from the Elasticsearch bulk insertion.

    """
    data_frame = pd.DataFrame(data_dict)
    return insert_dataframe(logger, index, data_frame)
