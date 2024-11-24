"""File: data_insertion.py.

Description:
- This file is used to insert data into Elasticsearch.
"""

from typing import Any

import pandas as pd
from elasticsearch.helpers import BulkIndexError, bulk

from CryptoFraudDetection.elasticsearch.elastic_client import (
    get_elasticsearch_client,
)
from CryptoFraudDetection.utils.logger import Logger

es = get_elasticsearch_client()


def insert_dataframe(
    logger: Logger,
    index: str,
    df: pd.DataFrame,
) -> tuple[int, int | list[dict[str, list]]]:
    """Insert a pandas DataFrame into Elasticsearch.

    Args:
    - logger (Logger): The logger to use for logging.
    - index (str): The Elasticsearch index to insert the data into.
    - df (pd.DataFrame): The DataFrame to insert.

    Returns:
    - dict: Elasticsearch bulk insert response.

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
        success_fail = bulk(client=es, actions=data, raise_on_error=False)
    except BulkIndexError as e:
        logger.handle_exception(
            BulkIndexError,
            f"Skipped some documents:\n{e}",
        )
    return success_fail


def insert_dict(
    logger: Logger,
    index: str,
    data_dict: dict[str, list[str]],
) -> tuple[int, int | list[dict[str, Any]]]:
    """Insert a list of dictionaries into Elasticsearch.

    Args:
    - logger (Logger): The logger to use for logging.
    - index (str): The Elasticsearch index to insert the data into.
    - data_dict (List[Dict[str, Any]]): The list of dictionaries to insert.

    Returns:
    - Tuple[int, int | List[Dict[str, Any]]]: Elasticsearch bulk insert response.

    """
    data_frame = pd.DataFrame(data_dict)
    return insert_dataframe(logger, index, data_frame)
