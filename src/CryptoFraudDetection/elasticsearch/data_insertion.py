"""
File: data_insertion.py

Description:
- This file is used to insert data into Elasticsearch.
"""

import pandas as pd
from elasticsearch.helpers import bulk

from CryptoFraudDetection.elasticsearch.elastic_client import get_elasticsearch_client


es = get_elasticsearch_client()


def insert_dataframe(index: str, df: pd.DataFrame) -> dict:
    """
    Insert a pandas DataFrame into Elasticsearch.

    Args:
    - index (str): The Elasticsearch index to insert the data into.
    - df (pd.DataFrame): The DataFrame to insert.

    Returns:
    - dict: Elasticsearch bulk insert response.
    """
    data = df.to_dict(orient="records")
    return bulk(client=es, actions=data, index=index)
