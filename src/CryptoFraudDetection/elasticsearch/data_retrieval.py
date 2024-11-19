"""
File: data_retrieval.py

Description:
- This file is used to retrieve data from Elasticsearch.
"""

from typing import Any
from elastic_transport import ObjectApiResponse
from CryptoFraudDetection.elasticsearch.elastic_client import get_elasticsearch_client

es = get_elasticsearch_client()


def search_data(index: str, q: str, size: int | None = None) -> ObjectApiResponse[Any]:
    """
    Search data in Elasticsearch.

    Args:
    - index (str): Index name.
    - q (str): Query string.

    Returns:
    - dict: Elasticsearch search results.
    """
    if size:
        return es.search(index=index, q=q, size=size)
    return es.search(index=index, q=q, size=10000)
