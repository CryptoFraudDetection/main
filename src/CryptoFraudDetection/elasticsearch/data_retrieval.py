"""
File: data_retrieval.py

Description:
- This file is used to retrieve data from Elasticsearch.
"""

from CryptoFraudDetection.elasticsearch.elastic_client import get_elasticsearch_client


def search_data(index, q):
    es = get_elasticsearch_client()
    response = es.search(index=index, q=q)
    return response
