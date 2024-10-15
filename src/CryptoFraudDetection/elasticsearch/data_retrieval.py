"""
File: data_retrieval.py

Description:
- This file is used to retrieve data from Elasticsearch.
"""

from CryptoFraudDetection.elasticsearch.elastic_client import get_elasticsearch_client

es = get_elasticsearch_client()


def search_data(index, q):
    return es.search(index=index, q=q)
