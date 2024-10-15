"""
File: data_insertion.py

Description:
- This file is used to insert data into Elasticsearch.
"""

from CryptoFraudDetection.elasticsearch.elastic_client import get_elasticsearch_client

from elasticsearch.helpers import bulk

es = get_elasticsearch_client()


def insert_dataframe(index, df):
    data = df.to_dict(orient="records")
    return bulk(client=es, actions=data, index=index)
