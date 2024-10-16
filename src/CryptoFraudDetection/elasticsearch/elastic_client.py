"""
File: elastic_client.py

Description:
- This file contains the Elasticsearch client.
"""

import os
import warnings

import urllib3
from elasticsearch import Elasticsearch

from CryptoFraudDetection.utils.exceptions import APIKeyNotSetException


# Our Tailscale connection is secure, so we can ignore these warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings(
    "ignore", message=".*using TLS with verify_certs=False is insecure"
)

ELASTICSEARCH_HOSTNAME = os.getenv("ELASTICSEARCH_HOSTNAME")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")


def get_elasticsearch_client() -> Elasticsearch:
    """
    Get the Elasticsearch client.

    Returns:
    - Elasticsearch: Elasticsearch client.
    """
    if ELASTICSEARCH_API_KEY == "changeme":
        raise APIKeyNotSetException()

    return Elasticsearch(
        hosts=ELASTICSEARCH_HOSTNAME,
        api_key=ELASTICSEARCH_API_KEY,
        verify_certs=False,
    )
