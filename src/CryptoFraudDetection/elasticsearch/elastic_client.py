"""File: elastic_client.py.

Description:
- This file contains the Elasticsearch client.
"""

import os
import warnings

import urllib3
from dotenv import find_dotenv, load_dotenv
from elasticsearch import Elasticsearch

from CryptoFraudDetection.utils.exceptions import APIKeyNotSetException

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path and os.getenv("ELASTICSEARCH_API_KEY") is None:
    load_dotenv(dotenv_path)

ELASTICSEARCH_HOSTNAME = os.getenv("ELASTICSEARCH_HOSTNAME")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

# Our Tailscale connection is secure, so we can ignore these warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings(
    "ignore",
    message=".*using TLS with verify_certs=False is insecure",
)


def get_elasticsearch_client(timeout: int = 60) -> Elasticsearch:
    """Get the Elasticsearch client.

    Returns:
    - Elasticsearch: Elasticsearch client.

    """
    if ELASTICSEARCH_API_KEY in ("changeme", None):
        raise APIKeyNotSetException

    return Elasticsearch(
        hosts=ELASTICSEARCH_HOSTNAME,
        api_key=ELASTICSEARCH_API_KEY,
        verify_certs=False,
        timeout=timeout
    )
