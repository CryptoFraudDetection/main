"""
File: data_retrieval.py.

Description:
    This file is used to retrieve data from Elasticsearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from CryptoFraudDetection.elasticsearch.elastic_client import (
    get_elasticsearch_client,
)

if TYPE_CHECKING:
    from elastic_transport import ObjectApiResponse

es = get_elasticsearch_client()


def search_data(
    index: str,
    q: str,
    size: int | None = None,
) -> ObjectApiResponse[Any] | dict[str, Any]:
    """
    Search data in Elasticsearch using the Scroll API if necessary.

    Attributes:
        index (str): The name of the Elasticsearch index to search.
        q (str): The query string to search.
        size (int): The number of results to return.

    Returns:
        response (ObjectApiResponse[Any] | dict[str, Any]): The response from the Elasticsearch search.

    """
    if not size or size <= 10000:
        return es.search(index=index, q=q, size=size) if size else es.search(index=index, q=q, size=10000)

    # use scroll API for large result sets
    scroll = "2m"
    batch_size = 1000
    total_size = size

    response = es.search(index=index, q=q, scroll=scroll, size=batch_size)

    sid = response["_scroll_id"]
    hits = response["hits"]["hits"]
    all_hits = hits.copy()

    while len(hits) > 0 and len(all_hits) < total_size:
        response = es.scroll(scroll_id=sid, scroll=scroll)
        sid = response["_scroll_id"]
        hits = response["hits"]["hits"]
        all_hits.extend(hits)

        if len(all_hits) >= total_size:
            break

    es.clear_scroll(scroll_id=sid)

    all_hits = all_hits[:total_size]

    return {
        "took": response["took"],
        "timed_out": response["timed_out"],
        "_shards": response["_shards"],
        "hits": {
            "total": {"value": len(all_hits), "relation": "eq"},
            "max_score": None,
            "hits": all_hits,
        },
    }
