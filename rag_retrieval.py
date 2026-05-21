import logging
from typing import Dict, List, Tuple

from embedding_utils import embed_text
from vector_store import get_all_offers, search_offers

logger = logging.getLogger(__name__)


def retrieve_for_query(user_query: str, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
    embedding = embed_text(user_query)
    results = search_offers(embedding, top_k=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not isinstance(metadatas, list):
        logger.error(f"Unexpected metadatas type: {type(metadatas)}")
        return [], []

    logger.info(f"Retrieved {len(documents)} documents for query: '{user_query}'")
    return documents, metadatas


def retrieve_all() -> Tuple[List[str], List[Dict]]:
    results = get_all_offers()
    return results.get("documents", []), results.get("metadatas", [])
