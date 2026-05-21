import hashlib
import logging
import os
from typing import Dict, List

import chromadb

from embedding_utils import embed_text

logger = logging.getLogger(__name__)

COLLECTION_NAME = "promotions"
_DB_PATH = "./chroma_store"


def _client() -> chromadb.ClientAPI:
    host = os.getenv("CHROMA_HOST")
    if host:
        port = int(os.getenv("CHROMA_PORT", "8000"))
        logger.info(f"Connecting to ChromaDB server at {host}:{port}")
        return chromadb.HttpClient(host=host, port=port)
    logger.info(f"Using local ChromaDB at {_DB_PATH}")
    return chromadb.PersistentClient(path=_DB_PATH)


def _collection(name: str = COLLECTION_NAME):
    return _client().get_or_create_collection(name=name)


def store_offers(offers: List[Dict], collection_name: str = COLLECTION_NAME) -> None:
    if not offers:
        logger.warning("No offers to store.")
        return

    col = _collection(collection_name)
    texts, metadatas, embeddings, ids = [], [], [], []

    for offer in offers:
        text = (
            f"{offer['title']} from {offer['brand']} "
            f"- Price: {offer['price']}, Discount: {offer['discount']}"
        )
        offer_id = hashlib.md5(
            f"{offer['title']}_{offer['price']}_{offer['link']}".encode()
        ).hexdigest()

        texts.append(text)
        embeddings.append(embed_text(text))
        ids.append(offer_id)
        metadatas.append({
            "title": offer["title"],
            "price": offer["price"],
            "discount": offer["discount"],
            "link": offer["link"],
            "brand": offer["brand"],
            "top_discount": offer["top_discount"],
        })

    existing = set(col.get(ids=ids)["ids"])
    new_idx = [i for i, oid in enumerate(ids) if oid not in existing]

    if new_idx:
        col.add(
            documents=[texts[i] for i in new_idx],
            metadatas=[metadatas[i] for i in new_idx],
            embeddings=[embeddings[i] for i in new_idx],
            ids=[ids[i] for i in new_idx],
        )
        logger.info(f"Stored {len(new_idx)} new offers in ChromaDB.")
    else:
        logger.info("No new offers; all already present in ChromaDB.")


def get_all_offers(collection_name: str = COLLECTION_NAME) -> Dict:
    return _collection(collection_name).get(include=["documents", "metadatas"])


def search_offers(
    query_embedding: List[float],
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> Dict:
    return _collection(collection_name).query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
