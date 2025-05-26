import logging
from typing import List, Dict, Any
import chromadb
from openai import OpenAI
import os

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str) -> List[float]:
    """
    Embeds the given text using OpenAI's text-embedding-ada-002 model.
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        return []


def ingest_offers(
    offers: List[Dict[str, Any]],
    collection_name: str = "offers"
) -> None:
    """
    Ingests a list of offer dictionaries into in-memory ChromaDB with OpenAI embeddings.
    Each offer must include: title, description, brand, category, expiry, link.
    """
    if not offers:
        logger.warning("No offers provided for ingestion.")
        return

    required_keys = {"title", "description", "brand", "category", "expiry", "link"}
    valid_offers = [offer for offer in offers if required_keys.issubset(offer.keys())]

    if not valid_offers:
        logger.error("No valid offers to ingest after validation.")
        return

    logger.info(f"Preparing to ingest {len(valid_offers)} offers into ChromaDB collection '{collection_name}'.")

    # Initialize ChromaDB client (in-memory)
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    for idx, offer in enumerate(valid_offers):
        doc_text = f"{offer['title']} - {offer['description']}"
        embedding = embed_text(doc_text)

        if not embedding:
            logger.warning(f"Skipping offer at index {idx} due to failed embedding.")
            continue

        try:
            collection.add(
                documents=[doc_text],
                embeddings=[embedding],
                metadatas=[{
                    "brand": offer["brand"],
                    "category": offer["category"],
                    "expiry": offer["expiry"],
                    "link": offer["link"]
                }],
                ids=[f"offer-{idx}"]
            )
        except Exception as e:
            logger.error(f"Failed to add offer {idx} to ChromaDB: {e}")

    logger.info(f"✅ Successfully ingested {len(valid_offers)} offers into ChromaDB.")


if __name__ == "__main__":
    example_offers = [
        {
            "title": "Puma Velocity Nitro at 50% Off",
            "description": "Running shoes from Puma at half price.",
            "brand": "Puma",
            "category": "Footwear",
            "expiry": "2025-06-01",
            "link": "https://www.puma.com/in/en/deals"
        },
        {
            "title": "Nykaa Summer Sale - Flat 40%",
            "description": "Skincare and makeup deals this summer.",
            "brand": "Nykaa",
            "category": "Beauty",
            "expiry": "2025-06-10",
            "link": "https://www.nykaa.com/offers"
        }
    ]

    ingest_offers(example_offers)
