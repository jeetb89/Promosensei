import logging
from typing import Dict, List

from data_pipeline import process_offers
from llm_api import summarize_brand_offers, summarize_search_results, summarize_top_discounts
from rag_retrieval import retrieve_all, retrieve_for_query
from scraper import scrape_generic_offers
from vector_store import get_all_offers, store_offers

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def handle_offers(url: str) -> List[Dict]:
    raw = scrape_generic_offers(url)
    if not raw:
        return []
    clean = process_offers(raw)
    store_offers(clean)
    return clean


def query_promotions(user_query: str, top_k: int = 5) -> str:
    try:
        documents, metadatas = retrieve_for_query(user_query, top_k=top_k)
        if not documents:
            return "Sorry, I couldn't find any relevant promotions."
        return summarize_search_results(user_query, documents, metadatas)
    except Exception as e:
        logger.error(f"Error in query_promotions: {e}", exc_info=True)
        return "Oops! Something went wrong while searching for promotions."


def get_discounted_summary() -> str:
    try:
        documents, metadatas = retrieve_all()
        if not documents or not metadatas:
            return "❗ Sorry, no promotions found at the moment."

        promotions = []
        for doc, meta in zip(documents, metadatas):
            try:
                discount = float(meta.get("discount", 0))
                price = float(meta.get("price", 0))
                if discount <= 0 or price <= 0:
                    continue
                promotions.append({
                    "title": meta.get("title", doc)[:100],
                    "brand": meta.get("brand", "Unknown"),
                    "price": price,
                    "discount_pct": discount,
                    "link": meta.get("link", "#"),
                    "top_discount": meta.get("top_discount", False),
                })
            except Exception as e:
                logger.warning(f"Skipping malformed entry: {e}")

        if not promotions:
            return "❗ No valid promotions found with discounts."

        top_promotions = sorted(promotions, key=lambda x: x["discount_pct"], reverse=True)[:7]
        return summarize_top_discounts(top_promotions)

    except Exception as e:
        logger.error(f"Error in get_discounted_summary: {e}")
        return "⚠️ Oops! Too many promotions to summarize right now. Please try again later."


def filter_by_brand(brand: str, top_k: int = 5) -> Dict:
    try:
        results = get_all_offers()
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return {"offers": [], "summary": "❗ No data available in the database."}

        offers = [
            m for m in metadatas
            if isinstance(m, dict) and m.get("brand", "").lower() == brand.lower()
        ]

        if not offers:
            return {"offers": [], "summary": f"🔍 No offers found for brand: `{brand}`."}

        top_offers = sorted(
            [o for o in offers if float(o.get("discount", 0)) > 0],
            key=lambda x: float(x.get("discount", 0)),
            reverse=True,
        )[:top_k]

        summary = summarize_brand_offers(brand, top_offers)
        return {"offers": offers, "summary": summary}

    except Exception as e:
        logger.error(f"Error filtering by brand '{brand}': {e}")
        return {"offers": [], "summary": f"❌ Error retrieving promotions for brand `{brand}`."}


def refresh_data() -> str:
    return "Promotion database refreshed successfully!"
