import os
import logging
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from scraper import scrape_nike_offers, push_to_chroma
from openai import OpenAI
from typing import List, Dict, Any,Tuple
import re
import numpy as np

load_dotenv()

# Configure logging for debug and error info
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables. Exiting.")
    raise EnvironmentError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

openai_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-ada-002"
)


try:
    chroma_client = Client(Settings())
except Exception as e:
    logger.error(f"Failed to initialize Chroma client: {e}")
    raise

try:
    collection = chroma_client.get_or_create_collection(name="promotions")
except Exception as e:
    logger.error(f"Failed to get or create Chroma collection: {e}")
    raise


def extract_brand(query: str, brands=["nike", "adidas", "reebok", "puma"]):
    query_lower = query.lower()
    for brand in brands:
        if brand in query_lower:
            return brand
    return None


def ingest_data() -> None:
    """
    Clears existing data in Chroma collection, scrapes fresh data, and ingests it.
    """
    try:
        all_items = collection.get()
        all_ids = all_items.get("ids", [])
        if all_ids:
            collection.delete(ids=all_ids)
            logger.info(f"Deleted {len(all_ids)} old items from ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to clear existing Chroma data: {e}")
        raise RuntimeError("Failed to clear ChromaDB data") from e

    try:
        offers = scrape_nike_offers()
        if not offers:
            logger.warning("No offers scraped from Nike.")
        push_to_chroma(offers)
        logger.info("Scraped offers pushed successfully to ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to scrape or push data to Chroma: {e}")
        raise RuntimeError("Failed to scrape or push data to Chroma") from e


def convert_embedding_to_list(embedding):
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    elif isinstance(embedding, list):
        return [convert_embedding_to_list(e) for e in embedding]
    else:
        return embedding
    

def query_promotions(user_query, top_k):
    try:
        embedded_query = openai_embed(user_query)  # could be nested with np.ndarray

        # Convert all np.ndarray in embedding to lists
        embedded_query = convert_embedding_to_list(embedded_query)

        # Ensure embeddings is a list of embeddings (list of lists of floats)
        # If embedding is a single embedding vector (list of floats), wrap in list
        if embedded_query and not isinstance(embedded_query[0], list):
            embedded_query = [embedded_query]

        results = collection.query(
            query_embeddings=embedded_query,
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        if not results.get("documents") or not results["documents"][0]:
            logger.warning("No documents returned from Chroma.")
            return []

        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            promo = {
                "title": doc,
                "brand": meta.get("brand"),
                "category": meta.get("category"),
                "expiry": meta.get("expiry"),
                "link": meta.get("link"),
                "current_price": meta.get("current_price"),
                "original_price": meta.get("original_price")
            }
            output.append(promo)

        return output

    except Exception as e:
        logger.error(f"Error in query_promotions: {e}")
        return []
    

def parse_price(price_str):
    if not price_str:
        return 0.0
    # Extract all digits and decimal points, ignoring spaces and non-numeric chars
    # This will find something like 5495.00 from "MRP : ₹ 5 495.00"
    price_str = str(price_str)
    matches = re.findall(r'[\d,.]+', price_str.replace(' ', ''))
    if not matches:
        return 0.0
    # Take the first match, replace commas if any, convert to float
    price_num = matches[0].replace(',', '')
    try:
        return float(price_num)
    except ValueError:
        return 0.0

def get_top_discounted_promotions(top_k=5):
    try:
        results = collection.get(include=["documents", "metadatas"])

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents or not metadatas:
            print("No documents or metadatas found!")
            return []

        docs_list = documents  # assuming single batch
        metas_list = metadatas

        promotions = []
        for doc, meta in zip(docs_list, metas_list):
            try:
                discount_raw = meta.get("discount", 0)
                current_price_raw = meta.get("current_price", 0)
                original_price_raw = meta.get("original_price", 0)

                discount = parse_price(discount_raw)
                current_price = parse_price(current_price_raw)
                original_price = parse_price(original_price_raw)

                print(f"Parsed prices - discount: {discount}, current: {current_price}, original: {original_price}")

                if discount <= 0 or current_price <= 0 or original_price <= 0:
                    print(f"Skipping invalid prices: discount={discount}, current={current_price}, original={original_price}")
                    continue

                promotions.append({
                    "title": doc,
                    "brand": meta.get("brand"),
                    "category": meta.get("category"),
                    "expiry": meta.get("expiry"),
                    "link": meta.get("link"),
                    "current_price": current_price,
                    "original_price": original_price,
                    "discount_pct": discount
                })
            except Exception as e:
                print(f"Warning: Skipping malformed entry: {e}")
                continue

        promotions.sort(key=lambda x: x["discount_pct"], reverse=True)
        return promotions[:top_k]

    except Exception as e:
        print(f"Error in get_top_discounted_promotions: {e}")
        return []


def generate_response(query: str, retrieved_texts: List[str]) -> str:
    """
    Generate a concise and friendly summary response based on retrieved promotional texts.
    """
    if not retrieved_texts:
        logger.warning("No retrieved texts to summarize.")
        return "Sorry, no relevant promotions were found."

    context_text = "\n".join(f"- {doc}" for doc in retrieved_texts)
    prompt = f"""
User asked: "{query}"
Relevant promotions found:
{context_text}

Please provide a concise, friendly summary of these promotions.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "Sorry, I couldn't generate a response at this time."


def format_top_deals(promotions):
    if not promotions:
        return "No top deals found."

    formatted = ""
    for promo in promotions:
        formatted += (
            f" *{promo['title']}* ({promo['brand']})\n"
            f" *Discount:* {promo['discount_pct']}%\n"
            f" *Price:* ₹{promo['current_price']} (was ₹{promo['original_price']})\n"
            f"🔗 {promo['link']}\n"
            f" *Expires:* {promo['expiry']}\n\n"
        )
    return formatted


def get_summary(text):

    prompt = (
        "Summarize the following top 5 promotional deals in a few sentences highlighting brands, discounts, and urgency:\n\n"
        f"{text}"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant summarizing shopping deals."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def refresh_data() -> str:
    """
    Refresh the promotions database by ingesting new data.
    """
    try:
        ingest_data()
        return "Promotion database refreshed successfully!"
    except Exception as e:
        logger.error(f"Failed to refresh promotion data: {e}")
        return "Failed to refresh promotion database."

    
def get_top_offers(documents: List[str], metadatas: List[dict], top_n: int = 10) -> List[str]:
    offers = []

    for doc, meta in zip(documents, metadatas):
        try:
            original_price = parse_price(meta.get("original_price", 0))
            current_price = parse_price(meta.get("current_price", 0))
            name = doc
            url = meta.get("link"),

            if original_price > current_price > 0:
                discount = original_price - current_price
                offers.append({
                    "name": name,
                    "current_price": current_price,
                    "original_price": original_price,
                    "discount": discount,
                    "url": url
                })
        except Exception:
            continue

    top_offers = sorted(offers, key=lambda x: x["discount"], reverse=True)[:top_n]

    if not top_offers:
      return ["No discounted offers found."]


    return (
     " *Top Nike Offers* \n\n" +
     "\n\n".join([
        f"{i+1}. *{offer['name']}*\n💰 ₹{int(offer['current_price'])} (was ₹{int(offer['original_price'])}, saved ₹{int(offer['discount'])})\n🔗 <{offer['url']}|View Product>"
        for i, offer in enumerate(top_offers)
     ])
    )

     
   

def filter_by_brand(brand: str) -> List[str]:
    """
    Filter promotions by brand name (case insensitive).
    Returns a list of top formatted offer strings for the brand.
    """
    try:
        results = collection.get(include=["documents", "metadatas"])

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents or not metadatas:
            logger.warning("No documents or metadatas found!")
            return ["No documents or metadatas found!"]

        brand_filtered_docs = []
        brand_filtered_meta = []

        for doc, meta in zip(documents, metadatas):
            if isinstance(meta, dict) and meta.get("brand", "").lower() == brand.lower():
                brand_filtered_docs.append(doc)
                brand_filtered_meta.append(meta)

        if not brand_filtered_docs:
            logger.info(f"No promotions found for brand: {brand}")
            return [f"No promotions found for brand: {brand}"]

        # Get top offers
        top_offers = get_top_offers(documents=brand_filtered_docs, metadatas=brand_filtered_meta)
        return top_offers

    except Exception as e:
        logger.error(f"Error filtering promotions by brand: {e}")
        return [f"Error retrieving promotions for brand: {brand}"]


def get_collection():
    return collection


if __name__ == "__main__":
    try:
        logger.info("Starting ingestion of promotion data...")
        ingest_data()
        logger.info("Ingestion complete.")

        user_query = "Looking for Nike shoes under 6000"
        logger.info(f"Querying promotions for: {user_query}")
        results = query_promotions(user_query)

        retrieved_texts = results.get('documents', [])
        if not retrieved_texts:
            logger.warning("No documents retrieved for the query.")
            print("No relevant promotions found.")
        else:
            answer = generate_response(user_query, retrieved_texts)
            print("AI Response:")
            print(answer)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print("An error occurred. Please check logs for details.")
