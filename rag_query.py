import os
import logging
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from scraper import scrape_generic_offers, push_to_chroma
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


def convert_embedding_to_list(embedding):
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    elif isinstance(embedding, list):
        return [convert_embedding_to_list(e) for e in embedding]
    else:
        return embedding
    

def query_promotions(user_query, top_k=5):
    try:
        logger.debug(f"Received user query: {user_query}")
        
        # Step 1: Embed query and retrieve from Chroma
        embedded_query = openai_embed(user_query)
        embedded_query = convert_embedding_to_list(embedded_query)
        logger.debug(f"Embedded query (post-conversion): {embedded_query}")
        
        if embedded_query and not isinstance(embedded_query[0], list):
            embedded_query = [embedded_query]

        results = collection.query(
            query_embeddings=embedded_query,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        logger.debug(f"Raw query results from Chroma: {results}")

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        logger.debug(f"Documents retrieved: {documents}")
        logger.debug(f"Metadatas retrieved: {metadatas}")

        if not documents:
            logger.warning("No documents returned from Chroma.")
            return "Sorry, I couldn't find any relevant promotions."

        # Defensive check on metadatas
        if not isinstance(metadatas, list):
            logger.error(f"Metadatas is not a list: {metadatas}")
            return "Sorry, promotion metadata is malformed."

        if len(metadatas) == 0:
            logger.warning("Metadatas list is empty.")
            return "Sorry, no metadata found for promotions."

        # Step 2: Format context for LLM using stored metadata keys
        context_lines = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            if not isinstance(meta, dict):
                logger.error(f"Metadata entry is not a dict: {meta}")
                continue
            context_lines.append(
                f"{i}. {meta.get('title')} from {meta.get('brand')} - Price: {meta.get('price')}, "
                f"Discount: {meta.get('discount')}, Link: {meta.get('link')}"
            )
        context_text = "\n".join(context_lines)
        logger.debug(f"Formatted context for LLM:\n{context_text}")

        # Step 3: Create LLM prompt
        prompt = f"""
You are Promo Sensei, a friendly and knowledgeable expert in finding the best product offers and deals.

A user is searching for: "{user_query}"

Below are some relevant promotions retrieved from the database:
{context_text}

Please provide a clear and concise summary of the best and most relevant offers based on this data. Include important details such as prices, discounts, and direct links to the offers. Use a helpful and engaging tone to guide the user toward the best deals.
"""

        logger.debug(f"Prompt sent to GPT:\n{prompt}")

        # Step 4: Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Promo Sensei, a helpful assistant specialized in recommending top product deals "},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        logger.debug("Received response from GPT")

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error in query_promotions with LLM: {e}", exc_info=True)
        return "Oops! Something went wrong while searching for promotions."

def get_discounted_summary():
    try:
        results = collection.get(include=["documents", "metadatas"])

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents or not metadatas:
            logger.warning("No documents or metadatas found in ChromaDB.")
            return "❗ Sorry, no promotions found at the moment."

        promotions = []
        for doc, meta in zip(documents, metadatas):
            try:
                discount = parse_price(meta.get("discount", 0))
                price = parse_price(meta.get("price", 0))

                if discount is None or price is None or discount <= 0 or price <= 0:
                    continue

                promotions.append({
                    "title": meta.get("title", doc)[:100],  # truncate long titles
                    "brand": meta.get("brand", "Unknown"),
                    "price": price,
                    "discount_pct": discount,
                    "link": meta.get("link", "#"),
                    "top_discount": meta.get("top_discount", False)
                })

            except Exception as e:
                logger.warning(f"Skipping malformed entry: {e}")
                continue

        if not promotions:
            return "❗ No valid promotions found with discounts."

        # Limit top 10 only to stay within token limits
        top_promotions = sorted(promotions, key=lambda x: x["discount_pct"], reverse=True)[:10]

        context_lines = []
        for i, p in enumerate(top_promotions, 1):
            context_lines.append(
                f"{i}. {p['title']} at {p['brand']} - ₹{p['price']} ({p['discount_pct']} off)\n"
                f"   🔗 Link: {p['link']}"
            )
        context_text = "\n".join(context_lines)

        prompt = f"""
You are Promo Sensei, a helpful assistant who summarizes the best deals.

Here are the top discounted promotions available:

{context_text}

Write a clear and friendly summary highlighting the best deals, why they’re good based on the discount, and include links. Be engaging, concise, and user-friendly.
"""

        logger.debug(f"Prompt sent to GPT:\n{prompt[:1000]}...")  # Log truncated prompt

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize best shopping offers and discounts for users in a helpful, concise tone."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        logger.debug("Received response from GPT")
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error in get_discounted_summary: {e}")
        return "⚠️ Oops! Too many promotions to summarize right now. Please try again later with fewer results."


def parse_discount(discount_str):
    try:
        if discount_str.endswith('%'):
            return float(discount_str.strip('%'))
        return None
    except Exception as e:
        logger.warning(f"Failed to parse discount value '{discount_str}': {e}")
        return None


def parse_price(value):
    try:
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return None

        # Remove currency symbols and common phrases
        cleaned = value.strip().replace('₹', '').replace(',', '')
        # Ignore ranges or non-numeric text
        if any(keyword in cleaned.lower() for keyword in ['under', 'over', 'less', 'more', 'n/a', 'na', 'free', 'varies']):
            return None

        # Extract the first number (handles cases like '₹500 off' or 'Up to 75%')
        match = re.search(r"\d+(\.\d+)?", cleaned)
        return float(match.group()) if match else None

    except Exception as e:
        logger.warning(f"Failed to parse price value '{value}': {e}")
        return None

def filter_by_brand(brand: str, top_k: int = 5) -> Dict[str, any]:
    """
    Return all structured offers for the given brand and an LLM-generated summary of the top ones.
    """
    def parse_discount(value):
        try:
            if isinstance(value, str):
                return float(value.replace('%', '').strip())
            return float(value)
        except Exception as e:
            logger.warning(f"Failed to parse discount value '{value}': {e}")
            return 0.0

    def parse_price(value):
        try:
            if isinstance(value, str):
                return float(value.replace('₹', '').replace(',', '').strip())
            return float(value)
        except Exception as e:
            logger.warning(f"Failed to parse price value '{value}': {e}")
            return 0.0

    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        if not metadatas:
            logger.warning("No metadata found in ChromaDB!")
            return {"offers": [], "summary": "❗ No data available in the database."}

        offers = []
        for meta in metadatas:
            if isinstance(meta, dict) and meta.get("brand", "").lower() == brand.lower():
                offers.append(meta)

        if not offers:
            return {
                "offers": [],
                "summary": f"🔍 No offers found for brand: `{brand}`."
            }

        # Select top_k by discount for summary
        sorted_offers = sorted(
            [o for o in offers if parse_discount(o.get("discount", 0)) > 0],
            key=lambda x: parse_discount(x.get("discount", 0)),
            reverse=True
        )
        top_offers = sorted_offers[:top_k]

        # Format for LLM prompt
        context_lines = []
        for i, o in enumerate(top_offers, 1):
            price = parse_price(o.get("price", 0))
            discount = parse_discount(o.get("discount", 0))
            context_lines.append(
                f"{i}. {o.get('title')} at {o.get('brand')} - ₹{price} ({discount} off)\n"
                f"   🔗 Link: {o.get('link')}"
            )
        context_text = "\n".join(context_lines)

        prompt = f"""
You are Promo Sensei, a smart shopping assistant.

Here are some great deals from `{brand}`:

{context_text}

Write a short, engaging summary that highlights why these offers are valuable. Mention discounts, prices, and encourage users to explore the deals.
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize shopping offers with clarity and persuasion."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()

        return {
            "offers": offers,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error retrieving offers for brand '{brand}': {e}")
        return {
            "offers": [],
            "summary": f"❌ Error retrieving promotions for brand `{brand}`."
        }

def refresh_data() -> str:
    """
    Refresh the promotions database by ingesting new data.
    """
    try:
        # ingest_data()
        return "Promotion database refreshed successfully!"
    except Exception as e:
        logger.error(f"Failed to refresh promotion data: {e}")
        return "Failed to refresh promotion database."

def handleOffers(url):
   return scrape_generic_offers(url)
     










# def parse_price(price_str):
#     if not price_str:
#         return 0.0
#     # Extract all digits and decimal points, ignoring spaces and non-numeric chars
#     # This will find something like 5495.00 from "MRP : ₹ 5 495.00"
#     price_str = str(price_str)
#     matches = re.findall(r'[\d,.]+', price_str.replace(' ', ''))
#     if not matches:
#         return 0.0
#     # Take the first match, replace commas if any, convert to float
#     price_num = matches[0].replace(',', '')
#     try:
#         return float(price_num)
#     except ValueError:
#         return 0.0

# # def get_top_discounted_promotions(top_k=5):
# #     try:
# #         results = collection.get(include=["documents", "metadatas"])

# #         documents = results.get("documents", [])
# #         metadatas = results.get("metadatas", [])

# #         if not documents or not metadatas:
# #             print("No documents or metadatas found!")
# #             return []

# #         docs_list = documents  # assuming single batch
# #         metas_list = metadatas

# #         promotions = []
# #         for doc, meta in zip(docs_list, metas_list):
# #             try:
# #                 discount_raw = meta.get("discount", 0)
# #                 current_price_raw = meta.get("current_price", 0)
# #                 original_price_raw = meta.get("original_price", 0)

# #                 discount = parse_price(discount_raw)
# #                 current_price = parse_price(current_price_raw)
# #                 original_price = parse_price(original_price_raw)

# #                 print(f"Parsed prices - discount: {discount}, current: {current_price}, original: {original_price}")

# #                 if discount <= 0 or current_price <= 0 or original_price <= 0:
# #                     print(f"Skipping invalid prices: discount={discount}, current={current_price}, original={original_price}")
# #                     continue

# #                 promotions.append({
# #                     "title": doc,
# #                     "brand": meta.get("brand"),
# #                     "category": meta.get("category"),
# #                     "expiry": meta.get("expiry"),
# #                     "link": meta.get("link"),
# #                     "current_price": current_price,
# #                     "original_price": original_price,
# #                     "discount_pct": discount
# #                 })
# #             except Exception as e:
# #                 print(f"Warning: Skipping malformed entry: {e}")
# #                 continue

# #         promotions.sort(key=lambda x: x["discount_pct"], reverse=True)
# #         return promotions[:top_k]

# #     except Exception as e:
# #         print(f"Error in get_top_discounted_promotions: {e}")
# #         return []


# def generate_response(query: str, retrieved_texts: List[str]) -> str:
#     """
#     Generate a concise and friendly summary response based on retrieved promotional texts.
#     """
#     if not retrieved_texts:
#         logger.warning("No retrieved texts to summarize.")
#         return "Sorry, no relevant promotions were found."

#     context_text = "\n".join(f"- {doc}" for doc in retrieved_texts)
#     prompt = f"""
# User asked: "{query}"
# Relevant promotions found:
# {context_text}

# Please provide a concise, friendly summary of these promotions.
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=200,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"Failed to generate response: {e}")
#         return "Sorry, I couldn't generate a response at this time."


# def format_top_deals(promotions):
#     if not promotions:
#         return "No top deals found."

#     formatted = ""
#     for promo in promotions:
#         formatted += (
#             f" *{promo['title']}* ({promo['brand']})\n"
#             f" *Discount:* {promo['discount_pct']}%\n"
#             f" *Price:* ₹{promo['current_price']} (was ₹{promo['original_price']})\n"
#             f"🔗 {promo['link']}\n"
#             f" *Expires:* {promo['expiry']}\n\n"
#         )
#     return formatted


# def get_summary(text):

#     prompt = (
#         "Summarize the following top 5 promotional deals in a few sentences highlighting brands, discounts, and urgency:\n\n"
#         f"{text}"
#     )

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant summarizing shopping deals."},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=150,
#         temperature=0.7,
#     )

#     return response.choices[0].message.content.strip()
    
# def get_top_offers(documents: List[str], metadatas: List[dict], top_n: int = 10) -> List[str]:
#     offers = []

#     for doc, meta in zip(documents, metadatas):
#         try:
#             original_price = parse_price(meta.get("original_price", 0))
#             current_price = parse_price(meta.get("current_price", 0))
#             name = doc
#             url = meta.get("link"),

#             if original_price > current_price > 0:
#                 discount = original_price - current_price
#                 offers.append({
#                     "name": name,
#                     "current_price": current_price,
#                     "original_price": original_price,
#                     "discount": discount,
#                     "url": url
#                 })
#         except Exception:
#             continue

#     top_offers = sorted(offers, key=lambda x: x["discount"], reverse=True)[:top_n]

#     if not top_offers:
#       return ["No discounted offers found."]


#     return (
#      " *Top Nike Offers* \n\n" +
#      "\n\n".join([
#         f"{i+1}. *{offer['name']}*\n💰 ₹{int(offer['current_price'])} (was ₹{int(offer['original_price'])}, saved ₹{int(offer['discount'])})\n🔗 <{offer['url']}|View Product>"
#         for i, offer in enumerate(top_offers)
#      ])
#     )


# # def filter_by_brand(brand: str) -> List[str]:
# #     """
# #     Filter promotions by brand name (case insensitive).
# #     Returns a list of top formatted offer strings for the brand.
# #     """
# #     try:
# #         results = collection.get(include=["documents", "metadatas"])

# #         documents = results.get("documents", [])
# #         metadatas = results.get("metadatas", [])

# #         if not documents or not metadatas:
# #             logger.warning("No documents or metadatas found!")
# #             return ["No documents or metadatas found!"]

# #         brand_filtered_docs = []
# #         brand_filtered_meta = []

# #         for doc, meta in zip(documents, metadatas):
# #             if isinstance(meta, dict) and meta.get("brand", "").lower() == brand.lower():
# #                 brand_filtered_docs.append(doc)
# #                 brand_filtered_meta.append(meta)

# #         if not brand_filtered_docs:
# #             logger.info(f"No promotions found for brand: {brand}")
# #             return [f"No promotions found for brand: {brand}"]

# #         # Get top offers
# #         top_offers = get_top_offers(documents=brand_filtered_docs, metadatas=brand_filtered_meta)
# #         return top_offers

# #     except Exception as e:
# #         logger.error(f"Error filtering promotions by brand: {e}")
# #         return [f"Error retrieving promotions for brand: {brand}"]


# def print_existing_offers(collection_name="promotions", limit=10):
#     client = Client()
#     collection = client.get_or_create_collection(name=collection_name)

#     # peek() returns a dict of lists: 'ids', 'documents', 'metadatas'
#     items = collection.peek(limit=limit)

#     ids = items.get("ids", [])
#     documents = items.get("documents", [])
#     metadatas = items.get("metadatas", [])

#     print(f"Showing up to {limit} offers in collection '{collection_name}':")

#     for i in range(len(ids)):
#         print(f"\nOffer {i+1}:")
#         print(f"  ID: {ids[i]}")
#         print(f"  Text: {documents[i]}")
#         print(f"  Metadata: {metadatas[i]}")

# def get_collection():
#     return collection  # Make sure 'collection' is defined somewhere





# def extract_brand(query: str, brands=["nike", "adidas", "reebok", "puma"]):
#     query_lower = query.lower()
#     for brand in brands:
#         if brand in query_lower:
#             return brand
#     return None


# def ingest_data() -> None:
#     """
#     Clears existing data in Chroma collection, scrapes fresh data, and ingests it.
#     """
#     try:
#         all_items = collection.get()
#         all_ids = all_items.get("ids", [])
#         if all_ids:
#             collection.delete(ids=all_ids)
#             logger.info(f"Deleted {len(all_ids)} old items from ChromaDB.")
#     except Exception as e:
#         logger.error(f"Failed to clear existing Chroma data: {e}")
#         raise RuntimeError("Failed to clear ChromaDB data") from e

#     try:
#         offers = scrape_generic_offers()
#         if not offers:
#             logger.warning("No offers scraped from Nike.")
#         push_to_chroma(offers)
#         logger.info("Scraped offers pushed successfully to ChromaDB.")
#     except Exception as e:
#         logger.error(f"Failed to scrape or push data to Chroma: {e}")
#         raise RuntimeError("Failed to scrape or push data to Chroma") from e


# if __name__ == "__main__":
#     offer = get_collection()
#     print_existing_offers()

#     # try:
#     #     logger.info("Starting ingestion of promotion data...")
#     #     ingest_data()
#     #     logger.info("Ingestion complete.")

#     #     user_query = "Looking for Nike shoes under 6000"
#     #     logger.info(f"Querying promotions for: {user_query}")
#     #     results = query_promotions(user_query)

#     #     retrieved_texts = results.get('documents', [])
#     #     if not retrieved_texts:
#     #         logger.warning("No documents retrieved for the query.")
#     #         print("No relevant promotions found.")
#     #     else:
#     #         answer = generate_response(user_query, retrieved_texts)
#     #         print("AI Response:")
#     #         print(answer)

#     # except Exception as e:
#     #     logger.error(f"Error in main execution: {e}")
#     #     print("An error occurred. Please check logs for details.")
