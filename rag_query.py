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

Please write a concise and beautifully formatted summary of the best deals. Use bullet points, emojis, and short sentences. Highlight the biggest discounts and include the product name, brand, price, discount, and a link for each. Be engaging and friendly.
"""

        logger.debug(f"Prompt sent to GPT:\n{prompt}")

        # Step 4: Get response from GPT
        response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are Promo Sensei, a helpful assistant specialized in recommending top product deals. Provide the results in a friendly tone using markdown formatting with emojis and links."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.7,
    # max_tokens=1000  # Optional: Helps avoid token limit errors
)

        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are Promo Sensei, a helpful assistant specialized in recommending top product deals. Provide the results in a beautiful format"},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.7
        # )
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
                    "title": meta.get("title", doc)[:100],
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

        # Limit top 7 promotions to stay within token limits
        top_promotions = sorted(promotions, key=lambda x: x["discount_pct"], reverse=True)[:7]

        # Build context lines
        context_lines = []
        for i, p in enumerate(top_promotions, 1):
            context_lines.append(
                f"{i}. {p['title']} at {p['brand']} - ₹{p['price']} ({p['discount_pct']} off)\n"
                f"   🔗 Link: {p['link']}"
            )
        context_text = "\n".join(context_lines)

        # GPT prompt
        prompt = f"""
You are Promo Sensei, a helpful assistant who summarizes the best deals.

Here are the top discounted promotions available:

{context_text}

Write a clear, friendly, and beautifully formatted summary of the best deals.
Use:
- Bullet points ✅
- Emojis 🎉
- Short, punchy sentences ✍️
- Include links 🔗
Focus on why the deal is great and keep it fun!
"""

        logger.debug(f"Prompt sent to GPT:\n{prompt[:1000]}...")  # Truncated log

        # GPT call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are Promo Sensei, a friendly assistant that summarizes top product deals using markdown, emojis, and a fun tone. Each deal should be clear and include title, brand, price, discount, and link."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800
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
                f"{i}. {o.get('title')} at {o.get('brand')} - ₹{price} ({discount}% off)\n"
                f"   🔗 Link: {o.get('link')}"
            )
        context_text = "\n".join(context_lines)

        prompt = f"""
You are Promo Sensei, a smart shopping assistant who crafts engaging summaries of the best deals.

Below are top promotions from the brand `{brand}`:

{context_text}

Please write a well-formatted, friendly summary that:
- Highlights the biggest discounts 💸
- Mentions price and brand clearly 🏷️
- Encourages users to check them out 🔗
- Uses bullet points ✅
- Includes emojis 🎉
- Keeps sentences short, energetic, and persuasive ✍️

Make it visually easy to scan and fun to read. Format the response using markdown so it looks great in a chat app like Slack.
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize shopping offers with clarity and persuasion using markdown, emojis, and short sentences."},
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
    try:
        # ingest_data()
        return "Promotion database refreshed successfully!"
    except Exception as e:
        logger.error(f"Failed to refresh promotion data: {e}")
        return "Failed to refresh promotion database."

def handleOffers(url):
   return scrape_generic_offers(url)
     




