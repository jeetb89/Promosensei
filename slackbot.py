import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from rag_query import (
    query_promotions,
    get_summary,
    refresh_data,
    filter_by_brand,
    get_top_discounted_promotions,
    format_top_deals,
    get_collection,
    
)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokens
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise EnvironmentError("SLACK_BOT_TOKEN or SLACK_APP_TOKEN is not set in the environment.")

# Initialize Slack Bolt app
app = App(token=SLACK_BOT_TOKEN)

# --- Command Handlers ---

@app.command("/promosensei-search")
def handle_search(ack, respond, command):
    ack()
    try:
        # Debug log the full incoming command
        logger.info(f"Received command: {command}")

        # Get and validate the text field
        user_query_raw = command.get('text')

        if not isinstance(user_query_raw, str):
            respond("⚠️ Invalid query format. Please enter text only.")
            logger.error(f"Expected string in command['text'], got {type(user_query_raw)}")
            return

        user_query = user_query_raw.strip()
        logger.info(f"Received search query: {user_query}")

        if not user_query:
            respond("❗ Please enter a search query.")
            return

        # Call the promotions search
        semantic_query, filters = parse_query(user_query)
        results = query_promotions(semantic_query, top_k=20)

        final_results = apply_filters(results, filters)


        logger.debug(f"Query results: {final_results}")

        # Format and respond
        formatted = format_response(final_results)
        respond(f"🔎 *Results for:* `{user_query}`\n{formatted}")

    except Exception as e:
        logger.exception("Search command failed:")
        respond(f"❌ Error during search: {str(e)}")


@app.command("/promosensei-summary")
def handle_summary(ack, respond):
    ack()
    try:

        promotions = get_top_discounted_promotions()
        if not promotions:
            respond("ℹ️ No promotions found in the database.")
            return

        # Format deals into readable string
        formatted_deals = format_top_deals(promotions)

        # Use AI to generate summary of just these deals
        ai_summary = get_summary(formatted_deals)

        final_message = (
            f"🧠 *AI Summary of Top 5 Deals:*\n{ai_summary}\n\n"
            f"🔥 *Top 5 Deals by Discount:*\n{formatted_deals}"
        )

        respond(final_message)

    except Exception as e:
        logger.error(f"Summary command failed: {e}")
        respond(f"❌ Error generating summary: {str(e)}")


@app.command("/promosensei-brand")
def handle_brand_filter(ack, respond, command):
    ack()
    try:
        brand = command['text'].strip()
        if not brand:
            respond("❗ Please specify a brand name.")
            return

        results = filter_by_brand(brand)
        logger.info(f"Filtered results for brand '{brand}': {results}")

        respond(f"🏷️ *Offers for:* `{brand}`\n{results}")
    except Exception as e:
        logger.error(f"Brand filter command failed: {e}")
        respond(f"❌ Error filtering by brand: {str(e)}")


@app.command("/promosensei-refresh")
def handle_refresh(ack, respond):
    ack()
    try:
        logger.info("Refreshing data...")
        refresh_data()
        respond("🔄 *Data refreshed successfully!*")
    except Exception as e:
        logger.error(f"Refresh command failed: {e}")
        respond(f"❌ Error refreshing data: {str(e)}")


# --- Helper ---
def format_response(results):
    if not results:
        return "No promotions found."

    try:
        formatted = []
        for promo in results:
            formatted.append(
                f"• *{promo.get('title', 'No Title')}*\n"
                f"  *Brand*: {promo.get('brand', '-')}\n"
                f"  *Category*: {promo.get('category', '-')}\n"
                f"  *Price*: {promo.get('current_price', '-')}/{promo.get('original_price', '-')}\n"
                f"  *Expires*: {promo.get('expiry', '-')}\n"
                f"  <{promo.get('link', '')}|View Offer>"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return "⚠️ Error formatting promotions."



def parse_query(user_query):
    filters = {}

    # Example regex to extract price filter
    price_match = re.search(r"(under|below|less than|<)\s*\$?(\d+)", user_query, re.I)
    if price_match:
        filters['max_price'] = float(price_match.group(2))

    # Extract brand if any known brand mentioned
    brands = ["nike", "adidas", "reebok", "puma"]
    for brand in brands:
        if brand.lower() in user_query.lower():
            filters['brand'] = brand
            break

    # Remove filter parts from query to get clean semantic part
    semantic_query = re.sub(r"(under|below|less than|<)\s*\$?\d+", "", user_query, flags=re.I).strip()

    return semantic_query, filters

def apply_filters(results, filters):
    filtered = []
    for item in results:
        # Filter by price
        if 'max_price' in filters:
            price_str = item.get("current_price")
            if price_str:
                try:
                    price_val = float(re.sub(r'[^\d.]', '', str(price_str)))
                    if price_val > filters['max_price']:
                        continue
                except Exception:
                    pass

        # Filter by brand
        if 'brand' in filters:
            if item.get("brand", "").lower() != filters['brand'].lower():
                continue

        # Add more filters here...

        filtered.append(item)

    return filtered


# --- Main Entrypoint ---

if __name__ == "__main__":
    logger.info("🚀 Starting Promo Sensei Slack Bot...")
    try:
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        handler.start()
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}")
