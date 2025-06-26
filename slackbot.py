import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import re
from rag_query import (
    query_promotions,
    refresh_data,
    filter_by_brand,
    handleOffers,
    get_discounted_summary,
    
)
import validators

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
@app.command("/promosensei-scrap")
def handle_scrap(ack, respond, command):
    ack()

    url = command.get('text', '').strip()

    if not url:
        respond("❗ Please provide a URL to scrape. Usage: `/promosensei-scrap https://example.com`")
        return

    # if not validators.url(url):
    #     respond("❗ The provided text is not a valid URL. Please provide a valid website URL.")
    #     return

    respond(f"🕵️‍♂️ Scraping offers from: <{url}> ... this may take a few seconds.")

    try:
        offers = handleOffers(url)  # You must ensure this is robust

        if not offers:
            respond(f"⚠️ No offers found at <{url}> or scraping failed.")
            return

        # Format and limit to top 5 offers
        lines = [f"*Offers successfully scraped from:* <{url}>"]
        # for offer in offers[:5]:
        #     lines.append(
        #         f"• *{offer.get('title', 'No Title')}*\n"
        #         f"  💰 Price: ₹{offer.get('price', '-')}, 🔻 Discount: {offer.get('discount', '-')}, 🏷️ Brand: {offer.get('brand', '-')}\n"
        #         f"  🔗 <{offer.get('link', url)}|View Offer>"
        #     )

        respond("\n\n".join(lines))

    except Exception as e:
        logger.exception(f"❌ Error scraping {url}: {e}")
        respond(f"❌ Failed to scrape the website: {str(e)}")


@app.command("/promosensei-search")
def handle_search(ack, respond, command):
    ack()  # Acknowledge immediately

    user_query = command.get("text", "").strip()
    if not user_query:
        respond("❗ Please provide a search query. For example: `/promosensei-search Nike shoes under 3000`")
        return

    respond(f"🔍 Searching promotions for: *{user_query}*...")

    try:
        # query_promotions returns a summary string
        summary = query_promotions(user_query)
        respond(summary)
    except Exception as e:
        logger.error(f"Error processing /promosensei-search: {e}")
        respond("⚠️ Sorry, something went wrong while processing your request.")

@app.command("/promosensei-summary")
def handle_summary(ack, respond):
    ack()
    try:
        summary = get_discounted_summary()

        if not summary or summary.startswith("❗") or summary.startswith("⚠️"):
            respond(summary)
            return

        respond(f"Summary of Top  Deals:*\n{summary}")

    except Exception as e:
        logger.error(f"Summary command failed: {e}")
        respond(f"❌ Error generating summary: {str(e)}")


@app.command("/promosensei-brand")
def handle_brand_filter(ack, respond, command):
    ack()
    try:
        brand = command['text'].strip()
        if not brand:
            respond("❗ Please specify a brand name, e.g. `/promosensei-brand Nike`")
            return

        result = filter_by_brand(brand)
        offers = result.get("offers", [])
        summary = result.get("summary", "")

        if not offers:
            respond(summary)
            return

        # Format offer list
        formatted = []
        for i, offer in enumerate(offers, 1):
            formatted.append(
                f"{i}. *{offer.get('title', 'No title')}* at `{offer.get('brand', 'Unknown')}`\n"
                f"   💰 Price: ₹{offer.get('price', '-')}, 🔻 Discount: {offer.get('discount', '-')}%\n"
                f"   🔗 <{offer.get('link', '#')}|View Offer>"
            )

        respond(f"🏷️ *Offers for:* `{brand}`\n\n📢 *Summary:* {summary}\n\n" + "\n\n".join(formatted))

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

if __name__ == "__main__":
    logger.info("🚀 Starting Promo Sensei Slack Bot...")
    try:
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        handler.start()
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}")
