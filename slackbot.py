import logging
import os

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from rag_query import (
    filter_by_brand,
    get_discounted_summary,
    handle_offers,
    query_promotions,
    refresh_data,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise EnvironmentError("SLACK_BOT_TOKEN or SLACK_APP_TOKEN is not set.")

app = App(token=SLACK_BOT_TOKEN)


@app.command("/promosensei-scrap")
def handle_scrap(ack, respond, command):
    ack()
    url = command.get("text", "").strip()
    if not url:
        respond("Please provide a URL. Usage: `/promosensei-scrap https://example.com`")
        return

    respond(f"Scraping offers from <{url}> ... this may take a few seconds.")
    try:
        offers = handle_offers(url)
        if not offers:
            respond(f"No offers found at <{url}>.")
            return
        respond(f"*Scraped {len(offers)} offers from:* <{url}>")
    except Exception as e:
        logger.exception(f"Error scraping {url}: {e}")
        respond(f"Failed to scrape the website: {str(e)}")


@app.command("/promosensei-search")
def handle_search(ack, respond, command):
    ack()
    user_query = command.get("text", "").strip()
    if not user_query:
        respond("Please provide a search query. Example: `/promosensei-search Nike shoes under 3000`")
        return

    respond(f"Searching promotions for: *{user_query}*...")
    try:
        respond(query_promotions(user_query))
    except Exception as e:
        logger.error(f"Error in /promosensei-search: {e}")
        respond("Sorry, something went wrong while processing your request.")


@app.command("/promosensei-summary")
def handle_summary(ack, respond):
    ack()
    try:
        respond(get_discounted_summary())
    except Exception as e:
        logger.error(f"Summary command failed: {e}")
        respond(f"Error generating summary: {str(e)}")


@app.command("/promosensei-brand")
def handle_brand_filter(ack, respond, command):
    ack()
    brand = command.get("text", "").strip()
    if not brand:
        respond("Please specify a brand name. Example: `/promosensei-brand Nike`")
        return

    try:
        result = filter_by_brand(brand)
        offers = result.get("offers", [])
        summary = result.get("summary", "")

        if not offers:
            respond(summary)
            return

        lines = []
        for i, offer in enumerate(offers, 1):
            lines.append(
                f"{i}. *{offer.get('title', 'No title')}* at `{offer.get('brand', 'Unknown')}`\n"
                f"   Price: ₹{offer.get('price', '-')}, Discount: {offer.get('discount', '-')}%\n"
                f"   <{offer.get('link', '#')}|View Offer>"
            )

        respond(f"*Offers for* `{brand}`\n\n*Summary:* {summary}\n\n" + "\n\n".join(lines))
    except Exception as e:
        logger.error(f"Brand filter command failed: {e}")
        respond(f"Error filtering by brand: {str(e)}")


@app.command("/promosensei-refresh")
def handle_refresh(ack, respond):
    ack()
    try:
        respond(refresh_data())
    except Exception as e:
        logger.error(f"Refresh command failed: {e}")
        respond(f"Error refreshing data: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Promo Sensei Slack Bot...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
