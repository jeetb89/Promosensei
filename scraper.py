import logging
from typing import List, Dict
from chromadb import Client
from playwright.sync_api import sync_playwright, TimeoutError
import re
from embedding_utils import embed_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_price(price_str: str) -> float:
    """Extracts numeric price from a string like '₹ 3 495.00'"""
    try:
        return float(re.sub(r"[^\d.]", "", price_str.replace(",", "")))
    except:
        return 0.0

def scrape_nike_offers() -> List[Dict]:
    offers = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Block unnecessary resources
            page.route("**/*", lambda route, request: route.abort()
                       if request.resource_type in ["image", "font", "stylesheet"]
                       else route.continue_())

            page.goto("https://www.nike.com/in/w/sale", timeout=60000)
            page.wait_for_selector("div.product-card", timeout=15000)

            scroll_to_bottom_until_all_loaded(page)

            cards = page.locator("div.product-card")
            count = cards.count()

            for i in range(count):
                card = cards.nth(i)
                try:
                    title = card.locator("div.product-card__title").inner_text()

                    subtitle = ""
                    subtitle_locator = card.locator("div.product-card__subtitle")
                    if subtitle_locator.count() > 0:
                        subtitle = subtitle_locator.nth(0).inner_text()

                    current_price_locator = card.locator('div[data-testid="product-price-reduced"], div.product-price.is--current-price')
                    current_price = current_price_locator.nth(0).inner_text() if current_price_locator.count() > 0 else ""

                    original_price_locator = card.locator('div[data-testid="product-price"], div.product-price.in__styling')
                    original_price = original_price_locator.nth(0).inner_text() if original_price_locator.count() > 0 else current_price

                    cp = extract_price(current_price)
                    op = extract_price(original_price)
                    discount = round((1 - (cp / op)) * 100, 2) if op > cp and op > 0 else 0.0

                    link = card.locator("a.product-card__link-overlay").get_attribute("href")

                    offers.append({
                        "title": title.strip(),
                        "description": subtitle.strip(),
                        "current_price": current_price.strip(),
                        "original_price": original_price.strip(),
                        "discount": discount,
                        "brand": "Nike",
                        "category": "Footwear",
                        "expiry": "N/A",
                        "link": link or "https://www.nike.com/in/w/sale",
                    })
                except Exception as e:
                    logger.warning(f"Skipping a product card at index {i} due to error: {e}")
                    continue

            browser.close()
    except Exception as e:
        logger.error(f"Error scraping Nike offers: {e}")
    return offers


def scroll_to_bottom_until_all_loaded(page, max_scrolls=50):
    prev_count = 0
    for i in range(max_scrolls):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        page.wait_for_timeout(2000)  # Wait for lazy-loaded content

        curr_count = page.locator("div.product-card").count()
        logger.debug(f"Scroll {i+1}: Product count = {curr_count}")

        if curr_count == prev_count:
            logger.info("Reached bottom of page, no new products loaded.")
            break
        prev_count = curr_count


def push_to_chroma(offers: List[Dict], collection_name="promotions") -> None:
    if not offers:
        logger.warning("No offers to push to ChromaDB.")
        return

    try:
        client = Client()
        collection = client.get_or_create_collection(name=collection_name)

        texts = []
        metadatas = []
        embeddings = []
        ids = []

        for idx, offer in enumerate(offers):
            text = f"{offer['title']} - {offer['description']} - Price: {offer.get('current_price', '')}"
            embedding = embed_text(text)
            texts.append(text)
           
            metadatas.append({
        "brand": offer["brand"],
        "category": offer["category"],
        "expiry": offer["expiry"],
        "link": offer["link"],
        "current_price": offer["current_price"],    # Add current price string
        "original_price": offer["original_price"],  # Optional: also original price
        "discount":offer["discount"],
      })
            embeddings.append(embedding)
            ids.append(f"offer-{idx}")

        collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        logger.info(f"✅ Successfully pushed {len(offers)} offers to ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to push offers to ChromaDB: {e}")
