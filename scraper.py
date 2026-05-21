from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import logging
import traceback
import unicodedata
from typing import List, Dict
from playwright.sync_api import sync_playwright, TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENCY_SYMBOLS = ['₹', '$', '£', '€', 'Rs', 'USD', 'INR']

def looks_like_price(text):
    pattern = r"(" + "|".join(re.escape(sym) for sym in CURRENCY_SYMBOLS) + r")\s?[\d,.]+"
    return re.search(pattern, text) is not None

BAD_TITLE_KEYWORDS = [
    'login', 'cart', 'seller', 'account', 'exploreplus', 'more', 'wishlist',
    'category', 'categories', 'home', 'offers', 'deals', 'shop', 'brands',
    'filters', 'sort', 'payment', 'shipping', 'return', 'about', 'sponsored',
    'm.r.p', 'mrp', 'ad', 'you are seeing', 'let us know', 'price', 'buy now',
    'free delivery', 'delivery'
]

def is_valid_title(title):
    title = unicodedata.normalize("NFKD", title)
    title_lower = title.lower()
    if len(title) < 10:
        return False
    if any(word in title_lower for word in BAD_TITLE_KEYWORDS):
        return False
    if (title.isupper() or title.islower()) and len(title) < 15:
        return False
    return True

def extract_product_candidates(soup):
    candidates = []
    for tag in soup.find_all(True):
        link = tag.find('a', href=True)
        if not link:
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text and looks_like_price(text):
            candidates.append(tag)
    logger.info(f"Found {len(candidates)} candidate product blocks")
    return candidates

def extract_discount(strings):
    discount_pattern = re.compile(r'(\d{1,3})\s?%')
    for s in strings:
        match = discount_pattern.search(s)
        if match:
            return match.group(1) + '%'
    return 'N/A'


TOP_DISCOUNT_KEYWORDS = [
    'top discount', 'best discount', 'hot deal', 'top offer', 'special discount', 'top saving', 'biggest discount', 'exclusive offer'
]

def has_top_discount_indicator(tag):
    # Check tag text and classes for top discount indicators
    text = tag.get_text(separator=" ", strip=True).lower()
    if any(keyword in text for keyword in TOP_DISCOUNT_KEYWORDS):
        return True
    
    # Also check class names for badges or discount labels
    classes = " ".join(tag.get('class', [])).lower()
    if any(keyword.replace(' ', '') in classes for keyword in TOP_DISCOUNT_KEYWORDS):
        return True
    
    # Check direct child spans/divs for discount badges
    for child in tag.find_all(['span', 'div']):
        child_text = child.get_text(strip=True).lower()
        if any(keyword in child_text for keyword in TOP_DISCOUNT_KEYWORDS):
            return True
    
    return False

def parse_product_block(tag, base_url):
    try:
        # Prefer heading tags or anchors with meaningful text for title
        possible_title = None
        for el in tag.find_all(['h1','h2','h3','h4','h5','h6','a']):
            text = el.get_text(strip=True)
            if text and is_valid_title(text):
                possible_title = text
                break

        if not possible_title:
            texts = [t.strip() for t in tag.stripped_strings if is_valid_title(t)]
            possible_title = max(texts, key=len) if texts else "Unknown product"

        price = 'N/A'
        for string in tag.stripped_strings:
            if looks_like_price(string):
                price = string
                break

        all_texts = list(tag.stripped_strings)
        discount = extract_discount(all_texts)

        a_tag = tag.find('a', href=True)
        link = urljoin(base_url, a_tag['href']) if a_tag else 'N/A'

        store_name = "N/A"
        for cls in ['store', 'seller', 'brand', 'shop']:
            store_tag = tag.find(class_=re.compile(cls, re.I))
            if store_tag and store_tag.get_text(strip=True):
                store_name = store_tag.get_text(strip=True)
                break

        if store_name == "N/A":
            candidates = []
            for el in tag.find_all(['span','div']):
                text = el.get_text(strip=True)
                if text and len(text) < 30 and not re.search(r'\d', text) and all(c.isalnum() or c.isspace() for c in text):
                    candidates.append(text)
            if candidates:
                store_name = sorted(candidates, key=len)[0]

        # Detect top discount badge
        top_discount = "Yes" if has_top_discount_indicator(tag) else "No"

        return {
            'title': possible_title.strip(),
            'price': price.strip(),
            'discount': discount,
            'link': link,
            'brand': store_name.strip(),
            'top_discount': top_discount
        }
    except Exception as e:
        logger.debug(f"Failed to parse product block: {e}")
        return None

def scrape_generic_offers(url: str, headful=False, screenshot_path=None):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()
        try:
            logger.info(f"Visiting: {url}")
            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            # Wait until at least 3 price-like elements appear
            page.wait_for_function("""
                () => {
                    return Array.from(document.querySelectorAll("*")).filter(el => {
                        return /₹|\\$|Rs|INR|USD/.test(el.innerText);
                    }).length > 3;
                }
            """, timeout=15000)

            if screenshot_path:
                page.screenshot(path=screenshot_path)
                logger.info(f"Saved screenshot to {screenshot_path}")

            content = page.content()
            soup = BeautifulSoup(content, "html.parser")

            candidates = extract_product_candidates(soup)
            offers = []
            seen = set()

            for tag in candidates:
                offer = parse_product_block(tag, url)
                if not offer:
                    continue
                key = (offer['title'], offer['price'], offer['link'])
                if key not in seen and offer['price'] != 'N/A' and is_valid_title(offer['title']):
                    offers.append(offer)
                    seen.add(key)

            logger.info(f"Scraped {len(offers)} unique offers from {url}")
            return offers
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}\n{traceback.format_exc()}")
        finally:
            browser.close()
