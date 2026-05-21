import hashlib
import logging
import re
import unicodedata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CURRENCY = ["₹", "$", "£", "€", "Rs", "USD", "INR", ","]


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).strip()


def normalize_price(raw: str) -> Optional[float]:
    if not raw or raw.strip() == "N/A":
        return None
    cleaned = raw
    for sym in _CURRENCY:
        cleaned = cleaned.replace(sym, "")
    match = re.search(r"\d+(\.\d+)?", cleaned)
    return float(match.group()) if match else None


def normalize_discount(raw: str) -> Optional[float]:
    if not raw or raw.strip() == "N/A":
        return None
    match = re.search(r"\d+(\.\d+)?", str(raw))
    return float(match.group()) if match else None


def _is_valid(offer: Dict) -> bool:
    return (
        bool(offer.get("title"))
        and len(offer.get("title", "")) >= 10
        and offer.get("price", 0.0) > 0
        and offer.get("link", "N/A") != "N/A"
    )


def _deduplicate(offers: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for offer in offers:
        key = hashlib.md5(
            f"{offer['title']}_{offer['price']}_{offer['link']}".encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(offer)
    return result


def process_offers(raw_offers: List[Dict]) -> List[Dict]:
    """
    Full cleaning pipeline: normalize → validate → deduplicate.
    price and discount are stored as floats (0.0 when unparseable).
    """
    cleaned = []
    for o in raw_offers:
        price = normalize_price(o.get("price", ""))
        discount = normalize_discount(o.get("discount", ""))
        cleaned.append({
            "title": _normalize(o.get("title", "")),
            "price": price if price is not None else 0.0,
            "discount": discount if discount is not None else 0.0,
            "link": o.get("link", "N/A"),
            "brand": _normalize(o.get("brand", "N/A")),
            "top_discount": o.get("top_discount", "No"),
        })

    valid = [o for o in cleaned if _is_valid(o)]
    deduped = _deduplicate(valid)
    logger.info(f"Pipeline: {len(raw_offers)} raw → {len(valid)} valid → {len(deduped)} unique")
    return deduped
