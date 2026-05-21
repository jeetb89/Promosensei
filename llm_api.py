import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")

_client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_BASE = (
    "You are Promo Sensei, a helpful assistant specialized in recommending top product deals. "
    "Provide results in a friendly tone using markdown formatting with emojis and links."
)


def _chat(system: str, user: str, temperature: float = 0.7, max_tokens: int = None) -> str:
    kwargs = dict(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return _client.chat.completions.create(**kwargs).choices[0].message.content.strip()


def _fmt_price(value) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_discount(value) -> str:
    try:
        v = float(value)
        return f"{v:.0f}%" if v > 0 else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def summarize_search_results(
    user_query: str, documents: List[str], metadatas: List[Dict]
) -> str:
    lines = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        if not isinstance(meta, dict):
            continue
        lines.append(
            f"{i}. {meta.get('title')} from {meta.get('brand')} "
            f"- Price: {_fmt_price(meta.get('price'))}, "
            f"Discount: {_fmt_discount(meta.get('discount'))}, "
            f"Link: {meta.get('link')}"
        )
    context = "\n".join(lines)
    prompt = (
        f'A user is searching for: "{user_query}"\n\n'
        f"Relevant promotions:\n{context}\n\n"
        "Write a concise, beautifully formatted summary using bullet points, emojis, and short "
        "sentences. Highlight the biggest discounts and include product name, brand, price, "
        "discount, and link. Be engaging and friendly."
    )
    return _chat(_SYSTEM_BASE, prompt)


def summarize_top_discounts(promotions: List[Dict]) -> str:
    lines = []
    for i, p in enumerate(promotions, 1):
        lines.append(
            f"{i}. {p['title']} at {p['brand']} - ₹{p['price']} ({p['discount_pct']}% off)\n"
            f"   🔗 Link: {p['link']}"
        )
    context = "\n".join(lines)
    prompt = (
        f"Here are the top discounted promotions:\n\n{context}\n\n"
        "Write a clear, friendly, beautifully formatted summary. Use bullet points ✅, "
        "emojis 🎉, short punchy sentences ✍️, and include links 🔗. "
        "Focus on why each deal is great."
    )
    return _chat(
        "You are Promo Sensei, a friendly assistant that summarizes top product deals using "
        "markdown, emojis, and a fun tone.",
        prompt,
        max_tokens=800,
    )


def summarize_brand_offers(brand: str, offers: List[Dict]) -> str:
    lines = []
    for i, o in enumerate(offers, 1):
        lines.append(
            f"{i}. {o.get('title')} at {o.get('brand')} "
            f"- {_fmt_price(o.get('price'))} ({_fmt_discount(o.get('discount'))} off)\n"
            f"   🔗 Link: {o.get('link')}"
        )
    context = "\n".join(lines)
    prompt = (
        f"Below are top promotions from the brand `{brand}`:\n\n{context}\n\n"
        "Write a well-formatted, friendly summary that highlights the biggest discounts 💸, "
        "mentions price and brand clearly 🏷️, encourages users to check them out 🔗, "
        "uses bullet points ✅, emojis 🎉, and short sentences ✍️. "
        "Format using markdown for Slack."
    )
    return _chat(
        "You summarize shopping offers with clarity and persuasion using markdown, emojis, "
        "and short sentences.",
        prompt,
    )
