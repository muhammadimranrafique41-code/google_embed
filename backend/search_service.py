"""
Search and filtering helpers for StyleMate.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from .supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - supports direct script execution
    from supabase_client import get_supabase_client

load_dotenv()

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
DEFAULT_MATCH_COUNT = 6


def search_by_text(query_embedding: List[float], match_count: int = DEFAULT_MATCH_COUNT) -> List[Dict]:
    """Run text-vector similarity search through Supabase RPC."""
    try:
        response = get_supabase_client().rpc(
            "match_products_by_text",
            {
                "query_embedding": query_embedding,
                "match_threshold": SIMILARITY_THRESHOLD,
                "match_count": match_count,
            },
        ).execute()
        results = response.data or []
        logger.info("Text search returned %s products", len(results))
        return results
    except Exception as exc:
        logger.error("Text search failed: %s", exc)
        return []


def search_by_image(query_embedding: List[float], match_count: int = DEFAULT_MATCH_COUNT) -> List[Dict]:
    """Run image-vector similarity search through Supabase RPC."""
    try:
        response = get_supabase_client().rpc(
            "match_products_by_image",
            {
                "query_embedding": query_embedding,
                "match_threshold": SIMILARITY_THRESHOLD,
                "match_count": match_count,
            },
        ).execute()
        results = response.data or []
        logger.info("Image search returned %s products", len(results))
        return results
    except Exception as exc:
        logger.error("Image search failed: %s", exc)
        return []


def merge_search_results(text_results: List[Dict], image_results: List[Dict], limit: int = DEFAULT_MATCH_COUNT) -> List[Dict]:
    """Merge text and image results, keeping the best similarity per product."""
    merged: dict[str, Dict] = {}

    for result in [*text_results, *image_results]:
        product_id = result["product_id"]
        if product_id not in merged or float(result.get("similarity", 0)) > float(
            merged[product_id].get("similarity", 0)
        ):
            merged[product_id] = result

    combined = sorted(
        merged.values(),
        key=lambda item: float(item.get("similarity", 0)),
        reverse=True,
    )
    return combined[:limit]


def extract_price_filter(message: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract inclusive min/max price bounds from natural-language queries."""
    text = message.lower().replace("\u20b9", "rs ")
    currency = r"(?:rs\.?|inr)?\s*"
    number = r"(\d+(?:\.\d+)?)"
    patterns = [
        (
            rf"between\s+{currency}{number}\s*(?:and|to|-)\s*{currency}{number}",
            lambda match: (float(match.group(1)), float(match.group(2))),
        ),
        (
            rf"{currency}{number}\s*(?:to|-)\s*{currency}{number}",
            lambda match: (float(match.group(1)), float(match.group(2))),
        ),
        (
            rf"(?:under|less\s+than|below|up\s+to|maximum|max|at\s+most)\s+{currency}{number}",
            lambda match: (None, float(match.group(1))),
        ),
        (
            rf"(?:above|more\s+than|over|at\s+least|minimum|min|starting\s+from)\s+{currency}{number}",
            lambda match: (float(match.group(1)), None),
        ),
    ]

    for pattern, extractor in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        min_price, max_price = extractor(match)
        if min_price is not None and max_price is not None and min_price > max_price:
            min_price, max_price = max_price, min_price
        return min_price, max_price

    return None, None


def apply_price_filter(
    products: List[Dict],
    min_price: Optional[float],
    max_price: Optional[float],
) -> List[Dict]:
    """Filter products using inclusive price bounds."""
    if min_price is None and max_price is None:
        return products

    filtered: List[Dict] = []
    for product in products:
        price = float(product["price"])
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        filtered.append(product)

    logger.info("Price filter reduced %s products to %s", len(products), len(filtered))
    return filtered


def fetch_all_products() -> List[Dict]:
    """Fetch the full product catalog without similarity scoring."""
    try:
        response = get_supabase_client().table("products").select("*").order("product_id").execute()
        results = response.data or []
        logger.info("Fetched %s products for fallback", len(results))
        return results
    except Exception as exc:
        logger.error("Fetching products failed: %s", exc)
        return []
