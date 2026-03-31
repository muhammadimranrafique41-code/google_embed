"""
Conversational response generation for StyleMate.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

CHAT_MODEL = "gemini-2.5-flash"

load_dotenv()

FASHION_ASSISTANT_PROMPT = """
You are StyleMate, a fashion shopping assistant for an online clothing store.
Help customers find products that fit their style, category, color, and budget.
Use only the provided product context when mentioning items.
Keep the tone warm, concise, and practical.
If there are no strong matches, say so clearly and suggest how the user can refine the search.
""".strip()

client: genai.Client | None = None


def _get_client() -> genai.Client:
    global client
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Load your .env file or export the variable before using chat generation."
            )
        client = genai.Client(api_key=api_key)
    return client


def _format_product_context(products: List[Dict]) -> str:
    if not products:
        return "No matching products were found."

    lines = []
    for product in products:
        similarity = product.get("similarity")
        similarity_text = f", similarity {float(similarity):.2f}" if similarity is not None else ""
        lines.append(
            f"- {product['product_name']} | category: {product['category']} | color: {product['color']} "
            f"| size: {product['size']} | price: Rs {product['price']} | rating: {product['rating']}"
            f"{similarity_text} | description: {product['description']}"
        )
    return "\n".join(lines)


def generate_response(user_message: str, products: List[Dict] | None = None) -> str:
    """Generate a chat response grounded in retrieved products."""
    catalog_context = _format_product_context(products or [])
    prompt = (
        f"Customer message:\n{user_message}\n\n"
        f"Retrieved products:\n{catalog_context}\n\n"
        "Write a helpful shopping reply. Mention a few best matches when available and explain why."
    )

    try:
        response = _get_client().models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=FASHION_ASSISTANT_PROMPT,
                temperature=0.6,
                max_output_tokens=250,
            ),
        )
        if response.text:
            return response.text.strip()
    except Exception as exc:
        logger.error("Chat generation failed: %s", exc)

    if products:
        top_names = ", ".join(product["product_name"] for product in products[:3])
        return f"I found a few promising matches for you: {top_names}. Let me know if you want a narrower style, color, or budget."
    return "I couldn't find a strong match yet. Try describing the item, color, category, or your budget."
