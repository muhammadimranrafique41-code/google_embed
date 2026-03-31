"""
Embedding helpers for StyleMate.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 3072

load_dotenv()


def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Load your .env file or export the variable before using embeddings."
        )
    return genai.Client(api_key=api_key)


client: genai.Client | None = None


def retry_with_backoff(max_attempts: int = 3, initial_delay: float = 1.0) -> Callable:
    """Retry transient embedding calls with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            delay = initial_delay
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # SDK-specific errors vary across releases.
                    last_error = exc
                    if attempt == max_attempts:
                        break
                    logger.warning(
                        "Embedding request failed on attempt %s/%s: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    time.sleep(delay)
                    delay *= 2
            raise RuntimeError(f"Embedding request failed after {max_attempts} attempts") from last_error

        return wrapper

    return decorator


def _get_client() -> genai.Client:
    global client
    if client is None:
        client = _build_client()
    return client


def _embedding_config() -> types.EmbedContentConfig:
    return types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )


@retry_with_backoff()
def get_text_embedding(text: str) -> list[float]:
    """Generate a Gemini embedding for text."""
    normalized = text.strip()
    if not normalized:
        raise ValueError("Text for embedding cannot be empty.")

    response = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=normalized,
        config=_embedding_config(),
    )
    return list(response.embeddings[0].values)


@retry_with_backoff()
def get_image_embedding(image_path: str) -> list[float]:
    """Generate a Gemini embedding for an image file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    with path.open("rb") as file_obj:
        image_part = types.Part.from_bytes(
            data=file_obj.read(),
            mime_type=mime_type or "image/jpeg",
        )

    response = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=types.Content(parts=[image_part]),
        config=_embedding_config(),
    )
    return list(response.embeddings[0].values)
