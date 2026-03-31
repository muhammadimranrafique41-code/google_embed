"""
FastAPI application for StyleMate.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .chat_service import generate_response
    from .embedding_service import get_image_embedding, get_text_embedding
    from .search_service import (
        apply_price_filter,
        extract_price_filter,
        fetch_all_products,
        merge_search_results,
        search_by_image,
        search_by_text,
    )
except ImportError:  # pragma: no cover - supports local direct execution
    from chat_service import generate_response
    from embedding_service import get_image_embedding, get_text_embedding
    from search_service import (
        apply_price_filter,
        extract_price_filter,
        fetch_all_products,
        merge_search_results,
        search_by_image,
        search_by_text,
    )

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
IMAGES_DIR = BASE_DIR / "data" / "images"

app = FastAPI(
    title="StyleMate API",
    description="AI-powered multimodal fashion shopping assistant",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


def serialize_products(products: list[dict]) -> list[dict]:
    """Normalize database records for frontend consumption."""
    normalized: list[dict] = []
    for product in products:
        normalized.append(
            {
                **product,
                "price": float(product["price"]),
                "rating": float(product["rating"]),
                "similarity": float(product["similarity"]) if product.get("similarity") is not None else None,
                "image_url": f"/images/{product['image_filename']}" if product.get("image_filename") else None,
            }
        )
    return normalized


@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve the single-page frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)) -> JSONResponse:
    """Handle text-only chat queries."""
    user_message = message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        min_price, max_price = extract_price_filter(user_message)
        logger.info("Received text query with price bounds min=%s max=%s", min_price, max_price)

        query_embedding = get_text_embedding(user_message)
        vector_results = search_by_text(query_embedding)

        if not vector_results and (min_price is not None or max_price is not None):
            vector_results = fetch_all_products()

        filtered_products = apply_price_filter(vector_results, min_price, max_price)
        products = serialize_products(filtered_products)
        assistant_reply = generate_response(user_message, products)

        return JSONResponse({"response": assistant_reply, "products": products})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Text chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process text query.")


@app.post("/api/chat-image")
async def chat_image_endpoint(
    message: str = Form(""),
    image: UploadFile = File(...),
) -> JSONResponse:
    """Handle multimodal chat queries using text and image inputs."""
    user_message = message.strip()
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image upload is required.")

    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(image.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        min_price, max_price = extract_price_filter(user_message)
        image_embedding = get_image_embedding(str(temp_path))
        image_results = search_by_image(image_embedding)

        text_results: list[dict] = []
        if user_message:
            text_embedding = get_text_embedding(user_message)
            text_results = search_by_text(text_embedding)

        merged_results = merge_search_results(text_results, image_results)
        filtered_products = apply_price_filter(merged_results, min_price, max_price)
        products = serialize_products(filtered_products)
        assistant_reply = generate_response(user_message or "Find similar items to this image.", products)

        return JSONResponse({"response": assistant_reply, "products": products})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process image query.")
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/products")
async def list_products() -> JSONResponse:
    """List all products for debugging."""
    try:
        products = serialize_products(fetch_all_products())
        return JSONResponse({"products": products, "count": len(products)})
    except Exception as exc:
        logger.exception("Product listing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch products.")


@app.get("/health")
async def health_check() -> JSONResponse:
    """Simple health endpoint."""
    return JSONResponse({"status": "healthy"})
