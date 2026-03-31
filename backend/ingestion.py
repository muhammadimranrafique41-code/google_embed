"""
Ingest the StyleMate product catalog into Supabase with multimodal embeddings.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from .embedding_service import get_image_embedding, get_text_embedding
    from .supabase_client import get_supabase_client
except ImportError:  # pragma: no cover - supports `python backend/ingestion.py`
    from embedding_service import get_image_embedding, get_text_embedding
    from supabase_client import get_supabase_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = ROOT_DIR / "data" / "products.csv"
IMAGES_DIR = ROOT_DIR / "data" / "images"


def load_products() -> pd.DataFrame:
    """Load and validate the product catalog CSV."""
    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(f"Products CSV not found: {PRODUCTS_CSV}")

    products = pd.read_csv(PRODUCTS_CSV)
    required_columns = {
        "product_id",
        "product_name",
        "category",
        "color",
        "size",
        "price",
        "rating",
        "description",
        "image_filename",
        "product_link",
    }
    missing = required_columns.difference(products.columns)
    if missing:
        raise ValueError(f"Products CSV is missing required columns: {sorted(missing)}")
    return products


def build_product_text(row: pd.Series) -> str:
    """Create a rich text representation for text embeddings."""
    return "\n".join(
        [
            f"Product Name: {row['product_name']}",
            f"Category: {row['category']}",
            f"Color: {row['color']}",
            f"Size: {row['size']}",
            f"Price: Rs {row['price']}",
            f"Rating: {row['rating']}",
            f"Description: {row['description']}",
        ]
    )


def build_product_record(row: pd.Series) -> dict:
    """Generate embeddings and shape a product row for upsert."""
    text_embedding = get_text_embedding(build_product_text(row))

    image_embedding = None
    image_path = IMAGES_DIR / str(row["image_filename"])
    if image_path.exists():
        image_embedding = get_image_embedding(str(image_path))
    else:
        logger.warning("Image not found for %s: %s", row["product_id"], image_path)

    return {
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "color": row["color"],
        "size": row["size"],
        "price": float(row["price"]),
        "rating": float(row["rating"]),
        "description": row["description"],
        "image_filename": row["image_filename"],
        "product_link": row["product_link"],
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
    }


def upsert_product(product: dict) -> None:
    """Upsert a single product into Supabase."""
    get_supabase_client().table("products").upsert(product, on_conflict="product_id").execute()


def main() -> None:
    """Run the ingestion pipeline."""
    logger.info("Starting StyleMate ingestion")
    products = load_products()
    total = len(products)
    success_count = 0
    failure_count = 0

    for index, (_, row) in enumerate(products.iterrows(), start=1):
        logger.info("Processing product %s of %s: %s", index, total, row["product_id"])
        try:
            product_record = build_product_record(row)
            upsert_product(product_record)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            logger.exception("Failed to process %s: %s", row["product_id"], exc)

    logger.info(
        "Ingestion complete. Successful: %s, Failed: %s, Total: %s",
        success_count,
        failure_count,
        total,
    )

    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
