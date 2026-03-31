"""
Shared Supabase client configuration for StyleMate.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create and cache a configured Supabase client."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not service_key:
        raise ValueError(
            "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env."
        )

    logger.info("Initializing Supabase client")
    return create_client(supabase_url, service_key)
