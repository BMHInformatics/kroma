# tasks.py
import os
from pathlib import Path
from django.conf import settings
from celery import shared_task
from google import genai


@shared_task
def refresh_gemini_kg_cache():
    """
    Runs every 45 minutes via Celery Beat to extend the cache TTL.
    """
    cache_record = Path(settings.MEDIA_ROOT) / "dravet_kg_cache_name.txt"

    if not cache_record.exists():
        return "No cache file found to refresh."

    with open(cache_record, "r") as f:
        cache_name = f.read().strip()

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        # Update the TTL for another 60 minutes
        client.caches.update(
            name=cache_name,
            ttl="3600s"
        )
        return f"Successfully refreshed cache: {cache_name}"
    except Exception as e:
        # If the cache died anyway (e.g. server was off), delete the tracker file.
        # The next user query will generate a fresh cache.
        cache_record.unlink(missing_ok=True)
        return f"Cache {cache_name} expired or failed to refresh. File deleted. Error: {str(e)}"