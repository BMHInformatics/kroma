import os
from datetime import datetime, timedelta
from pathlib import Path

from celery import shared_task
from django.conf import settings
from google import genai

from DSapp.pmc_pipeline import DEFAULT_SEARCH_TERM, ingest_search_results
from DSapp.triple_extraction_pipeline import extract_triples_for_eligible_articles


@shared_task
def refresh_gemini_kg_cache():
    cache_record = Path(settings.MEDIA_ROOT) / "dravet_kg_cache_name.txt"

    if not cache_record.exists():
        return "No cache file found to refresh."

    with open(cache_record, "r") as f:
        cache_name = f.read().strip()

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        client.caches.update(name=cache_name, ttl="3600s")
        return f"Successfully refreshed cache: {cache_name}"
    except Exception as e:
        cache_record.unlink(missing_ok=True)
        return f"Cache {cache_name} expired or failed to refresh. File deleted. Error: {str(e)}"


@shared_task
def sync_pmc_articles(
    search_term: str = DEFAULT_SEARCH_TERM,
    mindate: str = None,
    maxdate: str = None,
    retmax: int = 200,
    overwrite_existing: bool = False,
    download_pdf: bool = True,
):
    today = datetime.utcnow().date()
    if not maxdate:
        maxdate = today.strftime("%Y/%m/%d")
    if not mindate:
        mindate = (today - timedelta(days=1)).strftime("%Y/%m/%d")

    return ingest_search_results(
        term=search_term,
        mindate=mindate,
        maxdate=maxdate,
        retmax=retmax,
        db_alias="dsai",
        overwrite_existing=overwrite_existing,
        download_pdf=download_pdf,
    )


@shared_task
def daily_article_search():
    return sync_pmc_articles()


@shared_task
def extract_kroma_triples(limit: int = 20, overwrite_existing_refs: bool = False, pmcids: list[int] | None = None):
    return extract_triples_for_eligible_articles(
        db_alias="dsai",
        limit=limit,
        overwrite_existing_refs=overwrite_existing_refs,
        pmcids=pmcids,
    )
