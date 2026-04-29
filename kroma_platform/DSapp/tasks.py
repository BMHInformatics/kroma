import os
import json
from typing import Any
from datetime import datetime, timedelta
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from google import genai
from google.genai.types import CreateCachedContentConfig, UploadFileConfig

from DSapp.pmc_pipeline import DEFAULT_SEARCH_TERM, ingest_search_results
from DSapp.triple_extraction_pipeline import (
    extract_triples_automatic,
    extract_triples_for_eligible_articles,
)
from DSapp.kg_compact import build_compact_kg_files
from DSapp.views import get_system_instruction_for_role


KROMA_TRIPLE_QUEUE_FILENAME = "kroma_triple_extraction_queue.json"
KROMA_GEMINI_CACHE_RECORD_FILENAME = "dravet_kg_cache_records.json"

KROMA_CACHE_MODEL_NAME = "gemini-3.1-pro-preview"
KROMA_CACHE_ROLES = ["clinician", "patient", "scientist"]
KROMA_CACHE_TTL_SECONDS = 86400


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
        client.caches.update(name=cache_name, ttl="3600s")
        return f"Successfully refreshed cache: {cache_name}"
    except Exception as e:  # noqa: BLE001
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
    """
    Fetch PMC articles for a date range, skip duplicates by PMCID,
    download XML and OA PDF when available, and save metadata into dsai.article.
    """
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
    """
    Default daily incremental sync for Kroma.
    """
    return sync_pmc_articles()


@shared_task
def extract_kroma_triples(limit: int = 20, overwrite_existing_refs: bool = False, pmcids: list[int] | None = None):
    return extract_triples_for_eligible_articles(
        db_alias="dsai",
        limit=limit,
        overwrite_existing_refs=overwrite_existing_refs,
        pmcids=pmcids,
    )


@shared_task
def extract_kroma_triples(
    limit: int = 20,
    overwrite_existing_refs: bool = False,
    pmcids: list[int] | None = None,
):
    """
    Manual/queued KG extraction task.

    Use from Django shell:
        from DSapp.tasks import extract_kroma_triples
        extract_kroma_triples.delay(limit=5)
    """
    return extract_triples_for_eligible_articles(
        db_alias="dsai",
        limit=limit,
        overwrite_existing_refs=overwrite_existing_refs,
        pmcids=pmcids,
    )


@shared_task
def automatic_kroma_triple_extraction(limit: int = 20):
    """
    Scheduled incremental KG extraction.

    This skips articles already present in kg_triples_references.csv.
    """
    return extract_triples_automatic(db_alias="dsai", limit=limit)


def _media_file(filename: str) -> Path:
    return Path(settings.MEDIA_ROOT) / filename


def _load_triple_queue() -> list[int]:
    queue_path = _media_file(KROMA_TRIPLE_QUEUE_FILENAME)

    if not queue_path.exists():
        return []

    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [int(x) for x in data if str(x).isdigit()]
    except Exception:
        return []

    return []


def _save_triple_queue(pmcids: list[int]) -> None:
    queue_path = _media_file(KROMA_TRIPLE_QUEUE_FILENAME)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    # keep order, remove duplicates
    seen = set()
    cleaned = []
    for pmcid in pmcids:
        if pmcid not in seen:
            seen.add(pmcid)
            cleaned.append(int(pmcid))

    queue_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def _append_to_triple_queue(pmcids: list[int]) -> list[int]:
    current = _load_triple_queue()
    combined = current + [int(p) for p in pmcids]
    _save_triple_queue(combined)
    return _load_triple_queue()


def _load_cache_records() -> dict[str, Any]:
    cache_record_path = _media_file(KROMA_GEMINI_CACHE_RECORD_FILENAME)

    if not cache_record_path.exists():
        return {}

    try:
        data = json.loads(cache_record_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache_records(records: dict[str, Any]) -> None:
    cache_record_path = _media_file(KROMA_GEMINI_CACHE_RECORD_FILENAME)
    cache_record_path.parent.mkdir(parents=True, exist_ok=True)
    cache_record_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


@shared_task
def daily_pmc_sync_midnight():
    """
    Runs every day at midnight Eastern.

    It queries PMC, downloads XML/PDF when available, saves metadata into dsai.article,
    classifies articles, and queues newly downloaded eligible articles for triple extraction.
    """
    today = timezone.localdate()
    yesterday = today - timezone.timedelta(days=1)

    result = ingest_search_results(
        term=DEFAULT_SEARCH_TERM,
        mindate=yesterday.strftime("%Y/%m/%d"),
        maxdate=today.strftime("%Y/%m/%d"),
        retmax=200,
        db_alias="dsai",
        overwrite_existing=False,
        download_pdf=True,
    )

    queued_pmcids = []

    for item in result.get("items", []):
        # Queue only articles that were newly created/updated and are ready for KG extraction.
        if not (item.get("created") or item.get("updated")):
            continue

        if item.get("type") != "Original":
            continue

        if item.get("ds") != "Yes":
            continue

        if not item.get("has_xml"):
            continue

        pmcid = item.get("pmcid")
        if pmcid:
            queued_pmcids.append(int(pmcid))

    queue_after = _append_to_triple_queue(queued_pmcids)

    return {
        "pmc_sync": result,
        "newly_queued_for_triple_extraction": queued_pmcids,
        "queue_size_after": len(queue_after),
    }


@shared_task
def extract_next_queued_kroma_article():
    """
    Runs every 30 minutes.

    It pops ONE PMCID from the queue and sends only that article to Gemini
    for triple extraction and KG update.
    """
    queue = _load_triple_queue()

    if not queue:
        return {
            "status": "no_queued_articles",
            "message": "No queued articles are waiting for triple extraction.",
        }

    pmcid = queue.pop(0)
    _save_triple_queue(queue)

    result = extract_triples_for_eligible_articles(
        db_alias="dsai",
        pmcids=[pmcid],
        overwrite_existing_refs=False,
    )

    return {
        "processed_pmcid": pmcid,
        "remaining_queue_size": len(queue),
        "result": result,
    }


@shared_task
def reset_gemini_kg_cache_daily():
    """
    Runs every day at 8 AM Eastern.

    Deletes prior Gemini KG caches if possible, rebuilds compact KG files,
    uploads the latest KG files to Gemini cache, and stores cache names
    with a 24-hour TTL.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"status": "error", "message": "GOOGLE_API_KEY is not configured."}

    client = genai.Client(api_key=api_key)

    old_records = _load_cache_records()

    # Delete old caches if possible.
    deleted = []
    delete_errors = []

    for role, rec in old_records.get("roles", {}).items():
        cache_name = rec.get("cache_name")
        if not cache_name:
            continue
        try:
            client.caches.delete(name=cache_name)
            deleted.append(cache_name)
        except Exception as exc:
            delete_errors.append({
                "role": role,
                "cache_name": cache_name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Rebuild compact KG files from the latest CSVs.
    compact_files = build_compact_kg_files(force_rebuild=True)

    uploaded_files = []
    for path in compact_files.values():
        uploaded = client.files.upload(
            file=str(path),
            config=UploadFileConfig(
                mime_type="text/plain",
                display_name=Path(path).name,
            ),
        )
        uploaded_files.append(uploaded)

    created_roles = {}

    # Create one cache per role because the system instruction differs by role.
    for role in KROMA_CACHE_ROLES:
        system_instruction = get_system_instruction_for_role(role)

        cached_content = client.caches.create(
            model=KROMA_CACHE_MODEL_NAME,
            config=CreateCachedContentConfig(
                display_name=f"dravet-kg-{role}-{timezone.now().strftime('%Y%m%d-%H%M%S')}",
                ttl=f"{KROMA_CACHE_TTL_SECONDS}s",
                contents=uploaded_files,
                system_instruction=system_instruction,
            ),
        )

        created_roles[role] = {
            "cache_name": cached_content.name,
            "model_name": KROMA_CACHE_MODEL_NAME,
            "created_at": timezone.now().isoformat(),
            "ttl_seconds": KROMA_CACHE_TTL_SECONDS,
        }

    records = {
        "created_at": timezone.now().isoformat(),
        "model_name": KROMA_CACHE_MODEL_NAME,
        "ttl_seconds": KROMA_CACHE_TTL_SECONDS,
        "roles": created_roles,
    }

    _save_cache_records(records)

    return {
        "status": "success",
        "deleted_old_caches": deleted,
        "delete_errors": delete_errors,
        "created_roles": created_roles,
    }


