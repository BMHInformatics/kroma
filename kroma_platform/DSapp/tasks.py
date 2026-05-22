import os
import json
import hashlib
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
from DSapp.kg_compact import build_compact_kg_files, get_compact_kg_signature
from DSapp.pipeline_control import (
    load_pipeline_control,
    is_pmc_sync_enabled,
    is_triple_extraction_enabled,
    is_kg_cache_reset_enabled,
    kg_cache_ttl_seconds,
)


KROMA_TRIPLE_QUEUE_FILENAME = "kroma_triple_extraction_queue.json"
KROMA_GEMINI_CACHE_RECORD_FILENAME = "dravet_kg_cache_records.json"

# Keep this aligned with the chat model in views.py. The cache must be created
# for the same Gemini model that will consume it.
KROMA_CACHE_MODEL_NAME = "gemini-3-flash-preview"


@shared_task
def refresh_gemini_kg_cache():
    """
    Legacy task retained for compatibility.

    New KroMA cache behavior is controlled through kroma_pipeline_control.json.
    Prefer on-demand shared KG-only caching instead of refreshing a long-lived cache.
    """
    cache_record = Path(settings.MEDIA_ROOT) / "dravet_kg_cache_name.txt"

    if not cache_record.exists():
        return "No legacy cache file found to refresh."

    with open(cache_record, "r") as f:
        cache_name = f.read().strip()

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        client.caches.update(name=cache_name, ttl="3600s")
        return f"Successfully refreshed legacy cache: {cache_name}"
    except Exception as e:  # noqa: BLE001
        cache_record.unlink(missing_ok=True)
        return f"Legacy cache {cache_name} expired or failed to refresh. File deleted. Error: {str(e)}"


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
    """Default daily incremental sync for KroMA."""
    return sync_pmc_articles()


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

    # Keep order, remove duplicates.
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


def _kg_signature_hash() -> str:
    signature = get_compact_kg_signature()
    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()


def _cache_names_from_record(record: dict[str, Any]) -> list[str]:
    """Return cache names from both new shared records and old role records."""
    names = []
    if not isinstance(record, dict):
        return names

    cache_name = record.get("cache_name")
    if cache_name:
        names.append(cache_name)

    for rec in (record.get("roles") or {}).values():
        if isinstance(rec, dict) and rec.get("cache_name"):
            names.append(rec["cache_name"])

    return list(dict.fromkeys(names))


def delete_gemini_kg_cache() -> dict[str, Any]:
    """Delete current Gemini cache records when possible and clear the local record file."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"status": "error", "message": "GOOGLE_API_KEY is not configured."}

    client = genai.Client(api_key=api_key)
    records = _load_cache_records()

    deleted = []
    delete_errors = []
    for cache_name in _cache_names_from_record(records):
        try:
            client.caches.delete(name=cache_name)
            deleted.append(cache_name)
        except Exception as exc:  # noqa: BLE001
            delete_errors.append({
                "cache_name": cache_name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    _save_cache_records({
        "status": "empty",
        "deleted_at": timezone.now().isoformat(),
        "deleted_cache_names": deleted,
        "delete_errors": delete_errors,
    })

    return {
        "status": "success" if not delete_errors else "partial_success",
        "deleted_cache_names": deleted,
        "delete_errors": delete_errors,
    }


def warm_gemini_kg_cache(force_rebuild: bool = True, delete_existing: bool = True) -> dict[str, Any]:
    """
    Build/upload the compact KG and create one shared Gemini KG-only cache.

    The cache intentionally contains only the compact KG files. Clinician,
    patient/caregiver, and scientist instructions are sent separately at
    question time, so switching user role does not require another KG upload.
    """
    control = load_pipeline_control()

    if not control.get("kg_cache_enabled", True) or control.get("kg_cache_mode") == "off":
        return {
            "status": "disabled",
            "message": "Gemini KG cache is disabled from the KroMA admin dashboard.",
        }

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"status": "error", "message": "GOOGLE_API_KEY is not configured."}

    client = genai.Client(api_key=api_key)

    old_records = _load_cache_records()
    deleted = []
    delete_errors = []

    if delete_existing:
        for cache_name in _cache_names_from_record(old_records):
            try:
                client.caches.delete(name=cache_name)
                deleted.append(cache_name)
            except Exception as exc:  # noqa: BLE001
                delete_errors.append({
                    "cache_name": cache_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    compact_files = build_compact_kg_files(force_rebuild=force_rebuild)
    signature_hash = _kg_signature_hash()

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

    ttl_seconds = kg_cache_ttl_seconds(control)
    model_name = KROMA_CACHE_MODEL_NAME

    cached_content = client.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            display_name=f"dravet-kg-shared-{timezone.now().strftime('%Y%m%d-%H%M%S')}",
            ttl=f"{ttl_seconds}s",
            contents=uploaded_files,
        ),
    )

    created_at = timezone.now()
    records = {
        "status": "active",
        "cache_name": cached_content.name,
        "model_name": model_name,
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(seconds=ttl_seconds)).isoformat(),
        "ttl_seconds": ttl_seconds,
        "ttl_minutes": int(ttl_seconds / 60),
        "cache_contents": "shared KG-only compact TSV files",
        "role_instructions_cached": False,
        "kg_signature_hash": signature_hash,
        "deleted_old_caches": deleted,
        "delete_errors": delete_errors,
    }
    _save_cache_records(records)

    return {
        "status": "success",
        "cache_name": cached_content.name,
        "ttl_seconds": ttl_seconds,
        "ttl_minutes": int(ttl_seconds / 60),
        "deleted_old_caches": deleted,
        "delete_errors": delete_errors,
    }


@shared_task
def daily_pmc_sync_midnight():
    """
    Scheduled PMC sync.

    The dashboard can pause this task without code changes. When enabled, it
    queries PMC, downloads XML/PDF when available, saves metadata into dsai.article,
    classifies articles, and queues newly downloaded eligible articles for triple extraction.
    """
    control = load_pipeline_control()
    if not is_pmc_sync_enabled():
        return {
            "status": "paused",
            "message": "Automated PMC sync is paused from the KroMA admin dashboard.",
        }

    today = timezone.localdate()
    yesterday = today - timedelta(days=1)

    result = ingest_search_results(
        term=DEFAULT_SEARCH_TERM,
        mindate=yesterday.strftime("%Y/%m/%d"),
        maxdate=today.strftime("%Y/%m/%d"),
        retmax=int(control.get("pmc_sync_retmax", 200)),
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

        # Accept either usable full-text XML or PDF. The extraction pipeline can
        # fall back to PDF when XML is unavailable.
        if not (item.get("has_xml") or item.get("has_pdf")):
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
    Scheduled queued KG extraction.

    The dashboard can pause this task without code changes. When enabled, it
    pops queued PMCIDs and sends them to Gemini for triple extraction, validation,
    and KG CSV update.
    """
    control = load_pipeline_control()
    if not is_triple_extraction_enabled():
        return {
            "status": "paused",
            "message": "Automated KG extraction is paused from the KroMA admin dashboard.",
        }

    queue = _load_triple_queue()

    if not queue:
        return {
            "status": "no_queued_articles",
            "message": "No queued articles are waiting for triple extraction.",
        }

    limit = max(1, int(control.get("triple_extraction_limit_per_run", 1)))
    pmcids_to_process = queue[:limit]
    queue = queue[limit:]
    _save_triple_queue(queue)

    result = extract_triples_for_eligible_articles(
        db_alias="dsai",
        pmcids=pmcids_to_process,
        overwrite_existing_refs=False,
    )

    return {
        "processed_pmcids": pmcids_to_process,
        "remaining_queue_size": len(queue),
        "result": result,
    }


@shared_task
def reset_gemini_kg_cache_daily():
    """
    Scheduled Gemini KG cache warm/rebuild.

    This task now obeys the dashboard cache mode. It only runs when:
      kg_cache_enabled = true,
      kg_cache_mode = scheduled,
      automated_kg_cache_reset_enabled = true.

    It creates one shared KG-only cache that all user roles can reuse.
    Role-specific instructions are passed at question time.
    """
    if not is_kg_cache_reset_enabled():
        return {
            "status": "paused",
            "message": "Scheduled Gemini KG cache warm/rebuild is not enabled from the KroMA admin dashboard.",
        }

    return warm_gemini_kg_cache(force_rebuild=True, delete_existing=True)
