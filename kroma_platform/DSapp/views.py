import os
import json
import hashlib
import re
import uuid
import time
from datetime import datetime, timedelta
import zipfile
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.db import connections
from django.db import models
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

from google import genai
from google.genai.types import (
    CreateCachedContentConfig,
    GenerateContentConfig,
    UploadFileConfig,
)

from DSapp.kg_compact import (
    build_compact_kg_files,
    get_compact_kg_signature,
    resolve_reference_pmcids_from_triples,
    resolve_compact_triples_to_labels,
    compact_ids_from_any_triples,
    retrieve_relevant_kg_support,
    extract_query_reference_terms,
)
from DSapp.models import AccessRequest, Article, ChatLog



User = get_user_model()

# Gemini client
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


KG_COMPACT_GUIDE = (
    "You have access to a compact, lossless Dravet Syndrome knowledge graph through cached TSV files. "
    "Use it as structured background context when it is relevant, but do not treat it as the only source of knowledge. "
    "Do not expose file names, node IDs, predicate IDs, triples, or implementation details to the user. "
)

COMMON_OUTPUT_RULES = (
    "Answer the user's question directly and naturally. "
    "Use the provided structured Dravet Syndrome context when relevant, while also using your broader biomedical knowledge. "
    "Do not say that the knowledge graph is silent, incomplete, or missing information; if evidence is uncertain, "
    "explain the uncertainty in ordinary clinical or educational language. "
    "Do not mention knowledge graphs, triples, compact files, TSV files, node IDs, predicate IDs, cache files, or implementation details in the user-facing answer. "
    "Do not fabricate citations, PMCID values, trial identifiers, or source details. "
    "Do not provide personalized medical advice, diagnosis, or treatment instructions for a specific patient. "
    "After the user-facing answer, you may append an internal block beginning with [[RAW_TRIPLES]] and list compact KG edge rows that clearly support the answer. "
    "This block is hidden from users and is used only for article-link recovery. "
    "Use compact edge format only, such as n123 | p17 | n456, one edge per line. "
    "Prioritize edges involving the specific intervention, gene, mechanism, phenotype, or outcome named in the user's question. "
    "Do not list broad background edges if more specific supporting edges are available. "
)

CLINICIAN_INSTRUCTION = (
    "You are an expert assistant supporting clinicians and clinical researchers who care for or study patients with Dravet Syndrome. "
    "Answer at a professional level using precise clinical and scientific language, including seizure semiology, comorbidities, "
    "pharmacologic options, prognosis, mechanisms, and clinical trial evidence when relevant. "
    "Provide general educational information and describe typical clinical principles rather than patient-specific recommendations. "
    + KG_COMPACT_GUIDE
    + COMMON_OUTPUT_RULES
)

PATIENT_INSTRUCTION = (
    "You are an empathetic educator for people living with Dravet Syndrome, their parents, families, and caregivers. "
    "Use short paragraphs, clear explanations, and plain language. Define technical terms briefly when they are needed. "
    "Encourage discussion with the treating neurology team when appropriate. "
    + KG_COMPACT_GUIDE
    + COMMON_OUTPUT_RULES
)

SCIENTIST_INSTRUCTION = (
    "You are a technical consultant for basic and translational researchers working on Dravet Syndrome. "
    "Answer with mechanistic and translational depth, including genetics, models, electrophysiology, pharmacology, "
    "developmental biology, and cross-axis relationships when relevant. "
    + KG_COMPACT_GUIDE
    + COMMON_OUTPUT_RULES
)

def get_system_instruction_for_role(role: str) -> str:
    role = (role or "").lower()
    if role == "patient":
        return PATIENT_INSTRUCTION
    if role == "scientist":
        return SCIENTIST_INSTRUCTION
    return CLINICIAN_INSTRUCTION


# Process-local cache metadata for Gemini cached content.
KG_CACHED_CONTENT_NAME = None
KG_CACHED_CONTENT_MODEL = None
KG_SOURCE_SIGNATURE = None
KG_CACHED_SYSTEM_INSTRUCTION = None
KROMA_GEMINI_CACHE_RECORD_FILENAME = "dravet_kg_cache_records.json"

def get_kg_cached_content_name(model_name: str, system_instruction: str, role: str = "clinician") -> str:
    global KG_CACHED_CONTENT_NAME, KG_CACHED_CONTENT_MODEL, KG_SOURCE_SIGNATURE, KG_CACHED_SYSTEM_INSTRUCTION

    if not gemini_client:
        raise RuntimeError("Gemini client is not configured (missing API key).")

    role = (role or "clinician").lower()

    # First try to use the daily prebuilt cache created at 8 AM Eastern.
    daily_cache_record = _load_daily_gemini_cache_record()
    role_record = daily_cache_record.get("roles", {}).get(role)

    if role_record:
        daily_cache_name = role_record.get("cache_name")
        daily_cache_model = role_record.get("model_name")
        daily_system_hash = role_record.get("system_instruction_hash")
        current_system_hash = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()

        # Avoid reusing stale daily caches built with older, more restrictive prompts.
        # If old cache records do not include system_instruction_hash, skip them and
        # create a fresh lazy cache with the current instruction.
        if (
            daily_cache_name
            and daily_cache_model == model_name
            and daily_system_hash == current_system_hash
        ):
            try:
                gemini_client.caches.get(name=daily_cache_name)
                return daily_cache_name
            except Exception:
                # Fall through to lazy cache creation.
                pass

    # Fallback: create cache lazily if the 8 AM scheduled task has not run yet.
    compact_files = build_compact_kg_files()
    current_signature = get_compact_kg_signature()

    if (
        KG_CACHED_CONTENT_NAME
        and KG_CACHED_CONTENT_MODEL == model_name
        and KG_CACHED_SYSTEM_INSTRUCTION == system_instruction
        and KG_SOURCE_SIGNATURE == current_signature
    ):
        try:
            gemini_client.caches.get(name=KG_CACHED_CONTENT_NAME)
            return KG_CACHED_CONTENT_NAME
        except Exception:
            KG_CACHED_CONTENT_NAME = None

    uploaded_files = []
    for path in compact_files.values():
        uploaded = gemini_client.files.upload(
            file=str(path),
            config=UploadFileConfig(
                mime_type="text/plain",
                display_name=path.name,
            ),
        )
        uploaded_files.append(uploaded)

    cached_content = gemini_client.caches.create(
        model=model_name,
        config=CreateCachedContentConfig(
            display_name=f"dravet-kg-compact-cache-{uuid.uuid4().hex[:8]}",
            ttl="86400s",
            contents=uploaded_files,
            system_instruction=system_instruction,
        ),
    )

    KG_CACHED_CONTENT_NAME = cached_content.name
    KG_CACHED_CONTENT_MODEL = model_name
    KG_SOURCE_SIGNATURE = current_signature
    KG_CACHED_SYSTEM_INSTRUCTION = system_instruction

    # Persist lazy-created cache so other workers/processes can reuse it.
    # Include the system-instruction hash because cache records without this hash
    # are treated as stale by the cache lookup code above.
    current_system_hash = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
    existing = _load_daily_gemini_cache_record()
    roles = existing.get("roles", {}) if isinstance(existing, dict) else {}
    roles[role] = {
        "cache_name": cached_content.name,
        "model_name": model_name,
        "created_at": timezone.now().isoformat(),
        "ttl_seconds": 86400,
        "system_instruction_hash": current_system_hash,
    }
    _save_daily_gemini_cache_record({
        "created_at": timezone.now().isoformat(),
        "model_name": model_name,
        "ttl_seconds": 86400,
        "roles": roles,
    })

    return KG_CACHED_CONTENT_NAME


# KG_DOWNLOAD_FILENAMES = [
#     "kg_triples.csv",
#     "kg_triples_references.csv",
#     "kg_nodes.tsv",
#     "kg_predicates.tsv",
#     "kg_edges.tsv",
#     "kg_refs.tsv",
#     "kg_triples_rejected.csv",
# ]
#
# def _existing_kg_paths():
#     media_root = Path(settings.MEDIA_ROOT)
#     return [media_root / name for name in KG_DOWNLOAD_FILENAMES if (media_root / name).exists()]

KG_DOWNLOAD_FILENAME = "kg_triples.csv"

def _existing_kg_paths():
    kg_path = Path(settings.MEDIA_ROOT) / KG_DOWNLOAD_FILENAME
    return [kg_path] if kg_path.exists() else []

def _get_kg_last_updated_display() -> str:
    paths = _existing_kg_paths()
    if not paths:
        return ""

    latest_mtime = max(path.stat().st_mtime for path in paths)

    eastern = ZoneInfo("America/New_York")
    dt = datetime.fromtimestamp(latest_mtime, tz=eastern)

    return dt.strftime("%Y-%m-%d %I:%M %p %Z")


# @login_required
# def download_kg(request):
#     paths = _existing_kg_paths()
#     if not paths:
#         return HttpResponse("No KG files are available for download yet.", status=404)
#
#     buffer = BytesIO()
#     with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
#         for path in paths:
#             zf.write(path, arcname=path.name)
#
#     buffer.seek(0)
#     filename = f"kroma_kg_{timezone.now().strftime('%Y%m%d_%H%M%S')}.zip"
#     response = HttpResponse(buffer.getvalue(), content_type="application/zip")
#     response["Content-Disposition"] = f'attachment; filename="{filename}"'
#     return response

@login_required
def download_kg(request):
    kg_path = Path(settings.MEDIA_ROOT) / KG_DOWNLOAD_FILENAME

    if not kg_path.exists():
        return HttpResponse("KG file is not available for download yet.", status=404)

    filename = f"kroma_kg_triples_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with kg_path.open("rb") as f:
        response = HttpResponse(f.read(), content_type="text/csv")

    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response

@login_required
def index(request):
    qs = Article.objects.using("dsai").all()
    with connections["dsai"].cursor() as cursor:
        table_columns = {
            col.name for col in connections["dsai"].introspection.get_table_description(cursor, Article._meta.db_table)
        }
    available_fields = [
        field.name
        for field in Article._meta.concrete_fields
        if getattr(field, "column", field.name) in table_columns
    ]
    if available_fields:
        qs = qs.only(*available_fields)

    date_filter = request.GET.get("date_filter", "all")
    start_date = request.GET.get("start_date", "")
    end_date = request.GET.get("end_date", "")
    ds_filter = request.GET.get("ds", "all")
    organism_filter = request.GET.getlist("organism")
    type_filter = request.GET.getlist("type")
    axis_filter = request.GET.getlist("axis")
    search_query = request.GET.get("q", "").strip()

    today = timezone.now().date()

    if date_filter == "week":
        qs = qs.filter(date__gte=today - timedelta(days=7))
    elif date_filter == "month":
        qs = qs.filter(date__gte=today - timedelta(days=30))
    elif date_filter == "year":
        qs = qs.filter(date__gte=today - timedelta(days=365))
    elif date_filter == "custom":
        if start_date:
            qs = qs.filter(date__gte=start_date)
        if end_date:
            qs = qs.filter(date__lte=end_date)

    if ds_filter in ["Yes", "No"]:
        qs = qs.filter(ds=ds_filter)

    if organism_filter:
        qs = qs.filter(organism__in=organism_filter)

    if type_filter:
        qs = qs.filter(type__in=type_filter)

    if axis_filter:
        qs = qs.filter(axis__in=axis_filter)

    if search_query:
        qs = qs.filter(
            models.Q(title__icontains=search_query) |
            models.Q(abstract__icontains=search_query)
        )

    selected_articles = None
    if request.method == "POST":
        selected_ids = request.POST.getlist("selected_articles")
        if selected_ids:
            selected_articles = Article.objects.using("dsai").filter(pmcid__in=selected_ids)
            if available_fields:
                selected_articles = selected_articles.only(*available_fields)

    total = qs.count()

    # type_counts_qs = qs.values("type").annotate(count=Count("pmcid")).order_by("type")
    # type_labels = [item["type"] for item in type_counts_qs]
    # type_values = [item["count"] for item in type_counts_qs]
    # type_percentages = [round(100 * c / total, 1) for c in type_values] if total else [0] * len(type_values)

    type_counts_raw = qs.values("type").annotate(count=Count("pmcid")).order_by("type")

    type_grouped = {}

    for item in type_counts_raw:
        raw_type = (item["type"] or "").strip()
        count = item["count"]

        raw_type_lower = raw_type.lower()

        # Do not show null/blank article types in the pie chart
        if not raw_type_lower or raw_type_lower in ["null", "none", "unknown"]:
            continue

        # Merge case/case report/case series into one group
        if raw_type_lower in ["case", "case report", "case reports", "case series"]:
            label = "Case report"
        else:
            # Clean display label
            label = raw_type.title()

            # Optional nicer formatting
            if raw_type_lower == "non-english":
                label = "Non-English"
            elif raw_type_lower == "original":
                label = "Original"
            elif raw_type_lower == "review":
                label = "Review"

        type_grouped[label] = type_grouped.get(label, 0) + count

    type_labels = list(type_grouped.keys())
    type_values = list(type_grouped.values())

    type_total = sum(type_values)
    type_percentages = [
        round(100 * c / type_total, 1) for c in type_values
    ] if type_total else [0] * len(type_values)
    
    axis_counts_qs = qs.values("axis").annotate(count=Count("pmcid")).order_by("axis")
    axis_labels = [item["axis"] for item in axis_counts_qs]
    axis_values = [item["count"] for item in axis_counts_qs]
    axis_percentages = [round(100 * c / total, 1) for c in axis_values] if total else [0] * len(axis_values)

    org_counts_qs = qs.values("organism").annotate(count=Count("pmcid")).order_by("organism")
    org_labels = [item["organism"] for item in org_counts_qs]
    org_values = [item["count"] for item in org_counts_qs]
    org_percentages = [round(100 * c / total, 1) for c in org_values] if total else [0] * len(org_values)

    articles = qs.order_by("-date", "title")

    context = {
        "articles": articles,
        "selected_articles": selected_articles,
        "total_count": total,
        "date_filter": date_filter,
        "start_date": start_date,
        "end_date": end_date,
        "ds_filter": ds_filter,
        "organism_filter": organism_filter,
        "type_filter": type_filter,
        "axis_filter": axis_filter,
        "search_query": search_query,
        "type_choices": ["Original", "Review", "Case report", "Opinion", "Commentary", "Letter", "Other"],
        "axis_choices": [
            "Seizures", "Genetics", "Development", "Pharmacology",
            "Comorbidities", "Behavior", "SUDEP",
            "Drug Responsiveness", "Electrophysiology"
        ],
        "organism_choices": ["Human", "Zebrafish", "Mouse", "Primate", "Drosophila"],
        "type_labels_json": json.dumps(type_labels),
        "type_values_json": json.dumps(type_values),
        "type_percentages_json": json.dumps(type_percentages),
        "axis_labels_json": json.dumps(axis_labels),
        "axis_values_json": json.dumps(axis_values),
        "axis_percentages_json": json.dumps(axis_percentages),
        "org_labels_json": json.dumps(org_labels),
        "org_values_json": json.dumps(org_values),
        "org_percentages_json": json.dumps(org_percentages),
        "MEDIA_URL": settings.MEDIA_URL,
        "KROMA_FEEDBACK_URL": getattr(settings, "KROMA_FEEDBACK_URL", ""),
        "kg_last_updated": _get_kg_last_updated_display(),
    }

    return render(request, "DSapp/index.html", context)


def _sanitize_tier1(text: str) -> str:
    if not text:
        return text

    tier1 = text
    tier1 = re.sub(r"\((?:[^)]*?)KG-grounded[^)]*?\)", "", tier1, flags=re.IGNORECASE)
    tier1 = re.sub(r"\((?:[^)]*?)example\.org[^)]*?\)", "", tier1, flags=re.IGNORECASE)
    tier1 = re.sub(r"https?://\S+", "", tier1, flags=re.IGNORECASE)
    tier1 = re.sub(r"<https?://[^>]+>", "", tier1, flags=re.IGNORECASE)
    tier1 = re.sub(r"[ \t]{2,}", " ", tier1)
    tier1 = re.sub(r"\n{3,}", "\n\n", tier1).strip()
    return tier1


def _strip_markdown_for_plaintext(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"```+", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _finalize_output_for_user(text: str) -> str:
    if not text:
        return text

    # Hide any internal/provenance sections from the user-facing answer.
    # This protects against outputs from both the old 3-section prompt and the
    # newer optional [[RAW_TRIPLES]] hidden block.
    if "---***---" in text:
        text = text.split("---***---", 1)[0]

    raw_marker = re.search(r"\[\[\s*RAW_TRIPLES\s*\]\]", text, flags=re.IGNORECASE)
    if raw_marker:
        text = text[:raw_marker.start()]

    kg_marker = re.search(r"\[\[\s*KG_SUMMARY\s*\]\]", text, flags=re.IGNORECASE)
    if kg_marker:
        text = text[:kg_marker.start()]

    text = re.sub(r"\[\[\s*ANSWER\s*\]\]", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _split_kroma_sections(text: str):
    sections = {
        "answer": "",
        "kg_summary": "",
        "raw_triples": "",
    }

    if not text:
        return sections

    def extract(start_pattern, end_pattern=None):
        start = re.search(start_pattern, text, re.IGNORECASE)
        if not start:
            return ""
        s = text[start.end():]
        if end_pattern:
            end = re.search(end_pattern, s, re.IGNORECASE)
            if end:
                s = s[:end.start()]
        return s.strip()

    sections["answer"] = extract(
        r"\[\[\s*ANSWER\s*\]\]",
        r"\[\[\s*KG_SUMMARY\s*\]\]"
    )

    sections["kg_summary"] = extract(
        r"\[\[\s*KG_SUMMARY\s*\]\]",
        r"\[\[\s*RAW_TRIPLES\s*\]\]"
    )

    sections["raw_triples"] = extract(
        r"\[\[\s*RAW_TRIPLES\s*\]\]"
    )

    return sections


def _extract_raw_triples_fallback(full_text: str) -> str:
    """
    If the section splitter fails or the model outputs formatting noise,
    try to recover plausible triple lines from the full response.
    """
    if not full_text:
        return ""

    if "[[RAW_TRIPLES]]" in full_text:
        raw_part = full_text.split("[[RAW_TRIPLES]]", 1)[1]
        if "---***---" in raw_part:
            raw_part = raw_part.split("---***---", 1)[0]
    else:
        raw_part = full_text

    candidate_lines = []
    for raw_line in raw_part.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        line = line.strip("`")
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)

        if line.count("|") >= 2:
            parts = [p.strip() for p in line.split("|")]
            candidate_lines.append(" | ".join(parts[:3]))

    return "\n".join(candidate_lines)


def _normalize_raw_triples_text(raw_triples_text: str) -> str:
    if not raw_triples_text:
        return ""

    normalized_lines = []

    for raw_line in raw_triples_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = line.strip("`")
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)

        if line.count("|") >= 2:
            parts = [p.strip() for p in line.split("|")]
            normalized_lines.append(" | ".join(parts[:3]))

    return "\n".join(normalized_lines)


def _split_answer_and_internal_triples(text: str):
    """
    Split a Gemini response into the user-facing answer and an optional hidden
    RAW_TRIPLES block. This allows a single Gemini call to provide optional
    provenance hints without exposing KG internals to users.
    """
    text = text or ""
    match = re.search(r"\[\[\s*RAW_TRIPLES\s*\]\]", text, flags=re.IGNORECASE)
    if not match:
        return text.strip(), ""

    answer_part = text[:match.start()].strip()
    raw_part = text[match.end():].strip()

    # If the model accidentally adds another internal marker after RAW_TRIPLES,
    # ignore anything after it.
    next_marker = re.search(r"\n\s*\[\[", raw_part)
    if next_marker:
        raw_part = raw_part[:next_marker.start()].strip()

    return answer_part, raw_part


def _normalize_pmcid_to_int(value):
    """Normalize 10994531 / "10994531" / "PMC10994531" to int or None."""
    s = str(value or "").strip()
    if s.upper().startswith("PMC"):
        s = s[3:]
    return int(s) if s.isdigit() else None


def _split_author_names(authors: str) -> list[str]:
    authors = (authors or "").strip()
    if not authors:
        return []

    text = authors.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if ";" in text:
        parts = [p.strip() for p in text.split(";")]
    elif "|" in text:
        parts = [p.strip() for p in text.split("|")]
    elif "," in text:
        parts = [p.strip() for p in text.split(",")]
    elif " and " in text.lower():
        parts = [p.strip() for p in re.split(r"\s+and\s+", text, flags=re.IGNORECASE)]
    else:
        parts = [text]

    return [p for p in parts if p]


def _derive_first_author(first_author: str = "", authors: str = "") -> str:
    first_author = (first_author or "").strip()
    if first_author:
        return first_author
    names = _split_author_names(authors)
    return names[0] if names else ""


def _has_multiple_authors(first_author: str = "", authors: str = "") -> bool:
    names = _split_author_names(authors)
    if len(names) > 1:
        return True
    authors = (authors or "").strip()
    if authors and any(sep in authors for sep in [";", "|", " and "]):
        return True
    return False


def _get_reference_payload_from_pmcids(pmcids):
    if not pmcids:
        return []

    numeric_pmcids = []
    seen = set()
    for p in pmcids:
        p_int = _normalize_pmcid_to_int(p)
        if p_int is not None and p_int not in seen:
            seen.add(p_int)
            numeric_pmcids.append(p_int)

    if not numeric_pmcids:
        return []

    articles = list(
        Article.objects.using("dsai")
        .filter(pmcid__in=numeric_pmcids)
        .values(
            "pmcid",
            "pmid",
            "title",
            "first_author",
            "authors",
            "journal",
            "year",
        )
    )

    article_by_pmcid = {}
    for a in articles:
        p_int = _normalize_pmcid_to_int(a.get("pmcid"))
        if p_int is not None:
            article_by_pmcid[p_int] = a

    payload = []
    for p in numeric_pmcids:
        a = article_by_pmcid.get(p)
        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{p}/"
        if a:
            first_author = _derive_first_author(
                first_author=a.get("first_author", ""),
                authors=a.get("authors", ""),
            )
            payload.append({
                "pmcid": p,
                "pmid": a.get("pmid") or "",
                "title": a.get("title") or f"PubMed Central article PMC{p}",
                "url": pmc_url,
                "pmc_url": pmc_url,
                "pmcid_link_text": str(p),
                "first_author": first_author,
                "has_multiple_authors": _has_multiple_authors(
                    first_author=a.get("first_author", ""),
                    authors=a.get("authors", ""),
                ),
                "journal": a.get("journal") or "",
                "year": a.get("year") or "",
            })
        else:
            payload.append({
                "pmcid": p,
                "pmid": "",
                "title": f"PubMed Central article PMC{p}",
                "url": pmc_url,
                "pmc_url": pmc_url,
                "pmcid_link_text": str(p),
                "first_author": "",
                "has_multiple_authors": False,
                "journal": "",
                "year": "",
            })

    complete_payload = [
        item for item in payload
        if item.get("title")
        and not str(item.get("title", "")).startswith("PubMed Central article PMC")
    ]
    if len(complete_payload) >= 3:
        return complete_payload
    return payload


def _article_text_for_scoring(article: dict) -> str:
    parts = [
        article.get("title") or "",
        article.get("abstract") or "",
        article.get("journal") or "",
    ]
    text = " ".join(str(p) for p in parts if p)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return text.lower()


def _singularize_reference_token(token: str) -> str:
    """
    Conservative singular/plural normalization for reference retrieval.

    Important fix: do NOT turn "oligonucleotides" into "oligonucleotid".
    Most biomedical words ending in -ides should simply drop the final s:
      oligonucleotides -> oligonucleotide
      nucleotides -> nucleotide
      peptides -> peptide
    Use -es removal only for common English plural endings such as -ches/-shes/-xes.
    """
    token = (token or "").lower().strip()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _simple_reference_tokenize(text: str):
    return [
        _singularize_reference_token(t)
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3
    ]



def _build_article_search_phrases(question_text: str):
    """
    Build disease-agnostic but question-specific phrase anchors for Article-table
    lookup. This intentionally behaves closer to the visible Article Knowledge
    Base title/abstract search, while adding singular/plural normalization.

    Example:
      "antisense oligonucleotides prevent seizures in Dravet syndrome"
    yields high-priority phrases such as:
      "antisense oligonucleotide", "antisense oligonucleotides"
    """
    raw = question_text or ""
    raw_norm = re.sub(r"[^A-Za-z0-9]+", " ", raw).lower().strip()
    raw_tokens = [t for t in raw_norm.split() if len(t) >= 3]

    def canon(t):
        return _singularize_reference_token(t)

    # Very generic words that should not become anchor phrases by themselves.
    generic = {
        "what", "which", "when", "where", "whether", "about", "that", "this",
        "there", "their", "with", "without", "from", "into", "within", "between",
        "evidence", "study", "studies", "article", "paper", "papers", "effect",
        "effects", "prevent", "prevents", "preventing", "reduce", "reduces", "reduced",
        "seizure", "seizures", "epilepsy", "epilepsies", "dravet", "syndrome",
        "patient", "patients", "mouse", "model", "models", "clinical", "severe",
    }

    content_raw = [t for t in raw_tokens if t not in generic]
    content_canon = [canon(t) for t in content_raw if canon(t) not in generic]

    phrases = []
    seen = set()

    def add_phrase(parts):
        phrase = " ".join([p for p in parts if p]).strip()
        if not phrase or len(phrase) < 5:
            return
        if phrase in seen:
            return
        seen.add(phrase)
        phrases.append(phrase)

    # Adjacent raw/canonical ngrams preserve expressions like
    # "antisense oligonucleotide" without needing a DS-specific synonym list.
    for tokens in (content_raw, content_canon):
        for n in (4, 3, 2):
            for i in range(0, max(len(tokens) - n + 1, 0)):
                add_phrase(tokens[i:i+n])

    # Also add all adjacent ngrams from the full question, because sometimes the
    # user's key concept includes a term that was otherwise considered generic.
    for tokens in (raw_tokens, [canon(t) for t in raw_tokens]):
        for n in (3, 2):
            for i in range(0, max(len(tokens) - n + 1, 0)):
                parts = tokens[i:i+n]
                # keep only phrases containing at least one non-generic anchor
                if any(p not in generic for p in parts):
                    add_phrase(parts)

    return phrases[:20], content_canon[:10]


def _rank_article_pmcids_for_question(question_text: str, max_results: int = 10):
    """
    Fast focused Article-table search for references.

    Optimization strategy:
      1) Search title first using a small set of high-specificity phrases.
      2) If title search finds enough records, skip abstract scans entirely.
      3) Search abstracts only when title search finds too few references.
      4) Avoid broad DS/seizure/model terms as anchors.
    """
    focus = extract_query_reference_terms(question_text or "")
    strong_tokens = [
        _singularize_reference_token(t)
        for t in focus.get("strong_tokens", [])
        if len(t) >= 3
    ]
    intent_terms = [t for t in focus.get("intent_terms", []) if len(t) >= 3]
    article_phrases, phrase_anchor_tokens = _build_article_search_phrases(question_text or "")

    generic = {
        "dravet", "syndrome", "seizure", "seizures", "epilepsy", "epileptic", "patient", "patients",
        "study", "studies", "evidence", "prevent", "prevents", "preventing", "reduce", "reduces", "reduced",
        "clinical", "model", "models", "mouse", "mice", "rat", "rats", "zebrafish", "severe",
        "first", "year", "life", "other", "article", "paper", "papers", "treatment",
    }

    anchors = []
    for tok in phrase_anchor_tokens + strong_tokens:
        tok = _singularize_reference_token(tok)
        if tok and tok not in generic and tok not in anchors:
            anchors.append(tok)

    raw_tokens = [t for t in re.findall(r"[A-Za-z0-9]+", (question_text or "").lower()) if len(t) >= 3]
    canonical_tokens = [_singularize_reference_token(t) for t in raw_tokens]
    for toks in (raw_tokens, canonical_tokens):
        for i in range(max(len(toks) - 1, 0)):
            a, b = toks[i], toks[i + 1]
            if a not in generic or b not in generic:
                phrase = f"{a} {b}".strip()
                if phrase not in article_phrases:
                    article_phrases.insert(0, phrase)

    article_phrases = sorted(set(article_phrases), key=lambda p: (len(p.split()), len(p)), reverse=True)[:6]
    anchors = anchors[:4]

    candidate_by_pmcid = {}
    candidate_source = {}

    def add_candidates(q_obj, source_label, limit=80, include_abstract=False):
        if q_obj is None:
            return
        values = ["pmcid", "title", "journal", "year", "first_author", "authors"]
        if include_abstract:
            values.append("abstract")
        qs = (
            Article.objects.using("dsai")
            .filter(q_obj)
            .exclude(pmcid__isnull=True)
            .values(*values)[:limit]
        )
        for a in qs:
            pmcid = _normalize_pmcid_to_int(a.get("pmcid"))
            if pmcid is None:
                continue
            candidate_by_pmcid.setdefault(pmcid, a)
            candidate_source.setdefault(pmcid, set()).add(source_label)

    for phrase in article_phrases:
        if len(phrase) >= 5:
            add_candidates(models.Q(title__icontains=phrase), f"title_phrase:{phrase}", limit=40)

    if len(candidate_by_pmcid) < 2 and len(anchors) >= 2:
        title_and_q = models.Q()
        for tok in anchors[:2]:
            title_and_q &= models.Q(title__icontains=tok)
        add_candidates(title_and_q, "title_anchor_and_top2", limit=60)

    if len(candidate_by_pmcid) < 2:
        phrase_q = models.Q()
        phrase_count = 0
        for phrase in article_phrases[:3]:
            if len(phrase) >= 5:
                phrase_q |= models.Q(abstract__icontains=phrase)
                phrase_count += 1
        if phrase_count:
            add_candidates(phrase_q, "abstract_phrase", limit=80, include_abstract=False)

    if len(candidate_by_pmcid) < 2 and len(anchors) >= 2:
        abstract_and_q = models.Q()
        for tok in anchors[:2]:
            abstract_and_q &= models.Q(abstract__icontains=tok)
        add_candidates(abstract_and_q, "abstract_anchor_and_top2", limit=80, include_abstract=False)

    if len(candidate_by_pmcid) < 1 and anchors:
        title_or_q = models.Q()
        for tok in anchors[:4]:
            title_or_q |= models.Q(title__icontains=tok)
        add_candidates(title_or_q, "title_anchor_or", limit=60)

    if len(candidate_by_pmcid) < 1 and intent_terms:
        intent_q = models.Q()
        for tok in intent_terms[:4]:
            intent_q |= models.Q(title__icontains=tok)
        add_candidates(intent_q, "title_intent_or", limit=40)

    if not candidate_by_pmcid:
        return []

    scored = []
    for pmcid, a in candidate_by_pmcid.items():
        title_text = _article_text_for_scoring({"title": a.get("title") or ""})
        title_token_set = set(_simple_reference_tokenize(title_text))
        all_text = _article_text_for_scoring(a)
        token_set = set(_simple_reference_tokenize(all_text))

        phrase_title_hits = [p for p in article_phrases if p in title_text]
        phrase_any_hits = [p for p in article_phrases if p in all_text]
        title_anchor_hits = set(anchors) & title_token_set
        anchor_hits = set(anchors) & token_set

        if anchors and not phrase_any_hits and not anchor_hits:
            continue

        score = 0.0
        score += 800.0 * len(phrase_title_hits)
        score += 200.0 * max(0, len(phrase_any_hits) - len(phrase_title_hits))
        score += 120.0 * len(title_anchor_hits)
        score += 35.0 * len(anchor_hits)
        if len(anchor_hits) >= 2:
            score += 250.0

        sources = candidate_source.get(pmcid, set())
        if any(s.startswith("title_phrase") for s in sources):
            score += 500.0
        if "title_anchor_and_top2" in sources:
            score += 200.0
        if any(s.startswith("abstract") for s in sources):
            score += 40.0

        if phrase_any_hits or anchor_hits:
            for tok in intent_terms[:4]:
                if tok in title_token_set:
                    score += 4.0
                elif tok in token_set:
                    score += 1.0
            if "dravet" in token_set:
                score += 2.0

        if a.get("title"):
            score += 2.0
        if a.get("first_author") or a.get("authors"):
            score += 1.0
        if a.get("year"):
            score += 1.0

        if score > 0:
            scored.append((score, pmcid, sorted(sources), a.get("title") or ""))

    scored.sort(key=lambda x: x[0], reverse=True)
    pmcids = []
    seen = set()
    for _, pmcid, _, _ in scored:
        if pmcid not in seen:
            seen.add(pmcid)
            pmcids.append(pmcid)
        if len(pmcids) >= max_results:
            break

    return pmcids


def _merge_ranked_pmcids(*pmcid_lists, max_total: int = 15):
    merged = []
    seen = set()
    for pmcids in pmcid_lists:
        for p in pmcids or []:
            s = str(p).strip()
            if s.upper().startswith("PMC"):
                s = s[3:]
            if not s.isdigit():
                continue
            p_int = int(s)
            if p_int in seen:
                continue
            seen.add(p_int)
            merged.append(p_int)
            if len(merged) >= max_total:
                return merged
    return merged


@require_POST
@csrf_protect
def kg_chat_api(request):
    """
    KroMA chat endpoint with latency-aware reference recovery.

    The heavier KG support search is now conditional. The endpoint first uses
    faster, more focused sources:
      1) Article title/abstract search based on the user's exact question terms.
      2) References from Gemini's optional validated [[RAW_TRIPLES]] block.

    The KG fallback search runs only if those fast sources return too few
    references. Timing checkpoints are written to ChatLog.error_message so you
    can see which step is responsible for slow responses.
    """
    t0 = time.perf_counter()

    try:
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        user_message = request.POST.get("message", "").strip()
        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        if not GEMINI_API_KEY:
            return JsonResponse({"error": "Gemini API key missing"}, status=500)

        role = request.POST.get("role", "clinician")
        model_name = "gemini-3.1-pro-preview"
        db = "dsai"

        user_in_dsai, _ = User.objects.using(db).update_or_create(
            username=request.user.username,
            defaults={
                "email": request.user.email,
                "password": request.user.password,
                "is_active": request.user.is_active,
                "is_staff": request.user.is_staff,
                "is_superuser": request.user.is_superuser,
                "first_name": request.user.first_name,
                "last_name": request.user.last_name,
            },
        )

        log = ChatLog.objects.using(db).create(
            user=user_in_dsai,
            user_category=role,
            model_name=model_name,
            prompt=user_message,
            response="",
            was_success=False,
            error_message="",
        )

        system_instruction = get_system_instruction_for_role(role)
        cached_content_name = get_kg_cached_content_name(
            model_name=model_name,
            system_instruction=system_instruction,
            role=role,
        )
        t_cache = time.perf_counter()

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=[
                (
                    "User question:\n"
                    f"{user_message}\n\n"
                    "Answer naturally for the selected audience. Use the cached structured context when helpful, "
                    "but do not describe the context itself. If you can identify compact KG edge rows that support the answer, "
                    "append them after an internal [[RAW_TRIPLES]] marker. Prefer edges involving the specific intervention, gene, mechanism, phenotype, or outcome named in the question. The backend will hide that block from users."
                )
            ],
            config=GenerateContentConfig(
                cached_content=cached_content_name,
            ),
        )
        t_gemini = time.perf_counter()

        text = getattr(response, "text", "").strip()

        # The prompt asks for a natural answer plus an optional hidden [[RAW_TRIPLES]] block.
        # These helpers are retained so old/stale cache outputs with [[ANSWER]] or ---***---
        # are still cleaned safely.
        sections = _split_kroma_sections(text)
        answer_text = sections["answer"] if sections["answer"] else text
        answer_text = _sanitize_tier1(answer_text)
        answer_text = _finalize_output_for_user(answer_text)

        # Optional hidden model-reported triples. These are NOT shown to users.
        # They are only used as provenance hints after validation against kg_edges.tsv.
        model_raw_triples_text = sections.get("raw_triples", "")
        # Avoid scanning the whole answer for triples when the model did not
        # include the hidden marker. This was a measurable latency source.
        if not model_raw_triples_text.strip() and re.search(r"\[\[\s*RAW_TRIPLES\s*\]\]", text, flags=re.IGNORECASE):
            model_raw_triples_text = _extract_raw_triples_fallback(text)
        model_raw_triples_text = _normalize_raw_triples_text(model_raw_triples_text)
        model_compact_triples_text = compact_ids_from_any_triples(model_raw_triples_text)
        t_parse = time.perf_counter()

        # Fast reference retrieval. This avoids the heavier KG fallback when the
        # question-specific Article search or model-reported triples already find
        # enough references.
        article_pmcids = _rank_article_pmcids_for_question(user_message, max_results=10)
        model_reference_pmcids = resolve_reference_pmcids_from_triples(model_compact_triples_text)

        reference_pmcids = _merge_ranked_pmcids(
            article_pmcids,
            model_reference_pmcids,
            max_total=15,
        )
        t_fast_refs = time.perf_counter()

        # Conditional KG fallback: only run this if the fast, more specific sources
        # produced too few references. This reduces latency and prevents broad DS
        # matches from dominating when title/abstract search already found topic-specific papers.
        support = {}
        kg_fallback_used = False
        focused_backend_pmcids = []
        compact_triples_text = model_compact_triples_text
        resolved_triples_text = resolve_compact_triples_to_labels(model_compact_triples_text)

        if len(reference_pmcids) < 2:
            kg_fallback_used = True
            support = retrieve_relevant_kg_support(
                question_text=user_message,
                answer_text=answer_text,
                model_triples_text=model_compact_triples_text,
                max_triples=35,
                max_pmcids=10,
                initial_pool=30,
                expansion_per_node=0,
            )

            focused_backend_pmcids = support.get("pmcids", []) if isinstance(support, dict) else []
            compact_triples_text = support.get("compact_triples_text", model_compact_triples_text) if isinstance(support, dict) else model_compact_triples_text
            resolved_triples_text = support.get("resolved_triples_text", resolved_triples_text) if isinstance(support, dict) else resolved_triples_text

            reference_pmcids = _merge_ranked_pmcids(
                article_pmcids,
                model_reference_pmcids,
                focused_backend_pmcids,
                max_total=15,
            )
        t_kg_fallback = time.perf_counter()

        reference_payload = _get_reference_payload_from_pmcids(reference_pmcids)
        t_payload = time.perf_counter()

        log.response = text
        log.was_success = True
        log.error_message = (
            f"timing_cache={t_cache - t0:.2f}s | "
            f"timing_gemini={t_gemini - t_cache:.2f}s | "
            f"timing_parse={t_parse - t_gemini:.2f}s | "
            f"timing_fast_refs={t_fast_refs - t_parse:.2f}s | "
            f"timing_kg_fallback={t_kg_fallback - t_fast_refs:.2f}s | "
            f"timing_reference_payload={t_payload - t_kg_fallback:.2f}s | "
            f"timing_total={t_payload - t0:.2f}s | "
            f"kg_fallback_used={kg_fallback_used} | "
            f"backend_retrieval_debug={support.get('debug', {}) if isinstance(support, dict) else {}} | "
            f"model_compact_triples={model_compact_triples_text[:1000]} | "
            f"compact_triples_selected={compact_triples_text[:1000]} | "
            f"resolved_triples_selected={resolved_triples_text[:1000]} | "
            f"article_metadata_pmcids={article_pmcids} | "
            f"model_reference_pmcids={model_reference_pmcids} | "
            f"focused_backend_pmcids={focused_backend_pmcids} | "
            f"pmcids={reference_pmcids} | "
            f"reference_payload_count={len(reference_payload)} | "
            f"reference_payload_preview={reference_payload[:3]}"
        )
        log.save(using=db, update_fields=["response", "was_success", "error_message"])

        return JsonResponse({
            "response": answer_text,
            "references": reference_payload,
        })

    except Exception as e:
        return JsonResponse({"error": f"{type(e).__name__}: {str(e)}"}, status=500)


def request_access(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        email = request.POST.get("email", "").strip()
        affiliation = request.POST.get("affiliation", "").strip()
        reason = request.POST.get("reason", "").strip()

        if not (first_name and last_name and email):
            messages.error(request, "Please fill in your name and email.")
            return redirect("DSapp:login")

        access_req = AccessRequest.objects.create(
            first_name=first_name,
            last_name=last_name,
            email=email,
            affiliation=affiliation,
            reason=reason,
        )

        notify_email = getattr(settings, "KROMA_ACCESS_REQUEST_EMAIL", None)
        from_email = getattr(settings, "DEFAULT_FROM_EMAIL", notify_email)

        if notify_email and from_email:
            subject = "New KroMA access request"
            body_lines = [
                "A new KroMA access request has been submitted:",
                "",
                f"Name: {access_req.first_name} {access_req.last_name}",
                f"Email: {access_req.email}",
                f"Affiliation: {access_req.affiliation or '(not provided)'}",
                "",
                "Reason:",
                access_req.reason or "(not provided)",
                "",
                f"Submitted at: {access_req.created_at}",
            ]
            body = "\n".join(body_lines)

            send_mail(
                subject=subject,
                message=body,
                from_email=from_email,
                recipient_list=[notify_email],
                fail_silently=True,
            )

        messages.success(
            request,
            "Your request has been submitted. We will contact you via email."
        )
        return redirect("DSapp:login")

    return redirect("DSapp:login")


def _save_daily_gemini_cache_record(record: dict) -> None:
    cache_record_path = Path(settings.MEDIA_ROOT) / KROMA_GEMINI_CACHE_RECORD_FILENAME
    cache_record_path.parent.mkdir(parents=True, exist_ok=True)
    cache_record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _load_daily_gemini_cache_record():
    cache_record_path = Path(settings.MEDIA_ROOT) / KROMA_GEMINI_CACHE_RECORD_FILENAME

    if not cache_record_path.exists():
        return {}

    try:
        data = json.loads(cache_record_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
