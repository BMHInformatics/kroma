import os
import json
import re
import uuid
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
)
from DSapp.models import AccessRequest, Article, ChatLog



User = get_user_model()

# Gemini client
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


KG_COMPACT_GUIDE = (
    "You have access to the full Dravet Syndrome knowledge graph in a compact, lossless format "
    "distributed across four TSV files:\n"
    "1) kg_nodes.tsv with columns: node_id, label\n"
    "2) kg_predicates.tsv with columns: pred_id, label\n"
    "3) kg_edges.tsv with columns: s, p, o, r\n"
    "4) kg_refs.tsv with columns: ref_id, pmcid\n\n"
    "Interpretation rules:\n"
    "- Resolve subject and object node IDs through kg_nodes.tsv.\n"
    "- Resolve predicate IDs through kg_predicates.tsv.\n"
    "- Each row in kg_edges.tsv is one KG triple with a supporting reference ID.\n"
    "- Resolve each reference ID through kg_refs.tsv to obtain the PMCID.\n"
    "- The compact package is lossless; do not assume omitted information.\n"
    "- Reason over the entire KG, not only a single axis, because cross-axis dependencies matter.\n\n"
)

COMMON_OUTPUT_RULES = (
    "CRITICAL RULES:\n"
    "1) Always provide your output in the 3-portions format explained below.\n"
    "2) Always answer the user’s question directly first.\n"
    "3) In the first portion of your response (the [[ANSWER]] portion), do NOT show or reference raw KG IDs, "
    "URIs, URLs, file names, TSV internals, or any implementation details.\n"
    "4) Do not fabricate citations, trial identifiers, or PMCID values.\n"
    "5) If the KG is silent or incomplete, you may still answer from general knowledge, but clearly state uncertainty "
    "without discussing implementation details.\n"
    "OUTPUT FORMAT (INTERNAL — DO NOT EXPOSE TO USER):\n"
    "Use the separator `---***---\\n` to delineate sections.\n"
    "[[ANSWER]]\n"
    "Main response text only, in plain text.\n"
    "Provide a direct answer first. Do NOT mention the knowledge graph or compact files.\n\n"
    "---***---\n"
    "[[KG_SUMMARY]]\n"
    "Provide 3 to 6 short plain-text lines summarizing which KG relationships supported the answer and where coverage is thin.\n"
    "You may mention high-level cross-axis relationships, but do not expose node IDs, predicate IDs, or file internals.\n\n"
    "---***---\n"
    "[[RAW_TRIPLES]]\n"
    "Output resolved triples as plain lines only.\n"
    "Do NOT output compact IDs such as n123, p17, or r42.\n"
    "Do NOT use bullets, numbering, markdown, or code fences.\n"
    "Format each line exactly as: Subject | Predicate | Object\n"
    "Show at most 10 relevant resolved triples.\n"
)

CLINICIAN_INSTRUCTION = (
    "You are an expert assistant supporting clinicians and clinical researchers who care for "
    "or study patients with Dravet Syndrome. You answer at a professional level using precise "
    "clinical language (e.g., seizure semiology, comorbidities, pharmacologic options, prognosis, "
    "and clinical trial evidence). "
    "Use the provided full Dravet Syndrome KG as high-priority grounding, but you may also use your broader medical "
    "and scientific knowledge when needed. "
    "In your main answer, do NOT mention the knowledge graph or its implementation. "
    "Only provide general educational information, not personalized medical advice, diagnosis, or treatment "
    "recommendations for a specific patient. "
    "When asked about management or treatment, describe general principles, typical options, and supporting evidence, "
    "and remind the user to consult guidelines and the patient’s treating clinicians for decisions.\n\n"
    + KG_COMPACT_GUIDE
    + COMMON_OUTPUT_RULES
)

PATIENT_INSTRUCTION = (
    "You are an empathetic educator for people living with Dravet Syndrome, their parents, families, "
    "and caregivers. Your goal is to explain concepts clearly, in non-technical language, using short "
    "paragraphs and concrete examples. Avoid medical jargon whenever possible; when you must use a "
    "technical term, briefly define it in simple words. "
    "Use the provided full Dravet Syndrome KG as high-priority grounding, but you may also use broader medical knowledge "
    "when needed. "
    "Do not provide personal medical advice, diagnosis, or specific treatment instructions for an individual. "
    "Instead, offer general educational information, explain what kinds of questions they might ask their neurologist "
    "or care team, and encourage shared decision-making. "
    "Keep your tone calm, supportive, and realistic.\n\n"
    + KG_COMPACT_GUIDE
    + COMMON_OUTPUT_RULES
)

SCIENTIST_INSTRUCTION = (
    "You are a technical consultant for basic and translational researchers working on Dravet Syndrome. "
    "Use the provided full Dravet Syndrome KG as high-priority grounding, while also using your broader "
    "preclinical and mechanistic knowledge when the KG is incomplete. "
    "Treat the KG as a structured mechanistic resource and pay attention to cross-axis dependencies. "
    "In your main answer, do NOT mention the KG implementation details.\n\n"
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

        if daily_cache_name and daily_cache_model == model_name:
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

    if "---***---" in text:
        text = text.split("---***---", 1)[0]

    text = text.replace("[[ANSWER]]", "").strip()
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


def _get_reference_payload_from_pmcids(pmcids):
    if not pmcids:
        return []

    numeric_pmcids = []
    for p in pmcids:
        s = str(p).strip()
        if s.upper().startswith("PMC"):
            s = s[3:]
        if s.isdigit():
            numeric_pmcids.append(int(s))

    if not numeric_pmcids:
        return []

    articles = list(
        Article.objects.using("dsai")
        .filter(pmcid__in=numeric_pmcids)
        .values(
            "pmcid",
            "pmid",
            "title",
            "url",
            "first_author",
            "journal",
            "year",
        )
    )

    if articles:
        return [
            {
                "pmcid": a["pmcid"],
                "pmid": a["pmid"],
                "title": a["title"] or f"PMCID {a['pmcid']}",
                "url": a["url"] or f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{a['pmcid']}/",
                "first_author": a["first_author"] or "",
                "journal": a["journal"] or "",
                "year": a["year"] or "",
            }
            for a in articles
        ]

    # fallback if Article rows are missing
    return [
        {
            "pmcid": p,
            "pmid": "",
            "title": f"PMCID {p}",
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{p}/",
            "first_author": "",
            "journal": "",
            "year": "",
        }
        for p in numeric_pmcids
    ]


@require_POST
@csrf_protect
def kg_chat_api(request):
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
        
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=[f"User question: {user_message}"],
            config=GenerateContentConfig(
                cached_content=cached_content_name,
            ),
        )

        text = getattr(response, "text", "").strip()
        sections = _split_kroma_sections(text)

        if not sections["answer"]:
            sections["answer"] = text.split("[[", 1)[0].strip()

        answer_text = sections["answer"]
        raw_triples_text = sections["raw_triples"]

        if not raw_triples_text.strip():
            raw_triples_text = _extract_raw_triples_fallback(text)

        raw_triples_text = _normalize_raw_triples_text(raw_triples_text)

        answer_text = _sanitize_tier1(answer_text)
        answer_text = _finalize_output_for_user(answer_text)

        resolved_triples_text = resolve_compact_triples_to_labels(raw_triples_text)
        reference_pmcids = resolve_reference_pmcids_from_triples(raw_triples_text)
        reference_payload = _get_reference_payload_from_pmcids(reference_pmcids)

        log.response = text
        log.was_success = True
        log.error_message = (
            f"raw_triples_normalized={raw_triples_text[:1000]} | "
            f"resolved_triples={resolved_triples_text[:1000]} | "
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


def _load_daily_gemini_cache_record():
    cache_record_path = Path(settings.MEDIA_ROOT) / KROMA_GEMINI_CACHE_RECORD_FILENAME

    if not cache_record_path.exists():
        return {}

    try:
        data = json.loads(cache_record_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
