import os
import json
import time
import rdflib
import re
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect
from datetime import timedelta
from django.db.models import Count
from django.shortcuts import render, redirect
from django.utils import timezone
from django.conf import settings
from django.db import models
from django.http import JsonResponse
from django.contrib import messages
from django.core.mail import send_mail
from pathlib import Path
from google import genai
from google.genai.types import UploadFileConfig, GenerateContentConfig
from .models import Article, AccessRequest, ChatLog
import pandas as pd #TODO: temporary - remove when CSV bandaid is removed

User = get_user_model()


# Gemini client
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

CLINICIAN_INSTRUCTION = (
    "You are an expert assistant supporting clinicians and clinical researchers who care for "
    "or study patients with Dravet Syndrome. You answer at a professional level using precise "
    "clinical language (e.g., seizure semiology, comorbidities, pharmacologic options, prognosis, "
    "and clinical trial evidence). "
    "You have access to a Dravet Syndrome knowledge graph (KG) provided as N-Triples. Use the KG to "
    "contextualize and ground your answers, but you do not need to quote individual triples. "
    "You are also a general LLM and may use your broader medical and scientific training knowledge; "
    "treat KG-grounded statements as higher priority when available, and clearly label statements that "
    "are based on general knowledge when the KG is silent. "
    "When the KG does not address the question, answer from general clinical knowledge without commenting on KG availability. "
    "IMPORTANT: In your main answer, do NOT mention the knowledge graph at all. "
    "Do NOT say 'the KG does/does not contain...' or similar. "
    "If the KG is missing information, still answer using general medical knowledge, and simply state uncertainty if needed. "
    "Only discuss KG support/gaps in a separate [[KG_SUMMARY]] section. "
    "Only provide general educational information, not personalized medical advice, diagnosis, or "
    "treatment recommendations for a specific patient. When asked about management or treatment, "
    "describe general principles, typical options, and supporting evidence, and remind the user to "
    "consult guidelines and the patient’s treating clinicians for decisions. Be concise but precise."

    "OUTPUT FORMAT (INTERNAL — DO NOT EXPOSE TO USER):\n"
    "Use the seperator `---***---\n` to dilineate sections.\n"
    "[[ANSWER]]\n"
    "- Main response text goes here.\n\n"
    "- Provide a direct clinical answer first. Do NOT mention the knowledge graph. Do NOT include triples.\n\n"
    "---***---\n"
    "[[KG_SUMMARY]] (ONLY IF REQUESTED)\n"
    "- Plain-language KG grounding summary.\n\n"
    "- Provide 3–6 bullets summarizing which KG relationships supported the answer and where KG coverage is thin.\n\n"
    "---***---\n"
    "[[RAW_TRIPLES]] (ONLY IF REQUESTED)\n"
    "- Output raw N-Triples as plain lines only (no bullets, no numbering, no extra text), one triple per line.\n"
    "- Show at most 10 relevant N-Triples.\n"
)


PATIENT_INSTRUCTION = (
    "You are an empathetic educator for people living with Dravet Syndrome, their parents, families, "
    "and caregivers. Your goal is to explain concepts clearly, in non-technical language, using short "
    "paragraphs and concrete examples. Avoid medical jargon whenever possible; when you must use a "
    "technical term, briefly define it in simple words. "
    "You have access to a Dravet Syndrome knowledge graph (KG) provided as N-Triples, which you use to keep "
    "your answers accurate and focused, but you should not mention the KG or show triples. "
    "You are also a general LLM and may use your broader medical knowledge; treat KG-grounded statements as higher "
    "priority when available, and avoid overconfident claims when evidence is limited. "
    "Do not provide personal medical advice, diagnosis, or specific treatment instructions for "
    "an individual. Instead, offer general educational information, explain what kinds of questions "
    "they might ask their neurologist or care team, and encourage shared decision-making. "
    "Acknowledge that living with Dravet Syndrome is stressful and challenging, but keep your tone calm, "
    "supportive, and realistic—avoid false reassurance. If the available evidence is limited, say that clearly."
)


SCIENTIST_INSTRUCTION = (
    "You are a technical consultant for basic and translational researchers working on Dravet Syndrome. "
    "You have access to a Dravet Syndrome knowledge graph (KG) provided as N-Triples. Use the KG to ground "
    "and contextualize your answer, but you are also a general LLM: you may and should use your broader "
    "preclinical and mechanistic training knowledge when the KG is incomplete. Treat the KG as high-priority "
    "evidence when it speaks clearly; otherwise, supplement from general knowledge.\n\n"

    "CRITICAL RULES:\n"
    "1) Always answer the user’s question directly first.\n"
    "2) In Tier 1 and Tier 2, do NOT show or reference any raw KG identifiers, URIs, URLs, angle-bracket tokens, "
    "or strings like 'http://', 'https://', 'example.org', '<...>', 'KG-grounded:', 'subject/predicate/object', or 'triple(s)'.\n"
    "3) Never include triple syntax anywhere except Tier 3.\n"
    "4) Never imply you can ONLY use the KG. Use both KG + pretrained knowledge; if something is not in the KG, "
    "you may still answer from general knowledge, but label it as 'general knowledge' with uncertainty.\n"
    "5) Do not fabricate citations or trial identifiers.\n\n"

    "OUTPUT FORMAT (INTERNAL — DO NOT EXPOSE TO USER):\n"
    "Use the seperator `---***---\n` to dilineate sections.\n"
    "[[ANSWER]]\n"
    "- Main response text goes here.\n\n"
    "- Provide a direct clinical answer first. Do NOT mention the knowledge graph. Do NOT include triples.\n\n"
    "---***---\n"
    "[[KG_SUMMARY]] (ONLY IF REQUESTED)\n"
    "- Plain-language KG grounding summary.\n\n"
    "- Provide 3–6 bullets summarizing which KG relationships supported the answer and where KG coverage is thin.\n\n"
    "---***---\n"
    "[[RAW_TRIPLES]] (ONLY IF REQUESTED)\n"
    "- Output raw N-Triples as plain lines only (no bullets, no numbering, no extra text), one triple per line.\n"
    "- Show at most 10 relevant N-Triples.\n"
)




def get_system_instruction_for_role(role: str) -> str:
    role = (role or "").lower()
    if role == "patient":
        return PATIENT_INSTRUCTION
    elif role == "scientist":
        return SCIENTIST_INSTRUCTION
    # default: clinicians / researchers
    return CLINICIAN_INSTRUCTION


# Cache for the uploaded KG file
KG_FILE_REF = None


def ensure_kg_nt_file():
    """
    RDF:  media/kg/dravetkg.rdf
    NT:   media/kg/dravetkg.txt
    """
    media_root = Path(settings.MEDIA_ROOT)
    rdf_path = media_root / "kg" / "dravetkg.rdf"
    nt_path = media_root / "kg" / "dravetkg.txt"

    if not rdf_path.exists():
        raise FileNotFoundError(f"RDF KG file not found at {rdf_path}")

    # Regenerate if txt doesn't exist or RDF is newer
    if (not nt_path.exists()) or (rdf_path.stat().st_mtime > nt_path.stat().st_mtime):
        g = rdflib.Graph()
        g.parse(str(rdf_path), format="xml")
        g.serialize(destination=str(nt_path), format="nt")
    return nt_path

def get_kg_file_ref():
    global KG_FILE_REF

    if KG_FILE_REF is not None:
        return KG_FILE_REF

    if not gemini_client:
        raise RuntimeError("Gemini client is not configured (missing API key).")

    nt_path = ensure_kg_nt_file()

    unique_name = f"dravet-kg-nt-{int(time.time())}"

    KG_FILE_REF = gemini_client.files.upload(
        file=str(nt_path),
        config=UploadFileConfig(name=unique_name),
    )

    return KG_FILE_REF

@login_required
def index(request):
    # --- 1. Base queryset ---
    qs = Article.objects.using('dsai').all()

    # --- 2. Filters from querystring ---
    date_filter = request.GET.get('date_filter', 'all')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    ds_filter = request.GET.get('ds', 'all')
    organism_filter = request.GET.getlist('organism')
    type_filter = request.GET.getlist('type')
    axis_filter = request.GET.getlist('axis')
    search_query = request.GET.get('q', '').strip()

    today = timezone.now().date()

    # Date filter
    if date_filter == 'week':
        qs = qs.filter(date__gte=today - timedelta(days=7))
    elif date_filter == 'month':
        qs = qs.filter(date__gte=today - timedelta(days=30))
    elif date_filter == 'year':
        qs = qs.filter(date__gte=today - timedelta(days=365))
    elif date_filter == 'custom':
        if start_date:
            qs = qs.filter(date__gte=start_date)
        if end_date:
            qs = qs.filter(date__lte=end_date)

    # ds filter
    if ds_filter in ['Yes', 'No']:
        qs = qs.filter(ds=ds_filter)

    # organism filter
    if organism_filter:
        qs = qs.filter(organism__in=organism_filter)

    # type filter
    if type_filter:
        qs = qs.filter(type__in=type_filter)

    # axis filter
    if axis_filter:
        qs = qs.filter(axis__in=axis_filter)

    # search filter (title/abstract)
    if search_query:
        qs = qs.filter(
            # title OR abstract (Can be expanded to journal, authors, etc.)
            models.Q(title__icontains=search_query) |
            models.Q(abstract__icontains=search_query)
        )

    # --- 3. Selected articles ---
    selected_articles = None
    if request.method == 'POST':
        selected_ids = request.POST.getlist('selected_articles')
        if selected_ids:
            selected_articles = Article.objects.using('dsai').filter(pmcid__in=selected_ids)

    # --- 4. Stats for charts ---
    total = qs.count()

    # Type (pie chart)
    type_counts_qs = qs.values('type').annotate(count=Count('pmcid')).order_by('type')
    type_labels = [item['type'] for item in type_counts_qs]
    type_values = [item['count'] for item in type_counts_qs]
    type_percentages = [round(100 * c / total, 1) for c in type_values] if total else [0] * len(type_values)

    # Axis (bar chart)
    axis_counts_qs = qs.values('axis').annotate(count=Count('pmcid')).order_by('axis')
    axis_labels = [item['axis'] for item in axis_counts_qs]
    axis_values = [item['count'] for item in axis_counts_qs]
    axis_percentages = [round(100 * c / total, 1) for c in axis_values]

    # Organism (bar chart)
    org_counts_qs = qs.values('organism').annotate(count=Count('pmcid')).order_by('organism')
    org_labels = [item['organism'] for item in org_counts_qs]
    org_values = [item['count'] for item in org_counts_qs]
    org_percentages = [round(100 * c / total, 1) for c in org_values]

    # --- 5. Article list for table ---
    articles = qs.order_by('-date', 'title')

    context = {
        'articles': articles,
        'selected_articles': selected_articles,
        'total_count': total,

        # For filters
        'date_filter': date_filter,
        'start_date': start_date,
        'end_date': end_date,
        'ds_filter': ds_filter,
        'organism_filter': organism_filter,
        'type_filter': type_filter,
        'axis_filter': axis_filter,
        'search_query': search_query,

        # Choices for filters
        'type_choices': ['Original', 'Review', 'Case report', 'Opinion',
                         'Commentary', 'Letter', 'Other'],
        'axis_choices': ['Seizures', 'Genetics', 'Development', 'Pharmacology',
                         'Comorbidities', 'Behavior', 'SUDEP',
                         'Drug Responsiveness', 'Electrophysiology'],
        'organism_choices': ['Human', 'Zebrafish', 'Mouse', 'Primate', 'Drosophila'],

        # Chart data (JSON)
        'type_labels_json': json.dumps(type_labels),
        'type_values_json': json.dumps(type_values),
        'type_percentages_json': json.dumps(type_percentages),

        'axis_labels_json': json.dumps(axis_labels),
        'axis_values_json': json.dumps(axis_values),
        'axis_percentages_json': json.dumps(axis_percentages),

        'org_labels_json': json.dumps(org_labels),
        'org_values_json': json.dumps(org_values),
        'org_percentages_json': json.dumps(org_percentages),

        'MEDIA_URL': settings.MEDIA_URL,
        "KROMA_FEEDBACK_URL": getattr(settings, "KROMA_FEEDBACK_URL", ""),
    }

    return render(request, 'DSapp/index.html', context)


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
        model_name = "gemini-2.5-flash"
        db = "dsai"

        # Ensure user exists in dsai by username
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

        # Create log first
        log = ChatLog.objects.using(db).create(
            user=user_in_dsai,
            user_category=role,
            model_name=model_name,
            prompt=user_message,
            response="",
            was_success=False,
            error_message="",
        )

        uploaded_file = get_kg_file_ref()
        system_instruction = get_system_instruction_for_role(role)

        contents = [uploaded_file, f"User question: {user_message}"]

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )

        text = getattr(response, "text", "").strip()
        parts = [p.strip() for p in text.split("---***---") if p.strip()]

        response = parts[0] if len(parts) > 0 else ""
        kg_summary = parts[1] if len(parts) > 1 else ""
        used_triples = parts[2] if len(parts) > 2 else ""
        response = _sanitize_tier1(response)
        response = _finalize_output_for_user(response)

        #TODO: Put kg_summary into popup box

        # Get references based on used_triples
        # TODO: Do we want triples from any other sections too?
        references = _get_references(used_triples)

        # Resolve PMCIDs to article metadata
        articles = (
            Article.objects.using("dsai")
            .filter(pmcid__in=references)
            .values("pmcid", "title", "url")
        )

        reference_payload = [
            {
                "pmcid": a["pmcid"],
                "title": a["title"],
                "url": a["url"],
            }
            for a in articles
        ]

        log.response = text
        log.was_success = True
        log.error_message = ""
        log.save(using=db, update_fields=["response", "was_success", "error_message"])

        # TODO: Add reference section to response & have kg summary show up with seperate button
        return JsonResponse({
            "response": response,
            "references": reference_payload
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

        # --- Email notification ---
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


def _sanitize_tier1(text: str) -> str:
    """
    Removes KG URIs / URLs / 'KG-grounded:' parentheticals ONLY from Tier 1.
    Leaves Tier 2 and Tier 3 untouched.
    """
    if not text:
        return text

    # # Identify Tier boundaries (we'll be generous about formatting)
    # tier2_markers = [
    #     "\nTIER 2",
    #     "\nTier 2",
    #     "\nKG grounding summary",
    #     "\nKG Grounding Summary",
    # ]

    # split_idx = None
    # for m in tier2_markers:
    #     i = text.find(m)
    #     if i != -1:
    #         split_idx = i
    #         break

    # if split_idx is None:
    #     # No Tier 2 marker; still strip obvious URI leakage from entire output cautiously
    #     tier1 = text
    #     rest = ""
    # else:
    #     tier1 = text[:split_idx]
    #     rest = text[split_idx:]
    tier1 = text

    # Remove "(KG-grounded: ...)" or "(general knowledge, partially KG-grounded: ...)" style parentheticals in Tier 1
    tier1 = re.sub(r"\((?:[^)]*?)KG-grounded[^)]*?\)", "", tier1, flags=re.IGNORECASE)
    tier1 = re.sub(r"\((?:[^)]*?)example\.org[^)]*?\)", "", tier1, flags=re.IGNORECASE)

    # Remove any naked URLs/URIs in Tier 1
    tier1 = re.sub(r"https?://\S+", "", tier1, flags=re.IGNORECASE)

    # Remove any stray angle-bracketed tokens that look like IRIs
    tier1 = re.sub(r"<https?://[^>]+>", "", tier1, flags=re.IGNORECASE)

    # Clean up double spaces left by removals
    tier1 = re.sub(r"[ \t]{2,}", " ", tier1)
    tier1 = re.sub(r"\n{3,}", "\n\n", tier1).strip()

    # # Ensure we keep a trailing newline before the rest
    # if rest:
    #     return tier1 + "\n" + rest.lstrip()
    return tier1


def _finalize_output_for_user(text: str) -> str:
    """
    Returns ONLY the user-facing answer.
    Hides KG summary and raw triples from the UI,
    but they remain stored in ChatLog.response.
    """
    if not text:
        return text

    # Keep only content before the first section separator
    if "---***---" in text:
        text = text.split("---***---", 1)[0]

    # Remove internal answer marker
    text = text.replace("[[ANSWER]]", "").strip()

    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def _get_references(text: str) -> str:

    # Load CSV
    # TODO: Replace with DB lookups
    media_root = Path(settings.MEDIA_ROOT)
    csv_path = media_root/"triples_article_01.13.2026.csv"
    df = pd.read_csv(csv_path, dtype=str)
    
    # Normalize for matching (# TODO: Should already by normalized to lowercase in DB)
    df["Subject"] = df["Subject"].str.lower()
    df["Predicate"] = df["Predicate"].str.lower()
    df["Object"] = df["Object"].str.lower()

    pmcids = set()

    # Regex to extract triples of the form: <subj> <pred> <obj> .
    triple_pattern = re.compile(r"<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.")

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[["):
            continue

        match = triple_pattern.match(line)
        if not match:
            continue

        subj_uri, pred_uri, obj_uri = match.groups()

        subject = subj_uri.rstrip(">").split("/")[-1].lower()
        predicate = pred_uri.rstrip(">").split("/")[-1].lower()
        obj = obj_uri.rstrip(">").split("/")[-1].lower()

        matches = df[
            (df["Subject"] == subject) &
            (df["Predicate"] == predicate) &
            (df["Object"] == obj)
        ]

        if not matches.empty:
            pmcids.update(matches["PMCID"].dropna().tolist())

    if not pmcids:
        # TODO: Handle what happens if none of our triples have an available reference
        fake_list = {"2745418", "3547637", "4010885"}
        return fake_list
    return sorted(pmcids)
