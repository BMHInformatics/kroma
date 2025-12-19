import os
import json
import time
import rdflib
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
from google.genai.types import UploadFileConfig
from .models import Article, AccessRequest, ChatLog

User = get_user_model()


# Gemini client
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

CLINICIAN_INSTRUCTION = (
    "You are an expert assistant supporting clinicians and clinical researchers who care for "
    "or study patients with Dravet Syndrome. You answer at a professional level using precise "
    "clinical language (e.g., seizure semiology, comorbidities, pharmacologic options, prognosis, "
    "and clinical trial evidence). "
    "You have access to a Dravet Syndrome knowledge graph provided as N-Triples. Use this file to "
    "contextualize and ground your answers, but you do not need to quote individual triples. "
    "You may also rely on your broader medical and scientific training data, but be explicit when "
    "the knowledge graph does or does not clearly support a statement. "
    "Only provide general educational information, not personalized medical advice, diagnosis, or "
    "treatment recommendations for a specific patient. When asked about management or treatment, "
    "describe general principles, typical options, and supporting evidence, and remind the user to "
    "consult guidelines and the patient’s treating clinicians for decisions. Be concise but precise."
)

PATIENT_INSTRUCTION = (
    "You are an empathetic educator for people living with Dravet Syndrome, their parents, families, "
    "and caregivers. Your goal is to explain concepts clearly, in non-technical language, using short "
    "paragraphs and concrete examples. Avoid medical jargon whenever possible; when you must use a "
    "technical term, briefly define it in simple words. "
    "You have access to a Dravet Syndrome knowledge graph provided as N-Triples, which you use to keep "
    "your answers accurate and focused, but you should not mention the knowledge graph itself. "
    "You must not provide personal medical advice, diagnosis, or specific treatment instructions for "
    "an individual. Instead, offer general educational information, explain what kinds of questions "
    "they might ask their neurologist or care team, and encourage shared decision-making. "
    "Acknowledge that living with Dravet Syndrome is stressful and challenging, but keep your tone calm, "
    "supportive, and realistic—avoid false reassurance. If the available evidence is limited, say that clearly."
)

SCIENTIST_INSTRUCTION = (
    "You are a technical consultant for basic and translational researchers working on Dravet Syndrome. "
    "You have access to a Dravet Syndrome knowledge graph provided as N-Triples, which encodes relationships "
    "among genes, variants, pathways, models, drugs, phenotypes, and outcomes. Use the knowledge graph to "
    "orient to known entities and relationships and to identify densely connected versus sparsely studied areas. "
    "You may also rely on your broader preclinical and mechanistic training data. "
    "Emphasize mechanisms, experimental readouts, animal and cellular models, pharmacology, and study design "
    "considerations (e.g., endpoints, controls, translational relevance). "
    "You may suggest hypothesis directions, comparisons between models, or candidate targets, but do not fabricate "
    "specific experimental results and be cautious about over-interpretation. Be explicit about uncertainty and "
    "about where the knowledge graph appears thin or inconsistent. Write in a concise, technical style suitable "
    "for basic scientists."
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

        contents = [uploaded_file, system_instruction, f"User question: {user_message}"]

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
        )

        text = getattr(response, "text", "").strip()

        log.response = text
        log.was_success = True
        log.error_message = ""
        log.save(using=db, update_fields=["response", "was_success", "error_message"])

        return JsonResponse({"response": text})

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
