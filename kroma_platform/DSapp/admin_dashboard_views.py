from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from django.utils import timezone

from DSapp.admin_dashboard_forms import PipelineControlForm
from DSapp.kg_extraction_forms import KGTripleExtractionForm
from DSapp.pmc_sync_forms import PMCSyncForm
from DSapp.pmc_pipeline import ingest_search_results
from DSapp.pipeline_control import (
    load_pipeline_control,
    maybe_update_celery_beat_schedule,
    save_pipeline_control,
)
from DSapp.tasks import KROMA_GEMINI_CACHE_RECORD_FILENAME, KROMA_TRIPLE_QUEUE_FILENAME
from DSapp.triple_extraction_pipeline import extract_triples_manual, parse_pmcid_input
from DSapp.views import _get_kg_last_updated_display


def _is_kroma_admin(user) -> bool:
    return bool(
        user.is_authenticated
        and (
            user.is_superuser
            or user.is_staff
            or user.username in {"appadmin", "appadmin1", "appadmin2"}
        )
    )


def _read_json_media_file(filename: str, default):
    path = Path(settings.MEDIA_ROOT) / filename
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _get_queue_size() -> int:
    queue = _read_json_media_file(KROMA_TRIPLE_QUEUE_FILENAME, [])
    return len(queue) if isinstance(queue, list) else 0


def _get_cache_summary() -> dict:
    records = _read_json_media_file(KROMA_GEMINI_CACHE_RECORD_FILENAME, {})
    if not isinstance(records, dict):
        return {}
    return records


def _control_initial(control: dict) -> dict:
    return {
        "automated_pmc_sync_enabled": bool(control.get("automated_pmc_sync_enabled")),
        "pmc_sync_time": control.get("pmc_sync_time", "00:00"),
        "pmc_sync_retmax": control.get("pmc_sync_retmax", 200),
        "automated_triple_extraction_enabled": bool(control.get("automated_triple_extraction_enabled")),
        "triple_extraction_every_minutes": control.get("triple_extraction_every_minutes", 30),
        "triple_extraction_limit_per_run": control.get("triple_extraction_limit_per_run", 1),
        "automated_kg_cache_reset_enabled": bool(control.get("automated_kg_cache_reset_enabled")),
        "kg_cache_reset_time": control.get("kg_cache_reset_time", "08:00"),
        "timezone": control.get("timezone", "America/New_York"),
    }


@login_required
def admin_dashboard_view(request):
    if not _is_kroma_admin(request.user):
        return HttpResponseForbidden("You do not have permission to access the KroMA admin dashboard.")

    control = load_pipeline_control()
    pmc_result = None
    kg_result = None

    if request.method == "POST":
        action = request.POST.get("action", "")

        if action == "save_pipeline_control":
            control_form = PipelineControlForm(request.POST)
            pmc_form = PMCSyncForm()
            kg_form = KGTripleExtractionForm()

            if control_form.is_valid():
                cleaned = control_form.cleaned_data
                updated = save_pipeline_control(
                    {
                        "automated_pmc_sync_enabled": cleaned["automated_pmc_sync_enabled"],
                        "pmc_sync_time": cleaned["pmc_sync_time"].strftime("%H:%M"),
                        "pmc_sync_retmax": cleaned["pmc_sync_retmax"],
                        "automated_triple_extraction_enabled": cleaned["automated_triple_extraction_enabled"],
                        "triple_extraction_every_minutes": cleaned["triple_extraction_every_minutes"],
                        "triple_extraction_limit_per_run": cleaned["triple_extraction_limit_per_run"],
                        "automated_kg_cache_reset_enabled": cleaned["automated_kg_cache_reset_enabled"],
                        "kg_cache_reset_time": cleaned["kg_cache_reset_time"].strftime("%H:%M"),
                        "timezone": cleaned["timezone"],
                    },
                    username=request.user.username,
                )
                beat_ok, beat_msg = maybe_update_celery_beat_schedule(updated)
                if beat_ok:
                    messages.success(request, "Pipeline controls saved. " + beat_msg)
                else:
                    messages.warning(
                        request,
                        "Pipeline controls saved. Pause/resume switches are active, but timer updates were not pushed to Celery Beat. "
                        + beat_msg,
                    )
                return redirect("DSapp:admin_dashboard")

        elif action in {"toggle_pmc_sync", "toggle_triple_extraction", "toggle_kg_cache_reset"}:
            toggle_map = {
                "toggle_pmc_sync": (
                    "automated_pmc_sync_enabled",
                    "automated PMC query/download",
                ),
                "toggle_triple_extraction": (
                    "automated_triple_extraction_enabled",
                    "automated KG extraction/validation/KG CSV update",
                ),
                "toggle_kg_cache_reset": (
                    "automated_kg_cache_reset_enabled",
                    "automated Gemini KG cache reset/rebuild",
                ),
            }
            key, label = toggle_map[action]
            updated = save_pipeline_control(
                {key: not bool(control.get(key, False))},
                username=request.user.username,
            )
            beat_ok, beat_msg = maybe_update_celery_beat_schedule(updated)
            new_state = "enabled" if updated.get(key) else "paused"
            if beat_ok:
                messages.success(request, f"{label} is now {new_state}. " + beat_msg)
            else:
                messages.warning(
                    request,
                    f"{label} is now {new_state} through task-level guards. " + beat_msg,
                )
            return redirect("DSapp:admin_dashboard")

        elif action == "run_pmc_sync":
            pmc_form = PMCSyncForm(request.POST)
            kg_form = KGTripleExtractionForm()
            control_form = PipelineControlForm(initial=_control_initial(control))

            if pmc_form.is_valid():
                cleaned = pmc_form.cleaned_data
                try:
                    pmc_result = ingest_search_results(
                        term=cleaned["search_term"],
                        mindate=cleaned["mindate"].strftime("%Y/%m/%d"),
                        maxdate=cleaned["maxdate"].strftime("%Y/%m/%d"),
                        retmax=cleaned["retmax"],
                        db_alias="dsai",
                        overwrite_existing=cleaned["overwrite_existing"],
                        download_pdf=cleaned["download_pdf"],
                    )
                    counts = pmc_result.get("counts", {})
                    messages.success(
                        request,
                        f"Manual PMC sync finished. Found={counts.get('found', 0)}, "
                        f"Created={counts.get('created', 0)}, Updated={counts.get('updated', 0)}, "
                        f"Skipped={counts.get('skipped_duplicates', 0)}, "
                        f"Skipped preprints={counts.get('skipped_preprints', 0)}, Errors={counts.get('errors', 0)}.",
                    )
                except Exception as exc:  # noqa: BLE001
                    messages.error(request, f"Manual PMC sync failed: {type(exc).__name__}: {exc}")

        elif action == "run_kg_extract":
            kg_form = KGTripleExtractionForm(request.POST)
            pmc_form = PMCSyncForm()
            control_form = PipelineControlForm(initial=_control_initial(control))

            if kg_form.is_valid():
                cleaned = kg_form.cleaned_data
                pmcid_text = cleaned.get("pmcids") or ""
                pmcids = parse_pmcid_input(pmcid_text)
                limit = cleaned.get("limit")
                overwrite_existing_refs = cleaned.get("overwrite_existing_refs", False)
                run_async = cleaned.get("run_async", False)

                try:
                    if run_async:
                        from DSapp.tasks import extract_kroma_triples

                        task = extract_kroma_triples.delay(
                            limit=limit,
                            overwrite_existing_refs=overwrite_existing_refs,
                            pmcids=pmcids or None,
                        )
                        messages.success(request, f"Manual KG extraction task started. Celery task id: {task.id}")
                        return redirect("DSapp:admin_dashboard")

                    kg_result = extract_triples_manual(
                        db_alias="dsai",
                        pmcid_text=pmcid_text,
                        limit=limit if not pmcids else None,
                        overwrite_existing_refs=overwrite_existing_refs,
                    )
                    messages.success(
                        request,
                        "Manual KG extraction finished. "
                        f"Eligible={kg_result.get('eligible_count', 0)}, "
                        f"Processed={kg_result.get('processed_count', 0)}, "
                        f"Errors={kg_result.get('error_count', 0)}.",
                    )
                except Exception as exc:  # noqa: BLE001
                    messages.error(request, f"Manual KG extraction failed: {type(exc).__name__}: {exc}")

        else:
            messages.error(request, "Unknown dashboard action.")
            return redirect("DSapp:admin_dashboard")

    else:
        control_form = PipelineControlForm(initial=_control_initial(control))
        pmc_form = PMCSyncForm()
        kg_form = KGTripleExtractionForm()

    control = load_pipeline_control()
    context = {
        "control": control,
        "control_form": control_form,
        "pmc_form": pmc_form,
        "kg_form": kg_form,
        "pmc_result": pmc_result,
        "kg_result": kg_result,
        "queue_size": _get_queue_size(),
        "cache_summary": _get_cache_summary(),
        "kg_last_updated": _get_kg_last_updated_display(),
        "now": timezone.now(),
    }
    return render(request, "DSapp/admin_dashboard.html", context)
