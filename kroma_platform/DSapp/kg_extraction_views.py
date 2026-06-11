from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from DSapp.kg_extraction_forms import KGTripleExtractionForm
from DSapp.triple_extraction_pipeline import extract_triples_manual, parse_pmcid_input


@login_required
def kg_extract_view(request):
    """
    Manual web page for KG triple extraction.

    This page is intentionally separate from automation. It lets you:
    - process specific PMCIDs,
    - process the next N eligible articles,
    - optionally re-extract already processed articles,
    - optionally send the job to Celery if your worker is running.
    """
    result = None

    if request.method == "POST":
        form = KGTripleExtractionForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data
            pmcid_text = cleaned.get("pmcids") or ""
            pmcids = parse_pmcid_input(pmcid_text)
            limit = cleaned.get("limit")
            overwrite_existing_refs = cleaned.get("overwrite_existing_refs", False)
            run_async = cleaned.get("run_async", False)

            try:
                if run_async:
                    # Import here so this view still works even if Celery is not configured during development.
                    from DSapp.tasks import extract_kroma_triples

                    task = extract_kroma_triples.delay(
                        limit=limit,
                        overwrite_existing_refs=overwrite_existing_refs,
                        pmcids=pmcids or None,
                    )
                    messages.success(request, f"KG extraction task started. Celery task id: {task.id}")
                    return redirect("DSapp:kg_extract")

                result = extract_triples_manual(
                    db_alias="dsai",
                    pmcid_text=pmcid_text,
                    limit=limit if not pmcids else None,
                    overwrite_existing_refs=overwrite_existing_refs,
                )

                messages.success(
                    request,
                    "KG extraction finished. "
                    f"Eligible={result.get('eligible_count', 0)}, "
                    f"Processed={result.get('processed_count', 0)}, "
                    f"Errors={result.get('error_count', 0)}."
                )

            except Exception as exc:  # noqa: BLE001
                messages.error(request, f"KG extraction failed: {type(exc).__name__}: {exc}")
    else:
        form = KGTripleExtractionForm()

    return render(request, "DSapp/kg_extract.html", {"form": form, "result": result})
