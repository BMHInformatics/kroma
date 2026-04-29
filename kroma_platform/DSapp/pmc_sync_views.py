from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from DSapp.pmc_sync_forms import PMCSyncForm
from DSapp.pmc_pipeline import ingest_search_results


@login_required
def pmc_sync_view(request):
    if request.method == "POST":
        form = PMCSyncForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data
            messages.info(request,
                          f"overwrite_existing={cleaned['overwrite_existing']}, download_pdf={cleaned['download_pdf']}")

            try:
                result = ingest_search_results(
                    term=cleaned["search_term"],
                    mindate=cleaned["mindate"].strftime("%Y/%m/%d"),
                    maxdate=cleaned["maxdate"].strftime("%Y/%m/%d"),
                    retmax=cleaned["retmax"],
                    db_alias="dsai",
                    overwrite_existing=cleaned["overwrite_existing"],
                    download_pdf=cleaned["download_pdf"],
                )

                counts = result.get("counts", {})
                messages.success(
                    request,
                    f"PMC sync finished. Found={counts.get('found', 0)}, "
                    f"Created={counts.get('created', 0)}, "
                    f"Updated={counts.get('updated', 0)}, "
                    f"Skipped={counts.get('skipped_duplicates', 0)}, "
                    f"Skipped preprints={counts.get('skipped_preprints', 0)}, "
                    f"Errors={counts.get('errors', 0)}"
                )

                for err in result.get("errors", [])[:5]:
                    messages.error(request, f"PMCID {err.get('pmcid')}: {err.get('error')}")
                    tb = err.get("traceback")
                    if tb:
                        print(tb)

            except Exception as e:
                messages.error(request, f"PMC sync failed: {type(e).__name__}: {e}")

            return redirect("DSapp:pmc_sync")
    else:
        form = PMCSyncForm()

    return render(request, "DSapp/pmc_sync.html", {"form": form})