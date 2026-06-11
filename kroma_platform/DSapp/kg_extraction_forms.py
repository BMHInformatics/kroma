from django import forms


class KGTripleExtractionForm(forms.Form):
    """
    Manual KG extraction form.

    Use either:
    - specific PMCIDs, or
    - a limit for the next eligible unprocessed articles.
    """

    pmcids = forms.CharField(
        label="Specific PMCIDs",
        required=False,
        widget=forms.Textarea(attrs={
            "rows": 3,
            "placeholder": "Example: 13041439, PMC13052111\nLeave blank to process the next eligible articles.",
        }),
        help_text="Optional. Enter comma-, space-, semicolon-, or line-separated PMCIDs. You can include or omit the PMC prefix.",
    )

    limit = forms.IntegerField(
        label="Limit",
        required=False,
        min_value=1,
        max_value=200,
        initial=5,
        help_text="Used only when no specific PMCIDs are entered.",
    )

    overwrite_existing_refs = forms.BooleanField(
        label="Re-extract articles that already have KG references",
        required=False,
        initial=False,
        help_text="Check this when manually reprocessing specific PMCIDs after changing the prompt or QA rules.",
    )

    run_async = forms.BooleanField(
        label="Run with Celery in the background",
        required=False,
        initial=False,
        help_text="Use only if your Celery worker and broker are running. Leave unchecked for direct/manual testing.",
    )

    def clean(self):
        cleaned = super().clean()
        pmcids = (cleaned.get("pmcids") or "").strip()
        limit = cleaned.get("limit")

        if not pmcids and not limit:
            self.add_error("limit", "Enter a limit or provide specific PMCIDs.")

        return cleaned
