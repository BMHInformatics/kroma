from django import forms


class PipelineControlForm(forms.Form):
    automated_pmc_sync_enabled = forms.BooleanField(
        label="Enable automated PMC query/download",
        required=False,
        help_text="When unchecked, the scheduled PMC sync task exits without querying PMC or downloading articles.",
    )
    pmc_sync_time = forms.TimeField(
        label="PMC sync time",
        required=True,
        widget=forms.TimeInput(attrs={"type": "time"}),
        help_text="Daily time for automated PMC query/download.",
    )
    pmc_sync_retmax = forms.IntegerField(
        label="Max articles per automated PMC sync",
        min_value=1,
        max_value=1000,
        required=True,
    )

    automated_triple_extraction_enabled = forms.BooleanField(
        label="Enable automated KG extraction / validation / KG CSV update",
        required=False,
        help_text="When unchecked, queued articles will not be sent to Gemini and no new triples are appended automatically.",
    )
    triple_extraction_every_minutes = forms.IntegerField(
        label="Run queued KG extraction every N minutes",
        min_value=1,
        max_value=1440,
        required=True,
        help_text="The automated extractor processes one or more queued articles each time it runs.",
    )
    triple_extraction_limit_per_run = forms.IntegerField(
        label="Queued articles to process per extraction run",
        min_value=1,
        max_value=50,
        required=True,
        help_text="Use 1 to keep the previous behavior of processing one queued article per run.",
    )

    automated_kg_cache_reset_enabled = forms.BooleanField(
        label="Enable automated Gemini context-cache reset/rebuild",
        required=False,
        help_text="When unchecked, the scheduled cache reset task exits without deleting or replacing Gemini caches.",
    )
    kg_cache_reset_time = forms.TimeField(
        label="Gemini KG cache reset time",
        required=True,
        widget=forms.TimeInput(attrs={"type": "time"}),
    )
    timezone = forms.CharField(
        label="Scheduler timezone",
        max_length=80,
        required=True,
        initial="America/New_York",
    )
