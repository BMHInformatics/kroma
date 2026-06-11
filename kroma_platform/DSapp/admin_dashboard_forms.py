from datetime import time

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

    kg_cache_enabled = forms.BooleanField(
        label="Enable Gemini KG cache for KroMA chat",
        required=False,
        help_text="When unchecked, KroMA chat returns an error instead of creating or using a Gemini KG cache.",
    )
    kg_cache_mode = forms.ChoiceField(
        label="KG cache mode",
        required=True,
        choices=[
            ("on_demand", "On-demand: create only when chat needs it"),
            ("scheduled", "Scheduled warm cache"),
            ("manual", "Manual only"),
            ("off", "Off: do not use Gemini KG cache"),
        ],
        help_text="Recommended while usage is low: on-demand.",
    )
    kg_cache_ttl_minutes = forms.IntegerField(
        label="KG cache TTL, minutes",
        min_value=5,
        max_value=1440,
        required=True,
        help_text="How long the Gemini KG cache remains active after it is created. Recommended: 30 minutes.",
    )
    kg_cache_scheduled_time = forms.TimeField(
        label="Scheduled KG cache warm/rebuild time",
        required=False,
        widget=forms.TimeInput(attrs={"type": "time"}),
        help_text="Used only when KG cache mode is Scheduled warm cache.",
    )
    automated_kg_cache_reset_enabled = forms.BooleanField(
        label="Enable scheduled KG cache warm/rebuild",
        required=False,
        help_text="Used only when KG cache mode is Scheduled warm cache.",
    )
    timezone = forms.CharField(
        label="Scheduler timezone",
        max_length=80,
        required=False,
        initial="America/New_York",
        help_text="Used only when KG cache mode is Scheduled warm cache.",
    )

    def clean(self):
        cleaned = super().clean()
        mode = cleaned.get("kg_cache_mode") or "on_demand"

        # These controls are hidden in the dashboard unless scheduled mode is selected.
        # Keep safe defaults so saving on-demand/manual/off settings does not fail.
        if not cleaned.get("kg_cache_scheduled_time"):
            cleaned["kg_cache_scheduled_time"] = time(8, 0)
        if not cleaned.get("timezone"):
            cleaned["timezone"] = "America/New_York"

        if mode != "scheduled":
            cleaned["automated_kg_cache_reset_enabled"] = False

        return cleaned
