from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone


PIPELINE_CONTROL_FILENAME = "kroma_pipeline_control.json"

DEFAULT_PIPELINE_CONTROL = {
    # Default to paused so deploying this file immediately stops the automated pipeline
    # until an admin explicitly enables it from the dashboard.
    "automated_pmc_sync_enabled": False,
    "automated_triple_extraction_enabled": False,
    "automated_kg_cache_reset_enabled": False,

    # Schedule controls shown/edited on the dashboard. If django-celery-beat is
    # installed and configured, the dashboard will also push these values into
    # PeriodicTask records.
    "pmc_sync_time": "00:00",
    "triple_extraction_every_minutes": 30,
    "kg_cache_reset_time": "08:00",
    "timezone": "America/New_York",

    # Operational defaults for the automated tasks.
    "pmc_sync_retmax": 200,
    "triple_extraction_limit_per_run": 1,

    "updated_at": "",
    "updated_by": "",
}


@dataclass(frozen=True)
class PeriodicTaskNames:
    pmc_sync: str = "KroMA automated PMC sync"
    triple_extraction: str = "KroMA automated queued triple extraction"
    kg_cache_reset: str = "KroMA automated Gemini KG cache reset"


PERIODIC_TASK_NAMES = PeriodicTaskNames()


def _control_path() -> Path:
    path = Path(settings.MEDIA_ROOT) / PIPELINE_CONTROL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_pipeline_control() -> dict[str, Any]:
    """Load admin-editable pipeline controls from MEDIA_ROOT."""
    path = _control_path()
    data = deepcopy(DEFAULT_PIPELINE_CONTROL)

    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
        except Exception:
            # If the file is corrupt, keep safe defaults instead of running automation.
            pass

    data["triple_extraction_every_minutes"] = max(
        1,
        int(data.get("triple_extraction_every_minutes") or DEFAULT_PIPELINE_CONTROL["triple_extraction_every_minutes"]),
    )
    data["pmc_sync_retmax"] = max(1, int(data.get("pmc_sync_retmax") or DEFAULT_PIPELINE_CONTROL["pmc_sync_retmax"]))
    data["triple_extraction_limit_per_run"] = max(
        1,
        int(data.get("triple_extraction_limit_per_run") or DEFAULT_PIPELINE_CONTROL["triple_extraction_limit_per_run"]),
    )
    return data


def save_pipeline_control(updates: dict[str, Any], username: str = "") -> dict[str, Any]:
    """Persist dashboard control changes."""
    data = load_pipeline_control()
    data.update(updates or {})
    data["updated_at"] = timezone.now().isoformat()
    data["updated_by"] = username or ""

    path = _control_path()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def automation_enabled(key: str) -> bool:
    return bool(load_pipeline_control().get(key, False))


def is_pmc_sync_enabled() -> bool:
    return automation_enabled("automated_pmc_sync_enabled")


def is_triple_extraction_enabled() -> bool:
    return automation_enabled("automated_triple_extraction_enabled")


def is_kg_cache_reset_enabled() -> bool:
    return automation_enabled("automated_kg_cache_reset_enabled")


def maybe_update_celery_beat_schedule(control: dict[str, Any]) -> tuple[bool, str]:
    """
    Update django-celery-beat PeriodicTask rows if that package is installed.

    This makes timer changes effective from the dashboard without editing code.
    If django-celery-beat is not installed/configured, scheduled tasks will still
    obey the pause switches, but their run times must be changed wherever Celery
    Beat is configured for your deployment.
    """
    try:
        from django_celery_beat.models import CrontabSchedule, IntervalSchedule, PeriodicTask
    except Exception as exc:  # noqa: BLE001
        return False, f"django-celery-beat is not available: {type(exc).__name__}: {exc}"

    tz = control.get("timezone") or "America/New_York"

    pmc_hour, pmc_minute = str(control.get("pmc_sync_time") or "00:00").split(":")[:2]
    cache_hour, cache_minute = str(control.get("kg_cache_reset_time") or "08:00").split(":")[:2]
    extraction_minutes = max(1, int(control.get("triple_extraction_every_minutes") or 30))

    pmc_crontab, _ = CrontabSchedule.objects.get_or_create(
        minute=str(int(pmc_minute)),
        hour=str(int(pmc_hour)),
        day_of_week="*",
        day_of_month="*",
        month_of_year="*",
        timezone=tz,
    )
    PeriodicTask.objects.update_or_create(
        name=PERIODIC_TASK_NAMES.pmc_sync,
        defaults={
            "task": "DSapp.tasks.daily_pmc_sync_midnight",
            "crontab": pmc_crontab,
            "interval": None,
            "enabled": bool(control.get("automated_pmc_sync_enabled", False)),
        },
    )

    interval, _ = IntervalSchedule.objects.get_or_create(
        every=extraction_minutes,
        period=IntervalSchedule.MINUTES,
    )
    PeriodicTask.objects.update_or_create(
        name=PERIODIC_TASK_NAMES.triple_extraction,
        defaults={
            "task": "DSapp.tasks.extract_next_queued_kroma_article",
            "interval": interval,
            "crontab": None,
            "enabled": bool(control.get("automated_triple_extraction_enabled", False)),
        },
    )

    cache_crontab, _ = CrontabSchedule.objects.get_or_create(
        minute=str(int(cache_minute)),
        hour=str(int(cache_hour)),
        day_of_week="*",
        day_of_month="*",
        month_of_year="*",
        timezone=tz,
    )
    PeriodicTask.objects.update_or_create(
        name=PERIODIC_TASK_NAMES.kg_cache_reset,
        defaults={
            "task": "DSapp.tasks.reset_gemini_kg_cache_daily",
            "crontab": cache_crontab,
            "interval": None,
            "enabled": bool(control.get("automated_kg_cache_reset_enabled", False)),
        },
    )

    return True, "Celery Beat schedule updated."
