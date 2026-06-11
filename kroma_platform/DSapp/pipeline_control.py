from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone


PIPELINE_CONTROL_FILENAME = "kroma_pipeline_control.json"

KG_CACHE_MODES = {"off", "on_demand", "manual", "scheduled"}

DEFAULT_PIPELINE_CONTROL = {
    # Default to paused so deploying this file immediately stops the automated
    # article-ingestion/extraction pipeline until an admin explicitly enables it.
    "automated_pmc_sync_enabled": False,
    "automated_triple_extraction_enabled": False,

    # Scheduled cache warming is optional and is only used when
    # kg_cache_mode == "scheduled".
    "automated_kg_cache_reset_enabled": False,

    # Article-ingestion / KG-extraction schedule controls.
    "pmc_sync_time": "00:00",
    "triple_extraction_every_minutes": 30,
    "timezone": "America/New_York",

    # Gemini KG cache controls. KroMA now always uses one shared KG-only cache.
    # Role-specific clinician/patient/scientist instructions are sent at question
    # time and are not embedded in cached content.
    "kg_cache_enabled": True,
    "kg_cache_mode": "on_demand",
    "kg_cache_ttl_minutes": 30,
    "kg_cache_scheduled_time": "08:00",

    # Backward-compatible alias used by older templates/code. It is kept in sync
    # with kg_cache_scheduled_time by load/save helpers below.
    "kg_cache_reset_time": "08:00",

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
    kg_cache_reset: str = "KroMA scheduled Gemini KG cache warm/rebuild"


PERIODIC_TASK_NAMES = PeriodicTaskNames()


def _control_path() -> Path:
    path = Path(settings.MEDIA_ROOT) / PIPELINE_CONTROL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_int(value: Any, default: int, min_value: int, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default
    out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _normalize_control(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize old and new dashboard settings into one safe control dict."""
    # Migration from the previous cache form field.
    if not data.get("kg_cache_scheduled_time") and data.get("kg_cache_reset_time"):
        data["kg_cache_scheduled_time"] = data.get("kg_cache_reset_time")
    if not data.get("kg_cache_reset_time") and data.get("kg_cache_scheduled_time"):
        data["kg_cache_reset_time"] = data.get("kg_cache_scheduled_time")

    data.pop("kg_cache_scope", None)  # removed; shared KG-only cache is now hard-coded

    data["triple_extraction_every_minutes"] = _safe_int(
        data.get("triple_extraction_every_minutes"),
        DEFAULT_PIPELINE_CONTROL["triple_extraction_every_minutes"],
        1,
        1440,
    )
    data["pmc_sync_retmax"] = _safe_int(
        data.get("pmc_sync_retmax"),
        DEFAULT_PIPELINE_CONTROL["pmc_sync_retmax"],
        1,
        1000,
    )
    data["triple_extraction_limit_per_run"] = _safe_int(
        data.get("triple_extraction_limit_per_run"),
        DEFAULT_PIPELINE_CONTROL["triple_extraction_limit_per_run"],
        1,
        50,
    )
    data["kg_cache_ttl_minutes"] = _safe_int(
        data.get("kg_cache_ttl_minutes"),
        DEFAULT_PIPELINE_CONTROL["kg_cache_ttl_minutes"],
        5,
        1440,
    )

    mode = str(data.get("kg_cache_mode") or DEFAULT_PIPELINE_CONTROL["kg_cache_mode"]).strip()
    if mode not in KG_CACHE_MODES:
        mode = DEFAULT_PIPELINE_CONTROL["kg_cache_mode"]
    data["kg_cache_mode"] = mode

    data["kg_cache_enabled"] = bool(data.get("kg_cache_enabled", DEFAULT_PIPELINE_CONTROL["kg_cache_enabled"]))
    data["automated_pmc_sync_enabled"] = bool(data.get("automated_pmc_sync_enabled", False))
    data["automated_triple_extraction_enabled"] = bool(data.get("automated_triple_extraction_enabled", False))
    data["automated_kg_cache_reset_enabled"] = bool(data.get("automated_kg_cache_reset_enabled", False))

    if mode != "scheduled":
        data["automated_kg_cache_reset_enabled"] = False

    if not data.get("kg_cache_scheduled_time"):
        data["kg_cache_scheduled_time"] = DEFAULT_PIPELINE_CONTROL["kg_cache_scheduled_time"]
    data["kg_cache_reset_time"] = data.get("kg_cache_scheduled_time")

    if not data.get("timezone"):
        data["timezone"] = DEFAULT_PIPELINE_CONTROL["timezone"]

    return data


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

    return _normalize_control(data)


def save_pipeline_control(updates: dict[str, Any], username: str = "") -> dict[str, Any]:
    """Persist dashboard control changes."""
    data = load_pipeline_control()
    data.update(updates or {})
    data = _normalize_control(data)
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
    control = load_pipeline_control()
    return bool(
        control.get("kg_cache_enabled", True)
        and control.get("kg_cache_mode") == "scheduled"
        and control.get("automated_kg_cache_reset_enabled", False)
    )


def is_kg_cache_available_for_chat() -> bool:
    control = load_pipeline_control()
    return bool(control.get("kg_cache_enabled", True) and control.get("kg_cache_mode") != "off")


def kg_cache_ttl_seconds(control: dict[str, Any] | None = None) -> int:
    control = control or load_pipeline_control()
    minutes = _safe_int(
        control.get("kg_cache_ttl_minutes"),
        DEFAULT_PIPELINE_CONTROL["kg_cache_ttl_minutes"],
        5,
        1440,
    )
    return minutes * 60


def maybe_update_celery_beat_schedule(control: dict[str, Any]) -> tuple[bool, str]:
    """
    Update django-celery-beat PeriodicTask rows if that package is installed.

    Cache behavior is dashboard-driven:
      - on_demand/manual/off modes disable the scheduled cache task.
      - scheduled mode enables it only when automated_kg_cache_reset_enabled is true.
    """
    try:
        from django_celery_beat.models import CrontabSchedule, IntervalSchedule, PeriodicTask
    except Exception as exc:  # noqa: BLE001
        return False, f"django-celery-beat is not available: {type(exc).__name__}: {exc}"

    control = _normalize_control(dict(control or {}))
    tz = control.get("timezone") or "America/New_York"

    pmc_hour, pmc_minute = str(control.get("pmc_sync_time") or "00:00").split(":")[:2]
    cache_hour, cache_minute = str(control.get("kg_cache_scheduled_time") or "08:00").split(":")[:2]
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
            "enabled": is_kg_cache_reset_enabled(),
        },
    )

    return True, "Celery Beat schedule updated."
