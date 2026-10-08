"""Unit tests for ICAR/DAHD vaccination schedules, Pashu Aadhaar, and health status engine."""
from datetime import datetime, timedelta, timezone
import pytest
from app import health_tracker


def test_standard_schedules_cattle():
    """Verify ICAR/DAHD standard schedules for cattle contain major mandatory vaccines."""
    schedules = health_tracker.get_standard_schedules("cow")
    assert len(schedules) >= 5

    names = {s["name"] for s in schedules}
    assert any("FMD" in n for n in names)
    assert any("HS" in n for n in names)
    assert any("BQ" in n for n in names)
    assert any("Lumpy Skin" in n or "LSD" in n for n in names)


def test_standard_schedules_small_ruminants():
    """Verify sheep and goats receive species-appropriate schedules (e.g. PPR, Enterotoxaemia)."""
    schedules_goat = health_tracker.get_standard_schedules("goat")
    names = {s["name"] for s in schedules_goat}
    assert any("PPR" in n for n in names)
    assert any("Enterotoxaemia" in n or "ET" in n for n in names)


def test_standard_dewormers_rotation():
    """Verify that the standard dewormers rotation contains core veterinary anthelmintics."""
    dewormers = health_tracker.STANDARD_DEWORMERS
    assert len(dewormers) >= 4
    names = {d["name"] for d in dewormers}
    assert any("Albendazole" in n for n in names)
    assert any("Fenbendazole" in n for n in names)
    assert any("Ivermectin" in n for n in names)


def test_status_engine_overdue():
    """Verify that a vaccination whose next due date was in the past is flagged as overdue."""
    today = datetime.now(timezone.utc).date()
    past_date = (today - timedelta(days=20)).strftime("%Y-%m-%d")
    annotated = health_tracker.compute_health_status({"next_due_date": past_date})
    assert annotated["status_badge"] == "overdue"
    assert annotated["days_diff"] == -20


def test_status_engine_due_soon():
    """Verify that a vaccination due within 14 days is flagged as due_soon."""
    today = datetime.now(timezone.utc).date()
    soon_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    annotated = health_tracker.compute_health_status({"next_due_date": soon_date})
    assert annotated["status_badge"] == "due_soon"
    assert annotated["days_diff"] == 7


def test_status_engine_up_to_date():
    """Verify that a vaccination due > 14 days out is flagged as up_to_date."""
    today = datetime.now(timezone.utc).date()
    future_date = (today + timedelta(days=120)).strftime("%Y-%m-%d")
    annotated = health_tracker.compute_health_status({"next_due_date": future_date})
    assert annotated["status_badge"] == "up_to_date"
    assert annotated["days_diff"] == 120


def test_status_engine_completed():
    """Verify that a one-time lifetime vaccine with no next due date is marked completed."""
    annotated = health_tracker.compute_health_status({"next_due_date": None})
    assert annotated["status_badge"] == "completed"
    assert annotated["days_diff"] is None
