"""Shared helpers: load saved config.json fixtures (no network)."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_config(name: str) -> dict:
    return json.loads((FIXTURES / "configs" / f"{name}.json").read_text())


@pytest.fixture
def hub_cache(monkeypatch):
    """Point launchready's hub cache at the saved hub fixtures and block network access."""
    monkeypatch.setenv("LAUNCHREADY_CACHE", str(FIXTURES / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return FIXTURES / "hub"
