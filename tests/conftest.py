"""Shared helpers: load saved config.json fixtures, and keep every test off the network."""

import json
import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkBlocked(OSError):
    """A test tried to use the network."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Refuse every connection, loopback included, and fail the test even if code swallows it."""
    attempts = []

    def refuse(name):
        def blocked(*args, **kwargs):
            attempts.append(f"{name}{args[1:] if name.startswith('socket.') else args}")
            raise NetworkBlocked(f"tests have no network: {attempts[-1]}")

        return blocked

    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, method, refuse(f"socket.{method}"))
    for function in ("create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, function, refuse(function))
    # Clones run in a child process, where the patches above don't reach.
    monkeypatch.setattr("launchready.engines._git", refuse("git"))
    yield attempts
    assert not attempts, f"test tried to use the network: {attempts}"


def load_config(name: str) -> dict:
    return json.loads((FIXTURES / "configs" / f"{name}.json").read_text())


@pytest.fixture
def hub_cache(monkeypatch):
    """Point launchready's hub cache at the saved hub fixtures."""
    monkeypatch.setenv("LAUNCHREADY_CACHE", str(FIXTURES / "hub"))
    return FIXTURES / "hub"
