"""launchready.hub error handling, with the Hub calls stubbed (no network)."""

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from launchready import hub

RESPONSE = SimpleNamespace(
    status_code=404, headers={}, request=None, url="https://huggingface.co", text=""
)


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.fixture
def stub_info(monkeypatch):
    fake = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="abc"))
    monkeypatch.setattr(hub, "HfApi", lambda token=None: fake)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (GatedRepoError("x", response=RESPONSE), hub.GatedError),
        (RepositoryNotFoundError("x", response=RESPONSE), hub.HubError),
        (RemoteEntryNotFoundError("x", response=RESPONSE), hub.NoConfigError),
        (LocalEntryNotFoundError("x"), hub.HubUnavailableError),
        (HfHubHTTPError("429", response=RESPONSE), hub.HubUnavailableError),
    ],
)
def test_hub_failures_become_hub_errors(monkeypatch, stub_info, error, expected):
    monkeypatch.setattr(hub, "hf_hub_download", _raising(error))
    with pytest.raises(expected):
        hub._fetch("org/name", None)


def test_gated_message_is_unchanged_for_the_cli(monkeypatch, stub_info):
    error = GatedRepoError("x", response=RESPONSE)
    monkeypatch.setattr(hub, "hf_hub_download", _raising(error))
    with pytest.raises(hub.HubError, match="is gated: accept its license .* set HF_TOKEN"):
        hub._fetch("org/name", None)
