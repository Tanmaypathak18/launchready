"""The suite can't reach the network, even from code that swallows the error."""

import socket
import urllib.request

import pytest
from conftest import NetworkBlocked

from launchready import engines, hub
from launchready.hub import load_model


@pytest.mark.parametrize(
    "call",
    [
        lambda: socket.create_connection(("huggingface.co", 443)),
        lambda: socket.getaddrinfo("huggingface.co", 443),
        lambda: socket.socket().connect(("1.1.1.1", 443)),
        lambda: socket.socket().connect(("127.0.0.1", 7860)),
        lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b"x", ("1.1.1.1", 53)),
        lambda: engines._git("clone", "https://github.com/huggingface/transformers.git"),
    ],
)
def test_direct_connections_are_refused(no_network, call):
    with pytest.raises(NetworkBlocked):
        call()
    assert no_network
    no_network.clear()


def test_urllib_is_refused(no_network):
    with pytest.raises(OSError):
        urllib.request.urlopen("https://huggingface.co", timeout=1)
    assert no_network
    no_network.clear()


def test_a_hub_lookup_is_refused(no_network, tmp_path, monkeypatch):
    monkeypatch.setenv("LAUNCHREADY_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(Exception):  # noqa: B017 - huggingface_hub wraps the error as it likes
        hub.load_model("Qwen/Qwen3-4B")
    assert no_network
    no_network.clear()


def test_saved_fixtures_need_no_network(hub_cache):
    assert load_model("Qwen/Qwen3-4B").repo_id == "Qwen/Qwen3-4B"
