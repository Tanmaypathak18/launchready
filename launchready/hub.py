"""Hugging Face Hub access without downloading weights (Phase 1).

Fetches config.json, model info (release date, license, gated status), safetensors parameter
counts by dtype from file headers via `huggingface_hub.get_safetensors_metadata`, and total
safetensors file size. Results are cached as JSON under ~/.cache/launchready/hub (override with
LAUNCHREADY_CACHE). Reads HF_TOKEN from the environment for gated repos and raises HubError (or
a subclass, for front ends that word these cases themselves) instead of crashing.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    NotASafetensorsRepoError,
    RepositoryNotFoundError,
)


class HubError(Exception):
    """A repo couldn't be read: gated, missing, or unreachable."""


class GatedError(HubError):
    """The repo needs a license click and a token."""


class NoConfigError(HubError):
    """The repo has no config.json: GGUF-only, or not a Transformers model."""


class HubUnavailableError(HubError):
    """The Hub couldn't answer: rate limited, down, or offline."""


@dataclass
class HubModel:
    repo_id: str
    revision: str
    created_at: str | None
    last_modified: str | None
    license: str | None
    gated: str | bool
    parameters: dict[str, int]  # safetensors element counts by dtype, from headers
    safetensors_bytes: int
    config: dict
    warnings: list[str] = field(default_factory=list)


def cache_dir() -> Path:
    return Path(os.environ.get("LAUNCHREADY_CACHE", Path.home() / ".cache" / "launchready" / "hub"))


def _cache_path(repo_id: str, revision: str | None) -> Path:
    name = repo_id.replace("/", "__") + (f"@{revision}" if revision else "")
    return cache_dir() / f"{name}.json"


def load_model(repo_id: str, revision: str | None = None, refresh: bool = False) -> HubModel:
    """Return hub metadata for a repo, from the disk cache when present."""
    path = _cache_path(repo_id, revision)
    if path.exists() and not refresh:
        return HubModel(**json.loads(path.read_text()))
    model = _fetch(repo_id, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=2, sort_keys=True) + "\n")
    return model


def _fetch(repo_id: str, revision: str | None) -> HubModel:
    token = os.environ.get("HF_TOKEN")
    try:
        info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True)
        config_path = hf_hub_download(repo_id, "config.json", revision=info.sha, token=token)
    except GatedRepoError as e:
        raise GatedError(
            f"{repo_id} is gated: accept its license on huggingface.co and set HF_TOKEN"
        ) from e
    except RepositoryNotFoundError as e:
        raise HubError(f"{repo_id} not found (or private without HF_TOKEN)") from e
    except LocalEntryNotFoundError as e:
        raise HubUnavailableError(f"couldn't reach the Hub for {repo_id}: {e}") from e
    except EntryNotFoundError as e:
        raise NoConfigError(
            f"{repo_id} has no config.json, so it can't be sized: GGUF-only repos don't carry "
            "one, use the original model's repo"
        ) from e
    except HfHubHTTPError as e:
        raise HubUnavailableError(f"couldn't read {repo_id} from the Hub: {e}") from e

    warnings = []
    try:
        meta = get_safetensors_metadata(repo_id, revision=info.sha, token=token)
        parameters = dict(meta.parameter_count)
    except NotASafetensorsRepoError:
        parameters = {}
        warnings.append("no safetensors files: parameter count and weight sizes unavailable")
    except HfHubHTTPError as e:
        raise HubUnavailableError(f"couldn't read {repo_id} from the Hub: {e}") from e

    card = info.card_data.to_dict() if info.card_data else {}
    license_ = card.get("license_name") or card.get("license")
    return HubModel(
        repo_id=repo_id,
        revision=info.sha,
        created_at=info.created_at.isoformat() if info.created_at else None,
        last_modified=info.last_modified.isoformat() if info.last_modified else None,
        license=license_,
        gated=info.gated,
        parameters=parameters,
        safetensors_bytes=sum(
            s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors")
        ),
        config=json.loads(Path(config_path).read_text()),
        warnings=warnings,
    )
