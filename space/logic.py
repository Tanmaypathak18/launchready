"""What the LaunchReady Space shows: verdict, tables, wording. No Gradio, so tests run offline.

The math lives in launchready; this module picks a verdict from its numbers and writes Markdown
for a reader who isn't a developer. Everything here is projected. Text that comes from a repo
(warnings quote config values) is escaped before it reaches the page.
"""

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from launchready.hub import (
    GatedError,
    HubError,
    HubModel,
    HubUnavailableError,
    NoConfigError,
    load_model,
)
from launchready.memory import (
    BITS_PER_WEIGHT,
    GIB,
    GPU_MEMORY_UTILIZATION,
    RUNTIME_RESERVE_BYTES,
    Gpu,
    load_gpus,
)
from launchready.report import PRECISION_LABELS, FitReport, hub_file_sizes, project
from launchready.variants import Variant, load_variants

log = logging.getLogger("launchready.space")

KV_DTYPE = "fp16"
GITHUB = "https://github.com/Tanmaypathak18/launchready"
RAW = "https://raw.githubusercontent.com/Tanmaypathak18/launchready"
# Saved results, briefs and recipes don't change with the code, so links stay on this commit.
RESULTS_COMMIT = "768eed8691339d24bf61dfe38b8e85dda368bf31"
LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DEFAULT_CONTEXT = 32_768
DEFAULT_USERS = 8
EXAMPLES = [
    [LIGHTNING, "rtx-5090", 32_768, 8],
    [LIGHTNING, "rtx-4090", 32_768, 8],
    ["Qwen/Qwen3-4B", "rtx-4090", 32_768, 8],
    ["deepseek-ai/DeepSeek-V3", "rtx-5090", 32_768, 8],
    ["ibm-granite/granite-4.0-h-tiny", "tesla-t4", 4_096, 8],
    ["meta-llama/Llama-3.3-70B-Instruct", "rtx-5090", 32_768, 8],
]

FITS, TOO_FEW, NO_FIT, CANT_TELL = "fits", "too few users", "does not fit", "can't tell"
MEMORY_ONLY = "This is about memory only, not speed."
HYBRID_CAUTION = (
    "**Caution:** this model is a hybrid design, and projections for hybrids can be far too high. "
    "On the one hybrid measured so far (granite-4.0-h-tiny on 2× Tesla T4), real capacity was "
    "about 43% of the projection: 118 projected, 50.6 measured. Treat the number of users here "
    "as an upper limit."
)
ENGINE_NOTE = (
    "Whether inference software supports a model isn't checked here. The `launchready brief` "
    f"command does that; see the [Nemotron 3.5 Lightning brief]({GITHUB}/blob/{RESULTS_COMMIT}/"
    f"briefs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md) for an example."
)
_HUB_URL = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", re.IGNORECASE)
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+|~!>-])")


class BadInput(ValueError):
    """Something typed into the form can't be used; the message is shown as it is."""


@dataclass
class Verdict:
    state: str
    precision: str | None = None
    max_users: int = 0


@dataclass
class Page:
    verdict: str
    memory: str = ""
    warnings: str = ""
    variants: str = ""


def esc(text: object) -> str:
    """Escape repo-derived text for Markdown and HTML."""
    return _MARKDOWN.sub(r"\\\1", html.escape(str(text), quote=False))


def gib(n: float) -> str:
    return f"{n / GIB:,.2f}"


def normalize_repo_id(text: str | None) -> str:
    """`org/name` from what was typed: an id, or a huggingface.co link to a model."""
    value = (text or "").strip().strip("/")
    if not value:
        raise BadInput("Enter a model, like `Qwen/Qwen3-4B`.")
    if _HUB_URL.match(value):
        parts = _HUB_URL.sub("", value).split("?")[0].split("#")[0].split("/")
        if parts[0] in ("datasets", "spaces", "models"):
            parts = parts[1:] if parts[0] == "models" else []
        value = "/".join(parts[:2])
    try:
        validate_repo_id(value)
    except HFValidationError as e:
        raise BadInput(
            "That doesn't look like a Hugging Face model. Use the form `org/name`, like "
            "`Qwen/Qwen3-4B`, or paste the link to the model's page."
        ) from e
    return value


def whole_number(value: float | None, name: str) -> int:
    if value is None or value < 1 or value != int(value):
        raise BadInput(f"{name} must be a whole number of at least 1.")
    return int(value)


def decide(report: FitReport, gpu: Gpu, users: int) -> Verdict:
    """The highest-quality precision this GPU supports that reaches `users`.

    Precisions the GPU doesn't support never count: their memory assumes the stored size, and an
    upcast could use more. BITS_PER_WEIGHT is ordered from highest quality down.
    """
    if not report.weights:
        return Verdict(CANT_TELL)
    fits = report.fits[gpu.id]
    supported = [p for p in BITS_PER_WEIGHT if p in fits and fits[p]["native"]]
    for p in supported:
        if fits[p]["max_users"] >= users:
            return Verdict(FITS, p, fits[p]["max_users"])
    loadable = [p for p in supported if fits[p]["max_users"] >= 1]
    if loadable:
        best = max(loadable, key=lambda p: fits[p]["max_users"])
        return Verdict(TOO_FEW, best, fits[best]["max_users"])
    return Verdict(NO_FIT, min(supported, key=lambda p: report.weights[p]))


def verdict_text(
    verdict: Verdict,
    report: FitReport,
    gpu: Gpu,
    repo_id: str,
    users: int,
    context: int,
    hub_files: dict[str, tuple[str, int]],
) -> str:
    name = esc(repo_id.split("/")[-1])
    wanted = f"{users:,} user{'s' if users != 1 else ''} at {context:,} tokens each"
    label = PRECISION_LABELS.get(verdict.precision, "")
    if verdict.state == FITS:
        lines = [
            "## Fits (projected)",
            f"{name} should fit on the **{esc(gpu.name)}** with {wanted}, using the **{label}** "
            f"version. There is room for up to {verdict.max_users:,} users.",
        ]
    elif verdict.state == TOO_FEW:
        lines = [
            f"## Does not fit {users:,} users (projected)",
            f"{name} loads on the **{esc(gpu.name)}**, but there is only room for "
            f"{verdict.max_users:,} at {context:,} tokens each, using the **{label}** version.",
        ]
    elif verdict.state == NO_FIT:
        weights = report.weights[verdict.precision]
        room = report.fits[gpu.id][verdict.precision]["kv_budget_bytes"] + weights
        lines = [
            "## Does not fit (projected)",
            f"{name} does not fit on the **{esc(gpu.name)}**. The smallest version this GPU "
            f"supports, **{label}**, needs {gib(weights)} GiB plus {gib(report.seq.total)} GiB "
            f"per user, and {gib(room)} GiB is available.",
        ]
    else:
        return "\n\n".join(
            [
                "## Can't tell",
                f"{name} has no safetensors files, so its size can't be read without downloading "
                "it. LaunchReady never downloads model weights.",
            ]
        )
    if verdict.state != NO_FIT and verdict.precision not in hub_files:
        lines.append(f"No {label} version was found on the Hub, so its size is projected.")
    if report.arch.max_context and context > report.arch.max_context:
        lines.append(f"This model's limit is {report.arch.max_context:,} tokens per conversation.")
    if report.seq.mamba.dtype is not None and verdict.state != NO_FIT:
        lines.append(HYBRID_CAUTION)
    lines.append(
        f"{MEMORY_ONLY} Projected: usable memory = GPU memory × {GPU_MEMORY_UTILIZATION:.2f} − "
        f"weights − {RUNTIME_RESERVE_BYTES // GIB} GiB; users = usable memory ÷ memory per user."
    )
    return "\n\n".join(lines)


def memory_text(
    report: FitReport,
    gpu: Gpu,
    model: HubModel,
    context: int,
    hub_files: dict[str, tuple[str, int]],
) -> str:
    lines = [
        "## Memory (projected)",
        f"{esc(model.repo_id)} @ `{model.revision[:7]}`, license "
        f"{esc(model.license or 'not declared')}.",
    ]
    if report.weights:
        lines.append(
            f"On {esc(gpu.name)} ({gpu.memory_gib:g} GiB) at {context:,} tokens per user. Figures "
            "are memory only, not speed.\n\n"
            "| Version | Bits per weight | Weights GiB (projected) | Hub file GiB | "
            "Max users (projected) | Supported by this GPU |\n|---|--:|--:|--:|--:|---|\n"
            + "\n".join(
                f"| {PRECISION_LABELS[p]} | {BITS_PER_WEIGHT[p]:g} | "
                f"{gib(report.projected_weights[p])} | "
                f"{gib(hub_files[p][1]) if p in hub_files else '—'} | "
                f"{report.fits[gpu.id][p]['max_users']:,} | "
                f"{'Yes' if report.fits[gpu.id][p]['native'] else 'No, not counted'} |"
                for p in report.weights
            )
        )
        lines.append(
            "Max users uses the Hub file size where there is one, because real files are often "
            "larger than the projection. A file size is not measured GPU memory."
        )
    per_user = f"KV cache {gib(report.seq.attention_bytes)} GiB"
    if report.seq.mamba.dtype is not None:
        per_user += (
            f" + Mamba state {gib(report.seq.mamba_bytes)} GiB = {gib(report.seq.total)} GiB"
        )
    lines.append(
        f"Memory per user at {context:,} tokens (projected, KV cache {KV_DTYPE.upper()}): "
        f"{per_user}."
    )
    return "\n\n".join(lines)


def warnings_text(report: FitReport, extra: list[str]) -> str:
    if not (report.warnings or report.notes or extra):
        return ""
    lines = ["## Technical warnings", "The same warnings the `launchready fit` command gives."]
    items = [f"- **Warning:** {esc(w)}" for w in extra + report.warnings]
    items += [f"- Note: {esc(n)}" for n in report.notes]
    return "\n\n".join(lines) + "\n\n" + "\n".join(items)


def _variant_size(v: Variant) -> str:
    if v.gguf:
        shown = [f"{q} {gib(v.gguf[q])} GiB" for q in ("Q4_K_M", "Q8_0") if q in v.gguf]
        return " · ".join(shown) or f"{len(v.gguf)} quant types"
    return f"{gib(v.total_bytes)} GiB" if v.total_bytes else "—"


def _model_link(repo_id: str, text: str | None = None, path: str = "") -> str:
    return f"[{esc(text or repo_id)}](https://huggingface.co/{quote(repo_id)}{path})"


def variants_text(model: HubModel, variants: list[Variant]) -> str:
    lines = ["## Versions and links"]
    rows = [v for v in variants if v.format != "drafter"]
    if rows:
        lines.append(
            "Other versions of this model on the Hub, with file sizes:\n\n"
            "| Format | Repo | Size | Downloads |\n|---|---|--:|--:|\n"
            + "\n".join(
                f"| {esc(v.format)} | {_model_link(v.repo_id)} | {_variant_size(v)} | "
                f"{v.downloads:,} |"
                for v in rows
            )
        )
    else:
        lines.append("No other versions were found on the Hub.")
    links = [
        _model_link(model.repo_id, "Model page", f"/tree/{model.revision}"),
        _model_link(model.repo_id, "config.json", f"/blob/{model.revision}/config.json"),
        f"[All quantized versions](https://huggingface.co/models?other="
        f"base_model:quantized:{quote(model.repo_id)})",
    ]
    lines.append(" · ".join(links))
    lines.append(ENGINE_NOTE)
    return "\n\n".join(lines)


def error_text(error: Exception, repo_id: str) -> str:
    """What to tell a reader when a repo can't be sized. Never asks for a token."""
    name = esc(repo_id)
    if isinstance(error, GatedError):
        body = (
            f"The owner of {name} requires people to accept a license first. This free tool has "
            "no login, so it can't read the model. Accept the license on "
            f"{_model_link(repo_id, 'its Hugging Face page')}, then use the `launchready` "
            f"command-line tool ([GitHub]({GITHUB})) with your own account."
        )
        return f"## License required\n\n{body}"
    if isinstance(error, NoConfigError):
        body = (
            f"{name} has no config.json, so it can't be sized. GGUF download repos usually don't "
            "have one. Try the original model's repo instead."
        )
        return f"## Can't size this repo\n\n{body}"
    if isinstance(error, HubUnavailableError):
        body = "Hugging Face didn't answer, or this free tool is being rate limited. "
        return f"## Try again in a minute\n\n{body}Your input was fine."
    if isinstance(error, HubError):
        return (
            f"## Model not found\n\n{name} wasn't found on Hugging Face. Check the spelling. "
            "Private models can't be read here."
        )
    if isinstance(error, ValueError):
        return (
            f"## Can't size this repo\n\nLaunchReady can't read the config of {name}. It may "
            f"not be a text model, or it may use an older config format. Details: {esc(error)}."
        )
    return "## Something went wrong\n\nThis model couldn't be checked. Please try another one."


def check(repo_text: str | None, gpu_id: str, context: float | None, users: float | None) -> Page:
    """The whole page for one form submission. Never raises: failures become a message."""
    repo_id = ""
    try:
        repo_id = normalize_repo_id(repo_text)
        context = whole_number(context, "Context length")
        users = whole_number(users, "Number of users")
        gpu = next(g for g in load_gpus() if g.id == gpu_id)
        model = load_model(repo_id)
        extra, variants = [], []
        try:
            variants = load_variants(repo_id)
        except Exception as e:
            log.warning("variants lookup failed: %s", type(e).__name__)
            extra.append("couldn't list other versions: weights below are projected")
        hub_files = hub_file_sizes(model, variants)
        report = project(model, context, KV_DTYPE, [gpu], hub_files)
    except BadInput as e:
        return Page(f"## Check your input\n\n{e}")
    except (HubError, ValueError) as e:
        return Page(error_text(e, repo_id))
    except Exception as e:
        log.exception("unexpected failure: %s", type(e).__name__)
        return Page(error_text(e, repo_id))
    verdict = decide(report, gpu, users)
    return Page(
        verdict=verdict_text(verdict, report, gpu, repo_id, users, context, hub_files),
        memory=memory_text(report, gpu, model, context, hub_files),
        warnings=warnings_text(report, extra),
        variants=variants_text(model, variants),
    )


def how_calculated_text(gpus: list[Gpu]) -> str:
    lines = [
        "## How this is calculated",
        "Every number above is **projected**, not measured. No model weights are downloaded: "
        "only config.json and the safetensors file headers. Figures are memory only, not speed.",
        "- **Weights** = parameters × bits per weight ÷ 8, or the Hub file size where one exists\n"
        "- **Memory per user** = KV cache per token × context length + Mamba state, if any\n"
        f"- **Usable memory** = GPU memory × {GPU_MEMORY_UTILIZATION:.2f} − weights − "
        f"{RUNTIME_RESERVE_BYTES // GIB} GiB\n"
        "- **Max users** = usable memory ÷ memory per user, rounded down",
        "GPU memory is the size on NVIDIA's spec pages read as GiB, checked 2026-11-03, except "
        "the Tesla T4, which uses the measured 15 GiB.",
    ]
    unverified = [f"- **{esc(g.name)}:** {esc(g.unverified)}" for g in gpus if g.unverified]
    if unverified:
        lines.append("Unverified GPU details:\n\n" + "\n".join(unverified))
    return "\n\n".join(lines)


def measured_text() -> str:
    text = (Path(__file__).parent / "measured.md").read_text()
    return text.replace("{GITHUB}", GITHUB).replace("{RAW}", RAW).replace("{REF}", RESULTS_COMMIT)
