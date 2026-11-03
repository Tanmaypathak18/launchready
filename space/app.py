"""LaunchReady Space: will this model fit on this GPU? A thin Gradio page over logic.py.

Stays anonymous even if a token is added to the Space later: a public page holding a token would
let anyone read gated repos through it. Both lines must run before huggingface_hub is imported.
"""

import os

os.environ.pop("HF_TOKEN", None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr  # noqa: E402
from logic import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_USERS,
    EXAMPLES,
    LIGHTNING,
    check,
    how_calculated_text,
    load_gpus,
    measured_text,
)

INTRO = """# LaunchReady

Will an open model fit in your NVIDIA GPU's memory? Enter a model from Hugging Face and how you'd
use it. Nothing is downloaded, and every result is a projection, not a measurement.
"""
# Gradio breaks table cells mid-word on a phone; keep words whole and scroll the table instead.
CSS = """
.md table { display: block; overflow-x: auto; }
.md th, .md td { word-break: normal; overflow-wrap: normal; }
"""


def run(repo_id, gpu_id, context, users):
    page = check(repo_id, gpu_id, context, users)
    return page.verdict, page.memory, page.warnings, page.variants


gpus = load_gpus()

with gr.Blocks(title="LaunchReady") as demo:
    gr.Markdown(INTRO)
    repo = gr.Textbox(
        label="Model",
        value=LIGHTNING,
        info="A Hugging Face model id or link, like Qwen/Qwen3-4B",
        lines=1,
        max_lines=1,
    )
    with gr.Row():
        gpu = gr.Dropdown(label="GPU", choices=[(g.name, g.id) for g in gpus], value="rtx-5090")
        context = gr.Number(
            label="Context length",
            value=DEFAULT_CONTEXT,
            precision=0,
            minimum=1,
            info="Tokens per conversation",
        )
        users = gr.Number(
            label="Number of users",
            value=DEFAULT_USERS,
            precision=0,
            minimum=1,
            info="At the same time",
        )
    inputs = [repo, gpu, context, users]
    example_row = gr.Column()
    button = gr.Button("Check", variant="primary")
    verdict = gr.Markdown("Press **Check**, or pick an example above.")
    memory = gr.Markdown()
    warnings = gr.Markdown()
    variants = gr.Markdown()

    outputs = [verdict, memory, warnings, variants]
    button.click(run, inputs, outputs)
    repo.submit(run, inputs, outputs)
    with example_row:
        gr.Examples(
            examples=EXAMPLES,
            inputs=inputs,
            outputs=outputs,
            fn=run,
            run_on_click=True,
            cache_examples=False,
            label="Examples: click one to run it",
        )
    gr.Markdown(how_calculated_text(gpus))
    gr.Markdown(measured_text())

if __name__ == "__main__":
    demo.launch(css=CSS)
