"""Model brief rendering (Phase 2; BUILD_PLAN.md template 7.1).

Renders templates/brief.md.j2 with Jinja2 to briefs/<org>__<model>.md, combining hub metadata,
projected hardware requirements, engine readiness and available quantized variants. The
"Ecosystem impact" and "Open questions and risks" sections are left for hand-writing.
"""
