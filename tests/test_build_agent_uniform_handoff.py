from __future__ import annotations

import hashlib
import json

from scripts.build_agent_uniform_handoff import build


def test_build_agent_uniform_handoff_outputs_reproducible_bundle(tmp_path) -> None:
    first = build(tmp_path)
    first_markdown = (tmp_path / "agent-uniform-sampling-overview.md").read_bytes()
    first_html = (tmp_path / "agent-uniform-sampling-overview.html").read_bytes()

    second = build(tmp_path)

    assert first == second
    assert (tmp_path / "agent-uniform-sampling-overview.md").read_bytes() == first_markdown
    assert (tmp_path / "agent-uniform-sampling-overview.html").read_bytes() == first_html

    manifest = json.loads((tmp_path / "handoff-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "agent-uniform-handoff-v1"
    assert manifest["entry_point"] == "agent_uniform_sampling/README.md"
    assert len(manifest["files"]) == 9

    by_path = {row["path"]: row for row in manifest["files"]}
    markdown_row = by_path["agent-uniform-sampling-overview.md"]
    assert markdown_row["sha256"] == hashlib.sha256(first_markdown).hexdigest()
    assert markdown_row["bytes"] == len(first_markdown)

    html_text = first_html.decode("utf-8")
    assert "Representative membership first" in html_text
    assert "SessionCompletionSelector" in html_text