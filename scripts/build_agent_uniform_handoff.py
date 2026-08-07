from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs_agent_uniform_sampling"

OVERVIEW_MARKDOWN = """# Agent-Uniform Sampling With Bounded Evidence

## Purpose

Provide representative per-agent evaluation samples while operating under an LLM tokens-per-minute limit. Membership is selected uniformly within each tenant/agent population before token cost or session content is considered.

## Core Contract

1. Freeze eligible completed sessions and their versions.
2. Rank deterministically within each tenant/agent using identity and a seed, never token cost.
3. Persist selected membership, population size, sample size, and inclusion probability.
4. When enabled, materialize deterministic token-bounded evidence after selection.
5. Reserve evidence, prompt envelope, and completion tokens on one tokenizer basis.
6. Pace immutable selected requests under a rolling TPM limit.
7. Keep unserviceable, dropped, and nonresponse records in the selected denominator; never replace them.
8. Report response and truncation diagnostics per agent. Suppress probability-sampling intervals when selected response is incomplete.

## Evidence Priority

| Category | Mandatory when present | Weight |
|---|---:|---:|
| Final assistant outcome | Yes | 8 |
| Tool result | Yes | 7 |
| System context | Yes | 6 |
| Initial user goal | Yes | 4 |
| Later user refinement | Yes | 3 |
| Tool arguments | No | 2 |
| Earlier assistant content | No | 2 |

## Feature Flag

Bounded evidence defaults off. Flag-off queues preserve legacy raw-estimate scheduling and `OVERSIZED`. Flag-on queues require materialization before scheduling and use explicit `UNSERVICEABLE` reasons for pre-dispatch failures.

## Production Mapping

The Python JSON queue is a single-process reference adapter. In BIC Evaluations Service:

- retain the existing online rate gate;
- add completed-session uniform sampling near `SessionCompletionSelector` and `Services/Sampling`;
- persist membership and lifecycle through BJS/Cosmos patterns;
- materialize the exact judge request before `IGenAIService.ExecutePromptAsync`;
- keep CAPI as the provider path;
- add model-compatible tokenization, distributed TPM reservations, actual-usage reconciliation, and low-cardinality telemetry.

## Handoff Files

- `agent_uniform_sampling/README.md`: operator/developer entry point.
- `docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md`: normative behavior contract.
- `docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md`: target C# repository mapping and implementation sequence.
- `agent_uniform_sampling_walkthrough.ipynb`: executable, network-free reference.
- `tests/test_agent_uniform_sampling.py`: queue and sampling acceptance tests.
- `tests/test_token_representation.py`: weighted evidence acceptance tests.

## Validation

```powershell
py -3.11 -m pytest tests/test_agent_uniform_sampling.py tests/test_token_representation.py -q
py -3.11 scripts/build_agent_uniform_handoff.py
```
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(path: Path, output_dir: Path) -> str:
    if path.parent == output_dir:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.name
    return path.relative_to(ROOT).as_posix()


def _html_page(markdown_text: str) -> str:
    evidence_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{mandatory}</td><td>{weight}</td></tr>"
        for name, mandatory, weight in (
            ("Final assistant outcome", "Yes", 8),
            ("Tool result", "Yes", 7),
            ("System context", "Yes", 6),
            ("Initial user goal", "Yes", 4),
            ("Later user refinement", "Yes", 3),
            ("Tool arguments", "No", 2),
            ("Earlier assistant content", "No", 2),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent-Uniform Sampling Handoff</title>
<style>
:root{{--ink:#17242a;--muted:#55676d;--paper:#f4f6f2;--surface:#fff;--line:#ccd6d2;--green:#087f5b;--amber:#d99a1b;--red:#c95745;--blue:#2166a5}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif;line-height:1.55}}
.wrap{{width:min(1120px,calc(100% - 32px));margin:auto}} header{{padding:52px 0 38px;border-bottom:1px solid var(--line);background:linear-gradient(120deg,#eef7f1,#f8f3e8)}}
.eyebrow{{font:700 .75rem Consolas,monospace;color:var(--green);text-transform:uppercase}} h1{{font-size:clamp(2.4rem,6vw,5.2rem);line-height:.98;max-width:900px;margin:12px 0}} h2{{font-size:clamp(1.7rem,3vw,2.7rem);line-height:1.08}} h3{{margin:.2rem 0}}
.lead,.muted{{color:var(--muted)}} .lead{{font-size:1.2rem;max-width:760px}} section{{padding:48px 0;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card,.step{{background:var(--surface);border:1px solid var(--line);padding:18px;border-top:5px solid var(--green)}} .card:nth-child(2){{border-top-color:var(--amber)}} .card:nth-child(3){{border-top-color:var(--blue)}} .card:nth-child(4){{border-top-color:var(--red)}}
.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .step{{min-height:170px}} .number{{font:700 1.5rem Consolas,monospace;color:var(--green)}}
.callout{{padding:18px;background:#fff4df;border-left:5px solid var(--amber)}} table{{width:100%;border-collapse:collapse;background:var(--surface)}} th,td{{padding:10px;border:1px solid var(--line);text-align:left}} th{{font:700 .75rem Consolas,monospace;color:var(--green);text-transform:uppercase}}
code{{font-family:Consolas,monospace;color:var(--blue)}} .equation{{background:var(--ink);color:white;padding:20px;font:1rem Consolas,monospace}} footer{{padding:30px 0;color:var(--muted)}}
@media(max-width:800px){{.grid,.flow{{grid-template-columns:1fr 1fr}}}} @media(max-width:520px){{.grid,.flow{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><div class="wrap"><p class="eyebrow">Python reference handoff</p><h1>Representative membership first. Bounded execution second.</h1><p class="lead">A concrete implementation contract for bringing deterministic per-agent sampling and bounded LLM evidence into BIC Evaluations Service without letting token cost bias membership.</p></div></header>
<main>
<section><div class="wrap"><p class="eyebrow">Decision</p><h2>Freeze the sample before measuring execution cost.</h2><div class="grid"><article class="card"><h3>Sample</h3><p class="muted">Uniformly rank completed sessions inside each tenant/agent stratum.</p></article><article class="card"><h3>Persist</h3><p class="muted">Record N, n, p, seed, ranks, and selected identities.</p></article><article class="card"><h3>Materialize</h3><p class="muted">Build one deterministic bounded packet for each selected request.</p></article><article class="card"><h3>Schedule</h3><p class="muted">Reserve and pace immutable requests; never replace failures.</p></article></div></div></section>
<section><div class="wrap"><p class="eyebrow">Statistical invariant</p><h2>Every eligible session in agent a receives the same chance.</h2><div class="equation">p_a = n_a / N_a<br><br>rank = SHA256(seed || tenant || agent || session || version)</div><p class="muted">Estimated tokens, content, truncation, queue status, and judge outcome are absent from rank construction.</p></div></section>
<section><div class="wrap"><p class="eyebrow">Bounded evidence</p><h2>Spend context on task-completion evidence.</h2><table><thead><tr><th>Category</th><th>Mandatory</th><th>Weight</th></tr></thead><tbody>{evidence_rows}</tbody></table><p class="muted">If the full packet fits, it is unchanged. Otherwise mandatory segments receive a floor and remaining capacity is distributed deterministically.</p></div></section>
<section><div class="wrap"><p class="eyebrow">Lifecycle</p><h2>Pre-dispatch failures are not judge nonresponse.</h2><div class="flow"><article class="step"><span class="number">01</span><h3>Pending</h3><p class="muted">Selected and awaiting materialization or scheduling.</p></article><article class="step"><span class="number">02</span><h3>Unserviceable</h3><p class="muted">Structure, mandatory evidence, context, or TPM prevents dispatch.</p></article><article class="step"><span class="number">03</span><h3>Scheduled</h3><p class="muted">Immutable evidence and reservation are ready for dispatch.</p></article><article class="step"><span class="number">04</span><h3>Completed / Nonresponse</h3><p class="muted">Provider returned a usable result or failed after dispatch.</p></article></div><div class="callout"><strong>No replacement:</strong> every terminal record stays in the selected denominator. Partial response suppresses the reference confidence interval.</div></div></section>
<section><div class="wrap"><p class="eyebrow">BIC Evaluations Service</p><h2>Use existing service boundaries.</h2><table><thead><tr><th>Target seam</th><th>Implementation direction</th></tr></thead><tbody><tr><td><code>SessionCompletionSelector</code></td><td>Keep completion/watermark logic; replace percentage-plus-cap membership for this mode.</td></tr><tr><td><code>Services/Sampling</code></td><td>Add a completed-session uniform sampler; leave the online rate gate intact.</td></tr><tr><td>BJS / Cosmos</td><td>Persist membership, evidence readiness, reservations, attempts, and terminal reasons.</td></tr><tr><td><code>IGenAIService.ExecutePromptAsync</code></td><td>Materialize and count the exact request before this boundary; send the persisted artifact.</td></tr><tr><td>CAPI</td><td>Retain the provider path; confirm context and token-accounting semantics.</td></tr></tbody></table></div></section>
<section><div class="wrap"><p class="eyebrow">Start here</p><h2>Handoff sequence</h2><ol><li>Read <code>agent_uniform_sampling/README.md</code>.</li><li>Read <code>docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md</code>.</li><li>Use <code>docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md</code> for target files and acceptance tests.</li><li>Run <code>agent_uniform_sampling_walkthrough.ipynb</code>.</li><li>Run the focused pytest suite and compare the manifest hashes.</li></ol></div></section>
</main><footer><div class="wrap">Generated by <code>scripts/build_agent_uniform_handoff.py</code>. Markdown source length: {len(markdown_text):,} characters.</div></footer>
</body></html>"""


def build(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "agent-uniform-sampling-overview.md"
    html_path = output_dir / "agent-uniform-sampling-overview.html"
    markdown_path.write_text(OVERVIEW_MARKDOWN, encoding="utf-8")
    html_path.write_text(_html_page(OVERVIEW_MARKDOWN), encoding="utf-8")

    handoff_files = [
        ROOT / "agent_uniform_sampling" / "README.md",
        ROOT / "agent_uniform_sampling" / "prototype.py",
        ROOT / "agent_uniform_sampling_walkthrough.ipynb",
        ROOT / "docs" / "AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md",
        ROOT / "docs" / "BIC_EVALUATIONS_SERVICE_HANDOFF.md",
        ROOT / "tests" / "test_agent_uniform_sampling.py",
        ROOT / "tests" / "test_token_representation.py",
        markdown_path,
        html_path,
    ]
    manifest = {
        "schema_version": "agent-uniform-handoff-v1",
        "purpose": "Python reference handoff for BIC Evaluations Service",
        "target_repository": r"C:\Users\stangoodwin\BIC-Evaluations-Service",
        "entry_point": "agent_uniform_sampling/README.md",
        "validation_command": "py -3.11 -m pytest tests/test_agent_uniform_sampling.py tests/test_token_representation.py -q",
        "files": [
            {
                "path": _manifest_path(path, output_dir),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in handoff_files
        ],
    }
    manifest_path = output_dir / "handoff-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Agent-Uniform handoff overview artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output_dir.resolve())
    print(json.dumps({"schema_version": manifest["schema_version"], "files": len(manifest["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
