"""Self-contained HTML report generator for sampling v3 artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any

REPORT_VERSION = "agent365-sampling-v3-report-v1"
DEFAULT_OUTPUT_NAME = "agent365-sampling-v3-report.html"
DEFAULT_INPUT_DIR = Path("outputs_sampling_v3") / "runs"
REPORT_MANIFEST_NAME = "report_manifest.json"


V3_METHOD_LABELS = {
    "random_sampling_token_priority": "Random token-priority",
    "adaptive_minhash_32x4_token": "Adaptive MinHash 32x4 token",
    "adaptive_embedding_fullsession_token": "Adaptive embedding full-session token",
}

V3_COLORS = {
    "random_sampling_token_priority": "#2f74a8",
    "adaptive_minhash_32x4_token": "#177d65",
    "adaptive_embedding_fullsession_token": "#c7641d",
}


@dataclass(frozen=True)
class V3ReportInputs:
    aggregate: Path
    runs_jsonl: Path
    quadrant: Path
    throughput: Path
    corpus_audit: Path
    token_inventory: Path
    budget_manifest: Path
    embedding_ledger: Path
    selected_membership: Path
    methodology_delta: Path
    manifest: Path
    run_source_manifest: Path | None = None
    search_cleanup_audit: Path | None = None


@dataclass(frozen=True)
class LoadedV3Artifacts:
    aggregate: dict[str, Any]
    runs_jsonl: list[dict[str, Any]]
    quadrant: dict[str, Any]
    throughput: dict[str, Any]
    corpus_audit: dict[str, Any]
    token_inventory: list[dict[str, Any]]
    budget_manifest: dict[str, Any]
    embedding_ledger: dict[str, Any]
    selected_membership: dict[str, Any]
    methodology_delta_text: str
    manifest: dict[str, Any]
    run_source_manifest: dict[str, Any] | None
    search_cleanup_audit: dict[str, Any] | None
    source_paths: dict[str, Path]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object line in {path}")
        rows.append(row)
    return rows


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _method_label(name: str) -> str:
    return V3_METHOD_LABELS.get(name, name.replace("_", " ").title())


def _method_color(name: str) -> str:
    return V3_COLORS.get(name, "#546175")


def default_inputs(base_dir: Path) -> V3ReportInputs:
    return V3ReportInputs(
        aggregate=base_dir / "aggregate.json",
        runs_jsonl=base_dir / "runs.jsonl",
        quadrant=base_dir / "quadrant.json",
        throughput=base_dir / "throughput.json",
        corpus_audit=base_dir / "corpus_audit.json",
        token_inventory=base_dir / "token_inventory.jsonl",
        budget_manifest=base_dir / "budget_manifest.json",
        embedding_ledger=base_dir / "embedding_ledger.json",
        selected_membership=base_dir / "selected_membership.json",
        methodology_delta=base_dir / "methodology_delta.md",
        manifest=base_dir / "manifest.json",
        run_source_manifest=(base_dir / "run_source_manifest.json"),
        search_cleanup_audit=(base_dir / "search_cleanup_audit.json"),
    )


def _read_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object at {path}")
    return payload


def load_v3_artifacts(inputs: V3ReportInputs) -> LoadedV3Artifacts:
    source_paths = {
        "aggregate": inputs.aggregate,
        "runs_jsonl": inputs.runs_jsonl,
        "quadrant": inputs.quadrant,
        "throughput": inputs.throughput,
        "corpus_audit": inputs.corpus_audit,
        "token_inventory": inputs.token_inventory,
        "budget_manifest": inputs.budget_manifest,
        "embedding_ledger": inputs.embedding_ledger,
        "selected_membership": inputs.selected_membership,
        "methodology_delta": inputs.methodology_delta,
        "manifest": inputs.manifest,
    }
    if inputs.run_source_manifest is not None:
        source_paths["run_source_manifest"] = inputs.run_source_manifest
    if inputs.search_cleanup_audit is not None:
        source_paths["search_cleanup_audit"] = inputs.search_cleanup_audit
    return LoadedV3Artifacts(
        aggregate=_read_json(inputs.aggregate),
        runs_jsonl=_read_jsonl(inputs.runs_jsonl),
        quadrant=_read_json(inputs.quadrant),
        throughput=_read_json(inputs.throughput),
        corpus_audit=_read_json(inputs.corpus_audit),
        token_inventory=_read_jsonl(inputs.token_inventory),
        budget_manifest=_read_json(inputs.budget_manifest),
        embedding_ledger=_read_json(inputs.embedding_ledger),
        selected_membership=_read_json(inputs.selected_membership),
        methodology_delta_text=inputs.methodology_delta.read_text(encoding="utf-8"),
        manifest=_read_json(inputs.manifest),
        run_source_manifest=_read_json_optional(inputs.run_source_manifest),
        search_cleanup_audit=_read_json_optional(inputs.search_cleanup_audit),
        source_paths=source_paths,
    )


def _verify_artifact_manifest(artifacts: LoadedV3Artifacts) -> None:
    man = artifacts.manifest
    if str(man.get("version")) != "sampling-v3-manifest-v1":
        raise ValueError("manifest version must be sampling-v3-manifest-v1")

    listed = man.get("artifacts")
    if not isinstance(listed, dict):
        raise ValueError("manifest artifacts must be an object")

    key_to_path = {
        "aggregate": artifacts.source_paths["aggregate"],
        "runs_jsonl": artifacts.source_paths["runs_jsonl"],
        "quadrant": artifacts.source_paths["quadrant"],
        "throughput": artifacts.source_paths["throughput"],
        "corpus_audit": artifacts.source_paths["corpus_audit"],
        "token_inventory": artifacts.source_paths["token_inventory"],
        "budget_manifest": artifacts.source_paths["budget_manifest"],
        "embedding_ledger": artifacts.source_paths["embedding_ledger"],
        "selected_membership": artifacts.source_paths["selected_membership"],
        "methodology_delta": artifacts.source_paths["methodology_delta"],
    }
    missing = [key for key in key_to_path if key not in listed]
    if missing:
        raise ValueError(f"manifest is missing artifact entries: {','.join(missing)}")

    for key, expected_path in key_to_path.items():
        entry = listed.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry must be an object for {key}")
        recorded_sha = str(entry.get("sha256") or "")
        if len(recorded_sha) != 64:
            raise ValueError(f"manifest sha256 missing/invalid for {key}")
        computed_sha = _sha(expected_path)
        if computed_sha != recorded_sha:
            raise ValueError(f"manifest hash mismatch for {key}")
        recorded_bytes = int(entry.get("bytes") or -1)
        actual_bytes = int(expected_path.stat().st_size)
        if recorded_bytes != actual_bytes:
            raise ValueError(f"manifest size mismatch for {key}")

    optional_entries = {
        "run_source_manifest": artifacts.source_paths.get("run_source_manifest"),
        "search_cleanup_audit": artifacts.source_paths.get("search_cleanup_audit"),
    }
    for key, expected_path in optional_entries.items():
        if expected_path is None or not expected_path.exists():
            continue
        entry = listed.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest is missing artifact entry for optional file: {key}")
        recorded_sha = str(entry.get("sha256") or "")
        if len(recorded_sha) != 64:
            raise ValueError(f"manifest sha256 missing/invalid for {key}")
        computed_sha = _sha(expected_path)
        if computed_sha != recorded_sha:
            raise ValueError(f"manifest hash mismatch for {key}")
        recorded_bytes = int(entry.get("bytes") or -1)
        actual_bytes = int(expected_path.stat().st_size)
        if recorded_bytes != actual_bytes:
            raise ValueError(f"manifest size mismatch for {key}")


def validate_v3_artifacts(artifacts: LoadedV3Artifacts) -> None:
    if str(artifacts.aggregate.get("version")) != "sampling-v3-bundle-v1":
        raise ValueError("aggregate version must be sampling-v3-bundle-v1")
    if str(artifacts.corpus_audit.get("version")) != "sampling-v3-corpus-audit-v1":
        raise ValueError("corpus audit version must be sampling-v3-corpus-audit-v1")
    if str(artifacts.selected_membership.get("version")) != "sampling-v3-selected-membership-v1":
        raise ValueError("selected membership version must be sampling-v3-selected-membership-v1")

    quadrant_version = str(artifacts.quadrant.get("version"))
    if quadrant_version != "sampling-v3-quadrant-v1":
        raise ValueError("quadrant version must be sampling-v3-quadrant-v1")
    throughput_version = str(artifacts.throughput.get("version"))
    if throughput_version != "sampling-v3-throughput-v1":
        raise ValueError("throughput version must be sampling-v3-throughput-v1")

    budget_manifest = artifacts.budget_manifest
    if not isinstance(budget_manifest.get("outcome"), dict):
        raise ValueError("budget manifest missing outcome")
    if not isinstance(budget_manifest.get("quadrant"), dict):
        raise ValueError("budget manifest missing quadrant")
    if not isinstance(budget_manifest.get("throughput"), dict):
        raise ValueError("budget manifest missing throughput")

    if not artifacts.runs_jsonl:
        raise ValueError("runs.jsonl must contain at least one row")
    if not artifacts.token_inventory:
        raise ValueError("token_inventory.jsonl must contain at least one row")

    if artifacts.run_source_manifest is not None:
        if str(artifacts.run_source_manifest.get("version")) != "sampling-v3-run-source-manifest-v1":
            raise ValueError("run source manifest version must be sampling-v3-run-source-manifest-v1")
        if not isinstance(artifacts.run_source_manifest.get("source_hashes"), dict):
            raise ValueError("run source manifest source_hashes must be an object")

    if artifacts.search_cleanup_audit is not None:
        if str(artifacts.search_cleanup_audit.get("version")) != "sampling-v3-search-cleanup-audit-v1":
            raise ValueError("search cleanup audit version must be sampling-v3-search-cleanup-audit-v1")
        remaining_raw = artifacts.search_cleanup_audit.get("remaining_count")
        if remaining_raw is None or int(remaining_raw) < 0:
            raise ValueError("search cleanup audit remaining_count must be >= 0")
        if not isinstance(artifacts.search_cleanup_audit.get("scopes"), dict):
            raise ValueError("search cleanup audit scopes must be an object")

    _verify_artifact_manifest(artifacts)


def _svg_outcome_bars(rows: list[dict[str, Any]]) -> str:
    methods = sorted({str(row.get("method") or "") for row in rows})
    budgets = sorted({int(row.get("budget_tokens") or 0) for row in rows})
    if not methods or not budgets:
        return "<p>Outcome aggregates are unavailable.</p>"

    mae_map: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (str(row.get("method") or ""), int(row.get("budget_tokens") or 0))
        mae_map[key] = float((((row.get("mae") or {}).get("mean")) or 0.0))

    width = 920
    height = 320
    ml, mr, mt, mb = 70, 24, 28, 74
    pw = width - ml - mr
    ph = height - mt - mb
    slot = pw / max(1, len(budgets))
    bar_w = max(8.0, min(24.0, slot / (len(methods) + 1)))
    y_max = max([mae_map.get((m, b), 0.0) for m in methods for b in budgets] + [0.001])

    grid = []
    for i in range(6):
        v = y_max * i / 5
        y = mt + ph - (ph * (v / y_max if y_max else 0.0))
        grid.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 6}" y="{y + 4:.2f}" text-anchor="end" class="axis">{escape(_num(v, 3))}</text>'
        )

    bars = []
    xlabels = []
    for bi, budget in enumerate(budgets):
        base = ml + bi * slot
        xlabels.append(
            f'<text x="{base + slot / 2:.2f}" y="{height - 28}" text-anchor="middle" class="axis">{budget}</text>'
        )
        for mi, method in enumerate(methods):
            value = mae_map.get((method, budget), 0.0)
            h = ph * (value / y_max if y_max else 0.0)
            x = base + (slot - len(methods) * bar_w) / 2 + mi * bar_w
            y = mt + ph - h
            label = f"{_method_label(method)} budget {budget} MAE {_num(value, 3)}"
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 1:.2f}" height="{h:.2f}" '
                f'fill="{_method_color(method)}" aria-label="{escape(label)}" />'
            )

    legend = "".join(
        f'<span><i style="background:{_method_color(method)}"></i>{escape(_method_label(method))}</span>'
        for method in methods
    )

    return (
        "<figure class=\"chart\"><figcaption>Outcome MAE by exact token budget</figcaption>"
        f"<div class=\"chart-legend\">{legend}</div>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" "
        "aria-label=\"Outcome MAE by exact token budget\">"
        + "".join(grid)
        + "".join(bars)
        + "".join(xlabels)
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis">Exact budget tokens</text>'
        + f'<text x="16" y="{mt + ph / 2:.2f}" transform="rotate(-90 16 {mt + ph / 2:.2f})" class="axis">MAE</text>'
        + "</svg></div></figure>"
    )


def _topline_metrics(artifacts: LoadedV3Artifacts) -> dict[str, Any]:
    pop = int(artifacts.aggregate.get("population_count") or 0)
    runtime_s = float(artifacts.aggregate.get("runtime_seconds") or 0.0)
    generated_at = str(artifacts.aggregate.get("generated_at") or "")
    truncated = sum(1 for row in artifacts.token_inventory if bool(row.get("truncated")))
    emitted_total = sum(int(row.get("emitted_tokens") or 0) for row in artifacts.token_inventory)
    max_emitted_tokens = max((int(row.get("emitted_tokens") or 0) for row in artifacts.token_inventory), default=0)
    packet_count = len(artifacts.token_inventory)

    return {
        "population": pop,
        "runtime_seconds": runtime_s,
        "generated_at": generated_at,
        "truncated_count": truncated,
        "emitted_total": emitted_total,
        "packet_count": packet_count,
        "max_emitted_tokens": max_emitted_tokens,
    }


def _render_overview(artifacts: LoadedV3Artifacts) -> str:
    top = _topline_metrics(artifacts)
    source = artifacts.corpus_audit.get("source_files") or {}
    src_rows = []
    for name, row in sorted(source.items()):
        if not isinstance(row, dict):
            continue
        counts = row.get("counts") or {}
        src_rows.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td>{int(counts.get('units') or 0)}</td>"
            f"<td>{int(counts.get('agents') or 0)}</td>"
            f"<td>{escape(_pct(float(row.get('label_pass_rate') or 0.0), 1))}</td>"
            "</tr>"
        )

    outcome_tiers = ((artifacts.budget_manifest.get("outcome") or {}).get("legacy_outcome_tiers_pct") or [])
    quadrant_tiers = ((artifacts.budget_manifest.get("quadrant") or {}).get("legacy_quadrant_tiers_pct") or [])

    return (
        "<p class=\"lede\">This report validates persisted V3 artifacts, computes integrity checks from manifest hashes, and summarizes operational behavior on exact token-budget axes.</p>"
        "<div class=\"table-scroll\"><table><thead><tr><th>Source</th><th>Sessions</th><th>Agents</th><th>Label pass-rate</th></tr></thead>"
        f"<tbody>{''.join(src_rows)}</tbody></table></div>"
        "<p><strong>Population:</strong> "
        + f"{top['population']:,} sessions. <strong>Total emitted token mass:</strong> {top['emitted_total']:,}. "
        + f"<strong>Truncated packets:</strong> {top['truncated_count']:,}."
        + "</p>"
        + "<p><strong>Exact token tiers provenance:</strong> "
        + f"Outcome legacy percentages {escape(str(outcome_tiers))}, Quadrant legacy percentages {escape(str(quadrant_tiers))}."
        + " Percentages are provenance only; analysis axes are exact token budgets.</p>"
    )


def _render_v2_delta() -> str:
    return (
        "<h3>What changed from V2</h3>"
        "<ul>"
        "<li>V3 uses exact token-budget axes for outcome, quadrant, and throughput planes.</li>"
        "<li>V3 does not use Cochran sample sizing or finite-population correction; each arm packs as many whole sessions as its exact token budget permits.</li>"
        "<li>Token representation moved to packetized evidence with explicit emitted token accounting.</li>"
        "<li>Embedding lineage is live-profiled at 1536 dimensions and validated against deployed index schema.</li>"
        "<li>Adaptive selectors run native proposal followed by deterministic maximal fill under slack.</li>"
        "<li>Manifest hash validation and schema-version checks are mandatory for report generation.</li>"
        "<li>V3 bundle intentionally excludes V2 external snapshot artifacts.</li>"
        "</ul>"
    )


def _render_methods(artifacts: LoadedV3Artifacts) -> str:
    method_text = escape(artifacts.methodology_delta_text.strip())
    return (
        "<p>Method provenance is loaded from the persisted methodology delta artifact and summarized below.</p>"
        f"<pre>{method_text}</pre>"
    )


def _render_outcomes(artifacts: LoadedV3Artifacts) -> str:
    rows = ((artifacts.aggregate.get("outcome") or {}).get("aggregate") or [])
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mae = float((((row.get("mae") or {}).get("mean")) or 0.0))
        cov = float((((row.get("concept_coverage") or {}).get("mean")) or 0.0))
        saved = float((((row.get("fraction_saved") or {}).get("mean")) or 0.0))
        util = float((((row.get("token_utilization") or {}).get("mean")) or 0.0))
        n_count = float((((row.get("native_count") or {}).get("mean")) or 0.0))
        f_count = float((((row.get("fill_count") or {}).get("mean")) or 0.0))
        table_rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method') or '')))}</td>"
            f"<td>{int(row.get('budget_tokens') or 0):,}</td>"
            f"<td>{int(row.get('replays') or 0)}</td>"
            f"<td>{escape(_num(mae, 3))}</td>"
            f"<td>{escape(_pct(cov, 1))}</td>"
            f"<td>{escape(_pct(saved, 1))}</td>"
            f"<td>{escape(_pct(util, 1))}</td>"
            f"<td>{escape(_num(n_count, 1))}</td>"
            f"<td>{escape(_num(f_count, 1))}</td>"
            "</tr>"
        )

    return (
        "<p>Outcome metrics are aggregated over paired replays. All values are derived from exact token budgets, not percentage caps.</p>"
        + _svg_outcome_bars([row for row in rows if isinstance(row, dict)])
        + "<div class=\"table-scroll\"><table><thead><tr>"
        "<th>Method</th><th>Budget tokens</th><th>Replays</th><th>MAE mean</th><th>Concept coverage</th>"
        "<th>Fraction saved</th><th>Token utilization</th><th>Native count mean</th><th>Fill count mean</th>"
        "</tr></thead>"
        + f"<tbody>{''.join(table_rows)}</tbody></table></div>"
    )


def _render_token_embedding(artifacts: LoadedV3Artifacts) -> str:
    inv = artifacts.token_inventory
    original = sum(int(row.get("original_tokens") or 0) for row in inv)
    emitted = sum(int(row.get("emitted_tokens") or 0) for row in inv)
    truncated = sum(1 for row in inv if bool(row.get("truncated")))
    ledger = artifacts.embedding_ledger

    return (
        "<p>Token representation and embedding ledger track packet build/load behavior and embedding call footprint.</p>"
        "<div class=\"table-scroll\"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>Original tokens total</td><td>{original:,}</td></tr>"
        f"<tr><td>Emitted tokens total</td><td>{emitted:,}</td></tr>"
        f"<tr><td>Truncated packet count</td><td>{truncated:,}</td></tr>"
        f"<tr><td>Packet builds</td><td>{int(ledger.get('packet_builds') or 0):,}</td></tr>"
        f"<tr><td>Packet cache hits</td><td>{int(ledger.get('packet_cache_hits') or 0):,}</td></tr>"
        f"<tr><td>Embedding calls</td><td>{int(ledger.get('embedding_calls') or 0):,}</td></tr>"
        f"<tr><td>Embedding inputs</td><td>{int(ledger.get('embedding_inputs') or 0):,}</td></tr>"
        f"<tr><td>Embedding input tokens</td><td>{int(ledger.get('embedding_input_tokens') or 0):,}</td></tr>"
        f"<tr><td>Embedding latency seconds</td><td>{escape(_num(float(ledger.get('embedding_latency_seconds') or 0.0), 3))}</td></tr>"
        f"<tr><td>Embedding model id</td><td>{escape(str(ledger.get('embedding_model_id') or ''))}</td></tr>"
        f"<tr><td>Embedding deployment id</td><td>{escape(str(ledger.get('embedding_deployment_id') or ''))}</td></tr>"
        "</tbody></table></div>"
    )


def _render_quadrant(artifacts: LoadedV3Artifacts) -> str:
    if bool(artifacts.quadrant.get("skipped")):
        return "<p>Skipped in this run. Quadrant artifact exists and hash validation passed.</p>"

    groups = artifacts.quadrant.get("aggregate_groups") or []
    rows = []
    for row in groups:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method') or '')))}</td>"
            f"<td>{escape(str(row.get('quadrant') or ''))}</td>"
            f"<td>{int(row.get('budget_tokens') or 0):,}</td>"
            f"<td>{escape(_pct(float(row.get('representation_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_pct(float(row.get('budget_utilization_tokens_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_pct(float(row.get('zero_selection_agent_rate_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_num(float(row.get('mae_mean') or 0.0), 3))}</td>"
            "</tr>"
        )
    return (
        "<p>Quadrant plane uses post-membership labels for diagnostics only; selection itself remains label-blind.</p>"
        "<div class=\"table-scroll\"><table><thead><tr>"
        "<th>Method</th><th>Quadrant</th><th>Budget tokens</th><th>Representation</th>"
        "<th>Token utilization</th><th>Zero-selection agent rate</th><th>MAE mean</th></tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_throughput(artifacts: LoadedV3Artifacts) -> str:
    if bool(artifacts.throughput.get("skipped")):
        return "<p>Skipped in this run. Throughput artifact exists and hash validation passed.</p>"

    cfg = artifacts.throughput.get("config") or {}
    eps_map = cfg.get("eval_tokens_per_second_map") or {}
    grid = artifacts.throughput.get("aggregate_grid") or []

    rows = []
    for row in grid:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method') or '')))}</td>"
            f"<td>{escape(_num(float(row.get('arrival_rate_sessions_per_second') or 0.0), 2))}</td>"
            f"<td>{escape(_num(float(row.get('eval_capacity_sessions_per_second') or 0.0), 2))}</td>"
            f"<td>{int(row.get('budget_tokens') or 0):,}</td>"
            f"<td>{escape(_num(float(row.get('queue_admitted_tokens_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_num(float(row.get('queue_max_tokens_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_num(float(row.get('token_pressure_ratio_mean') or 0.0), 3))}</td>"
            f"<td>{escape(_pct(float(row.get('budget_utilization_tokens_mean') or 0.0), 1))}</td>"
            "</tr>"
        )

    eps_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(_num(float(v), 3))}</td></tr>"
        for k, v in sorted(eps_map.items(), key=lambda kv: float(kv[0]))
    )

    return (
        "<p>Throughput token queue summaries report native proposal admission pressure under queue capacity equal to budget tokens.</p>"
        "<div class=\"table-scroll\"><table><thead><tr><th>Capacity rate (sessions/s)</th><th>Eval tokens/s</th></tr></thead>"
        f"<tbody>{eps_rows}</tbody></table></div>"
        "<div class=\"table-scroll\"><table><thead><tr>"
        "<th>Method</th><th>Arrival rate/s</th><th>Eval capacity/s</th><th>Budget tokens</th>"
        "<th>Queue admitted tokens mean</th><th>Queue max tokens mean</th><th>Token pressure ratio mean</th><th>Token utilization mean</th>"
        "</tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_repro(artifacts: LoadedV3Artifacts) -> str:
    man = artifacts.manifest
    art = man.get("artifacts") or {}
    rows = []
    for key, row in sorted(art.items()):
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(str(key))}</td>"
            f"<td>{escape(str(row.get('path') or ''))}</td>"
            f"<td>{escape(str(row.get('sha256') or '')[:16])}</td>"
            f"<td>{int(row.get('bytes') or 0):,}</td>"
            "</tr>"
        )

    prov = (artifacts.aggregate.get("provenance") or {}).get("code_hashes") or {}
    prov_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v)[:16] if v else '')}</td></tr>"
        for k, v in sorted(prov.items())
    )

    optional_rows = []
    if artifacts.run_source_manifest is not None:
        optional_rows.append(
            "<tr>"
            "<td>run_source_manifest</td>"
            f"<td>{escape(str(artifacts.run_source_manifest.get('captured_at') or ''))}</td>"
            f"<td>{escape(str(artifacts.run_source_manifest.get('branch') or ''))}</td>"
            f"<td>{len((artifacts.run_source_manifest.get('source_hashes') or {})):,}</td>"
            "</tr>"
        )
    if artifacts.search_cleanup_audit is not None:
        optional_rows.append(
            "<tr>"
            "<td>search_cleanup_audit</td>"
            f"<td>{escape(str(artifacts.search_cleanup_audit.get('checked_at') or ''))}</td>"
            f"<td>{escape(str(artifacts.search_cleanup_audit.get('tenant_id') or ''))}</td>"
            f"<td>{int(artifacts.search_cleanup_audit.get('remaining_count') or 0):,}</td>"
            "</tr>"
        )

    optional_table = ""
    if optional_rows:
        optional_table = (
            "<h3>Optional post-run artifacts</h3>"
            "<div class=\"table-scroll\"><table><thead><tr>"
            "<th>Artifact</th><th>Captured/Checked At</th><th>Scope</th><th>Count</th></tr></thead>"
            f"<tbody>{''.join(optional_rows)}</tbody></table></div>"
        )

    return (
        "<p>Reproducibility and storage checks are constrained to persisted artifacts and manifest attestations.</p>"
        "<div class=\"table-scroll\"><table><thead><tr><th>Artifact</th><th>Path</th><th>SHA-256 prefix</th><th>Bytes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<h3>Code hash provenance</h3>"
        "<div class=\"table-scroll\"><table><thead><tr><th>File</th><th>SHA-256 prefix</th></tr></thead>"
        f"<tbody>{prov_rows}</tbody></table></div>"
        f"{optional_table}"
    )


def _render_caveats_and_conclusions(artifacts: LoadedV3Artifacts) -> str:
    top = _topline_metrics(artifacts)
    outcome = ((artifacts.aggregate.get("outcome") or {}).get("aggregate") or [])
    mean_mae = mean(
        float((((row.get("mae") or {}).get("mean")) or 0.0))
        for row in outcome
        if isinstance(row, dict)
    ) if outcome else 0.0

    by_budget: dict[int, float] = {}
    for row in outcome:
        if not isinstance(row, dict):
            continue
        budget = int(row.get("budget_tokens") or 0)
        mae = float((((row.get("mae") or {}).get("mean")) or 0.0))
        if budget not in by_budget or mae < by_budget[budget]:
            by_budget[budget] = mae
    budget_rows = "".join(
        f"<li>{budget:,} tokens: lowest observed MAE {_num(mae, 3)} (descriptive within-budget only).</li>"
        for budget, mae in sorted(by_budget.items())
    )
    cap_binding = "non-binding" if int(top["truncated_count"]) == 0 else "binding"

    return (
        "<h3>Explicit caveats</h3>"
        "<ul>"
        "<li>Token-mass percentages are provenance only; all reported axes are exact token budgets.</li>"
        "<li>No Cochran sizing or finite-population correction is applied; sessions are selected until no remaining whole session fits the token slack.</li>"
        "<li>Random method results are descriptive; no design-based confidence intervals are claimed.</li>"
        "<li>Adaptive rates are diagnostic mechanism outputs, not population estimators.</li>"
        "<li>Tau 0.55 is an uncalibrated assumption for this deployment context.</li>"
        "<li>Live Azure Search HNSW behavior and latency are environment-specific and can drift by index/service state.</li>"
        "<li>A 4096-entry exact recent-leader buffer resolves many decisions before HNSW and can mask indexing-lag effects in observed decision paths.</li>"
        "<li>Labels are joined only post-membership; selection membership is label-blind.</li>"
        f"<li>Runtime packet-cap check from token inventory: {int(top['truncated_count'])}/{int(top['packet_count'])} packets truncated, max emitted tokens {int(top['max_emitted_tokens'])}, cap is {cap_binding}.</li>"
        "<li>No V2 external snapshot artifacts are present in V3 bundle outputs.</li>"
        "</ul>"
        "<h3>Conclusions from persisted values</h3>"
        f"<p>Population analyzed: {top['population']:,}. Mean MAE across aggregate cells: {_num(mean_mae, 3)}. "
        "Comparisons are matched exact-token-budget descriptive diagnostics; no universal best row is declared across budgets. "
        "Conclusions remain bounded by the caveats above and by any skipped planes in this run.</p>"
        "<p>Lowest observed MAE per exact budget (descriptive only):</p>"
        f"<ul>{budget_rows}</ul>"
    )


def _tab_button(tab_id: str, title: str, selected: bool = False) -> str:
    return (
        f'<button class="tab-button" role="tab" id="tab-{escape(tab_id)}" '
        f'aria-controls="panel-{escape(tab_id)}" aria-selected="{str(selected).lower()}" '
        f'tabindex="{0 if selected else -1}">{escape(title)}</button>'
    )


def _tab_panel(tab_id: str, title: str, content: str, selected: bool = False) -> str:
    hidden = "" if selected else " hidden"
    return (
        f'<section class="tab-panel" role="tabpanel" id="panel-{escape(tab_id)}" '
        f'aria-labelledby="tab-{escape(tab_id)}" tabindex="0"{hidden}>'
        f"<h2>{escape(title)}</h2>{content}</section>"
    )


def render_v3_html_report(artifacts: LoadedV3Artifacts) -> str:
    validate_v3_artifacts(artifacts)
    top = _topline_metrics(artifacts)

    tabs = [
        ("overview", "Overview", _render_overview(artifacts)),
        ("delta", "What Changed from V2", _render_v2_delta()),
        ("methods", "Methods", _render_methods(artifacts)),
        ("outcomes", "Outcomes", _render_outcomes(artifacts)),
        ("ledger", "Token and Embedding", _render_token_embedding(artifacts)),
        ("quadrant", "Quadrant", _render_quadrant(artifacts)),
        ("throughput", "Throughput Queue", _render_throughput(artifacts)),
        ("repro", "Reproducibility and Storage", _render_repro(artifacts)),
        ("caveats", "Caveats and Conclusions", _render_caveats_and_conclusions(artifacts)),
    ]

    tab_buttons = []
    tab_panels = []
    for idx, (tab_id, title, content) in enumerate(tabs):
        selected = idx == 0
        tab_buttons.append(_tab_button(tab_id, title, selected=selected))
        tab_panels.append(_tab_panel(tab_id, title, content, selected=selected))

    generated = str(artifacts.aggregate.get("generated_at") or "")

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Agent365 Sampling V3 Report</title>
<style>
:root {{
  --bg: #f4f6f8;
  --card: #ffffff;
  --ink: #172029;
  --muted: #5c6c7b;
  --line: #d7dee5;
  --blue: #2f74a8;
  --teal: #177d65;
  --orange: #c7641d;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); background: linear-gradient(180deg, #f7f9fb 0%, #edf2f6 100%); }}
.page {{ max-width: 1180px; margin: 0 auto; padding: 18px 18px 32px; }}
header {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
h1 {{ margin: 0; font-size: 1.35rem; letter-spacing: 0.01em; }}
.status-row {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
.status {{ border: 1px solid var(--line); border-left: 4px solid var(--blue); border-radius: 8px; padding: 8px 10px; background: #fafcfe; }}
.status b {{ display: block; font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 3px; }}
.status span {{ font-size: 0.96rem; }}
.status:nth-child(2) {{ border-left-color: var(--teal); }}
.status:nth-child(3) {{ border-left-color: var(--orange); }}
.status:nth-child(4) {{ border-left-color: #4f87a2; }}
.status:nth-child(5) {{ border-left-color: #5d7a96; }}
.tabs {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 7px; }}
.tab-button {{ border: 1px solid var(--line); border-radius: 7px; padding: 8px 10px; background: #f7fafc; color: var(--ink); cursor: pointer; font-weight: 600; }}
.tab-button[aria-selected=\"true\"] {{ background: #eaf3fb; border-color: #95bfdc; }}
.tab-panel {{ margin-top: 10px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
.tab-panel h2 {{ margin-top: 0; font-size: 1.1rem; }}
.lede {{ color: var(--muted); margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; vertical-align: top; text-align: left; }}
th {{ background: #f6f9fc; }}
.table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
ul {{ margin: 0.2rem 0 0.7rem; }}
pre {{ background: #f3f7fb; border: 1px solid var(--line); border-radius: 8px; padding: 10px; overflow-x: auto; white-space: pre-wrap; font-size: 0.86rem; }}
.chart {{ margin: 10px 0; }}
.chart figcaption {{ font-weight: 700; margin-bottom: 6px; }}
.chart-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
.chart svg {{ min-width: 700px; width: 100%; height: auto; display: block; }}
.chart-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 8px; font-size: 0.86rem; color: var(--muted); }}
.chart-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.chart-legend i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.axis {{ fill: #5c6c7b; font-size: 11px; }}
.grid {{ stroke: #e5ecf3; stroke-width: 1; }}
.footer {{ margin-top: 10px; color: var(--muted); font-size: 0.82rem; }}
@media (max-width: 980px) {{
  .status-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 680px) {{
  .page {{ padding: 10px; }}
  .status-row {{ grid-template-columns: 1fr; }}
  .tab-button {{ width: 100%; text-align: left; }}
}}
</style>
</head>
<body>
<div class=\"page\">
<header>
  <h1>Agent365 Sampling V3 Operational Report</h1>
  <div class=\"status-row\" aria-label=\"status and provenance summary\">
    <div class=\"status\"><b>Status</b><span>Validated (hash + schema)</span></div>
    <div class=\"status\"><b>Generated At</b><span>{escape(str(top['generated_at']))}</span></div>
    <div class=\"status\"><b>Population</b><span>{top['population']:,} sessions</span></div>
    <div class=\"status\"><b>Runtime</b><span>{_num(float(top['runtime_seconds']), 2)} seconds</span></div>
    <div class=\"status\"><b>Exact Token Tiers</b><span>{escape(str((artifacts.budget_manifest.get('outcome') or {}).get('legacy_outcome_tiers_pct') or []))}</span></div>
  </div>
</header>

<nav class=\"tabs\" role=\"tablist\" aria-label=\"Report sections\">
  {''.join(tab_buttons)}
</nav>

{''.join(tab_panels)}

<p class=\"footer\">Report version: {REPORT_VERSION}. Rendered at {escape(generated)}. Self-contained output with inline CSS, JS, and SVG only.</p>
</div>
<script>
(() => {{
  const tabs = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  function activate(tab) {{
    tabs.forEach((t) => {{
      const selected = t === tab;
      t.setAttribute('aria-selected', selected ? 'true' : 'false');
      t.tabIndex = selected ? 0 : -1;
      const panel = document.getElementById(t.getAttribute('aria-controls'));
      if (panel) panel.hidden = !selected;
    }});
  }}
  tabs.forEach((tab, i) => {{
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (e) => {{
      let next = i;
      if (e.key === 'ArrowRight') next = (i + 1) % tabs.length;
      if (e.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
      if (e.key === 'Home') next = 0;
      if (e.key === 'End') next = tabs.length - 1;
      if (next !== i) {{
        e.preventDefault();
        tabs[next].focus();
        activate(tabs[next]);
      }}
    }});
  }});
}})();
</script>
</body>
</html>
"""

    if "http://" in html or "https://" in html or "cdn" in html.lower():
        raise ValueError("report must be self-contained and avoid external assets")
    return html


def build_report_manifest_payload(
    *,
    output_path: Path,
    aggregate_generated_at: str,
    bundle_manifest_sha256: str,
    report_generator_source_sha256: str,
) -> dict[str, Any]:
    return {
        "version": "sampling-v3-report-manifest-v1",
        "report_filename": output_path.name,
        "report_sha256": _sha(output_path),
        "report_bytes": int(output_path.stat().st_size),
        "aggregate_generated_at": str(aggregate_generated_at),
        "bundle_manifest_sha256": str(bundle_manifest_sha256),
        "report_generator_source_sha256": str(report_generator_source_sha256),
    }


def validate_report_manifest(*, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    if str(payload.get("version")) != "sampling-v3-report-manifest-v1":
        raise ValueError("report manifest version must be sampling-v3-report-manifest-v1")
    if str(payload.get("report_filename")) != report_path.name:
        raise ValueError("report manifest filename mismatch")
    if str(payload.get("report_sha256")) != _sha(report_path):
        raise ValueError("report manifest sha256 mismatch")
    if int(payload.get("report_bytes") or -1) != int(report_path.stat().st_size):
        raise ValueError("report manifest bytes mismatch")
    for key in ("aggregate_generated_at", "bundle_manifest_sha256", "report_generator_source_sha256"):
        value = str(payload.get(key) or "")
        if not value:
            raise ValueError(f"report manifest field missing: {key}")
    return payload


def write_v3_html_report(*, output_path: Path, inputs: V3ReportInputs) -> Path:
    artifacts = load_v3_artifacts(inputs)
    html = render_v3_html_report(artifacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    aggregate_generated_at = str(artifacts.aggregate.get("generated_at") or "")
    manifest_payload = build_report_manifest_payload(
        output_path=output_path,
        aggregate_generated_at=aggregate_generated_at,
        bundle_manifest_sha256=_sha(inputs.manifest),
        report_generator_source_sha256=_sha(Path(__file__)),
    )
    report_manifest_path = output_path.with_name(REPORT_MANIFEST_NAME)
    report_manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    validate_report_manifest(report_path=output_path, manifest_path=report_manifest_path)
    return output_path
