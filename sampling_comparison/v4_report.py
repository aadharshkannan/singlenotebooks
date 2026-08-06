"""Self-contained HTML report generator for sampling v4 artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REPORT_VERSION = "agent365-sampling-v4-report-v2"
REPORT_MANIFEST_VERSION = "sampling-v4-report-manifest-v1"
DEFAULT_OUTPUT_NAME = "agent365-sampling-v4-report.html"
DEFAULT_INPUT_DIR = Path("outputs_sampling_v4") / "runs"
REPORT_MANIFEST_NAME = "report_manifest.json"

V4_BUNDLE_VERSION = "sampling-v4-bundle-v1"
V4_MANIFEST_VERSION = "sampling-v4-manifest-v1"
V4_IDW_CONFIG_VERSION = "sampling-v4-idw-config-v1"
V4_SOURCE_LINEAGE_VERSION = "sampling-v4-source-lineage-v1"
V4_OUTCOME_VERSION = "sampling-v4-outcome-v1"
V3_MANIFEST_VERSION = "sampling-v3-manifest-v1"
V3_BUNDLE_VERSION = "sampling-v3-bundle-v1"

RANDOM_METHOD = "random_sampling_token_priority"
MINHASH_METHOD = "adaptive_minhash_32x4_token"
EMBEDDING_METHOD = "adaptive_embedding_fullsession_token"
SELECTED_ONLY_MODE = "selected_only"
MODEL_ASSISTED_MODE = "model_assisted_idw"

METHOD_LABELS = {
    RANDOM_METHOD: "Random selected-only",
    MINHASH_METHOD: "MinHash selected-only",
    EMBEDDING_METHOD: "Embedding selected-only",
}

METHOD_SHORT = {
    RANDOM_METHOD: "Random",
    MINHASH_METHOD: "MinHash",
    EMBEDDING_METHOD: "Embedding",
}

METHOD_COLORS = {
    RANDOM_METHOD: "#8f5f2a",
    MINHASH_METHOD: "#2d7c6d",
    EMBEDDING_METHOD: "#1f6d8c",
    "idw": "#b04a36",
    "fallback": "#6f7f36",
}


@dataclass(frozen=True)
class V4ReportInputs:
    aggregate: Path
    runs_jsonl: Path
    idw_config: Path
    methodology_delta: Path
    source_lineage: Path
    manifest: Path


@dataclass(frozen=True)
class LoadedV4Artifacts:
    aggregate: dict[str, Any]
    runs_jsonl: list[dict[str, Any]]
    idw_config: dict[str, Any]
    methodology_delta_text: str
    source_lineage: dict[str, Any]
    manifest: dict[str, Any]
    source_v3_manifest: dict[str, Any]
    source_v3_aggregate: dict[str, Any] | None
    source_v3_token_inventory: list[dict[str, Any]] | None
    source_v3_quadrant: dict[str, Any] | None
    source_v3_throughput: dict[str, Any] | None
    source_paths: dict[str, Path]


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def _duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _finite(value: Any, *, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def _safe_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{field_name} must be an integer") from exc


def _safe_prob(value: Any, *, field_name: str) -> float:
    out = _finite(value, field_name=field_name)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return out


def _method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name.replace("_", " ").title())


def _friendly_source_name(key: str) -> str:
    names = {
        "aggregate": "V4 aggregate bundle",
        "runs_jsonl": "V4 run rows",
        "idw_config": "IDW configuration",
        "methodology_delta": "Methodology notes",
        "source_lineage": "Selection lineage",
        "manifest": "V4 manifest",
        "source_v3_manifest": "Selection-stage manifest",
        "source_v3_aggregate": "Selection-stage aggregate",
        "source_v3_token_inventory": "Selection-stage token inventory",
        "source_v3_quadrant": "Selection-stage quadrant artifact",
        "source_v3_throughput": "Selection-stage throughput artifact",
    }
    return names.get(key, key)


def default_inputs(base_dir: Path) -> V4ReportInputs:
    return V4ReportInputs(
        aggregate=base_dir / "aggregate.json",
        runs_jsonl=base_dir / "runs.jsonl",
        idw_config=base_dir / "idw_config.json",
        methodology_delta=base_dir / "methodology_delta.md",
        source_lineage=base_dir / "source_lineage.json",
        manifest=base_dir / "manifest.json",
    )


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object at {path}")
    return payload


def _load_optional_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists() or not path.is_file():
        return None
    return _read_jsonl(path)


def _resolve_source_v3_manifest_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be an object")
    entry = artifacts.get("source_v3_manifest")
    if not isinstance(entry, dict):
        raise ValueError("manifest missing source_v3_manifest entry")
    rel = str(entry.get("path") or "")
    if not rel:
        raise ValueError("source_v3_manifest path must be non-empty")
    return manifest_path.parent / rel


def _resolve_source_entry_path(source_root: Path, entry: dict[str, Any]) -> Path:
    raw = str(entry.get("path") or "")
    if not raw:
        raise ValueError("source manifest artifact path must be non-empty")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    name_fallback = source_root / candidate.name
    if name_fallback.exists():
        return name_fallback
    return source_root / candidate


def load_v4_artifacts(inputs: V4ReportInputs) -> LoadedV4Artifacts:
    manifest = _read_json(inputs.manifest)
    source_v3_manifest_path = _resolve_source_v3_manifest_path(inputs.manifest, manifest)

    source_v3_root = source_v3_manifest_path.parent
    source_v3_manifest = _read_json(source_v3_manifest_path)
    source_v3_artifacts = source_v3_manifest.get("artifacts")
    if not isinstance(source_v3_artifacts, dict):
        raise ValueError("source V3 manifest artifacts must be an object")

    source_v3_aggregate_entry = source_v3_artifacts.get("aggregate")
    source_v3_inventory_entry = source_v3_artifacts.get("token_inventory")
    source_v3_quadrant_entry = source_v3_artifacts.get("quadrant")
    source_v3_throughput_entry = source_v3_artifacts.get("throughput")

    source_v3_aggregate_path = None
    if isinstance(source_v3_aggregate_entry, dict):
        source_v3_aggregate_path = _resolve_source_entry_path(source_v3_root, source_v3_aggregate_entry)

    source_v3_inventory_path = None
    if isinstance(source_v3_inventory_entry, dict):
        source_v3_inventory_path = _resolve_source_entry_path(source_v3_root, source_v3_inventory_entry)

    source_v3_quadrant_path = None
    if isinstance(source_v3_quadrant_entry, dict):
        source_v3_quadrant_path = _resolve_source_entry_path(source_v3_root, source_v3_quadrant_entry)

    source_v3_throughput_path = None
    if isinstance(source_v3_throughput_entry, dict):
        source_v3_throughput_path = _resolve_source_entry_path(source_v3_root, source_v3_throughput_entry)

    source_v3_aggregate = _load_optional_json(source_v3_aggregate_path) if source_v3_aggregate_path else None
    source_v3_inventory = _load_optional_jsonl(source_v3_inventory_path) if source_v3_inventory_path else None
    source_v3_quadrant = _load_optional_json(source_v3_quadrant_path) if source_v3_quadrant_path else None
    source_v3_throughput = _load_optional_json(source_v3_throughput_path) if source_v3_throughput_path else None

    source_paths = {
        "aggregate": inputs.aggregate,
        "runs_jsonl": inputs.runs_jsonl,
        "idw_config": inputs.idw_config,
        "methodology_delta": inputs.methodology_delta,
        "source_lineage": inputs.source_lineage,
        "manifest": inputs.manifest,
        "source_v3_manifest": source_v3_manifest_path,
    }
    if source_v3_aggregate_path is not None and source_v3_aggregate is not None:
        source_paths["source_v3_aggregate"] = source_v3_aggregate_path
    if source_v3_inventory_path is not None and source_v3_inventory is not None:
        source_paths["source_v3_token_inventory"] = source_v3_inventory_path
    if source_v3_quadrant_path is not None and source_v3_quadrant is not None:
        source_paths["source_v3_quadrant"] = source_v3_quadrant_path
    if source_v3_throughput_path is not None and source_v3_throughput is not None:
        source_paths["source_v3_throughput"] = source_v3_throughput_path

    return LoadedV4Artifacts(
        aggregate=_read_json(inputs.aggregate),
        runs_jsonl=_read_jsonl(inputs.runs_jsonl),
        idw_config=_read_json(inputs.idw_config),
        methodology_delta_text=inputs.methodology_delta.read_text(encoding="utf-8"),
        source_lineage=_read_json(inputs.source_lineage),
        manifest=manifest,
        source_v3_manifest=source_v3_manifest,
        source_v3_aggregate=source_v3_aggregate,
        source_v3_token_inventory=source_v3_inventory,
        source_v3_quadrant=source_v3_quadrant,
        source_v3_throughput=source_v3_throughput,
        source_paths=source_paths,
    )


def _validate_manifest_entry(*, key: str, entry: dict[str, Any], expected_path: Path) -> None:
    recorded_sha = str(entry.get("sha256") or "")
    if len(recorded_sha) != 64:
        raise ValueError(f"manifest sha256 missing/invalid for {key}")
    if _sha(expected_path) != recorded_sha:
        raise ValueError(f"manifest hash mismatch for {key}")
    recorded_bytes = _safe_int(entry.get("bytes"), field_name=f"manifest.artifacts.{key}.bytes")
    if recorded_bytes != int(expected_path.stat().st_size):
        raise ValueError(f"manifest size mismatch for {key}")


def _verify_v4_manifest_integrity(artifacts: LoadedV4Artifacts) -> None:
    manifest = artifacts.manifest
    if str(manifest.get("version")) != V4_MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {V4_MANIFEST_VERSION}")

    listed = manifest.get("artifacts")
    if not isinstance(listed, dict):
        raise ValueError("manifest artifacts must be an object")

    required = (
        "aggregate",
        "runs_jsonl",
        "idw_config",
        "methodology_delta",
        "source_lineage",
        "source_v3_manifest",
    )
    missing = [key for key in required if key not in listed]
    if missing:
        raise ValueError(f"manifest is missing artifact entries: {','.join(missing)}")

    for key in required:
        entry = listed.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry must be an object for {key}")
        _validate_manifest_entry(key=key, entry=entry, expected_path=artifacts.source_paths[key])


def _validate_optional_source_shape(name: str, payload: dict[str, Any]) -> None:
    if name == "quadrant":
        if not isinstance(payload.get("aggregate_groups"), list):
            raise ValueError("source_v3 quadrant.aggregate_groups must be a list")
        if not isinstance(payload.get("config"), dict):
            raise ValueError("source_v3 quadrant.config must be an object")
        if not isinstance(payload.get("quadrants"), dict):
            raise ValueError("source_v3 quadrant.quadrants must be an object")
    if name == "throughput":
        if not isinstance(payload.get("aggregate_grid"), list):
            raise ValueError("source_v3 throughput.aggregate_grid must be a list")
        if not isinstance(payload.get("config"), dict):
            raise ValueError("source_v3 throughput.config must be an object")


def _verify_source_linkage(artifacts: LoadedV4Artifacts) -> None:
    source_manifest = artifacts.source_v3_manifest
    if str(source_manifest.get("version")) != V3_MANIFEST_VERSION:
        raise ValueError(f"source V3 manifest version must be {V3_MANIFEST_VERSION}")

    source_lineage = artifacts.source_lineage
    if str(source_lineage.get("version")) != V4_SOURCE_LINEAGE_VERSION:
        raise ValueError(f"source lineage version must be {V4_SOURCE_LINEAGE_VERSION}")

    source_meta = source_lineage.get("source_manifest")
    if not isinstance(source_meta, dict):
        raise ValueError("source_lineage.source_manifest must be an object")

    source_manifest_sha = _sha(artifacts.source_paths["source_v3_manifest"])
    if str(source_meta.get("version") or "") != V3_MANIFEST_VERSION:
        raise ValueError("source_lineage source manifest version mismatch")
    if str(source_meta.get("sha256") or "") != source_manifest_sha:
        raise ValueError("source_lineage source manifest hash mismatch")

    aggregate_source = artifacts.aggregate.get("source_v3") or {}
    if not isinstance(aggregate_source, dict):
        raise ValueError("aggregate.source_v3 must be an object")

    aggregate_manifest = aggregate_source.get("manifest")
    if not isinstance(aggregate_manifest, dict):
        raise ValueError("aggregate.source_v3.manifest must be an object")

    if str(aggregate_manifest.get("version") or "") != V3_MANIFEST_VERSION:
        raise ValueError("aggregate.source_v3 manifest version mismatch")
    if str(aggregate_manifest.get("sha256") or "") != source_manifest_sha:
        raise ValueError("aggregate.source_v3 manifest hash mismatch")

    source_block = artifacts.manifest.get("source")
    if not isinstance(source_block, dict):
        raise ValueError("manifest.source must be an object")

    if str(source_block.get("source_manifest_version") or "") != V3_MANIFEST_VERSION:
        raise ValueError("manifest.source source_manifest_version mismatch")
    if str(source_block.get("source_manifest_sha256") or "") != source_manifest_sha:
        raise ValueError("manifest.source source_manifest_sha256 mismatch")

    source_artifacts = source_manifest.get("artifacts")
    if not isinstance(source_artifacts, dict):
        raise ValueError("source V3 manifest artifacts must be an object")

    for source_key, loaded_key in (
        ("aggregate", "source_v3_aggregate"),
        ("token_inventory", "source_v3_token_inventory"),
        ("quadrant", "source_v3_quadrant"),
        ("throughput", "source_v3_throughput"),
    ):
        expected_path = artifacts.source_paths.get(loaded_key)
        if expected_path is None:
            continue
        entry = source_artifacts.get(source_key)
        if not isinstance(entry, dict):
            raise ValueError(f"source V3 manifest missing {source_key} entry")
        _validate_manifest_entry(
            key=f"source_v3.{source_key}",
            entry=entry,
            expected_path=expected_path,
        )

    if artifacts.source_v3_quadrant is not None:
        _validate_optional_source_shape("quadrant", artifacts.source_v3_quadrant)
    if artifacts.source_v3_throughput is not None:
        _validate_optional_source_shape("throughput", artifacts.source_v3_throughput)


def _validate_run_row(row: dict[str, Any], idx: int, population_count: int) -> None:
    method = str(row.get("method") or "")
    mode = str(row.get("estimation_mode") or "")

    allowed_methods = {RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD}
    if method not in allowed_methods:
        raise ValueError(f"runs[{idx}].method is unsupported: {method}")

    _safe_int(row.get("budget_tokens"), field_name=f"runs[{idx}].budget_tokens")
    _safe_int(row.get("legacy_tier_pct"), field_name=f"runs[{idx}].legacy_tier_pct")
    _safe_int(row.get("repetition"), field_name=f"runs[{idx}].repetition")

    selected_ids_raw = row.get("selected_ids")
    if not isinstance(selected_ids_raw, list):
        raise ValueError(f"runs[{idx}].selected_ids must be a list")
    selected_ids = [str(value) for value in selected_ids_raw]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"runs[{idx}].selected_ids contains duplicates")

    selected_count = _safe_int(row.get("selected_count"), field_name=f"runs[{idx}].selected_count")
    if selected_count != len(selected_ids):
        raise ValueError(f"runs[{idx}].selected_count does not match selected_ids")
    if selected_count < 0 or selected_count > population_count:
        raise ValueError(f"runs[{idx}].selected_count out of bounds")

    selected_pass_rate = _safe_prob(
        row.get("selected_only_pass_rate", row.get("selected_pass_rate", 0.0)),
        field_name=f"runs[{idx}].selected_only_pass_rate",
    )
    census_pass_rate = _safe_prob(row.get("census_pass_rate"), field_name=f"runs[{idx}].census_pass_rate")
    selected_error = _safe_prob(
        row.get("selected_only_absolute_error", row.get("absolute_error", 0.0)),
        field_name=f"runs[{idx}].selected_only_absolute_error",
    )

    row["selected_only_pass_rate"] = selected_pass_rate
    row["census_pass_rate"] = census_pass_rate
    row["selected_only_absolute_error"] = selected_error

    if method in {RANDOM_METHOD, MINHASH_METHOD}:
        if mode != SELECTED_ONLY_MODE:
            raise ValueError("invalid method/mode combination: random/minhash must be selected_only")
        if row.get("model_assisted") is not None:
            raise ValueError("invalid method/mode combination: random/minhash must not include model_assisted")
        return

    if method == EMBEDDING_METHOD:
        if mode != MODEL_ASSISTED_MODE:
            raise ValueError("invalid method/mode combination: embedding must be model_assisted_idw")
        model_assisted = row.get("model_assisted")
        if not isinstance(model_assisted, dict):
            raise ValueError("embedding rows must include model_assisted object")
        rates = model_assisted.get("rates")
        counts = model_assisted.get("counts")
        metrics = model_assisted.get("metrics")
        if not isinstance(rates, dict) or not isinstance(counts, dict) or not isinstance(metrics, dict):
            raise ValueError("embedding model_assisted must include rates/counts/metrics objects")

        _safe_prob(
            rates.get("estimated_pass_rate"),
            field_name=f"runs[{idx}].model_assisted.rates.estimated_pass_rate",
        )
        _safe_prob(
            rates.get("absolute_aggregate_rate_error"),
            field_name=f"runs[{idx}].model_assisted.rates.absolute_aggregate_rate_error",
        )

        pop = _safe_int(
            counts.get("population_count"),
            field_name=f"runs[{idx}].model_assisted.counts.population_count",
        )
        obs = _safe_int(counts.get("observed_count"), field_name=f"runs[{idx}].model_assisted.counts.observed_count")
        imp = _safe_int(counts.get("imputed_count"), field_name=f"runs[{idx}].model_assisted.counts.imputed_count")
        if pop != population_count:
            raise ValueError("population_count mismatch between run model_assisted counts and aggregate")
        if obs + imp != pop:
            raise ValueError("model_assisted observed_count + imputed_count must equal population_count")

        cal_bins = metrics.get("calibration_bins")
        dist_bins = metrics.get("nearest_distance_error_bins")
        per_agent = metrics.get("per_agent")
        if not isinstance(cal_bins, list) or not isinstance(dist_bins, list) or not isinstance(per_agent, list):
            raise ValueError(
                "embedding metrics must include calibration_bins, nearest_distance_error_bins, and per_agent lists"
            )


def validate_v4_artifacts(artifacts: LoadedV4Artifacts) -> None:
    if str(artifacts.aggregate.get("version")) != V4_BUNDLE_VERSION:
        raise ValueError(f"aggregate version must be {V4_BUNDLE_VERSION}")
    if str((artifacts.aggregate.get("outcome") or {}).get("version")) != V4_OUTCOME_VERSION:
        raise ValueError(f"aggregate outcome version must be {V4_OUTCOME_VERSION}")
    if str(artifacts.idw_config.get("version")) != V4_IDW_CONFIG_VERSION:
        raise ValueError(f"idw config version must be {V4_IDW_CONFIG_VERSION}")

    runs = artifacts.runs_jsonl
    if not runs:
        raise ValueError("runs.jsonl must contain at least one row")

    population_count = _safe_int(artifacts.aggregate.get("population_count"), field_name="aggregate.population_count")
    if population_count <= 0:
        raise ValueError("aggregate.population_count must be > 0")

    _verify_v4_manifest_integrity(artifacts)
    _verify_source_linkage(artifacts)

    outcome_aggregate = ((artifacts.aggregate.get("outcome") or {}).get("aggregate") or [])
    if not isinstance(outcome_aggregate, list):
        raise ValueError("aggregate.outcome.aggregate must be a list")

    for idx, row in enumerate(runs):
        _validate_run_row(row, idx, population_count)

    methods = {str(row.get("method") or "") for row in runs}
    required_methods = {RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD}
    if methods != required_methods:
        raise ValueError("runs.jsonl must contain random, minhash, and embedding methods")

    for idx, row in enumerate(outcome_aggregate):
        if not isinstance(row, dict):
            continue
        method = str(row.get("method") or "")
        if method != EMBEDDING_METHOD:
            continue
        counts_sum = row.get("model_assisted_counts_sum")
        if not isinstance(counts_sum, dict):
            raise ValueError(f"aggregate.outcome.aggregate[{idx}] missing model_assisted_counts_sum")
        pop_sum = _safe_int(
            counts_sum.get("population_count"),
            field_name="model_assisted_counts_sum.population_count",
        )
        replays = _safe_int(row.get("replays"), field_name="aggregate.outcome.aggregate.replays")
        if pop_sum != population_count * replays:
            raise ValueError("aggregate embedding model_assisted population_count sum mismatch")

    source_agg = artifacts.source_v3_aggregate
    if source_agg is not None and str(source_agg.get("version") or "") != V3_BUNDLE_VERSION:
        raise ValueError(f"source_v3 aggregate version must be {V3_BUNDLE_VERSION}")


def _summarize_outcomes(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runs:
        key = (str(row.get("method")), int(row.get("budget_tokens") or 0))
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, bucket in grouped.items():
        selected_only = [
            _finite(row.get("selected_only_pass_rate"), field_name="selected_only_pass_rate") for row in bucket
        ]
        census = [_finite(row.get("census_pass_rate"), field_name="census_pass_rate") for row in bucket]
        selected_mae = [
            _finite(row.get("selected_only_absolute_error"), field_name="selected_only_absolute_error") for row in bucket
        ]
        selected_count = [_safe_int(row.get("selected_count"), field_name="selected_count") for row in bucket]
        fraction_saved = [_finite(row.get("fraction_saved", 0.0), field_name="fraction_saved") for row in bucket]
        concept_coverage = [
            _finite(row.get("concept_coverage", 0.0), field_name="concept_coverage") for row in bucket
        ]
        representation = [_finite(row.get("representation", 0.0), field_name="representation") for row in bucket]

        idw_error_vals = []
        idw_vals = []
        observed_shares = []
        idw_shares = []
        fallback_shares = []
        for row in bucket:
            model_assisted = row.get("model_assisted")
            if isinstance(model_assisted, dict):
                rates = model_assisted.get("rates")
                counts = model_assisted.get("counts")
                if isinstance(rates, dict):
                    idw_vals.append(_finite(rates.get("estimated_pass_rate"), field_name="estimated_pass_rate"))
                    idw_error_vals.append(
                        _finite(
                            rates.get("absolute_aggregate_rate_error", 0.0),
                            field_name="absolute_aggregate_rate_error",
                        )
                    )
                if isinstance(counts, dict):
                    prov_counts = counts.get("provenance_counts") or {}
                    if isinstance(prov_counts, dict):
                        pop = max(1, _safe_int(counts.get("population_count", 1), field_name="population_count"))
                        observed_count = _safe_int(prov_counts.get("observed", 0), field_name="observed")
                        idw_count = sum(
                            _safe_int(prov_counts.get(name, 0), field_name=name)
                            for name in ("idw", "exact_match")
                        )
                        fallback_count = sum(
                            _safe_int(prov_counts.get(name, 0), field_name=name)
                            for name in ("agent_mean", "global_mean", "prior")
                        )
                        observed_shares.append(float(observed_count / pop))
                        idw_shares.append(float(idw_count / pop))
                        fallback_shares.append(float(fallback_count / pop))

        out[key] = {
            "selected_only": float(mean(selected_only)) if selected_only else 0.0,
            "census": float(mean(census)) if census else 0.0,
            "selected_mae": float(mean(selected_mae)) if selected_mae else 0.0,
            "selected_mae_low": min(selected_mae) if selected_mae else 0.0,
            "selected_mae_high": max(selected_mae) if selected_mae else 0.0,
            "selected_count": float(mean(selected_count)) if selected_count else 0.0,
            "fraction_saved": float(mean(fraction_saved)) if fraction_saved else 0.0,
            "concept_coverage": float(mean(concept_coverage)) if concept_coverage else 0.0,
            "representation": float(mean(representation)) if representation else 0.0,
            "idw": float(mean(idw_vals)) if idw_vals else 0.0,
            "idw_error": float(mean(idw_error_vals)) if idw_error_vals else 0.0,
            "observed_share": float(mean(observed_shares)) if observed_shares else 0.0,
            "idw_share": float(mean(idw_shares)) if idw_shares else 0.0,
            "fallback_share": float(mean(fallback_shares)) if fallback_shares else 0.0,
        }
    return out


def _summarize_mae(aggregate_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row in aggregate_rows:
        method = str(row.get("method") or "")
        budget = int(row.get("budget_tokens") or 0)
        selected = row.get("selected_only_mae") or {}
        if not isinstance(selected, dict):
            continue

        rec = {
            "selected_mean": _finite(selected.get("mean", 0.0), field_name="selected_only_mae.mean"),
            "selected_low": _finite(selected.get("empirical_low", 0.0), field_name="selected_only_mae.empirical_low"),
            "selected_high": _finite(
                selected.get("empirical_high", 0.0), field_name="selected_only_mae.empirical_high"
            ),
            "replays": _safe_int(row.get("replays", 0), field_name="replays"),
        }

        if method == EMBEDDING_METHOD:
            idw = row.get("idw_absolute_error") or {}
            if isinstance(idw, dict):
                rec["idw_mean"] = _finite(idw.get("mean", 0.0), field_name="idw_absolute_error.mean")
                rec["idw_low"] = _finite(
                    idw.get("empirical_low", 0.0), field_name="idw_absolute_error.empirical_low"
                )
                rec["idw_high"] = _finite(
                    idw.get("empirical_high", 0.0), field_name="idw_absolute_error.empirical_high"
                )

        out[(method, budget)] = rec
    return out


def _representative_embedding_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in runs
        if str(row.get("method")) == EMBEDDING_METHOD and isinstance(row.get("model_assisted"), dict)
    ]
    if not candidates:
        return None

    def _sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            -int(row.get("legacy_tier_pct") or 0),
            int(row.get("repetition") or 0),
            int(row.get("budget_tokens") or 0),
            str(row.get("order_hash") or ""),
        )

    candidates.sort(key=_sort_key)
    return candidates[0]


def _chart_axis_labels(height: int, ml: int, mt: int, ph: int, *, as_percent: bool = False) -> str:
    labels = []
    ticks = [0, 25, 50, 75, 100]
    for tick in ticks:
        y = mt + ph - (ph * (tick / 100.0))
        label = f"{tick}%" if as_percent else str(tick)
        labels.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml - 5}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{escape(label)}</text>'
        )
    return "".join(labels)


def _plot_grouped_mae(
    *,
    budgets: list[int],
    mae: dict[tuple[str, int], dict[str, float]],
    title: str,
) -> str:
    if not budgets:
        return "<p>MAE comparison unavailable.</p>"

    width = 980
    height = 360
    ml, mr, mt, mb = 72, 30, 26, 88
    pw = width - ml - mr
    ph = height - mt - mb

    categories = [
        (RANDOM_METHOD, "selected_mean", _method_label(RANDOM_METHOD), METHOD_COLORS[RANDOM_METHOD]),
        (MINHASH_METHOD, "selected_mean", _method_label(MINHASH_METHOD), METHOD_COLORS[MINHASH_METHOD]),
        (EMBEDDING_METHOD, "idw_mean", "Full-session embedding + IDW", METHOD_COLORS["idw"]),
    ]

    y_max = 0.001
    for budget in budgets:
        for method, mean_key, _label, _color in categories:
            row = mae.get((method, budget), {})
            y_max = max(y_max, float(row.get(mean_key, 0.0)))

    slot = pw / max(1, len(budgets))
    bar_slot = slot / max(1, len(categories))
    bar_width = max(12.0, min(34.0, bar_slot * 0.62))

    grid = []
    for step in range(6):
        v = y_max * (step / 5.0)
        y = mt + ph - (ph * (v / y_max))
        grid.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{escape(_num(v, 3))}</text>'
        )

    shapes = []
    labels = []
    for bidx, budget in enumerate(budgets):
        bx = ml + bidx * slot
        labels.append(
            f'<text x="{bx + slot / 2:.2f}" y="{height - 34}" text-anchor="middle" class="axis">{budget:,}</text>'
        )
        for cidx, (method, mean_key, label, color) in enumerate(categories):
            row = mae.get((method, budget), {})
            mean_val = float(row.get(mean_key, 0.0))

            cx = bx + cidx * bar_slot + bar_slot / 2.0
            y_mean = mt + ph - (ph * (mean_val / y_max))
            bar_height = mt + ph - y_mean
            x = cx - bar_width / 2.0
            shapes.append(
                f'<rect x="{x:.2f}" y="{y_mean:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{escape(color)}" '
                f'aria-label="budget {budget} {escape(label)} mean absolute error {_num(mean_val, 4)}" />'
            )
            shapes.append(
                f'<text x="{cx:.2f}" y="{max(mt + 11.0, y_mean - 5.0):.2f}" text-anchor="middle" '
                f'class="value-label">{escape(_num(mean_val, 3))}</text>'
            )

    legend = "".join(
        f'<span><i style="background:{escape(color)}"></i>{escape(label)}</span>'
        for _method, _mk, label, color in categories
    )

    return (
        f"<figure class=\"chart\"><figcaption>{escape(title)}</figcaption>"
        f"<div class=\"chart-legend\">{legend}</div>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(title)}\">"
        + "".join(grid)
        + "".join(shapes)
        + "".join(labels)
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Exact token budget</text>'
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">Absolute error</text>'
        + "</svg></div></figure>"
    )


def _plot_outcome_percentage(
    *,
    budgets: list[int],
    summary: dict[tuple[str, int], dict[str, float]],
    metric_key: str,
    title: str,
    y_label: str,
) -> str:
    if not budgets:
        return f"<p>{escape(title)} unavailable.</p>"

    width = 980
    height = 360
    ml, mr, mt, mb = 76, 34, 28, 90
    pw = width - ml - mr
    ph = height - mt - mb

    methods = [RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD]

    slot = pw / max(1, len(budgets))
    group_slot = slot / max(1, len(methods))
    bar_w = max(7.0, group_slot * 0.72)

    grid = []
    for tick in [0, 25, 50, 75, 100]:
        y = mt + ph - (ph * (tick / 100.0))
        grid.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{tick}</text>'
        )

    bars = []
    labels = []
    for bidx, budget in enumerate(budgets):
        bx = ml + bidx * slot
        labels.append(
            f'<text x="{bx + slot / 2:.2f}" y="{height - 36}" text-anchor="middle" class="axis">{budget:,}</text>'
        )
        for midx, method in enumerate(methods):
            rec = summary.get((method, budget), {})
            value = max(0.0, min(1.0, float(rec.get(metric_key, 0.0))))
            x = bx + midx * group_slot + (group_slot - bar_w) / 2.0
            h = ph * value
            y = mt + ph - h

            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{escape(METHOD_COLORS[method])}" '
                f'aria-label="{escape("Full-session embedding + IDW" if method == EMBEDDING_METHOD else _method_label(method))} '
                f'budget {budget} {escape(y_label)} {_pct(value, 1)}" />'
            )

    legend = "".join(
        f'<span><i style="background:{escape(METHOD_COLORS[m])}"></i>'
        f'{escape("Full-session embedding + IDW" if m == EMBEDDING_METHOD else _method_label(m))}</span>'
        for m in methods
    )

    return (
        f"<figure class=\"chart\"><figcaption>{escape(title)}</figcaption>"
        f"<div class=\"chart-legend\">{legend}</div>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(title)}\">"
        + "".join(grid)
        + "".join(bars)
        + "".join(labels)
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Exact token budget</text>'
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">{escape(y_label)} (%)</text>'
        + "</svg></div></figure>"
    )


def _plot_provenance_stacked(
    *,
    budgets: list[int],
    summary: dict[tuple[str, int], dict[str, float]],
) -> str:
    if not budgets:
        return "<p>Provenance coverage unavailable.</p>"

    width = 980
    height = 360
    ml, mr, mt, mb = 74, 32, 26, 84
    pw = width - ml - mr
    ph = height - mt - mb

    slot = pw / max(1, len(budgets))
    bar_w = min(90.0, slot * 0.58)

    bars = []
    labels = []
    for idx, budget in enumerate(budgets):
        rec = summary.get((EMBEDDING_METHOD, budget), {})
        obs = max(0.0, float(rec.get("observed_share", 0.0)))
        idw = max(0.0, float(rec.get("idw_share", 0.0)))
        prior = max(0.0, float(rec.get("fallback_share", 0.0)))
        total = obs + idw + prior
        if total <= 0.0:
            obs, idw, prior = 1.0, 0.0, 0.0
            total = 1.0
        obs /= total
        idw /= total
        prior /= total

        x = ml + idx * slot + (slot - bar_w) / 2.0
        y = mt + ph
        for value, color, label in (
            (obs, METHOD_COLORS[EMBEDDING_METHOD], "Observed"),
            (idw, METHOD_COLORS["idw"], "IDW / exact match"),
            (prior, METHOD_COLORS["fallback"], "Mean / prior fallback"),
        ):
            h = ph * value
            y -= h
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{escape(color)}" '
                f'aria-label="budget {budget} {escape(label)} share {_pct(value, 1)}" />'
            )
        labels.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 34}" text-anchor="middle" class="axis">{budget:,}</text>'
        )
        labels.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 18}" text-anchor="middle" class="axis">{escape(_pct(obs, 0))}/{escape(_pct(idw, 0))}/{escape(_pct(prior, 0))}</text>'
        )

    legend = (
        f'<span><i style="background:{METHOD_COLORS[EMBEDDING_METHOD]}"></i>Observed</span>'
        f'<span><i style="background:{METHOD_COLORS["idw"]}"></i>IDW / exact match</span>'
        f'<span><i style="background:{METHOD_COLORS["fallback"]}"></i>Mean / prior fallback</span>'
    )

    return (
        "<figure class=\"chart\"><figcaption>Embedding provenance shares by exact budget (population counts)</figcaption>"
        f"<div class=\"chart-legend\">{legend}</div>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Embedding provenance shares by exact budget\">"
        + _chart_axis_labels(height, ml, mt, ph, as_percent=True)
        + "".join(bars)
        + "".join(labels)
        + f'<line x1="{ml}" y1="{mt + ph:.2f}" x2="{ml + pw}" y2="{mt + ph:.2f}" class="grid" />'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 4}" text-anchor="middle" class="axis">Exact token budget</text>'
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">Population share (%)</text>'
        + "</svg></div></figure>"
    )


def _render_executive_summary(
    artifacts: LoadedV4Artifacts,
    summary: dict[tuple[str, int], dict[str, float]],
    mae_summary: dict[tuple[str, int], dict[str, float]],
    budgets: list[int],
) -> str:
    population = _safe_int(artifacts.aggregate.get("population_count", 0), field_name="population_count")
    runtime = _finite(artifacts.aggregate.get("runtime_seconds", 0.0), field_name="runtime_seconds")

    idw_improved = 0
    idw_regressed = 0
    idw_tied = 0
    for row in artifacts.runs_jsonl:
        if str(row.get("method") or "") != EMBEDDING_METHOD:
            continue
        rates = ((row.get("model_assisted") or {}).get("rates") or {})
        delta = _finite(rates.get("delta_vs_selected_only_absolute_error", 0.0), field_name="idw delta")
        if delta < -1e-12:
            idw_improved += 1
        elif delta > 1e-12:
            idw_regressed += 1
        else:
            idw_tied += 1

    no_candidate_novel = 0
    full_scan_fallbacks = 0
    for row in artifacts.runs_jsonl:
        if str(row.get("method") or "") != MINHASH_METHOD:
            continue
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        no_candidate_novel += _safe_int(telemetry.get("no_candidate_novel", 0), field_name="no_candidate_novel")
        full_scan_fallbacks += _safe_int(
            telemetry.get("full_scan_fallbacks", 0),
            field_name="full_scan_fallbacks",
        )

    winner_counts: dict[str, int] = {}
    for budget in budgets:
        candidates = []
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            rec = mae_summary.get((method, budget), {})
            error_key = "idw_mean" if method == EMBEDDING_METHOD else "selected_mean"
            label = "Full-session embedding + IDW" if method == EMBEDDING_METHOD else _method_label(method)
            candidates.append((float(rec.get(error_key, 1.0)), label))
        candidates.sort(key=lambda x: x[0])
        winner_counts[candidates[0][1]] = winner_counts.get(candidates[0][1], 0) + 1

    winner_summary = "; ".join(
        f"{label.replace(' selected-only', '')} won {count} of {len(budgets)} budgets"
        for label, count in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
    )

    if 20 in {int(r.get("legacy_tier_pct") or 0) for r in artifacts.runs_jsonl}:
        rep_budget = min(
            int(r.get("budget_tokens") or 0)
            for r in artifacts.runs_jsonl
            if int(r.get("legacy_tier_pct") or 0) == 20
        )
    else:
        rep_budget = sorted(budgets)[len(budgets) // 2] if budgets else 0

    rep_rows = []
    for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
        row = summary.get((method, rep_budget), {})
        mae = mae_summary.get((method, rep_budget), {})
        error_key = "idw_mean" if method == EMBEDDING_METHOD else "selected_mean"
        label = "Full-session embedding + IDW" if method == EMBEDDING_METHOD else _method_label(method)
        rep_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{rep_budget:,}</td>"
            f"<td>{escape(_num(float(row.get('selected_count',0.0)),2))}</td>"
            f"<td>{escape(_num(float(mae.get(error_key,0.0)),4))}</td>"
            f"<td>{escape(_pct(float(row.get('fraction_saved',0.0)),1))}</td>"
            f"<td>{escape(_pct(float(row.get('concept_coverage',0.0)),1))}</td>"
            "</tr>"
        )

    return (
        "<p class=\"tight\">This report answers one question: under fixed exact token budgets, which sampling method best approximates census outcome rates while preserving coverage. "
        f"The current run covers {population:,} sessions and completed in {_duration(runtime)}. "
        "Percent tier labels are provenance labels only; exact token budgets are the authoritative comparison axis.</p>"
        "<div class=\"summary-grid\">"
        "<div class=\"summary-panel\"><h3>Experiment Design</h3>"
        f"<p>Three methods compared at matched exact budgets across {len(budgets)} budget levels with persisted memberships and deterministic fills.</p></div>"
        "<div class=\"summary-panel\"><h3>Outcome Winner Pattern</h3>"
        f"<p>{escape(winner_summary)}. Exact values are in the Outcomes tab.</p></div>"
        "<div class=\"summary-panel\"><h3>IDW Verdict</h3>"
        f"<p>Embedding IDW improved {idw_improved} cells, regressed {idw_regressed}, tied {idw_tied}.</p></div>"
        "<div class=\"summary-panel\"><h3>MinHash Verdict</h3>"
        f"<p>{no_candidate_novel:,} bucket misses were treated as novel; {full_scan_fallbacks:,} exhaustive scans were used.</p></div>"
        "</div>"
        "<h3>Representative Exact-Budget Table</h3>"
        "<p class=\"tight\">Representative budget uses 20% legacy provenance when available; otherwise median exact budget.</p>"
        "<div class=\"table-scroll\"><table><thead><tr>"
        "<th>Reported method</th><th>Exact token budget</th><th>Selected sessions mean</th><th>Reported MAE</th><th>Fraction saved</th><th>Concept coverage</th>"
        "</tr></thead><tbody>"
        + "".join(rep_rows)
        + "</tbody></table></div>"
        "<section class=\"print-keep\"><h3>Metric Definitions</h3>"
        "<ul>"
        "<li>Selected-only absolute aggregate rate error = |selected pass rate - census|.</li>"
        "<li>IDW aggregate error = |observed+imputed population mean - census|.</li>"
        "<li>Fraction saved = 1 - (selected count / population count).</li>"
        "<li>Concept coverage = covered concepts / total concepts in the run aggregate.</li>"
        "<li>Deterministic expected labels are pseudo-judge outputs; adaptive methods are not design-unbiased estimators.</li>"
        "</ul></section>"
    )


def _render_outcomes(
    artifacts: LoadedV4Artifacts,
    summary: dict[tuple[str, int], dict[str, float]],
    mae_summary: dict[tuple[str, int], dict[str, float]],
    budgets: list[int],
) -> str:
    chart1 = _plot_grouped_mae(
        budgets=budgets,
        mae=mae_summary,
        title="Mean absolute error versus census by exact token budget",
    )

    chart2 = _plot_outcome_percentage(
        budgets=budgets,
        summary=summary,
        metric_key="fraction_saved",
        title="Fraction saved by exact token budget",
        y_label="Fraction saved",
    )
    chart3 = _plot_outcome_percentage(
        budgets=budgets,
        summary=summary,
        metric_key="concept_coverage",
        title="Concept coverage by exact token budget",
        y_label="Concept coverage",
    )

    dense_rows = []
    for budget in budgets:
        best = None
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            key = "idw_mean" if method == EMBEDDING_METHOD else "selected_mean"
            v = float(mae_summary.get((method, budget), {}).get(key, 1.0))
            if best is None or v < best:
                best = v
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            row = summary.get((method, budget), {})
            mrow = mae_summary.get((method, budget), {})
            error_key = "idw_mean" if method == EMBEDDING_METHOD else "selected_mean"
            error = float(mrow.get(error_key, 0.0))
            best_text = "best reported MAE" if abs(error - float(best or 0.0)) < 1e-12 else ""

            tier = ""
            tier_rows = [
                int(r.get("legacy_tier_pct", 0))
                for r in artifacts.runs_jsonl
                if str(r.get("method")) == method and int(r.get("budget_tokens", 0)) == budget
            ]
            if tier_rows:
                tier = f"{min(tier_rows)}%"

            dense_rows.append(
                "<tr>"
                f"<td>{escape(tier)}</td>"
                f"<td>{budget:,}</td>"
                f"<td>{escape('Full-session embedding + IDW' if method == EMBEDDING_METHOD else _method_label(method))}</td>"
                f"<td>{escape(_num(float(row.get('selected_count',0.0)),2))}</td>"
                f"<td>{escape(_num(error,4))}</td>"
                f"<td>{escape(_pct(float(row.get('fraction_saved',0.0)),1))}</td>"
                f"<td>{escape(_pct(float(row.get('concept_coverage',0.0)),1))}</td>"
                f"<td>{escape(best_text)}</td>"
                "</tr>"
            )

    interp1 = []
    for budget in budgets:
        entries = []
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            key = "idw_mean" if method == EMBEDDING_METHOD else "selected_mean"
            label = "Full-session embedding + IDW" if method == EMBEDDING_METHOD else _method_label(method)
            entries.append((float(mae_summary.get((method, budget), {}).get(key, 1.0)), label))
        entries.sort(key=lambda x: x[0])
        interp1.append(f"{budget:,}: {entries[0][1]} lowest MAE {_num(entries[0][0],4)}")

    saved_interpretation = []
    coverage_interpretation = []
    for budget in budgets:
        best_saved = None
        best_cov = None
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            rec = summary.get((method, budget), {})
            fs = float(rec.get("fraction_saved", 0.0))
            cv = float(rec.get("concept_coverage", 0.0))
            label = "Full-session embedding + IDW" if method == EMBEDDING_METHOD else _method_label(method)
            if best_saved is None or fs > best_saved[0]:
                best_saved = (fs, label)
            if best_cov is None or cv > best_cov[0]:
                best_cov = (cv, label)
        saved_interpretation.append(f"{budget:,}: {best_saved[1]} {_pct(best_saved[0],1)}")
        coverage_interpretation.append(f"{budget:,}: {best_cov[1]} {_pct(best_cov[0],1)}")

    return (
        "<p class=\"tight\">Outcome comparisons keep exact budget parity. Random and MinHash report selected-only error; full-session embedding reports the observed-plus-IDW population estimate.</p>"
        + chart1
        + f"<p class=\"tight\">Interpretation: {escape('; '.join(interp1))}.</p>"
        + chart2
        + f"<p class=\"tight\">Highest fraction saved: {escape('; '.join(saved_interpretation))}.</p>"
        + chart3
        + f"<p class=\"tight\">Highest concept coverage: {escape('; '.join(coverage_interpretation))}.</p>"
        + "<h3>Dense Outcome Table</h3>"
        + "<div class=\"table-scroll\"><table><thead><tr>"
        + "<th>Legacy provenance tier</th><th>Exact tokens</th><th>Reported method</th><th>Selected count mean</th><th>Reported MAE</th><th>Fraction saved</th><th>Concept coverage</th><th>Note</th>"
        + "</tr></thead><tbody>"
        + "".join(dense_rows)
        + "</tbody></table></div>"
    )


def _render_method_panel(title: str, steps: list[str], facts: list[tuple[str, str]]) -> str:
    step_rows = "".join(f"<li>{escape(item)}</li>" for item in steps)
    fact_rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>" for label, value in facts
    )
    return (
        f"<div class=\"method-panel\"><h3>{escape(title)}</h3><ol>{step_rows}</ol>"
        f"<div class=\"table-scroll\"><table><tbody>{fact_rows}</tbody></table></div></div>"
    )


def _render_sampling_methods(artifacts: LoadedV4Artifacts) -> str:
    idw_cfg = artifacts.idw_config.get("idw_config") or {}
    k = _safe_int(idw_cfg.get("k", 0), field_name="idw_config.k") if isinstance(idw_cfg, dict) else 0
    power = _finite(idw_cfg.get("power", 2.0), field_name="idw_config.power") if isinstance(idw_cfg, dict) else 2.0
    eps = _finite(idw_cfg.get("eps", 0.0), field_name="idw_config.eps") if isinstance(idw_cfg, dict) else 0.0
    prior = _finite(idw_cfg.get("prior", 0.0), field_name="idw_config.prior") if isinstance(idw_cfg, dict) else 0.0

    panels = [
        _render_method_panel(
            "Random token-priority selection",
            [
                "Build an exact token budget from persisted budget manifests.",
                "Rank sessions deterministically from the paired randomized replay order.",
                "Pack maximal whole sessions without splitting.",
                "Seal labels until membership is frozen.",
            ],
            [
                ("Estimation mode", "Selected-only"),
                ("Budget rule", "Exact token budget is authoritative"),
                ("Fill behavior", "No separate adaptive fill stage"),
            ],
        ),
        _render_method_panel(
            "MinHash LSH 32x4 selection",
            [
                "Generate a 128-value MinHash signature split into 32 bands of 4 rows.",
                "Propose sessions from candidate collisions under exact budget.",
                "If no candidates collide, mark proposal as novel/no-candidate.",
                "Do not execute exhaustive full-corpus fallback scans.",
            ],
            [
                ("LSH shape", "128 values; 32 bands x 4 rows"),
                ("No-candidate handling", "Novel proposal; no exhaustive scan"),
                ("Estimation mode", "Selected-only"),
            ],
        ),
        _render_method_panel(
            "Embedding selection",
            [
                "Embed sessions under the same exact budget process.",
                "Select native embedding proposals with deterministic budget fill.",
                "Freeze membership before any outcome estimation.",
                "Track observed coverage and unjudged units for IDW stage.",
            ],
            [
                ("Estimation mode", "Selected-only baseline + IDW extension"),
                ("Label handling", "Labels remain sealed until membership freeze"),
                ("Budget packing", "Maximal whole-session packing"),
            ],
        ),
        _render_method_panel(
            "Same-agent IDW estimation",
            [
                "After membership freeze, impute only unjudged units.",
                "Use same-agent donor neighborhoods with inverse-distance weighting.",
                "Fallback to prior only when no donor is available.",
                "Combine observed and imputed means into aggregate rate estimate.",
            ],
            [
                ("k", str(k)),
                ("power", _num(power, 3)),
                ("distance epsilon", _num(eps, 8)),
                ("prior probability", _num(prior, 3)),
            ],
        ),
    ]

    return (
        "<p class=\"tight\">The complete sampling and estimation workflow is defined below.</p>"
        + "<div class=\"method-grid\">"
        + "".join(panels)
        + "</div>"
    )


def _quadrant_population_table(quadrants: dict[str, Any]) -> str:
    summary = quadrants.get("quadrant_summary") if isinstance(quadrants, dict) else {}
    if not isinstance(summary, dict) or not summary:
        return "<p>Quadrant population summary unavailable.</p>"

    rows = []
    for name, payload in sorted(summary.items()):
        if not isinstance(payload, dict):
            continue
        sessions = int(
            payload.get("unit_count")
            or payload.get("sessions")
            or payload.get("session_count")
            or payload.get("count")
            or 0
        )
        agents = int(payload.get("agents") or payload.get("agent_count") or 0)
        source_counts = payload.get("corpus_counts") or payload.get("source_counts")
        source_text = "-"
        if isinstance(source_counts, dict):
            parts = [f"{k}:{int(v)}" for k, v in sorted(source_counts.items())]
            source_text = ", ".join(parts)
        rows.append(
            "<tr>"
            f"<td>{escape(str(name))}</td><td>{sessions:,}</td><td>{agents:,}</td><td>{escape(source_text)}</td>"
            "</tr>"
        )

    return (
        "<div class=\"table-scroll\"><table><thead><tr><th>Quadrant</th><th>Sessions</th><th>Agents</th><th>Source counts</th></tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _plot_quadrant_metric(
    *,
    aggregate_groups: list[dict[str, Any]],
    metric_key: str,
    title: str,
    y_label: str,
) -> str:
    if not aggregate_groups:
        return f"<p>{escape(title)} unavailable.</p>"

    width = 980
    height = 360
    ml, mr, mt, mb = 88, 30, 28, 90
    pw = width - ml - mr
    ph = height - mt - mb

    methods = [RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD]
    quadrants = sorted({str(r.get("quadrant") or "") for r in aggregate_groups if str(r.get("quadrant") or "")})
    if not quadrants:
        return f"<p>{escape(title)} unavailable.</p>"

    quadrant_budgets = sorted(
        {
            (str(row.get("quadrant") or ""), int(row.get("budget_tokens") or 0))
            for row in aggregate_groups
        }
    )
    cells = [(quadrant, budget, method) for quadrant, budget in quadrant_budgets for method in methods]
    slot = pw / max(1, len(cells))
    bar_w = max(2.0, min(7.0, slot * 0.75))

    agg: dict[tuple[str, int, str], list[float]] = {}
    for row in aggregate_groups:
        key = (str(row.get("quadrant") or ""), int(row.get("budget_tokens") or 0), str(row.get("method") or ""))
        value = _finite(row.get(metric_key, 0.0), field_name=metric_key)
        agg.setdefault(key, []).append(value)

    bars = []
    x_labels = []
    for idx, (quadrant, budget, method) in enumerate(cells):
        values = agg.get((quadrant, budget, method), [])
        val = float(mean(values)) if values else 0.0
        val = max(0.0, min(1.0, val))
        h = ph * val
        x = ml + idx * slot + (slot - bar_w) / 2.0
        y = mt + ph - h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{escape(METHOD_COLORS.get(method, "#666"))}" '
            f'aria-label="{escape(quadrant)} budget {budget} {escape(_method_label(method))} {escape(metric_key)} {_pct(val,1)}" />'
        )

        if idx % len(methods) == 1:
            q_short = quadrant.replace("high_variety_", "HV-").replace("low_variety_", "LV-").replace("high_velocity", "HVel").replace("low_velocity", "LVel")
            x_labels.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{height - 42}" text-anchor="middle" class="axis">{escape(q_short)}</text>'
            )
            x_labels.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{height - 28}" text-anchor="middle" class="axis">{budget:,}</text>'
            )

    legend = "".join(
        f'<span><i style="background:{escape(METHOD_COLORS[m])}"></i>{escape(_method_label(m))}</span>' for m in methods
    )

    return (
        f"<figure class=\"chart\"><figcaption>{escape(title)}</figcaption>"
        f"<div class=\"chart-legend\">{legend}</div>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(title)}\">"
        + _chart_axis_labels(height, ml, mt, ph, as_percent=True)
        + "".join(bars)
        + "".join(x_labels)
        + f'<line x1="{ml}" y1="{mt + ph:.2f}" x2="{ml + pw}" y2="{mt + ph:.2f}" class="grid" />'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis">Quadrant/budget/method small multiples</text>'
        + f'<text x="20" y="{mt + ph / 2:.2f}" transform="rotate(-90 20 {mt + ph / 2:.2f})" class="axis">{escape(y_label)} (%)</text>'
        + "</svg></div></figure>"
    )


def _render_quadrant_behavior(artifacts: LoadedV4Artifacts) -> str:
    quadrant = artifacts.source_v3_quadrant
    if quadrant is None:
        return "<p class=\"tight\">Quadrant artifact is unavailable in the selection-stage bundle; this panel cannot render persisted quadrant diagnostics.</p>"

    aggregate_groups = [row for row in quadrant.get("aggregate_groups", []) if isinstance(row, dict)]
    quadrants_block = quadrant.get("quadrants") if isinstance(quadrant.get("quadrants"), dict) else {}
    population_table = _quadrant_population_table(quadrants_block)

    rep_chart = _plot_quadrant_metric(
        aggregate_groups=aggregate_groups,
        metric_key="representation_mean",
        title="Representation ratio by quadrant, exact budget, and method",
        y_label="Representation ratio",
    )

    zero_chart = _plot_quadrant_metric(
        aggregate_groups=aggregate_groups,
        metric_key="zero_selection_agent_rate_mean",
        title="Zero-selection agent rate by quadrant, budget, and method",
        y_label="Zero-selection agent rate",
    )

    rep_vals = [float(row.get("representation_mean", 0.0)) for row in aggregate_groups]
    zero_vals = [float(row.get("zero_selection_agent_rate_mean", 0.0)) for row in aggregate_groups]
    rep_text = "-"
    zero_text = "-"
    if rep_vals:
        rep_text = f"representation range {_pct(min(rep_vals),1)} to {_pct(max(rep_vals),1)}"
    if zero_vals:
        zero_text = f"zero-selection rate range {_pct(min(zero_vals),1)} to {_pct(max(zero_vals),1)}"

    detail_rows = []
    for row in sorted(
        aggregate_groups,
        key=lambda item: (
            str(item.get("quadrant") or ""),
            int(item.get("budget_tokens") or 0),
            str(item.get("method") or ""),
        ),
    ):
        tier_values = row.get("legacy_tier_pct_provenance") or []
        tier_text = ", ".join(f"{int(value)}%" for value in tier_values)
        detail_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('quadrant') or ''))}</td>"
            f"<td>{escape(tier_text)}</td>"
            f"<td>{int(row.get('budget_tokens') or 0):,}</td>"
            f"<td>{escape(_method_label(str(row.get('method') or '')))}</td>"
            f"<td>{escape(_pct(float(row.get('representation_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_pct(float(row.get('concept_coverage_mean') or 0.0), 1))}</td>"
            f"<td>{escape(_pct(float(row.get('zero_selection_agent_rate_mean') or 0.0), 1))}</td>"
            "</tr>"
        )

    return (
        "<p class=\"tight\">Quadrant behavior tests the variety/velocity mechanism under fixed exact budgets.</p>"
        + population_table
        + rep_chart
        + f"<p class=\"tight\">Interpretation: {escape(rep_text)} across persisted quadrants and methods.</p>"
        + zero_chart
        + f"<p class=\"tight\">Interpretation: {escape(zero_text)}; high zero-selection pockets indicate admission sparsity rather than judge latency.</p>"
        + "<h3>Exact Quadrant Results</h3>"
        + _render_table(
            detail_rows,
            [
                "Quadrant",
                "Provenance tier",
                "Exact token budget",
                "Method",
                "Representation",
                "Concept coverage",
                "Zero-selection agents",
            ],
        )
    )


def _render_table(rows: list[str], headers: list[str]) -> str:
    return (
        "<div class=\"table-scroll\"><table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _heatmap(
    *,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    value_fmt: str,
) -> str:
    if not rows:
        return f"<p>{escape(title)} unavailable.</p>"

    arrivals = sorted({float(r.get("arrival_rate_sessions_per_second") or 0.0) for r in rows})
    capacities = sorted({float(r.get("eval_capacity_sessions_per_second") or 0.0) for r in rows})
    matrix: dict[tuple[float, float], float] = {}
    for r in rows:
        key = (float(r.get("arrival_rate_sessions_per_second") or 0.0), float(r.get("eval_capacity_sessions_per_second") or 0.0))
        matrix[key] = float(r.get(metric_key) or 0.0)

    v_max = max(matrix.values()) if matrix else 1.0
    v_min = min(matrix.values()) if matrix else 0.0
    span = max(1e-9, v_max - v_min)

    width = 980
    height = 360
    ml, mr, mt, mb = 104, 28, 36, 88
    pw = width - ml - mr
    ph = height - mt - mb

    cell_w = pw / max(1, len(arrivals))
    cell_h = ph / max(1, len(capacities))

    rects = []
    for cy, cap in enumerate(capacities):
        for cx, arr in enumerate(arrivals):
            val = matrix.get((arr, cap), 0.0)
            t = (val - v_min) / span
            r = int(235 - 110 * t)
            g = int(245 - 120 * t)
            b = int(250 - 160 * t)
            x = ml + cx * cell_w
            y = mt + (len(capacities) - 1 - cy) * cell_h
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="rgb({r},{g},{b})" stroke="#dce3e8" '
                f'aria-label="arrival {arr} capacity {cap} value {value_fmt.format(val)}" />'
            )
            rects.append(
                f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="middle" class="axis">{escape(value_fmt.format(val))}</text>'
            )

    x_labels = []
    for idx, arr in enumerate(arrivals):
        x = ml + idx * cell_w + cell_w / 2.0
        x_labels.append(f'<text x="{x:.2f}" y="{height - 34}" text-anchor="middle" class="axis">{arr:g}</text>')

    y_labels = []
    for idx, cap in enumerate(reversed(capacities)):
        y = mt + idx * cell_h + cell_h / 2.0
        y_labels.append(f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{cap:g}</text>')

    return (
        f"<figure class=\"chart\"><figcaption>{escape(title)}</figcaption>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{escape(title)}\">"
        + "".join(rects)
        + "".join(x_labels)
        + "".join(y_labels)
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Arrival rate (sessions/s)</text>'
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">Evaluation capacity (sessions/s)</text>'
        + "</svg></div></figure>"
    )


def _render_throughput(artifacts: LoadedV4Artifacts) -> str:
    throughput = artifacts.source_v3_throughput
    if throughput is None:
        return "<p class=\"tight\">Throughput artifact is unavailable in the selection-stage bundle; admission pressure and queue diagnostics cannot render.</p>"

    grid = [row for row in throughput.get("aggregate_grid", []) if isinstance(row, dict)]
    if not grid:
        return "<p class=\"tight\">Throughput aggregate grid is empty.</p>"
    budgets = sorted({int(row.get("budget_tokens") or 0) for row in grid})

    rows = []
    for budget in budgets:
        for method in (RANDOM_METHOD, MINHASH_METHOD, EMBEDDING_METHOD):
            scoped = [r for r in grid if int(r.get("budget_tokens") or 0) == budget and str(r.get("method") or "") == method]
            if not scoped:
                continue
            worst_pressure = max(float(r.get("token_pressure_ratio_mean") or 0.0) for r in scoped)
            worst_latency = max(float(r.get("decision_runtime_ms_p95_mean") or 0.0) for r in scoped)
            rows.append(
                "<tr>"
                f"<td>{budget:,}</td><td>{escape(_method_label(method))}</td><td>{escape(_num(worst_pressure,3))}</td><td>{escape(_num(worst_latency,1))}</td>"
                "</tr>"
            )

    focus_budget = budgets[-1] if budgets else 0
    embedding_rows = [
        r
        for r in grid
        if int(r.get("budget_tokens") or 0) == focus_budget and str(r.get("method") or "") == EMBEDDING_METHOD
    ]

    pressure_chart = _heatmap(
        rows=embedding_rows,
        metric_key="token_pressure_ratio_mean",
        title=f"Token pressure heatmap (embedding, exact budget {focus_budget:,})",
        value_fmt="{:.2f}",
    )
    latency_chart = _heatmap(
        rows=embedding_rows,
        metric_key="decision_runtime_ms_p95_mean",
        title=f"P95 decision latency heatmap (embedding, exact budget {focus_budget:,})",
        value_fmt="{:.0f}",
    )

    return (
        "<p class=\"tight\">Throughput varies arrival rate against evaluation capacity under queue capacity policy. Metrics describe admission mechanics, not LLM generation latency.</p>"
        + _render_table(
            rows,
            [
                "Exact budget",
                "Method",
                "Worst token pressure ratio",
                "Worst p95 decision latency (ms)",
            ],
        )
        + pressure_chart
        + "<p class=\"tight\">Interpretation: pressure ratios near 1.0 indicate sustained budget saturation for embedding under this budget.</p>"
        + latency_chart
        + "<p class=\"tight\">Interpretation: p95 latency peaks align with capacity-constrained cells; this is queueing pressure, not model response time.</p>"
    )


def _plot_reliability(calibration_bins: list[dict[str, Any]]) -> str:
    width = 980
    height = 360
    ml, mr, mt, mb = 72, 28, 28, 82
    pw = width - ml - mr
    ph = height - mt - mb

    if not calibration_bins:
        return "<p>Calibration reliability plot unavailable.</p>"

    points = []
    for row in calibration_bins:
        x = _safe_prob(row.get("avg_prediction", 0.0), field_name="avg_prediction")
        y = _safe_prob(row.get("avg_label", 0.0), field_name="avg_label")
        px = ml + pw * x
        py = mt + ph - ph * y
        points.append((px, py, int(float(row.get("count", 0.0)))))

    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{max(2.5, min(8.0, 2.0 + math.sqrt(max(0, c)))):.2f}" '
        f'fill="{METHOD_COLORS[EMBEDDING_METHOD]}" opacity="0.75" aria-label="calibration count {c}" />'
        for x, y, c in points
    )

    line = (
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt}" '
        'stroke="#b04a36" stroke-width="1.6" stroke-dasharray="6 4" />'
    )

    return (
        "<figure class=\"chart\"><figcaption>Calibration reliability plot (embedding representative run)</figcaption>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Calibration reliability plot\">"
        + line
        + circles
        + f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" class="grid-box" />'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Predicted probability</text>'
        + f'<text x="16" y="{mt + ph / 2:.2f}" transform="rotate(-90 16 {mt + ph / 2:.2f})" class="axis">Observed pass-rate</text>'
        + "</svg></div></figure>"
    )


def _plot_distance_error(distance_bins: list[dict[str, Any]]) -> str:
    width = 980
    height = 360
    ml, mr, mt, mb = 72, 28, 28, 82
    pw = width - ml - mr
    ph = height - mt - mb

    rows = [row for row in distance_bins if row.get("mae") is not None and row.get("avg_distance") is not None]
    if not rows:
        return "<p>Nearest-distance versus error plot unavailable.</p>"

    x_max = max(_finite(row.get("avg_distance"), field_name="avg_distance") for row in rows)
    y_max = max(_finite(row.get("mae"), field_name="mae") for row in rows)
    x_max = max(x_max, 0.01)
    y_max = max(y_max, 0.01)

    dots = []
    for row in rows:
        x = _finite(row.get("avg_distance"), field_name="avg_distance")
        y = _finite(row.get("mae"), field_name="mae")
        c = int(row.get("count") or 0)
        px = ml + pw * (x / x_max)
        py = mt + ph - ph * (y / y_max)
        radius = max(2.5, min(7.5, 2.0 + math.sqrt(max(0, c))))
        dots.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{METHOD_COLORS[RANDOM_METHOD]}" opacity="0.78" '
            f'aria-label="distance {_num(x, 3)} mae {_num(y, 3)} count {c}" />'
        )

    return (
        "<figure class=\"chart\"><figcaption>Nearest-distance versus error (embedding representative run)</figcaption>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Nearest-distance versus error\">"
        + "".join(dots)
        + f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" class="grid-box" />'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Average nearest distance</text>'
        + f'<text x="16" y="{mt + ph / 2:.2f}" transform="rotate(-90 16 {mt + ph / 2:.2f})" class="axis">Mean absolute error</text>'
        + "</svg></div></figure>"
    )


def _plot_agent_scatter(per_agent_rows: list[dict[str, Any]]) -> str:
    width = 980
    height = 360
    ml, mr, mt, mb = 72, 28, 28, 82
    pw = width - ml - mr
    ph = height - mt - mb

    if not per_agent_rows:
        return "<p>Per-agent estimated versus census dots unavailable.</p>"

    dots = []
    for row in per_agent_rows:
        est = _safe_prob(row.get("estimated_pass_rate"), field_name="per_agent.estimated_pass_rate")
        cen = _safe_prob(row.get("census_pass_rate"), field_name="per_agent.census_pass_rate")
        count = int(row.get("population_count") or 0)
        px = ml + pw * est
        py = mt + ph - ph * cen
        radius = max(2.8, min(8.5, 2.1 + math.sqrt(max(0, count))))
        dots.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{METHOD_COLORS[MINHASH_METHOD]}" opacity="0.8" '
            f'aria-label="agent {escape(str(row.get("agent_id") or ""))} estimated {_num(est, 3)} census {_num(cen, 3)}" />'
        )

    diagonal = (
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt}" '
        'stroke="#485760" stroke-width="1.6" stroke-dasharray="6 4" />'
    )

    return (
        "<figure class=\"chart\"><figcaption>Per-agent estimated versus census dots (embedding representative run)</figcaption>"
        f"<div class=\"chart-scroll\"><svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-agent estimated versus census dots\">"
        + diagonal
        + "".join(dots)
        + f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" class="grid-box" />'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="axis">Estimated pass-rate</text>'
        + f'<text x="16" y="{mt + ph / 2:.2f}" transform="rotate(-90 16 {mt + ph / 2:.2f})" class="axis">Census pass-rate</text>'
        + "</svg></div></figure>"
    )


def _render_embedding_diagnostics(
    artifacts: LoadedV4Artifacts,
    summary: dict[tuple[str, int], dict[str, float]],
) -> str:
    representative = _representative_embedding_run(artifacts.runs_jsonl)
    if representative is None:
        return "<p class=\"tight\">No embedding representative run available.</p>"

    budget = int(representative.get("budget_tokens") or 0)
    tier = int(representative.get("legacy_tier_pct") or 0)
    rep = int(representative.get("repetition") or 0)
    order_hash = str(representative.get("order_hash") or "")

    model_assisted = representative.get("model_assisted") or {}
    rates = model_assisted.get("rates") if isinstance(model_assisted, dict) else {}
    counts = model_assisted.get("counts") if isinstance(model_assisted, dict) else {}
    metrics = model_assisted.get("metrics") if isinstance(model_assisted, dict) else {}

    rates = rates if isinstance(rates, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}

    calibration_bins = [row for row in metrics.get("calibration_bins", []) if isinstance(row, dict)]
    distance_bins = [row for row in metrics.get("nearest_distance_error_bins", []) if isinstance(row, dict)]
    per_agent = [row for row in metrics.get("per_agent", []) if isinstance(row, dict)]

    inventory = artifacts.source_v3_token_inventory or []
    emitted = sum(int(row.get("emitted_tokens") or 0) for row in inventory)
    original = sum(int(row.get("original_tokens") or 0) for row in inventory)
    truncated = sum(1 for row in inventory if bool(row.get("truncated")))

    source_agg = artifacts.source_v3_aggregate or {}
    ledger = {}
    runtime_block = source_agg.get("runtime") if isinstance(source_agg, dict) else {}
    if isinstance(runtime_block, dict):
        ledger = runtime_block.get("embedding_ledger") or {}
    if not isinstance(ledger, dict):
        cfg_block = source_agg.get("config") if isinstance(source_agg, dict) else {}
        runtime_cfg = cfg_block.get("runtime") if isinstance(cfg_block, dict) else {}
        ledger = runtime_cfg.get("embedding_ledger") if isinstance(runtime_cfg, dict) else {}
    if not isinstance(ledger, dict):
        ledger = {}

    counts_row = [
        "<tr>"
        f"<td>{budget:,}</td><td>{tier}%</td><td>{rep}</td><td>{escape(order_hash)}</td>"
        f"<td>{_safe_int(counts.get('observed_count',0), field_name='observed'):,}</td><td>{_safe_int(counts.get('imputed_count',0), field_name='imputed'):,}</td>"
        f"<td>{escape(_num(float(rates.get('absolute_aggregate_rate_error',0.0)),4))}</td>"
        f"<td>{escape(_num(float(rates.get('delta_vs_selected_only_absolute_error',0.0)),4))}</td>"
        f"<td>{escape(_num(float(metrics.get('unjudged_only_mae',0.0)),4))}</td>"
        f"<td>{escape(_num(float(metrics.get('unjudged_only_brier',0.0)),4))}</td>"
        f"<td>{escape(_num(float(metrics.get('expected_calibration_error',0.0)),4))}</td>"
        f"<td>{escape(_num(float((metrics.get('leave_one_out') or {}).get('mae',0.0)),4))}</td>"
        f"<td>{escape(_num(float((metrics.get('leave_one_out') or {}).get('brier_score',0.0)),4))}</td>"
        f"<td>{_safe_int(counts.get('zero_donor_agent_count',0), field_name='zero_donor'):,}</td>"
        f"<td>{_safe_int(counts.get('prior_count',0), field_name='prior_count'):,}</td>"
        "</tr>"
    ]

    ledger_rows = [
        f"<tr><td>Token inventory rows</td><td>{len(inventory):,}</td></tr>",
        f"<tr><td>Emitted tokens</td><td>{emitted:,}</td></tr>",
        f"<tr><td>Original tokens</td><td>{original:,}</td></tr>",
        f"<tr><td>Truncation count</td><td>{truncated:,}</td></tr>",
        f"<tr><td>Embedding calls</td><td>{int(ledger.get('embedding_calls') or 0):,}</td></tr>",
        f"<tr><td>Embedding input tokens</td><td>{int(ledger.get('embedding_input_tokens') or 0):,}</td></tr>",
        f"<tr><td>Embedding model</td><td>{escape(str(ledger.get('embedding_model_id') or ''))}</td></tr>",
        f"<tr><td>Embedding deployment</td><td>{escape(str(ledger.get('embedding_deployment_id') or ''))}</td></tr>",
    ]

    return (
        "<p class=\"tight\">Representative embedding run selection is deterministic: highest legacy tier, then lowest repetition, then lowest exact budget, then lexical order hash.</p>"
        + _render_table(
            counts_row,
            [
                "Exact budget",
                "Legacy tier",
                "Repetition",
                "Order hash",
                "Observed",
                "Imputed",
                "Aggregate IDW error",
                "Delta vs selected-only",
                "Unjudged MAE",
                "Unjudged Brier",
                "ECE",
                "LOO MAE",
                "LOO Brier",
                "Zero-donor agents",
                "Prior counts",
            ],
        )
        + _plot_reliability(calibration_bins)
        + _plot_distance_error(distance_bins)
        + _plot_agent_scatter(per_agent)
        + "<h3>Token and Embedding Ledger</h3>"
        + _render_table(ledger_rows, ["Metric", "Value"])
    )


def _render_lineage_integrity(artifacts: LoadedV4Artifacts) -> str:
    source_link = artifacts.source_lineage
    source_manifest_meta = source_link.get("source_manifest") if isinstance(source_link.get("source_manifest"), dict) else {}

    rows = [
        f"<tr><td>Selection bundle subdir</td><td>{escape(str(source_link.get('source_bundle_subdir') or ''))}</td></tr>",
        f"<tr><td>Selection bundle version</td><td>{escape(str(source_link.get('source_bundle_version') or ''))}</td></tr>",
        f"<tr><td>Selection outcome version</td><td>{escape(str(source_link.get('source_outcome_version') or ''))}</td></tr>",
        f"<tr><td>Selection manifest version</td><td>{escape(str(source_manifest_meta.get('version') or ''))}</td></tr>",
        f"<tr><td>Selection manifest sha256</td><td>{escape(str(source_manifest_meta.get('sha256') or ''))}</td></tr>",
        f"<tr><td>Selection rerun</td><td>{escape(str(source_link.get('selection_rerun')))}</td></tr>",
        f"<tr><td>Augmentation function</td><td>{escape(str(source_link.get('augmentation') or ''))}</td></tr>",
    ]

    cfg = artifacts.idw_config.get("idw_config") if isinstance(artifacts.idw_config.get("idw_config"), dict) else {}
    methodology = [
        "Membership was frozen in the selection-stage bundle before estimation.",
        "Exact token budgets were preserved cell-for-cell for outcome comparisons.",
        "Random and MinHash remain selected-only outcome estimators.",
        "Embedding received post-freeze same-agent IDW estimation over unjudged units.",
        f"IDW parameters: k={cfg.get('k')}, power={cfg.get('power')}, eps={cfg.get('eps')}, exact_cosine_eps={cfg.get('exact_cosine_eps')}, prior={cfg.get('prior')}.",
        "No exhaustive MinHash full-scan fallback was introduced in this report stage.",
    ]

    return (
        "<p class=\"tight\">Lineage and integrity records are technical and hash-validated. Reader-facing interpretation is fully contained in this report.</p>"
        + _render_table(rows, ["Lineage field", "Value"])
        + "<h3>Standalone Methodology Summary</h3><ul>"
        + "".join(f"<li>{escape(line)}</li>" for line in methodology)
        + "</ul>"
    )


def _render_repro_caveats(artifacts: LoadedV4Artifacts) -> str:
    listed = artifacts.manifest.get("artifacts") or {}
    rows = []
    if isinstance(listed, dict):
        for key, payload in sorted(listed.items()):
            if not isinstance(payload, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(_friendly_source_name(str(key)))}</td>"
                f"<td>{escape(str(payload.get('path') or ''))}</td>"
                f"<td>{int(payload.get('bytes') or 0):,}</td>"
                f"<td>{escape(str(payload.get('sha256') or '')[:18])}</td>"
                "</tr>"
            )

    caveats = [
        "The selection stage did not compute IDW; IDW was applied after membership freeze.",
        "Adaptive methods are not design-unbiased estimators in this setup.",
        "Expected labels are pseudo-judge outputs, not human gold labels.",
        "Synthetic corpus scope may not transfer to production distributions.",
        "Same-agent donor fallback can increase prior-driven estimates when donor coverage is sparse.",
        "Only three paired outcome repetitions were run for this bundle.",
        "Empirical low/high are observed replay ranges, not confidence intervals.",
    ]

    return (
        "<p class=\"tight\">Reproducibility uses persisted manifests and source hashes with friendly display labels.</p>"
        + _render_table(rows, ["Artifact", "Relative path", "Bytes", "SHA-256 prefix"])
        + "<h3>Caveats</h3><ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in caveats)
        + "</ul>"
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


def render_v4_html_report(
    artifacts: LoadedV4Artifacts,
    *,
    section_ids: tuple[str, ...] | None = None,
    report_title: str = "Agent365 Sampling V4 Operational Report",
) -> str:
    validate_v4_artifacts(artifacts)

    aggregate_rows = [
        row for row in ((artifacts.aggregate.get("outcome") or {}).get("aggregate") or []) if isinstance(row, dict)
    ]
    summary = _summarize_outcomes(artifacts.runs_jsonl)
    mae_summary = _summarize_mae(aggregate_rows)
    budgets = sorted({int(row.get("budget_tokens") or 0) for row in artifacts.runs_jsonl})

    tabs = [
        ("executive", "Executive Summary", _render_executive_summary(artifacts, summary, mae_summary, budgets)),
        ("outcomes", "Outcomes", _render_outcomes(artifacts, summary, mae_summary, budgets)),
        ("methods", "Sampling Methods", _render_sampling_methods(artifacts)),
        ("quadrant", "Quadrant Behavior", _render_quadrant_behavior(artifacts)),
        ("throughput", "Throughput", _render_throughput(artifacts)),
        ("embedding", "Embedding Diagnostics", _render_embedding_diagnostics(artifacts, summary)),
        ("lineage", "Lineage & Integrity", _render_lineage_integrity(artifacts)),
        ("repro", "Reproducibility & Caveats", _render_repro_caveats(artifacts)),
    ]
    if section_ids is not None:
        requested = set(section_ids)
        available = {tab_id for tab_id, _title, _content in tabs}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown report section ids: {','.join(sorted(unknown))}")
        tabs = [tab for tab in tabs if tab[0] in requested]
        if not tabs:
            raise ValueError("section_ids must include at least one report section")

    tab_buttons = []
    tab_panels = []
    for idx, (tab_id, title, content) in enumerate(tabs):
        selected = idx == 0
        tab_buttons.append(_tab_button(tab_id, title, selected=selected))
        tab_panels.append(_tab_panel(tab_id, title, content, selected=selected))

    generated_at = str(artifacts.aggregate.get("generated_at") or "")
    population = int(artifacts.aggregate.get("population_count") or 0)
    runtime = _duration(float(artifacts.aggregate.get("runtime_seconds") or 0.0))

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{escape(report_title)}</title>
<style>
:root {{
  --bg-a: #f6f8f7;
  --bg-b: #eef4f2;
  --card: #ffffff;
  --ink: #182027;
  --muted: #4f5b63;
  --line: #d3dde3;
  --accent-a: #1f6d8c;
  --accent-b: #2d7c6d;
  --accent-c: #b04a36;
  --radius: 8px;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
  font-family: Bahnschrift, Aptos, "Segoe UI", sans-serif;
}}
.page {{ max-width: 1280px; margin: 0 auto; padding: 14px; }}
header {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: start;
}}
h1 {{ margin: 0; font-size: 1.34rem; letter-spacing: 0.01em; }}
.header-tools {{ display: flex; gap: 8px; align-items: center; }}
.print-btn {{
  border: 1px solid #9fb4c1;
  border-radius: 8px;
  background: #eef5f9;
  color: #0f3345;
  padding: 7px 10px;
  font-weight: 700;
  cursor: pointer;
}}
.meta-row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }}
.meta {{ border: 1px solid var(--line); border-radius: 8px; padding: 7px 9px; background: #fbfdfd; }}
.meta b {{ display: block; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
.meta span {{ font-size: 0.92rem; }}
.tabs-wrap {{
  margin-top: 10px;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f8fbfb;
}}
.tabs {{ display: flex; gap: 6px; padding: 7px; min-width: max-content; }}
.tab-button {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f8fbfc;
  color: var(--ink);
  padding: 7px 10px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}}
.tab-button[aria-selected=\"true\"] {{ background: #e8f3f9; border-color: #9ec0d4; }}
.tab-panel {{
  margin-top: 10px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
}}
.tab-panel h2 {{ margin: 0 0 8px; font-size: 1.08rem; }}
.tight {{ margin: 0.25rem 0 0.72rem; color: var(--muted); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 8px 0; }}
.summary-panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fafdfc; }}
.summary-panel h3 {{ margin: 0 0 6px; font-size: 0.95rem; }}
.summary-panel p {{ margin: 0; font-size: 0.9rem; }}
.method-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
.method-panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 9px; background: #fcfefe; }}
.method-panel h3 {{ margin: 0 0 6px; font-size: 0.96rem; }}
.method-panel ol {{ margin: 0 0 8px 1.2rem; padding: 0; }}
.method-panel li {{ margin-bottom: 3px; }}
.table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
thead th {{ position: sticky; top: 0; background: #f3f8fa; z-index: 1; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
.chart {{ margin: 10px 0; }}
.chart figcaption {{ font-weight: 700; margin-bottom: 6px; }}
.chart-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; }}
.chart svg {{ width: 980px; height: 360px; display: block; }}
.chart-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 8px; color: var(--muted); font-size: 0.86rem; }}
.chart-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.chart-legend i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.axis {{ fill: #4f5b63; font-size: 11px; }}
.value-label {{ fill: #26343d; font-size: 9px; font-weight: 700; }}
.grid {{ stroke: #e4eaee; stroke-width: 1; }}
.grid-box {{ stroke: #d9e1e6; stroke-width: 1; }}
.footer {{ margin-top: 10px; color: var(--muted); font-size: 0.82rem; }}
@media (max-width: 1000px) {{
  .meta-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .method-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 700px) {{
  .page {{ padding: 10px; }}
  header {{ grid-template-columns: 1fr; }}
  .meta-row {{ grid-template-columns: 1fr; }}
  .summary-grid {{ grid-template-columns: 1fr; }}
}}
@media print {{
    @page {{ size: A4 landscape; margin: 10mm; }}
  body {{ background: #fff; }}
  .tabs-wrap {{ display: none; }}
  .tab-panel[hidden] {{ display: block !important; }}
  .print-btn {{ display: none; }}
  .page {{ max-width: 100%; padding: 0; }}
    header {{ border: none; padding: 0 0 8px; }}
    .tab-panel {{
        display: block !important;
        break-inside: auto;
        page-break-inside: auto;
        border: none;
        margin: 0;
        padding: 8px 0 0;
    }}
    .tab-panel + .tab-panel {{ break-before: page; page-break-before: always; }}
    .chart, .summary-panel, .method-panel, .print-keep {{
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    .chart-scroll, .table-scroll {{ overflow: visible; border-color: #cbd3d8; }}
    .chart svg {{ width: 100%; max-width: 100%; height: auto; }}
    table {{ font-size: 8pt; }}
    th, td {{ padding: 4px 5px; }}
    thead {{ display: table-header-group; }}
    tr {{ break-inside: avoid; page-break-inside: avoid; }}
    .footer {{ display: none; }}
}}
</style>
</head>
<body>
<div class=\"page\">
<header>
  <div>
    <h1>{escape(report_title)}</h1>
    <div class=\"meta-row\" aria-label=\"topline status\">
      <div class=\"meta\"><b>Status</b><span>Validated hash, schema, and selection-stage linkage</span></div>
      <div class=\"meta\"><b>Generated at</b><span>{escape(generated_at)}</span></div>
      <div class=\"meta\"><b>Population</b><span>{population:,} sessions</span></div>
    <div class=\"meta\"><b>Runtime</b><span>{escape(runtime)}</span></div>
    </div>
  </div>
  <div class=\"header-tools\"><button type=\"button\" class=\"print-btn\" id=\"print-report\">Print Report</button></div>
</header>
<div class=\"tabs-wrap\"><nav class=\"tabs\" role=\"tablist\" aria-label=\"Report sections\">{''.join(tab_buttons)}</nav></div>
{''.join(tab_panels)}
<p class=\"footer\">Report version: {REPORT_VERSION}. Self-contained output with inline CSS, JS, and SVG only.</p>
</div>
<script>
(() => {{
  const tabs = Array.from(document.querySelectorAll('.tab-button'));
  const printButton = document.getElementById('print-report');
  if (printButton) {{
    printButton.addEventListener('click', () => window.print());
  }}
  function activate(tab) {{
    tabs.forEach((item) => {{
      const selected = item === tab;
      item.setAttribute('aria-selected', selected ? 'true' : 'false');
      item.tabIndex = selected ? 0 : -1;
      const panel = document.getElementById(item.getAttribute('aria-controls'));
      if (panel) panel.hidden = !selected;
    }});
  }}
  tabs.forEach((tab, index) => {{
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {{
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== index) {{
        event.preventDefault();
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
    source_paths: dict[str, Path],
    report_generator_source_sha256: str,
) -> dict[str, Any]:
    source_inputs: dict[str, dict[str, Any]] = {}
    for key, path in sorted(source_paths.items()):
        source_inputs[key] = {
            "path": str(path),
            "sha256": _sha(path),
            "bytes": int(path.stat().st_size),
        }

    return {
        "version": REPORT_MANIFEST_VERSION,
        "report_filename": output_path.name,
        "report_sha256": _sha(output_path),
        "report_bytes": int(output_path.stat().st_size),
        "aggregate_generated_at": str(aggregate_generated_at),
        "source_input_hashes": source_inputs,
        "report_generator_source_sha256": str(report_generator_source_sha256),
    }


def validate_report_manifest(*, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    if str(payload.get("version")) != REPORT_MANIFEST_VERSION:
        raise ValueError(f"report manifest version must be {REPORT_MANIFEST_VERSION}")
    if str(payload.get("report_filename")) != report_path.name:
        raise ValueError("report manifest filename mismatch")
    if str(payload.get("report_sha256")) != _sha(report_path):
        raise ValueError("report manifest sha256 mismatch")
    if _safe_int(payload.get("report_bytes"), field_name="report_manifest.report_bytes") != int(report_path.stat().st_size):
        raise ValueError("report manifest bytes mismatch")

    source_inputs = payload.get("source_input_hashes")
    if not isinstance(source_inputs, dict) or not source_inputs:
        raise ValueError("report manifest source_input_hashes must be a non-empty object")

    for key, entry in source_inputs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"report manifest entry must be object for {key}")
        path = Path(str(entry.get("path") or ""))
        if not path.exists():
            raise ValueError(f"report manifest source input missing on disk: {key}")
        if str(entry.get("sha256") or "") != _sha(path):
            raise ValueError(f"report manifest source sha mismatch for {key}")
        if _safe_int(entry.get("bytes"), field_name=f"report manifest {key} bytes") != int(path.stat().st_size):
            raise ValueError(f"report manifest source bytes mismatch for {key}")

    generator_sha = str(payload.get("report_generator_source_sha256") or "")
    if len(generator_sha) != 64:
        raise ValueError("report manifest report_generator_source_sha256 missing/invalid")
    return payload


def write_v4_html_report(
    *,
    output_path: Path,
    inputs: V4ReportInputs,
    section_ids: tuple[str, ...] | None = None,
    report_title: str = "Agent365 Sampling V4 Operational Report",
    manifest_name: str = REPORT_MANIFEST_NAME,
) -> Path:
    artifacts = load_v4_artifacts(inputs)
    html = render_v4_html_report(
        artifacts,
        section_ids=section_ids,
        report_title=report_title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    manifest_payload = build_report_manifest_payload(
        output_path=output_path,
        aggregate_generated_at=str(artifacts.aggregate.get("generated_at") or ""),
        source_paths=artifacts.source_paths,
        report_generator_source_sha256=_sha(Path(__file__)),
    )
    report_manifest_path = output_path.with_name(manifest_name)
    report_manifest_path.write_text(_canonical_json(manifest_payload) + "\n", encoding="utf-8")
    validate_report_manifest(report_path=output_path, manifest_path=report_manifest_path)
    return output_path
