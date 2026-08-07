from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v2_experiment import load_combined_dataset  # noqa: E402
from sampling_comparison.v3_experiment import (  # noqa: E402
    V3_EMBEDDING_MODEL,
    V3_MAX_SESSION_PACKET_TOKENS,
    build_v3_runtime,
)
from sampling_comparison.v3_report import (  # noqa: E402
    default_inputs,
    load_v3_artifacts,
    validate_v3_artifacts,
)
from sampling_comparison.v4_idw import (  # noqa: E402
    IDWConfig,
    estimate_embedding_population,
    freeze_membership,
    leave_one_out_donor_diagnostics,
    validate_embedding_population,
)
from trace_sampling.azure_config import AzureConfig  # noqa: E402
from trace_sampling.embedding import AzureOpenAIEmbedder  # noqa: E402
from trace_sampling.session_embedding import TiktokenTokenizer  # noqa: E402

ORACLE_VERSION = "sampling-v4-oracle-v1"


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _embedding_only_config_from_env() -> AzureConfig:
    return AzureConfig(
        openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        embedding_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT", "https://unused.search.windows.net"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX", "unused-index"),
    )


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_expected_label_map(*, unit_ids: Sequence[str], labels_by_unit: Mapping[str, bool]) -> dict[str, float]:
    out: dict[str, float] = {}
    for uid in unit_ids:
        if uid not in labels_by_unit:
            raise ValueError(f"expected label missing for unit_id={uid}")
        out[uid] = 1.0 if bool(labels_by_unit[uid]) else 0.0
    return out


def _validate_selected_membership_payload(
    *,
    membership_payload: Mapping[str, Any],
    aggregate_population_count: int,
    full_dataset_count: int,
) -> list[str]:
    if str(membership_payload.get("version")) != "sampling-v3-selected-membership-v1":
        raise ValueError("selected_membership version must be sampling-v3-selected-membership-v1")
    if int(membership_payload.get("legacy_tier_pct_provenance") or -1) != 20:
        raise ValueError("selected_membership legacy provenance tier must be 20")

    membership = membership_payload.get("membership")
    if not isinstance(membership, Mapping):
        raise ValueError("selected_membership.membership must be an object")
    if str(membership.get("method") or "") != "adaptive_embedding_fullsession_token":
        raise ValueError("selected_membership method must be adaptive_embedding_fullsession_token")

    selected_ids_raw = membership.get("selected_ids")
    if not isinstance(selected_ids_raw, list) or not selected_ids_raw:
        raise ValueError("selected_membership must include non-empty selected_ids")
    selected_ids = [str(x) for x in selected_ids_raw]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected_membership selected_ids contains duplicates")

    declared_selected_count = int(membership.get("selected_count") or -1)
    if declared_selected_count != len(selected_ids):
        raise ValueError("selected_membership selected_count does not match selected_ids")

    if int(aggregate_population_count) != int(full_dataset_count):
        raise ValueError("V3 aggregate population_count must match full dataset count")

    return selected_ids


def _build_agent_and_vectors(
    *,
    eligible_ids: Sequence[str],
    trace_by_unit_id: Mapping[str, Any],
    embedding_vector_by_trace_id: Mapping[int, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    agent_id_by_unit: dict[str, str] = {}
    vector_by_unit: dict[str, Any] = {}
    for uid in eligible_ids:
        trace = trace_by_unit_id.get(uid)
        if trace is None:
            raise ValueError(f"trace missing for eligible unit_id={uid}")
        trace_id = int(trace.trace_id)
        vector = embedding_vector_by_trace_id.get(trace_id)
        if vector is None:
            raise ValueError(f"embedding vector missing for trace_id={trace_id} unit_id={uid}")
        agent_id_by_unit[uid] = str(trace.agent_id)
        vector_by_unit[uid] = vector
    return agent_id_by_unit, vector_by_unit


def compute_v4_oracle(
    *,
    data: Any,
    runtime: Any,
    selected_membership_payload: Mapping[str, Any],
    aggregate_population_count: int,
    v3_manifest_hash: str,
    v3_report_hash: str,
    idw_config: IDWConfig,
    created_at: str | None = None,
    oracle_version: str = ORACLE_VERSION,
) -> dict[str, Any]:
    selected_ids = _validate_selected_membership_payload(
        membership_payload=selected_membership_payload,
        aggregate_population_count=int(aggregate_population_count),
        full_dataset_count=len(data.unit_ids),
    )

    eligible_ids = [str(uid) for uid in data.unit_ids]
    if not set(selected_ids).issubset(set(eligible_ids)):
        raise ValueError("selected_ids must be a subset of full dataset unit_ids")

    membership = freeze_membership(
        cell_id="v4-oracle-selected-membership-20",
        eligible_ids=eligible_ids,
        selected_ids=selected_ids,
    )

    agent_id_by_unit, vector_by_unit = _build_agent_and_vectors(
        eligible_ids=membership.eligible_ids,
        trace_by_unit_id=data.trace_by_unit_id,
        embedding_vector_by_trace_id=runtime.embedding_vector_by_trace_id,
    )

    # Persist/hold membership hash before labels are materialized.
    frozen_membership_hash = str(membership.membership_hash)

    expected_labels_all = _to_expected_label_map(unit_ids=membership.eligible_ids, labels_by_unit=data.labels_by_unit)
    judged_values_selected = {uid: expected_labels_all[uid] for uid in membership.selected_ids}

    estimates = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent_id_by_unit,
        vector_by_unit=vector_by_unit,
        judged_values_by_unit=judged_values_selected,
        config=idw_config,
    )
    validation = validate_embedding_population(estimates, expected_labels_all)
    loo = leave_one_out_donor_diagnostics(
        membership=membership,
        agent_id_by_unit=agent_id_by_unit,
        vector_by_unit=vector_by_unit,
        judged_values_by_unit=judged_values_selected,
        config=idw_config,
    )

    selected_only_rate = _safe_float(validation.judged_only_pass_rate)
    selected_only_abs_error = _safe_float(validation.judged_only_absolute_rate_error)
    idw_abs_error = float(validation.absolute_aggregate_rate_error)
    if selected_only_abs_error is None:
        raise ValueError("selected-only absolute error is unexpectedly missing")

    embedding_ledger = {
        "packet_builds": int(runtime.ledger.packet_builds),
        "packet_cache_hits": int(runtime.ledger.packet_cache_hits),
        "embedding_calls": int(runtime.ledger.embedding_calls),
        "embedding_inputs": int(runtime.ledger.embedding_inputs),
        "embedding_input_tokens": int(runtime.ledger.embedding_input_tokens),
        "embedding_latency_seconds": float(runtime.ledger.embedding_latency_seconds),
        "embedding_content_hash_count": len(runtime.ledger.embedding_content_hashes),
        "embedding_model_id": str(runtime.ledger.embedding_model_id),
        "embedding_deployment_id": str(runtime.ledger.embedding_deployment_id),
        "embedding_embedder_class": str(runtime.ledger.embedding_embedder_class),
    }

    return {
        "version": str(oracle_version),
        "created_at": str(created_at or _utc_now_iso()),
        "v3": {
            "manifest_hash": str(v3_manifest_hash),
            "report_hash": str(v3_report_hash),
            "selected_membership_provenance_tier_pct": 20,
            "selected_membership_method": "adaptive_embedding_fullsession_token",
        },
        "hashes": {
            "membership_hash": frozen_membership_hash,
            "population_hash": str(membership.population_hash),
        },
        "config": {
            "idw": asdict(idw_config),
        },
        "embedding_ledger": embedding_ledger,
        "counts": {
            "population_count": int(estimates.aggregate.population_count),
            "selected_count": int(estimates.aggregate.observed_count),
            "imputed_count": int(estimates.aggregate.imputed_count),
            "zero_donor_agent_count": int(estimates.aggregate.zero_donor_agent_count),
            "prior_count": int(estimates.aggregate.prior_count),
        },
        "rates": {
            "expected_census_rate": float(validation.census_pass_rate),
            "selected_only_rate": float(selected_only_rate),
            "selected_only_abs_error": float(selected_only_abs_error),
            "idw_estimated_rate": float(estimates.aggregate.estimated_pass_rate),
            "idw_abs_error": idw_abs_error,
            "delta_aggregate_mae_idw_minus_selected_only": float(idw_abs_error - selected_only_abs_error),
            "provenance_counts": {k: int(v) for k, v in estimates.aggregate.provenance_counts.items()},
            "provenance_population_weighted_rates": {
                k: float(v) for k, v in estimates.aggregate.provenance_population_weighted_rates.items()
            },
        },
        "metrics": {
            "per_unit_mae": float(validation.per_unit_mae),
            "brier_score": float(validation.brier_score),
            "expected_calibration_error": float(validation.expected_calibration_error),
            "macro_per_agent_mae": float(validation.macro_per_agent_mae),
            "unjudged_only_mae": _safe_float(validation.unjudged_only_mae),
            "unjudged_only_brier": _safe_float(validation.unjudged_only_brier),
            "leave_one_out": {
                "judged_count": len(judged_values_selected),
                "mae": float(loo.mae),
                "brier_score": float(loo.brier_score),
            },
        },
        "gate": {
            "idw_abs_error_le_selected_only_abs_error": bool(idw_abs_error <= selected_only_abs_error),
        },
        "notes": [
            "Expected labels are treated as deterministic pseudo-judge outputs for this oracle run.",
            "This oracle is an upper-bound validation check, not a production online judgment path.",
            "No Azure Search operations are used in this run.",
            "No raw vectors, packet text, labels, or per-unit donor rows are persisted.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run narrow V4 oracle check from retained V3 artifacts")
    parser.add_argument("--v3-dir", default=str(Path("outputs_sampling_v3") / "v3"))
    parser.add_argument("--output", default=str(Path("outputs_sampling_v4") / "runs" / "oracle-20.json"))
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--idw-eps", type=float, default=1e-6)
    parser.add_argument("--idw-exact-cosine-eps", type=float, default=1e-8)
    parser.add_argument("--idw-prior", type=float, default=0.5)
    parser.add_argument("--embedding-model", default=V3_EMBEDDING_MODEL)
    parser.add_argument("--embedding-deployment", default="")
    return parser


def _v3_report_hash_from_loaded_artifacts(artifacts: Any) -> str:
    payload = {
        "aggregate_version": artifacts.aggregate.get("version"),
        "runs_count": len(artifacts.runs_jsonl),
        "quadrant_version": artifacts.quadrant.get("version"),
        "throughput_version": artifacts.throughput.get("version"),
        "corpus_audit_version": artifacts.corpus_audit.get("version"),
        "token_inventory_rows": len(artifacts.token_inventory),
        "budget_manifest": artifacts.budget_manifest,
        "embedding_ledger": artifacts.embedding_ledger,
        "selected_membership": artifacts.selected_membership,
        "manifest": artifacts.manifest,
    }
    return _sha256_text(_canonical_json(payload))


def main() -> None:
    args = _build_parser().parse_args()
    v3_dir = Path(args.v3_dir)
    output_path = Path(args.output)

    inputs = default_inputs(v3_dir)
    artifacts = load_v3_artifacts(inputs)
    validate_v3_artifacts(artifacts)

    data = load_combined_dataset(enforce_integrity_counts=True)

    config = _embedding_only_config_from_env()
    deployment = str(args.embedding_deployment).strip() or str(config.embedding_deployment)

    tokenizer = TiktokenTokenizer(model_name=str(args.embedding_model), encoding_name="cl100k_base")
    embedder = AzureOpenAIEmbedder(config)
    runtime = build_v3_runtime(
        data,
        tokenizer=tokenizer,
        embedder=embedder,
        embedding_model_id=str(args.embedding_model),
        embedding_deployment_id=deployment,
        embedding_batch_size=int(args.embedding_batch_size),
        max_session_packet_tokens=V3_MAX_SESSION_PACKET_TOKENS,
    )

    manifest_path = v3_dir / "manifest.json"
    manifest_hash = _sha256_file(manifest_path)
    report_hash = _v3_report_hash_from_loaded_artifacts(artifacts)

    idw_config = IDWConfig(
        k=int(args.idw_k),
        power=float(args.idw_power),
        eps=float(args.idw_eps),
        exact_cosine_eps=float(args.idw_exact_cosine_eps),
        prior=float(args.idw_prior),
    )

    oracle = compute_v4_oracle(
        data=data,
        runtime=runtime,
        selected_membership_payload=artifacts.selected_membership,
        aggregate_population_count=int(artifacts.aggregate.get("population_count") or 0),
        v3_manifest_hash=manifest_hash,
        v3_report_hash=report_hash,
        idw_config=idw_config,
    )

    _write_json_atomic(output_path, oracle)
    output_sha = _sha256_file(output_path)

    print(
        json.dumps(
            {
                "version": oracle["version"],
                "output": str(output_path),
                "sha256": output_sha,
                "pass_gate": oracle["gate"]["idw_abs_error_le_selected_only_abs_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
