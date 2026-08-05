from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v2_experiment import load_combined_dataset, slice_dataset  # noqa: E402
from sampling_comparison.v3_experiment import (  # noqa: E402
    V3_EMBEDDING_DIMENSIONS,
    V3_EMBEDDING_MODEL,
    V3_MAX_SESSION_PACKET_TOKENS,
    build_v3_runtime,
)
from sampling_comparison.v3_outputs import run_v3_experiment_bundle  # noqa: E402
from sampling_comparison.v3_outputs import write_run_source_manifest  # noqa: E402
from sampling_comparison.v3_outputs import write_search_cleanup_audit  # noqa: E402
from sampling_comparison.v3_outputs import _code_hashes  # noqa: E402
from sampling_comparison.v3_report import default_inputs, write_v3_html_report  # noqa: E402
from trace_sampling.azure_config import AzureConfig  # noqa: E402
from trace_sampling.embedding import AzureOpenAIEmbedder  # noqa: E402
from trace_sampling.session_embedding import TiktokenTokenizer  # noqa: E402
from trace_sampling.vector_store import AzureSearchVectorStore  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_float_csv(value: str) -> tuple[float, ...]:
    vals = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(float(v) for v in vals)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    vals = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(v) for v in vals)


def _current_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _remaining_scope_count(*, store: AzureSearchVectorStore, tenant_id: str) -> int:
    flt = store._build_filter(tenant_id=tenant_id)
    ids = store._search_ids(filter_expr=flt, page_size=1000, max_scan=20000)
    return len(ids)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live sampling v3 experiment bundle")
    parser.add_argument(
        "--output",
        default=str(Path("outputs_sampling_v3") / "runs" / _utc_stamp()),
        help="Output directory for V3 artifacts only",
    )
    parser.add_argument("--canary-limit", type=int, default=0, help="Optional dataset slice size for canary run")
    parser.add_argument("--outcome-repetitions", type=int, default=3)
    parser.add_argument("--quadrant-replays", type=int, default=3)
    parser.add_argument("--throughput-replays", type=int, default=2)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--cleanup-max-attempts", type=int, default=10)
    parser.add_argument("--cleanup-settle-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-quadrant", action="store_true")
    parser.add_argument("--outcome-tiers", default="5,10,20,30,50")
    parser.add_argument("--quadrant-tiers", default="15,30")
    parser.add_argument("--arrival-rates", default="0.25,1,4,16")
    parser.add_argument("--capacity-rates", default="0.25,1,4,16")
    parser.add_argument("--skip-report", action="store_true", help="Skip HTML report generation")
    return parser


def _get_attr(obj: object, *names: str):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _validate_non_secret_schema(config: AzureConfig, *, index_client=None) -> dict[str, object]:
    if not config.search_endpoint.startswith("https://"):
        raise ValueError("AZURE_SEARCH_ENDPOINT must be an https endpoint")
    if not config.search_index.strip():
        raise ValueError("AZURE_SEARCH_INDEX is required")
    if not config.embedding_deployment.strip():
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required")

    if index_client is None:
        from azure.search.documents.indexes import SearchIndexClient
        from trace_sampling.azure_config import get_credential

        index_client = SearchIndexClient(config.search_endpoint, get_credential())

    index = index_client.get_index(config.search_index)
    fields = list(getattr(index, "fields", []) or [])
    field_map = {str(_get_attr(f, "name") or ""): f for f in fields}

    if "cluster_id" not in field_map:
        raise ValueError("search index schema invalid: missing key field cluster_id")
    cluster_id = field_map["cluster_id"]
    if not bool(_get_attr(cluster_id, "key", "is_key")):
        raise ValueError("search index schema invalid: cluster_id must be key=true")

    required_filterable = ("tenant_id", "agent_id", "semantic_scope", "run_scope")
    for fname in required_filterable:
        field = field_map.get(fname)
        if field is None:
            raise ValueError(f"search index schema invalid: missing field {fname}")
        if not bool(_get_attr(field, "filterable", "is_filterable")):
            raise ValueError(f"search index schema invalid: {fname} must be filterable")

    last_seen = field_map.get("last_seen")
    if last_seen is None:
        raise ValueError("search index schema invalid: missing field last_seen")
    if not bool(_get_attr(last_seen, "filterable", "is_filterable")):
        raise ValueError("search index schema invalid: last_seen must be filterable")
    if not bool(_get_attr(last_seen, "sortable", "is_sortable")):
        raise ValueError("search index schema invalid: last_seen must be sortable")

    vector = field_map.get("vector")
    if vector is None:
        raise ValueError("search index schema invalid: missing vector field")
    if not bool(_get_attr(vector, "searchable", "is_searchable")):
        raise ValueError("search index schema invalid: vector must be searchable")
    if bool(_get_attr(vector, "retrievable", "is_retrievable")):
        raise ValueError("search index schema invalid: vector must be non-retrievable")

    dims = _get_attr(vector, "vector_search_dimensions", "dimensions")
    if int(dims or 0) != V3_EMBEDDING_DIMENSIONS:
        raise ValueError(
            "search index schema invalid: vector dimensions must be "
            f"{V3_EMBEDDING_DIMENSIONS}"
        )

    vector_profile_name = str(_get_attr(vector, "vector_search_profile_name") or "")
    if vector_profile_name != "hnsw-cosine":
        raise ValueError("search index schema invalid: vector profile must be hnsw-cosine")

    vector_search = getattr(index, "vector_search", None)
    profiles = list(getattr(vector_search, "profiles", []) or [])
    if not profiles:
        raise ValueError("search index schema invalid: vector_search.profiles missing")

    profile = None
    for p in profiles:
        if str(_get_attr(p, "name") or "") == "hnsw-cosine":
            profile = p
            break
    if profile is None:
        raise ValueError("search index schema invalid: profile hnsw-cosine missing")

    algorithm_ref = str(_get_attr(profile, "algorithm_configuration_name") or "")
    algorithms = list(getattr(vector_search, "algorithms", []) or [])
    algorithm = None
    for alg in algorithms:
        if str(_get_attr(alg, "name") or "") == algorithm_ref:
            algorithm = alg
            break
    if algorithm is None:
        raise ValueError("search index schema invalid: profile algorithm reference missing")

    kind = str(_get_attr(algorithm, "kind", "kind_name", "type") or "").lower()
    if "hnsw" not in kind:
        raise ValueError("search index schema invalid: profile algorithm must be HNSW")

    metric_obj = _get_attr(algorithm, "parameters")
    metric = str(_get_attr(metric_obj, "metric") or _get_attr(algorithm, "metric") or "").lower()
    if metric and "cosine" not in metric:
        raise ValueError("search index schema invalid: HNSW metric must be cosine")

    store = AzureSearchVectorStore(config, dim=V3_EMBEDDING_DIMENSIONS, ensure_index=False)
    probe = store._build_filter(
        agent_id="schema-agent",
        semantic_scope="schema-scope",
        tenant_id="schema-tenant",
        run_scope="schema-run",
    )
    if probe is None:
        raise ValueError("failed to build scoped filter for required search fields")

    return {
        "index": config.search_index,
        "endpoint": config.search_endpoint,
        "key_field": "cluster_id",
        "required_filter_fields": list(required_filterable),
        "last_seen_filterable_sortable": True,
        "vector_field": {
            "name": "vector",
            "dimensions": int(dims),
            "profile": vector_profile_name,
            "searchable": True,
            "retrievable": False,
        },
        "algorithm": {
            "name": str(_get_attr(algorithm, "name") or ""),
            "kind": kind,
            "metric": metric or "unspecified",
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output)
    if "outputs_sampling_v2" in out_dir.as_posix():
        raise ValueError("V3 run output must not target V2 output paths")

    config = AzureConfig.from_env()
    schema_info = _validate_non_secret_schema(config)

    data = load_combined_dataset(enforce_integrity_counts=False)
    if int(args.canary_limit) > 0:
        data = slice_dataset(data, limit=int(args.canary_limit))

    tokenizer = TiktokenTokenizer(model_name=V3_EMBEDDING_MODEL, encoding_name="cl100k_base")
    embedder = AzureOpenAIEmbedder(config)

    runtime = build_v3_runtime(
        data,
        tokenizer=tokenizer,
        embedder=embedder,
        embedding_model_id=V3_EMBEDDING_MODEL,
        embedding_deployment_id=config.embedding_deployment,
        embedding_batch_size=int(args.embedding_batch_size),
        max_session_packet_tokens=V3_MAX_SESSION_PACKET_TOKENS,
    )

    tenant_scope = "sampling-v3-experiment"

    result = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=out_dir,
        vector_store_factory=lambda _tenant, _scope: AzureSearchVectorStore(
            config,
            dim=V3_EMBEDDING_DIMENSIONS,
            ensure_index=False,
        ),
        outcome_repetitions=int(args.outcome_repetitions),
        quadrant_replays=int(args.quadrant_replays),
        throughput_replays=int(args.throughput_replays),
        legacy_outcome_tiers_pct=_parse_int_csv(args.outcome_tiers),
        legacy_quadrant_tiers_pct=_parse_int_csv(args.quadrant_tiers),
        throughput_arrival_rates_sessions_per_second=_parse_float_csv(args.arrival_rates),
        throughput_capacity_rates_sessions_per_second=_parse_float_csv(args.capacity_rates),
        seed=int(args.seed),
        tenant_id=tenant_scope,
        cleanup_max_attempts=int(args.cleanup_max_attempts),
        cleanup_settle_seconds=float(args.cleanup_settle_seconds),
        skip_quadrant=bool(args.skip_quadrant),
        skip_throughput=bool(args.skip_throughput),
        aggregate_config={
            "embedding": {
                "model": V3_EMBEDDING_MODEL,
                "deployment": config.embedding_deployment,
                "dimensions": V3_EMBEDDING_DIMENSIONS,
            },
            "tokenizer": {
                "encoding_name": "cl100k_base",
                "encoding_id": tokenizer.encoding_id,
                "version": tokenizer.version,
                "max_packet_tokens": V3_MAX_SESSION_PACKET_TOKENS,
            },
            "search": {
                "endpoint": config.search_endpoint,
                "index": config.search_index,
                "ensure_index": False,
                "schema_validation": schema_info,
            },
            "tau": 0.55,
            "tenant_scope": tenant_scope,
            "cleanup": {
                "max_attempts": int(args.cleanup_max_attempts),
                "settle_seconds": float(args.cleanup_settle_seconds),
                "quiet_scans_required": 2,
            },
        },
    )

    total_emitted_tokens = sum(int(row["emitted_tokens"]) for row in result["token_inventory"])
    estimated_embedding_calls = int(runtime.ledger.embedding_calls)
    summary = {
        "version": result["aggregate"]["version"],
        "population_count": result["aggregate"]["population_count"],
        "runtime_seconds": result["aggregate"]["runtime_seconds"],
        "output_paths": result["output_paths"],
        "schema_validation": schema_info,
        "cost_warning_estimate": {
            "note": "Estimate from token inventory only; no pricing claim.",
            "total_emitted_tokens": int(total_emitted_tokens),
            "embedding_calls": estimated_embedding_calls,
            "unique_embedding_inputs": int(runtime.ledger.embedding_inputs),
        },
    }

    audit_store = AzureSearchVectorStore(
        config,
        dim=V3_EMBEDDING_DIMENSIONS,
        ensure_index=False,
    )
    remaining_count = _remaining_scope_count(
        store=audit_store,
        tenant_id=tenant_scope,
    )
    cleanup_audit = write_search_cleanup_audit(
        output_dir=out_dir,
        tenant_id=tenant_scope,
        checked_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        remaining_count=remaining_count,
        scopes={
            "tenant_id": tenant_scope,
        },
        manifest_path=Path(result["output_paths"]["manifest"]),
        allow_nonzero=False,
    )
    run_source = write_run_source_manifest(
        output_dir=out_dir,
        pre_run_source_hashes=_code_hashes(),
        branch=_current_branch(),
        captured_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        note="Post-run exact capture for retained-bundle reproducibility.",
        manifest_path=Path(result["output_paths"]["manifest"]),
    )
    summary["search_cleanup_audit"] = cleanup_audit
    summary["run_source_manifest"] = run_source

    if not bool(args.skip_report):

        report_path = out_dir / "agent365-sampling-v3-report.html"
        written = write_v3_html_report(
            output_path=report_path,
            inputs=default_inputs(out_dir),
        )
        summary["report_html"] = str(written)
        summary["report_manifest"] = str(written.with_name("report_manifest.json"))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
