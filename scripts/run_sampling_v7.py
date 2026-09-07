from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v7_experiment import run_v7_experiment
from sampling_comparison.v3_experiment import V3_EMBEDDING_MODEL
from sampling_comparison.v6_runner import _build_runtime_with_cache
from trace_sampling.azure_config import AzureConfig
from trace_sampling.embedding import AzureOpenAIEmbedder
from trace_sampling.session_embedding import TiktokenTokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the transductive Sampling V7 artifact bundle")
    parser.add_argument("--output", default="outputs_sampling_v7/runs/v7-live", help="Output directory for the artifact bundle")
    parser.add_argument("--repetitions", type=int, default=10, help="Bootstrap replay repetition count")
    parser.add_argument("--base-seed", type=int, default=13, help="Base seed used to derive replay seeds")
    parser.add_argument("--seeds", default="", help="Optional CSV replay-seed override; one seed per repetition")
    parser.add_argument("--budgets", default="1,3,5,10,20", help="CSV of budget percentages")
    parser.add_argument("--partial-label", type=int, default=0, choices=(0, 1), help="Partial-label mapping override")
    parser.add_argument("--skip-search-sync", action="store_true", help="Only for tests/debug; production runs should require real Search sync")
    parser.add_argument("--dataset-historical", default=None, help="Optional historical_300 dataset path override")
    parser.add_argument("--dataset-dense", default=None, help="Optional dense_2500 dataset path override")
    parser.add_argument("--dataset-cosmos-root", default="", help="Optional cosmos_otel root override")
    parser.add_argument("--runtime-cache-root", default="outputs_sampling_v7/cache", help="Directory for v3 runtime embedding caches")
    parser.add_argument("--search-index-cosine", default="trace-clusters-sampling-v7-cosine")
    parser.add_argument("--search-index-euclidean", default="trace-clusters-sampling-v7-euclidean")
    return parser


def _parse_csv(value: str) -> tuple[int, ...]:
    if not value or not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _build_parser().parse_args()
    explicit_seeds = _parse_csv(args.seeds)
    if explicit_seeds and (int(args.repetitions) != 10 or int(args.base_seed) != 13):
        raise ValueError("--seeds cannot be combined with non-default --repetitions/--base-seed; use one seed contract")
    budgets = _parse_csv(args.budgets) or (1, 3, 5, 10, 20)
    from sampling_comparison.v7_dataset import load_v7_datasets, to_combined_dataset

    datasets = load_v7_datasets(
        historical_path=args.dataset_historical or "synthetic_data/a365_historical_300/synthetic_observability.a365-otel.json",
        dense_path=args.dataset_dense or "synthetic_data/a365_dense_2500/corpus/a365.synthetic.strict.otlp.json",
        cosmos_root=args.dataset_cosmos_root or r"C:\Users\stangoodwin\synth-data\data\cosmos_otel",
        partial_label=args.partial_label,
    )
    azure = AzureConfig.from_env()
    embedder = AzureOpenAIEmbedder(azure)
    tokenizer = TiktokenTokenizer(model_name=V3_EMBEDDING_MODEL, encoding_name="cl100k_base")

    payload = {}
    cache_root = Path(args.runtime_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    for dataset_id, dataset in datasets.items():
        combined = to_combined_dataset(dataset)
        source_hashes: dict[str, str] = {}
        for key, value in sorted(dataset.source_paths.items()):
            path = Path(value)
            if path.exists() and path.is_file():
                source_hashes[key] = _sha256_file(path)

        runtime, cache_meta = _build_runtime_with_cache(
            combined,
            tokenizer=tokenizer,
            embedder=embedder,
            embedding_model_id=V3_EMBEDDING_MODEL,
            embedding_deployment_id=azure.embedding_deployment,
            embedding_batch_size=32,
            embedding_dimensions=1536,
            max_session_packet_tokens=8191,
            cache_base_path=cache_root / dataset_id,
            source_hashes=source_hashes,
        )

        full_vectors_by_unit: dict[str, list[float]] = {}
        for uid in dataset.ordered_unit_ids:
            trace_id = int(dataset.traces_by_unit_id[uid].trace_id)
            if trace_id not in runtime.embedding_vector_by_trace_id:
                raise ValueError(f"missing runtime embedding vector for dataset={dataset_id} unit_id={uid}")
            full_vectors_by_unit[uid] = runtime.embedding_vector_by_trace_id[trace_id].tolist()
        for uid in dataset.ordered_unit_ids:
            if uid not in runtime.token_cost_by_unit_id:
                raise ValueError(f"missing runtime token count for dataset={dataset_id} unit_id={uid}")

        payload[dataset_id] = {
            "ordered_unit_ids": dataset.ordered_unit_ids,
            "labels_by_unit": dataset.labels_by_unit,
            "full_vectors_by_unit": full_vectors_by_unit,
            "traces_by_unit_id": dataset.traces_by_unit_id,
            "agent_id_by_unit": dataset.agent_id_by_unit,
            "concept_key_by_unit": dataset.concept_key_by_unit,
            "use_case_id_by_unit": dataset.use_case_id_by_unit,
            "business_use_case_guid_by_unit": dataset.business_use_case_guid_by_unit,
            "metadata_by_unit": dataset.metadata_by_unit,
            "representation_source_by_unit": dataset.representation_source_by_unit,
            "source_paths": dataset.source_paths,
            "token_count_by_unit": dict(runtime.token_cost_by_unit_id),
            "minhash_records_by_unit_id": dict(runtime.minhash_records_by_unit_id),
            "runtime_ledger": {
                "embedding_model_id": runtime.ledger.embedding_model_id,
                "embedding_deployment_id": runtime.ledger.embedding_deployment_id,
                "embedding_calls": runtime.ledger.embedding_calls,
                "embedding_inputs": runtime.ledger.embedding_inputs,
                "embedding_input_tokens": runtime.ledger.embedding_input_tokens,
                "embedding_latency_seconds": runtime.ledger.embedding_latency_seconds,
                "packet_cache_hits": runtime.ledger.packet_cache_hits,
                "cache_hit": bool(cache_meta.get("cache_hit") or False),
                "cache_rows": int(cache_meta.get("cache_rows") or 0),
                "cache_packet_hash_count": int(cache_meta.get("cache_packet_hash_count") or 0),
                "cache_provenance": dict(cache_meta.get("provenance") or {}),
            },
        }
    result = run_v7_experiment(
        datasets=payload,
        output_dir=args.output,
        seed_values=explicit_seeds if explicit_seeds else None,
        repetitions=int(args.repetitions),
        base_seed=int(args.base_seed),
        budget_levels=budgets,
        method_ids=None,
        partial_label=args.partial_label,
        azure_config=azure,
        skip_search_sync=args.skip_search_sync,
        cosine_index_name=args.search_index_cosine,
        euclidean_index_name=args.search_index_euclidean,
    )
    print(f"v7 bundle complete: {result['output_dir']}")
    print(f"runs: {result['summary']['run_count']}")
    print(f"report: {result['report_path']}")


if __name__ == "__main__":
    main()
