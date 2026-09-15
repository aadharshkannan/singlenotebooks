from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_inputs import AzureBatchEmbedder, load_tau2_expected_dataset, prepare_live_input
from sampling_comparison.v2_experiment import load_combined_dataset
from sampling_comparison.v7_dataset import _subset_combined_by_corpus, build_v7_dataset_from_combined, load_cosmos_otel_expected_dataset


def read_settings(env_file: Path, environ: Mapping[str, str]) -> dict[str, str]:
    # Parse only the explicitly named file; never walk to a different checkout's .env.
    values = dotenv_values(env_file) if env_file.is_file() else {}
    names = (
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_TENANT_ID",
    )
    return {name: environ.get(name) or values.get(name) or "" for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage expected labels and session packets; --live explicitly enables embedding APIs only.")
    parser.add_argument("--dataset", required=True, choices=("historical_300", "dense_2500", "cosmos_otel", "tau2_bench"))
    parser.add_argument("--cosmos-root")
    parser.add_argument("--tau2-source", help="Recorded Tau2 simulations JSON, with embedded tasks and messages.")
    parser.add_argument("--tau2-labels", help="Required explicit expected-label JSON mapping raw simulation IDs to binary 0/1; NOT reward_info.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default=".env", help="Explicit dotenv file; ignored by Git and never printed or copied.")
    parser.add_argument("--endpoint", help="Overrides AZURE_OPENAI_ENDPOINT from the environment or dotenv file.")
    parser.add_argument("--tenant-id", help="Optional explicit Azure CLI tenant for keyless authentication.")
    parser.add_argument("--deployment", help="Overrides AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")
    parser.add_argument("--api-version", help="Classic Azure endpoint API version; modern Foundry uses /openai/v1/.")
    parser.add_argument("--live", action="store_true", help="Make paid embedding calls for uncached batches; NEVER calls a judge.")
    args = parser.parse_args()
    settings = read_settings(Path(args.env_file), os.environ)
    endpoint = args.endpoint or settings["AZURE_OPENAI_ENDPOINT"]
    if not endpoint:
        parser.error("provide --endpoint or AZURE_OPENAI_ENDPOINT in the explicit dotenv file/environment")
    deployment = args.deployment or settings["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] or "text-embedding-3-small"
    tenant_id = args.tenant_id or settings["AZURE_TENANT_ID"] or None
    api_key = settings["AZURE_OPENAI_API_KEY"] or None
    if api_key and api_key.strip() in ("<key>", "your-api-key-here"):
        parser.error("dotenv contains a placeholder rather than a usable API key")
    if args.dataset == "tau2_bench":
        if not args.tau2_source or not args.tau2_labels:
            parser.error("Tau2 requires --tau2-source AND --tau2-labels; recorded rewards are not substituted")
        data = load_tau2_expected_dataset(Path(args.tau2_source), Path(args.tau2_labels))
        label_source = "User-supplied explicit binary expected task-completion labels keyed by simulation ID. Recorded rewards and evaluators are not used."
    elif args.dataset == "cosmos_otel":
        if not args.cosmos_root:
            parser.error("--cosmos-root is required for cosmos_otel")
        data = load_cosmos_otel_expected_dataset(args.cosmos_root, partial_label=0)
        label_source = "Existing task-design expected_outcome: good=1, bad=0, partial=0. Not fresh judge results or verified actual completion."
    else:
        data = build_v7_dataset_from_combined(
            _subset_combined_by_corpus(load_combined_dataset(), corpus_id=args.dataset), dataset_id=args.dataset,
        )
        label_source = "Retained synthetic expected binary task-completion labels; no fresh LLM judgments."
    embedder = AzureBatchEmbedder(
        endpoint=endpoint, deployment=deployment, tenant_id=tenant_id, api_key=api_key,
        api_version=args.api_version or settings["AZURE_OPENAI_API_VERSION"] or "2024-02-01",
    ) if args.live else None
    result = prepare_live_input(
        data, Path(args.output), embedder=embedder, endpoint=endpoint,
        deployment=deployment, tenant_id=tenant_id, label_source=label_source,
    )
    print(result["status"])


if __name__ == "__main__":
    main()
