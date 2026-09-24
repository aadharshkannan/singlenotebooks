from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_inputs import prepare_input
from sampling_comparison.matryoshka_inputs import AzureBatchEmbedder
from scripts.prepare_matryoshka_live_input import read_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official IMDb corpus; --live enables paid embedding calls, never an LLM judge.")
    parser.add_argument("--archive", type=Path, default=Path("external_data/imdb/aclImdb_v1.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs_imdb/cache/imdb-50000"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    embedder, identity = None, None
    if args.live:
        settings = read_settings(args.env_file, os.environ)
        endpoint = settings["AZURE_OPENAI_ENDPOINT"]
        if not endpoint:
            parser.error("AZURE_OPENAI_ENDPOINT is missing from the explicit .env file/environment; no API call was made")
        deployment = settings["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] or "text-embedding-3-small"
        api_version = settings["AZURE_OPENAI_API_VERSION"] or "2024-02-01"
        embedder = AzureBatchEmbedder(
            endpoint=endpoint, deployment=deployment, api_version=api_version,
            api_key=settings["AZURE_OPENAI_API_KEY"] or None,
            tenant_id=settings["AZURE_TENANT_ID"] or None,
        )
        identity = f"{endpoint.rstrip('/')}|{deployment}|{api_version}"
    result = prepare_input(
        args.archive, args.output, embedder=embedder, provider_identity=identity,
        batch_size=args.batch_size,
    )
    print(f"{result['status']}: {result['sessions']} reviews; {result['unique_texts']} unique embedding inputs")


if __name__ == "__main__":
    main()
