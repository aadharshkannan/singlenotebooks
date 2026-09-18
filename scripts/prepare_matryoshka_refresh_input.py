"""Explicitly authorized embedding-only Cosmos refresh, aggregate output only."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import load_input, sha256_file
from sampling_comparison.matryoshka_refresh_inputs import (
    ENDPOINT, MODEL, MeteredAzureEmbedder, RefreshError, prepare_refresh_input, require_target, safe_summary,
)
from scripts.prepare_matryoshka_live_input import read_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh exact cached Cosmos inputs; offline unless --live.")
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, help="Read only this explicit dotenv file; required with --live.")
    parser.add_argument("--live", action="store_true", help="Authorize only missing native embedding inputs to the pinned Azure endpoint.")
    args = parser.parse_args(argv)
    if args.live and (args.env_file is None or not args.env_file.is_file()):
        parser.error("--live requires an existing explicit --env-file")
    logging.disable(logging.CRITICAL)  # SDK diagnostics must not echo credentials or payloads.
    embedder = None
    try:
        repo = Path(__file__).resolve().parents[1]
        output = args.output.resolve()
        if not output.is_relative_to(repo / "outputs_matryoshka" / "cache"):
            raise RefreshError("Output must be inside this worktree's ignored Matryoshka cache.")
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", str(output / "units.json")], cwd=repo, capture_output=True,
        )
        if ignored.returncode != 0:
            raise RefreshError("Sensitive input cache output is not Git ignored.")
        if args.live:
            # No ambient environment/provider fallback and no dotenv discovery.
            settings = read_settings(args.env_file, {})
            require_target(settings["AZURE_OPENAI_ENDPOINT"], settings["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
            key = settings["AZURE_OPENAI_API_KEY"]
            if not key or key.strip() in ("<key>", "your-api-key-here"):
                raise RefreshError("The explicit dotenv file lacks a usable API key.")
            embedder = MeteredAzureEmbedder(
                endpoint=ENDPOINT, deployment=MODEL, api_key=key,
                api_version=settings["AZURE_OPENAI_API_VERSION"] or "2024-02-01",
            )
        result = prepare_refresh_input(
            args.cosmos_root, output, preflight_path=args.preflight, reuse_manifest=args.reuse_manifest,
            embedder=embedder, progress=lambda message: print(message, flush=True),
        )
        summary = safe_summary(result)
        if result["status"] == "complete":
            data = load_input(output / "manifest.json")
            summary["verified_vector_shape"] = list(data.vectors.shape)
            summary["input_manifest"] = str((output / "manifest.json").relative_to(repo))
            summary["artifact_sha256"] = {
                name: sha256_file(output / name) for name in ("manifest.json", "units.json", "vectors.npz", "preparation.json")
            }
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except RefreshError as exc:
        print(f"Refresh stopped: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        print("Refresh stopped: input, cache, configuration or API validation failed; details suppressed for privacy.",
              file=sys.stderr, flush=True)
        return 1
    finally:
        if embedder is not None:
            embedder.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
