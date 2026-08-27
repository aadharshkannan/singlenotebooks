from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v4_idw import IDWConfig  # noqa: E402
from sampling_comparison.v6_experiment import NOMINAL_TOKENS_PER_SESSION, SAMPLE_CAPS, TRIAL_SEEDS  # noqa: E402
from sampling_comparison.v6_runner import (  # noqa: E402
    DEFAULT_MAVEN_CENTROIDS_DB,
    DEFAULT_MAVEN_TAXONOMY_DB,
    default_output_dir,
    run_sampling_v6_bundle,
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _format_progress_bar(event: dict[str, object]) -> str:
    phase = str(event.get("phase") or "progress")
    percent = float(event.get("percent") or 0.0)
    total_replays = int(event.get("total_replays") or 1)
    replay_session_current = int(event.get("replay_session_current") or 0)
    replay_session_total = int(event.get("replay_session_total") or 1)
    current_seed = event.get("current_seed")
    current_cap = event.get("current_cap")
    current_method = event.get("current_method")
    current_replay = event.get("current_replay")

    width = 28
    pct = max(0.0, min(100.0, percent))
    if pct <= 0.0:
        bar_inner = "." * width
    elif pct >= 100.0:
        bar_inner = "=" * width
    else:
        filled = int(round((pct / 100.0) * width))
        filled = max(1, min(width - 1, filled))
        bar_inner = "=" * (filled - 1) + ">" + "." * (width - filled)
    bar = f"[{bar_inner}]"

    src = f"{pct:5.1f}%"
    replay_text = f"replay {current_replay}/{total_replays}" if current_replay is not None else f"replay {event.get('completed_replays', 0)}/{total_replays}"
    seed_cap = f"seed {current_seed} cap {current_cap}" if current_seed is not None and current_cap is not None else "seed ? cap ?"
    method_label = str(current_method or "unknown")
    if phase.lower().startswith("replay") or phase.lower() == "search-replay":
        search_text = "Search current/total" if replay_session_total <= 1 else f"Search {replay_session_current}/{replay_session_total}"
    elif phase.lower() == "method-evaluation":
        search_text = "done"
    else:
        search_text = f"phase {phase}"
    suffix = f" | {replay_text} | {seed_cap} | {method_label} | {search_text}"
    return f"{bar} | {src}{suffix}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sampling V6 live runner and artifact bundle")
    parser.add_argument("--output", default=str(default_output_dir()))
    parser.add_argument("--caps", default=",".join(str(x) for x in SAMPLE_CAPS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in TRIAL_SEEDS))
    parser.add_argument("--avg-tokens", type=int, default=NOMINAL_TOKENS_PER_SESSION)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--cleanup-max-attempts", type=int, default=10)
    parser.add_argument("--cleanup-settle-seconds", type=float, default=0.0)
    parser.add_argument("--ensure-search-index", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Suppress the live ASCII progress bar while still writing progress.json")

    parser.add_argument("--centroids-db", default=DEFAULT_MAVEN_CENTROIDS_DB)
    parser.add_argument("--taxonomy-db", default=DEFAULT_MAVEN_TAXONOMY_DB)

    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--idw-eps", type=float, default=1e-6)
    parser.add_argument("--idw-exact-cosine-eps", type=float, default=1e-8)
    parser.add_argument("--idw-prior", type=float, default=0.5)

    parser.add_argument("--classifications-cache", default=None)
    parser.add_argument("--embeddings-cache", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.output)
    idw = IDWConfig(
        k=int(args.idw_k),
        power=float(args.idw_power),
        eps=float(args.idw_eps),
        exact_cosine_eps=float(args.idw_exact_cosine_eps),
        prior=float(args.idw_prior),
    )
    last_phase = None
    last_rendered = ""

    def _on_progress(event: dict[str, object]) -> None:
        nonlocal last_phase, last_rendered
        if args.no_progress:
            return
        phase = str(event.get("phase") or "progress")
        status = str(event.get("status") or "running")
        if phase != last_phase and phase not in {"preprocessing", "complete"}:
            print(f"\n[{phase}] {event.get('message', '')}", flush=True)
            last_phase = phase
        line = _format_progress_bar(event)
        if line != last_rendered:
            print(f"\r{line:<120}", end="", flush=True)
            last_rendered = line
        if status in {"complete", "failed"}:
            print("\n", end="", flush=True)
            last_phase = None
            last_rendered = ""

    try:
        result = run_sampling_v6_bundle(
            output_dir=out_dir,
            caps=_parse_int_csv(args.caps),
            seeds=_parse_int_csv(args.seeds),
            avg_tokens_per_session=int(args.avg_tokens),
            embedding_batch_size=int(args.embedding_batch_size),
            cleanup_max_attempts=int(args.cleanup_max_attempts),
            cleanup_settle_seconds=float(args.cleanup_settle_seconds),
            ensure_search_index=bool(args.ensure_search_index),
            idw_config=idw,
            centroids_db_path=str(args.centroids_db),
            taxonomy_db_path=str(args.taxonomy_db),
            embeddings_cache_path=args.embeddings_cache,
            classifications_cache_path=args.classifications_cache,
            baseline_dir=args.baseline_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume=not bool(args.no_resume),
            skip_report=bool(args.skip_report),
            progress_callback=_on_progress,
        )
    except Exception:
        if not args.no_progress:
            print("\n[failed] Sampling V6 bundle failed", flush=True)
        raise

    if not args.no_progress:
        print("", flush=True)
    print("sampling-v6 bundle complete")
    for key, path in sorted(result["output_paths"].items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
