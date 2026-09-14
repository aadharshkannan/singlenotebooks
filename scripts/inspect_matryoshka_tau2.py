from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import ijson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


def inspect(source: Path) -> dict:
    with source.open("rb") as stream:
        tasks = list(ijson.items(stream, "tasks.item"))
    task_ids = {task["id"] for task in tasks}
    ids = set()
    per_task: Counter = Counter()
    rewards: Counter = Counter()
    reward_bases: Counter = Counter()
    expected_fields: Counter = Counter()
    total = 0
    with source.open("rb") as stream:
        for simulation in ijson.items(stream, "simulations.item"):
            total += 1
            if simulation["id"] in ids:
                raise ValueError("duplicate Tau2 simulation ID")
            ids.add(simulation["id"])
            per_task[simulation["task_id"]] += 1
            reward = simulation.get("reward_info") or {}
            rewards[str(reward.get("reward"))] += 1
            reward_bases["+".join(sorted(reward.get("reward_basis") or []))] += 1
            for name in ("expected_label", "expected_outcome", "expected_task_completion"):
                if name in simulation:
                    expected_fields[name] += 1
    if set(per_task) - task_ids:
        raise ValueError("simulated task IDs do not match embedded task snapshot")
    return {
        "version": "matryoshka-tau2-readiness-v1", "dataset_id": "tau2_bench",
        "status": "blocked_expected_labels", "sessions": total,
        "agents": 1, "agent_scope": "One model/configuration in the retained banking_knowledge run; not one agent per task.",
        "positive_count": None, "task_count": len(tasks), "simulated_tasks": len(per_task),
        "simulations_per_task_histogram": dict(Counter(per_task.values())),
        "source_paths": {"simulations": str(source)},
        "source_hashes": {"simulations": sha256_file(source)},
        "observed_reward_counts_NOT_expected_labels": dict(rewards),
        "recorded_reward_basis_counts": dict(reward_bases),
        "recognized_explicit_expected_label_fields": dict(expected_fields),
        "tasks_with_nl_assertion_basis": [
            task["id"] for task in tasks
            if "NL_ASSERTION" in ((task.get("evaluation_criteria") or {}).get("reward_basis") or [])
        ],
        "label_source": "No verified explicit expected task-completion label mapping. Recorded reward_info is observed benchmark evaluation, not a task-design expected label.",
        "reason": "Expected-label source/mapping required; do not substitute recorded rewards (some criteria include legacy LLM NL_ASSERTION checks) or invoke a judge. API embeddings also require configured credentials for the approved endpoint.",
        "live_embedding_calls": 0, "live_judge_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Tau2 metadata without embedding, judging, or treating rewards as expected labels.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect(Path(args.source))
    write_json(Path(args.output), result)
    print(f"{result['sessions']} simulations; expected-label mapping unavailable; no API calls.")


if __name__ == "__main__":
    main()
