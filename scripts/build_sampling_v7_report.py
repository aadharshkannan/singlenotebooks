from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v7_report_rich import (
    build_v7_final_report,
    build_v7_print_report_html,
    build_v7_report_html,
)


def _assert_expected_run_shape(payload: dict) -> None:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("aggregate payload must include a 'runs' list")

    seen_keys: set[tuple[str, str, int, int]] = set()
    duplicate_keys: list[tuple[str, str, int, int]] = []
    for row in runs:
        if not isinstance(row, dict):
            continue
        dataset_id = row.get("dataset_id")
        method_id = row.get("method_id")
        budget_pct = row.get("budget_pct")
        repetition_index = row.get("repetition_index")
        if dataset_id is None or method_id is None or budget_pct is None or repetition_index is None:
            raise ValueError("each run row must include dataset_id, method_id, budget_pct, and repetition_index")
        key = (str(dataset_id), str(method_id), int(budget_pct), int(repetition_index))
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise ValueError(f"duplicate Cartesian run keys detected: {duplicate_keys[:10]}")

    dataset_ids = {str(row.get("dataset_id")) for row in runs if isinstance(row, dict) and row.get("dataset_id") is not None}
    methods = {str(row.get("method_id")) for row in runs if isinstance(row, dict) and row.get("method_id") is not None}
    budgets = {int(row.get("budget_pct")) for row in runs if isinstance(row, dict) and row.get("budget_pct") is not None}
    repetitions = {
        int(row.get("repetition_index"))
        for row in runs
        if isinstance(row, dict) and row.get("repetition_index") is not None
    }
    expected_keyspace = {
        (dataset_id, method_id, budget_pct, repetition_index)
        for dataset_id in dataset_ids
        for method_id in methods
        for budget_pct in budgets
        for repetition_index in repetitions
    }
    observed_shape = len(runs)
    unique_shape = len(seen_keys)
    if unique_shape != len(expected_keyspace):
        missing = sorted(expected_keyspace - seen_keys)[:10]
        raise ValueError(
            "inconsistent run shape: expected unique Cartesian product "
            f"{len(expected_keyspace)} across dataset_id x method_id x budget_pct x repetition_index, "
            f"observed unique rows {unique_shape} and total rows {observed_shape}; "
            f"missing keys: {missing}"
        )
    summary = payload.get("summary")
    if isinstance(summary, dict):
        expected_default_shape = summary.get("expected_default_shape")
        if expected_default_shape is not None and int(expected_default_shape) != observed_shape:
            raise ValueError(
                f"summary expected_default_shape={int(expected_default_shape)} does not match observed run rows={observed_shape}"
            )
        configured_shape = summary.get("configured_shape")
        if configured_shape is not None and int(configured_shape) != observed_shape:
            raise ValueError(
                f"summary configured_shape={int(configured_shape)} does not match observed run rows={observed_shape}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _update_manifest(
    manifest_path: Path,
    *,
    interactive_path: Path,
    final_json_path: Path,
    print_path: Path,
) -> None:
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    hashes = manifest.get("hashes") if isinstance(manifest.get("hashes"), dict) else {}

    files["interactive_report"] = str(interactive_path)
    files["final_report"] = str(final_json_path)
    files["print_report"] = str(print_path)
    hashes["interactive_report"] = _sha256_file(interactive_path)
    hashes["final_report"] = _sha256_file(final_json_path)
    hashes["print_report"] = _sha256_file(print_path)

    manifest["files"] = files
    manifest["hashes"] = hashes
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the Sampling V7 interactive HTML report")
    parser.add_argument("--input", default="outputs_sampling_v7/runs/latest/aggregate.json", help="Artifact aggregate path")
    parser.add_argument("--output", default=None, help="Output HTML path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    aggregate_path = Path(args.input)
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    _assert_expected_run_shape(payload)
    runs = payload.get("runs")
    summary = payload.get("summary")
    datasets = payload.get("datasets")
    if not isinstance(runs, list):
        raise ValueError("aggregate payload must include a 'runs' list")
    if not isinstance(summary, dict):
        raise ValueError("aggregate payload must include a 'summary' object")
    report_payload = {
        "runs": runs,
        "summary": summary,
        "datasets": datasets or [],
        "aggregate_metrics": payload.get("aggregate_metrics") or {},
        "paired_differences": payload.get("paired_differences") or {},
    }

    html = build_v7_report_html(report_payload)
    out_path = Path(args.output) if args.output else aggregate_path.with_name("interactive_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    manifest_path = aggregate_path.with_name("manifest.json")
    source_artifacts: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_artifacts["manifest"] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
        for key, value in (manifest.get("files") or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                source_artifacts[key] = _artifact_record(path)

    final_report = build_v7_final_report(
        report_payload,
        artifacts={"source": source_artifacts},
    )
    final_json_path = out_path.with_name("final_report.json")
    _write_json(final_json_path, final_report)

    print_html = build_v7_print_report_html(report_payload, final_report)
    print_path = out_path.with_name("print_report.html")
    print_path.write_text(print_html, encoding="utf-8")

    provenance = final_report.setdefault("provenance", {})
    artifacts = provenance.setdefault("artifacts", {})
    artifacts["generated"] = {
        "interactive_report": _artifact_record(out_path),
        "print_report": _artifact_record(print_path),
    }
    _write_json(final_json_path, final_report)

    _update_manifest(
        manifest_path,
        interactive_path=out_path,
        final_json_path=final_json_path,
        print_path=print_path,
    )

    print(out_path)


if __name__ == "__main__":
    main()
