from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.run_sampling_v3 import _build_parser, _validate_non_secret_schema


class _Field:
    def __init__(self, name: str, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


class _IndexClient:
    def __init__(self, index_obj):
        self._index_obj = index_obj

    def get_index(self, _name: str):
        return self._index_obj


def _fake_config():
    return SimpleNamespace(
        search_endpoint="https://example.search.windows.net",
        search_index="maven-session-sampling-v3",
        embedding_deployment="embed-dep",
    )


def _valid_index():
    fields = [
        _Field("cluster_id", key=True),
        _Field("tenant_id", filterable=True),
        _Field("agent_id", filterable=True),
        _Field("semantic_scope", filterable=True),
        _Field("run_scope", filterable=True),
        _Field("last_seen", filterable=True, sortable=True),
        _Field(
            "vector",
            searchable=True,
            retrievable=False,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-cosine",
        ),
    ]
    algorithms = [
        SimpleNamespace(
            name="hnsw",
            kind="hnsw",
            parameters=SimpleNamespace(metric="cosine"),
        )
    ]
    profiles = [SimpleNamespace(name="hnsw-cosine", algorithm_configuration_name="hnsw")]
    return SimpleNamespace(fields=fields, vector_search=SimpleNamespace(profiles=profiles, algorithms=algorithms))


def test_cli_defaults_to_v3_output_path_and_never_v2_default():
    parser = _build_parser()
    args = parser.parse_args([])
    out = Path(args.output).as_posix()
    assert "outputs_sampling_v3" in out
    assert "outputs_sampling_v2" not in out


def test_cli_parser_flags_for_replays_cleanup_and_skips_exist():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--canary-limit",
            "10",
            "--outcome-repetitions",
            "2",
            "--quadrant-replays",
            "1",
            "--throughput-replays",
            "1",
            "--embedding-batch-size",
            "8",
            "--cleanup-max-attempts",
            "12",
            "--cleanup-settle-seconds",
            "0.0",
            "--skip-throughput",
            "--skip-quadrant",
            "--skip-report",
        ]
    )
    assert args.canary_limit == 10
    assert args.outcome_repetitions == 2
    assert args.quadrant_replays == 1
    assert args.throughput_replays == 1
    assert args.embedding_batch_size == 8
    assert args.cleanup_max_attempts == 12
    assert args.cleanup_settle_seconds == 0.0
    assert args.skip_throughput is True
    assert args.skip_quadrant is True
    assert args.skip_report is True


def test_schema_validation_accepts_required_deployed_index_shape(monkeypatch):
    class _FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def _build_filter(self, **_kwargs):
            return "tenant_id eq 'x'"

    monkeypatch.setattr("scripts.run_sampling_v3.AzureSearchVectorStore", _FakeStore)
    cfg = _fake_config()
    summary = _validate_non_secret_schema(cfg, index_client=_IndexClient(_valid_index()))
    assert summary["key_field"] == "cluster_id"
    assert summary["vector_field"]["dimensions"] == 1536
    assert summary["vector_field"]["profile"] == "hnsw-cosine"
    assert summary["vector_field"]["retrievable"] is False


def test_schema_validation_fails_closed_on_invalid_vector_shape(monkeypatch):
    class _FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def _build_filter(self, **_kwargs):
            return "tenant_id eq 'x'"

    monkeypatch.setattr("scripts.run_sampling_v3.AzureSearchVectorStore", _FakeStore)
    idx = _valid_index()
    for field in idx.fields:
        if field.name == "vector":
            field.vector_search_dimensions = 1024
            break

    with pytest.raises(ValueError, match="vector dimensions must be 1536"):
        _validate_non_secret_schema(_fake_config(), index_client=_IndexClient(idx))


def test_cli_runs_cleanup_and_run_source_before_report(monkeypatch, tmp_path, capsys):
    import scripts.run_sampling_v3 as cli

    calls: list[str] = []
    out_dir = tmp_path / "outputs_sampling_v3" / "runs" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "aggregate.json",
        "runs.jsonl",
        "quadrant.json",
        "throughput.json",
        "corpus_audit.json",
        "token_inventory.jsonl",
        "budget_manifest.json",
        "embedding_ledger.json",
        "selected_membership.json",
        "methodology_delta.md",
        "manifest.json",
    ):
        p = out_dir / name
        if name.endswith(".jsonl"):
            p.write_text('{"x":1}\n', encoding="utf-8")
        elif name.endswith(".md"):
            p.write_text("# m\n", encoding="utf-8")
        else:
            p.write_text('{"version":"sampling-v3-manifest-v1","artifacts":{}}\n' if name == "manifest.json" else '{"x":1}\n', encoding="utf-8")

    monkeypatch.setattr(cli, "AzureConfig", SimpleNamespace(from_env=lambda: _fake_config()))
    monkeypatch.setattr(cli, "_validate_non_secret_schema", lambda _cfg: {"ok": True})
    monkeypatch.setattr(cli, "load_combined_dataset", lambda enforce_integrity_counts=False: object())
    monkeypatch.setattr(cli, "slice_dataset", lambda data, limit=0: data)
    monkeypatch.setattr(cli, "TiktokenTokenizer", lambda **kwargs: SimpleNamespace(encoding_id="enc", version="v"))
    monkeypatch.setattr(cli, "AzureOpenAIEmbedder", lambda _cfg: object())
    monkeypatch.setattr(
        cli,
        "build_v3_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            embedding_semantic_scope="scope-a",
            ledger=SimpleNamespace(embedding_calls=1, embedding_inputs=1),
        ),
    )

    def _bundle(**kwargs):
        calls.append("bundle")
        return {
            "aggregate": {"version": "sampling-v3-bundle-v1", "population_count": 1, "runtime_seconds": 1.0},
            "token_inventory": [{"emitted_tokens": 1}],
            "output_paths": {
                "aggregate": str(out_dir / "aggregate.json"),
                "runs_jsonl": str(out_dir / "runs.jsonl"),
                "quadrant": str(out_dir / "quadrant.json"),
                "throughput": str(out_dir / "throughput.json"),
                "corpus_audit": str(out_dir / "corpus_audit.json"),
                "token_inventory": str(out_dir / "token_inventory.jsonl"),
                "budget_manifest": str(out_dir / "budget_manifest.json"),
                "embedding_ledger": str(out_dir / "embedding_ledger.json"),
                "selected_membership": str(out_dir / "selected_membership.json"),
                "methodology_delta": str(out_dir / "methodology_delta.md"),
                "manifest": str(out_dir / "manifest.json"),
            },
        }

    monkeypatch.setattr(cli, "run_v3_experiment_bundle", _bundle)
    monkeypatch.setattr(cli, "AzureSearchVectorStore", lambda *_args, **_kwargs: SimpleNamespace(_build_filter=lambda **_k: "f", _search_ids=lambda **_k: []))
    monkeypatch.setattr(cli, "_remaining_scope_count", lambda **kwargs: 0)
    monkeypatch.setattr(cli, "_code_hashes", lambda: {"sampling_comparison/v3_outputs.py": "a" * 64})
    monkeypatch.setattr(cli, "_current_branch", lambda: "stangoodwin/sampling-experiment-v3")

    def _cleanup(**kwargs):
        calls.append("cleanup")
        return {"path": str(out_dir / "search_cleanup_audit.json"), "manifest_entry": {"sha256": "b" * 64, "bytes": 1}}

    def _source(**kwargs):
        calls.append("source")
        return {"path": str(out_dir / "run_source_manifest.json"), "manifest_entry": {"sha256": "c" * 64, "bytes": 1}}

    def _report(**kwargs):
        calls.append("report")
        return out_dir / "agent365-sampling-v3-report.html"

    monkeypatch.setattr(cli, "write_search_cleanup_audit", _cleanup)
    monkeypatch.setattr(cli, "write_run_source_manifest", _source)
    monkeypatch.setattr(cli, "write_v3_html_report", _report)
    monkeypatch.setattr(cli, "default_inputs", lambda _out: object())

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sampling_v3.py",
            "--output",
            str(out_dir),
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert calls == ["bundle", "cleanup", "source", "report"]
    assert payload["report_manifest"].endswith("report_manifest.json")
