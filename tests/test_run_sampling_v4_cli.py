from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.run_sampling_v4 import _build_parser, _validate_non_secret_schema


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
        search_index="maven-session-sampling-v4",
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


def test_cli_defaults_to_v4_output_path_and_never_v2_or_v3_default():
    parser = _build_parser()
    args = parser.parse_args([])
    out = Path(args.output).as_posix()
    assert "outputs_sampling_v4" in out
    assert "outputs_sampling_v2" not in out
    assert "outputs_sampling_v3" not in out


def test_cli_parser_flags_include_v3_controls_and_idw_defaults():
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
            "--idw-k",
            "9",
            "--idw-power",
            "3",
            "--idw-eps",
            "0.0001",
            "--idw-exact-cosine-eps",
            "0.00000002",
            "--idw-prior",
            "0.4",
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
    assert args.idw_k == 9
    assert args.idw_power == 3.0
    assert args.idw_eps == 0.0001
    assert args.idw_exact_cosine_eps == 0.00000002
    assert args.idw_prior == 0.4
    assert args.skip_report is True


def test_schema_validation_accepts_required_deployed_index_shape(monkeypatch):
    class _FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def _build_filter(self, **_kwargs):
            return "tenant_id eq 'x'"

    monkeypatch.setattr("scripts.run_sampling_v4.AzureSearchVectorStore", _FakeStore)
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

    monkeypatch.setattr("scripts.run_sampling_v4.AzureSearchVectorStore", _FakeStore)
    idx = _valid_index()
    for field in idx.fields:
        if field.name == "vector":
            field.vector_search_dimensions = 1024
            break

    with pytest.raises(ValueError, match="vector dimensions must be 1536"):
        _validate_non_secret_schema(_fake_config(), index_client=_IndexClient(idx))


def test_cli_rejects_v2_or_v3_output_paths(monkeypatch):
    import scripts.run_sampling_v4 as cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sampling_v4.py",
            "--output",
            "outputs_sampling_v2/runs/x",
        ],
    )
    with pytest.raises(ValueError, match="must not target V2"):
        cli.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sampling_v4.py",
            "--output",
            "outputs_sampling_v3/runs/x",
        ],
    )
    with pytest.raises(ValueError, match="must not target V3"):
        cli.main()


def test_cli_runs_v4_bundle_with_idw_and_report(monkeypatch, tmp_path, capsys):
    import scripts.run_sampling_v4 as cli

    out_dir = tmp_path / "outputs_sampling_v4" / "runs" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

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
            ledger=SimpleNamespace(embedding_calls=1, embedding_inputs=2),
        ),
    )
    monkeypatch.setattr(cli, "_code_hashes", lambda: {"sampling_comparison/v3_outputs.py": "a" * 64})
    monkeypatch.setattr(cli, "_current_branch", lambda: "main")

    def _bundle(**kwargs):
        captured["bundle_kwargs"] = kwargs
        return {
            "aggregate": {
                "version": "sampling-v4-bundle-v1",
                "population_count": 1,
                "runtime_seconds": 1.0,
            },
            "source_v3": {
                "token_inventory": [{"emitted_tokens": 7}],
                "embedding_ledger": {"embedding_calls": 5, "embedding_inputs": 4},
            },
            "output_paths": {
                "aggregate": str(out_dir / "aggregate.json"),
                "runs_jsonl": str(out_dir / "runs.jsonl"),
                "idw_config": str(out_dir / "idw_config.json"),
                "methodology_delta": str(out_dir / "methodology_delta.md"),
                "source_lineage": str(out_dir / "source_lineage.json"),
                "manifest": str(out_dir / "manifest.json"),
                "source_v3_manifest": str(out_dir / "source_v3" / "manifest.json"),
            },
        }

    monkeypatch.setattr(cli, "run_v4_experiment_bundle", _bundle)
    monkeypatch.setattr(
        cli,
        "AzureSearchVectorStore",
        lambda *_args, **_kwargs: SimpleNamespace(_build_filter=lambda **_k: "f", _search_ids=lambda **_k: []),
    )
    monkeypatch.setattr(cli, "_remaining_scope_count", lambda **kwargs: 0)

    class _ReportPath:
        def __init__(self, path: Path):
            self._path = path

        def with_name(self, name: str):
            return self._path.with_name(name)

        def __str__(self):
            return str(self._path)

    def _write_report(*, output_path, inputs):
        captured["report_output_path"] = str(output_path)
        captured["report_inputs"] = inputs
        return _ReportPath(Path(output_path))

    fake_v4_report = SimpleNamespace(
        default_inputs=lambda base_dir: {"base_dir": str(base_dir)},
        write_v4_html_report=_write_report,
    )

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "sampling_comparison.v4_report", fake_v4_report)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sampling_v4.py",
            "--output",
            str(out_dir),
            "--idw-k",
            "11",
            "--idw-power",
            "1.5",
            "--idw-eps",
            "1e-5",
            "--idw-exact-cosine-eps",
            "1e-9",
            "--idw-prior",
            "0.6",
        ],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)

    bundle_kwargs = dict(captured["bundle_kwargs"])
    idw = bundle_kwargs["idw_config"]
    assert idw.k == 11
    assert idw.power == 1.5
    assert idw.eps == 1e-5
    assert idw.exact_cosine_eps == 1e-9
    assert idw.prior == 0.6
    assert bundle_kwargs["tenant_id"] == "sampling-v4-experiment"
    assert bundle_kwargs["aggregate_config"]["embedding"]["dimensions"] == 1536
    assert bundle_kwargs["aggregate_config"]["search"]["ensure_index"] is False
    assert bundle_kwargs["aggregate_config"]["source_v3_pre_run"]["branch"] == "main"

    assert payload["cost_warning_estimate"]["total_emitted_tokens"] == 7
    assert payload["cost_warning_estimate"]["embedding_calls"] == 5
    assert payload["cost_warning_estimate"]["unique_embedding_inputs"] == 4
    assert payload["search_cleanup"]["remaining_count"] == 0
    assert payload["search_cleanup"]["persisted"] is False
    assert payload["report_manifest"].endswith("report_manifest.json")


def test_cli_skip_report_and_cleanup_nonzero_raises(monkeypatch, tmp_path):
    import scripts.run_sampling_v4 as cli

    out_dir = tmp_path / "outputs_sampling_v4" / "runs" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            ledger=SimpleNamespace(embedding_calls=1, embedding_inputs=2),
        ),
    )

    monkeypatch.setattr(
        cli,
        "run_v4_experiment_bundle",
        lambda **kwargs: {
            "aggregate": {
                "version": "sampling-v4-bundle-v1",
                "population_count": 1,
                "runtime_seconds": 1.0,
            },
            "source_v3": {
                "token_inventory": [{"emitted_tokens": 1}],
                "embedding_ledger": {"embedding_calls": 1, "embedding_inputs": 1},
            },
            "output_paths": {
                "aggregate": str(out_dir / "aggregate.json"),
                "runs_jsonl": str(out_dir / "runs.jsonl"),
                "idw_config": str(out_dir / "idw_config.json"),
                "methodology_delta": str(out_dir / "methodology_delta.md"),
                "source_lineage": str(out_dir / "source_lineage.json"),
                "manifest": str(out_dir / "manifest.json"),
                "source_v3_manifest": str(out_dir / "source_v3" / "manifest.json"),
            },
        },
    )
    monkeypatch.setattr(
        cli,
        "AzureSearchVectorStore",
        lambda *_args, **_kwargs: SimpleNamespace(_build_filter=lambda **_k: "f", _search_ids=lambda **_k: ["x"]),
    )
    monkeypatch.setattr(cli, "_remaining_scope_count", lambda **kwargs: 1)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sampling_v4.py",
            "--output",
            str(out_dir),
            "--skip-report",
        ],
    )

    with pytest.raises(ValueError, match="remaining_count == 0"):
        cli.main()
