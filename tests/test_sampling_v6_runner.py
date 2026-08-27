from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from random_sampling.models import EvaluationUnit, Turn
from sampling_comparison.v2_experiment import CombinedDataset
from sampling_comparison.v4_idw import IDWConfig
from sampling_comparison.v6_business_use_case import (
    BusinessUseCaseInfo,
    BusinessUseCaseDetermination,
    SessionSelection,
)
from sampling_comparison.v6_experiment import METHOD_IDS
from sampling_comparison.v6_runner import (
    _Arm2SelectionResult,
    _ClassificationRow,
    _aggregate_trial_rows,
    _sha256_file,
    _select_arm2_exact_count,
    default_output_dir,
    run_sampling_v6_bundle,
)
from trace_sampling.azure_config import AzureConfig

import scripts.run_sampling_v6 as cli_script


def _make_unit(*, unit_id: str, tenant: str, agent: str, user: str, assistant: str, label: bool) -> tuple[EvaluationUnit, dict[str, object], bool]:
    unit = EvaluationUnit(
        tenant_id=tenant,
        agent_id=agent,
        conversation_id=f"conv-{unit_id}",
        session_id=f"sess-{unit_id}",
        channel="chat",
        source_trace_ids=(f"trace-{unit_id}",),
        started_at=None,
        ended_at=None,
        had_error=False,
        turns=(Turn(user_text=user, assistant_text=assistant),),
        tool_calls=(),
        unit_id=unit_id,
    )
    meta = {
        "corpus_id": "historical_300" if unit_id.endswith("0") else "dense_2500",
        "domain": "ops",
        "task": "triage",
        "difficulty": "easy",
    }
    return unit, meta, label


def _tiny_dataset(tmp_path: Path) -> CombinedDataset:
    src_a = tmp_path / "hist.json"
    src_b = tmp_path / "dense.json"
    src_a.write_text("{\"x\":1}\n", encoding="utf-8")
    src_b.write_text("{\"x\":2}\n", encoding="utf-8")

    rows = [
        _make_unit(unit_id="u0", tenant="t1", agent="a1", user="User 0", assistant="Asst 0", label=True),
        _make_unit(unit_id="u1", tenant="t1", agent="a2", user="User 1", assistant="Asst 1", label=False),
        _make_unit(unit_id="u2", tenant="t2", agent="a1", user="User 2", assistant="Asst 2", label=True),
        _make_unit(unit_id="u3", tenant="t2", agent="a2", user="User 3", assistant="Asst 3", label=False),
        _make_unit(unit_id="u4", tenant="t3", agent="a3", user="User 4", assistant="Asst 4", label=True),
    ]

    units = tuple(row[0] for row in rows)
    unit_ids = tuple(u.unit_id or "" for u in units)
    labels_by_unit = {row[0].unit_id or "": row[2] for row in rows}
    metadata_by_unit = {row[0].unit_id or "": dict(row[1]) for row in rows}

    from sampling_comparison.v2_experiment import _trace  # type: ignore

    traces = []
    trace_by_id = {}
    for idx, unit in enumerate(units, start=1):
        uid = unit.unit_id or ""
        tr = _trace(unit, unit_id=uid, ordinal=idx, meta=metadata_by_unit[uid])
        traces.append(tr)
        trace_by_id[uid] = tr

    return CombinedDataset(
        units=units,
        unit_ids=unit_ids,
        traces=tuple(traces),
        trace_by_unit_id=trace_by_id,
        labels_by_unit=labels_by_unit,
        metadata_by_unit=metadata_by_unit,
        corpus_id_by_unit={"u0": "historical_300", "u1": "dense_2500", "u2": "dense_2500", "u3": "dense_2500", "u4": "dense_2500"},
        original_unit_id_by_unit={uid: uid for uid in unit_ids},
        scoped_identities=tuple(sorted({f"{u.tenant_id}|{u.agent_id}" for u in units})),
        source_paths={"historical_300": str(src_a), "dense_2500": str(src_b)},
    )


class _FakeTokenizer:
    encoding_name = "cl100k_base"
    encoding_id = "cl100k_base:test"
    version = "test"

    def count(self, text: str) -> int:
        return max(1, len(text.split()))


class _FakeEmbedder:
    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = sum(ord(ch) for ch in text) % 997
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim)
            norm = float(np.linalg.norm(vec))
            out[i] = np.asarray(vec / (norm or 1.0), dtype=np.float32)
        return out


class _FakeArtifacts:
    def __init__(self) -> None:
        self.metadata = type(
            "Meta",
            (),
            {
                "taxonomy_version": "6.0",
                "dimensions": 1536,
                "taxonomy_count": 11,
                "request_centroid_count": 7,
                "response_centroid_count": 9,
                "taxonomy_db_path": "taxonomy.db",
                "taxonomy_db_sha256": "t" * 64,
                "centroids_db_path": "centroids.db",
                "centroids_db_sha256": "c" * 64,
            },
        )()


class _FakeClassifier:
    def enumerate_unique_clean_texts(self, *, steps, token_counter=None, max_input_tokens=8191):
        out = []
        for step in steps:
            for text in (step.request, step.response):
                value = (text or "").strip()
                if value and value not in out:
                    out.append(value)
        return tuple(out)

    def classify_sessions_from_text_embeddings(self, *, sessions_by_unit_id, embeddings_by_text, token_counter=None, max_input_tokens=8191):
        selected: dict[str, SessionSelection | None] = {}
        for unit_id in sessions_by_unit_id:
            guid = UUID("11111111-1111-1111-1111-111111111111") if unit_id != "u3" else UUID("9a6df217-0865-486d-93da-519ebcd37a70")
            info = BusinessUseCaseInfo(
                guid=guid,
                domain="ops",
                segment="seg",
                category="cat",
                sub_category="sub",
                sub_subcategory="leaf",
                business_task="task",
            )
            determination = BusinessUseCaseDetermination(
                guid=guid,
                use_case=info,
                status="Agree",
                reason="fake",
                combined_cosine_similarity=0.9,
                input_matches=(),
                output_matches=(),
                combined_best=None,
            )
            selected[unit_id] = SessionSelection(step_index=0, provenance="threshold", determination=determination)
        return selected


class _NonFiniteClassifier(_FakeClassifier):
    def classify_sessions_from_text_embeddings(self, *, sessions_by_unit_id, embeddings_by_text, token_counter=None, max_input_tokens=8191):
        selected = super().classify_sessions_from_text_embeddings(
            sessions_by_unit_id=sessions_by_unit_id,
            embeddings_by_text=embeddings_by_text,
            token_counter=token_counter,
            max_input_tokens=max_input_tokens,
        )
        patched: dict[str, SessionSelection | None] = {}
        for unit_id, row in selected.items():
            if row is None:
                patched[unit_id] = None
                continue
            det = row.determination
            patched[unit_id] = SessionSelection(
                step_index=row.step_index,
                provenance=row.provenance,
                determination=BusinessUseCaseDetermination(
                    guid=det.guid,
                    use_case=det.use_case,
                    status=det.status,
                    reason=det.reason,
                    combined_cosine_similarity=float("nan"),
                    input_matches=det.input_matches,
                    output_matches=det.output_matches,
                    combined_best=det.combined_best,
                ),
            )
        return patched


class _TrackingEmbedder(_FakeEmbedder):
    def __init__(self, dim: int = 1536) -> None:
        super().__init__(dim=dim)
        self.batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        return super().embed(texts)


class _ExplodingClassifier(_FakeClassifier):
    def classify_sessions_from_text_embeddings(self, *, sessions_by_unit_id, embeddings_by_text, token_counter=None, max_input_tokens=8191):
        raise AssertionError("baseline extension should not reclassify when cached baseline classifications are valid")


class _FakeStore:
    def __init__(self, *args, **kwargs):
        self.search_queries = 0
        self.writes = 0
        self.cleanup_deleted = 0

    def delete_scope_settled(self, tenant_id, run_scope, semantic_scope=None, max_attempts=3, settle_seconds=0.0):
        return [], 0

    def count_scope(self, *, tenant_id: str, run_scope: str, semantic_scope: str) -> int:
        return 0


def _fake_arm2_selector(**kwargs):
    ordered = list(kwargs["ordered_unit_ids"])
    cap = int(kwargs["cap"])
    selected = tuple(ordered[:cap])
    from sampling_comparison.v6_experiment import SelectionRecord

    records = tuple(
        SelectionRecord(
            unit_id=uid,
            method_id=METHOD_IDS["arm2"],
            stratum="fake",
            inclusion_probability=None,
            weight=None,
            reason="fake-arm2",
        )
        for uid in selected
    )
    return _Arm2SelectionResult(
        selected_ids=selected,
        records=records,
        telemetry={"fallbacks": 0, "search_queries": 0, "writes": 0, "cleanup_deleted": 0},
    )


def _run_bundle(
    tmp_path: Path,
    *,
    out_name: str,
    seeds=(13, 14, 15),
    caps=(1, 2, 3, 4, 5),
    embedder=None,
    embeddings_cache=None,
    classifications_cache=None,
    classifier=None,
    baseline_dir=None,
    checkpoint_dir=None,
    resume=True,
    arm2_selector=None,
    progress_callback=None,
):
    data = _tiny_dataset(tmp_path)
    out_dir = tmp_path / out_name
    cfg = AzureConfig(
        openai_endpoint="https://example.openai.azure.com",
        openai_api_version="2024-02-01",
        embedding_deployment="embed-test",
        search_endpoint="https://example.search.windows.net",
        search_index="trace-clusters-sampling-v6",
        openai_api_key="x",
        search_api_key="y",
    )
    fake_embedder = embedder or _FakeEmbedder(dim=1536)

    result = run_sampling_v6_bundle(
        output_dir=out_dir,
        caps=caps,
        seeds=seeds,
        avg_tokens_per_session=100,
        embedding_batch_size=2,
        cleanup_max_attempts=3,
        cleanup_settle_seconds=0.0,
        ensure_search_index=False,
        idw_config=IDWConfig(k=2, power=2.0, eps=1e-6, exact_cosine_eps=1e-8, prior=0.5),
        embeddings_cache_path=embeddings_cache,
        classifications_cache_path=classifications_cache,
        skip_report=True,
        data=data,
        azure_config=cfg,
        tokenizer=_FakeTokenizer(),
        embedder=fake_embedder,
        classifier=classifier or _FakeClassifier(),
        use_case_artifacts=_FakeArtifacts(),
        vector_store_factory=lambda _tenant, _scope: _FakeStore(),
        arm2_selector=arm2_selector or _fake_arm2_selector,
        enforce_integrity_counts=False,
        baseline_dir=baseline_dir,
        checkpoint_dir=checkpoint_dir,
        resume=bool(resume),
        progress_callback=progress_callback,
    )
    return result, fake_embedder


def _make_agent_floor_fixture() -> tuple[tuple, ...]:
    descriptors = []
    for agent_idx in range(1, 61):
        for unit_idx in range(1, 7):
            unit_id = f"agent-{agent_idx}-u{unit_idx}"
            descriptors.append(
                {
                    "unit_id": unit_id,
                    "agent_id": f"agent-{agent_idx}",
                    "use_case_id": f"use-case-{(unit_idx % 3) + 1}",
                    "concept_key": f"concept-{(agent_idx + unit_idx) % 4}",
                    "label": bool((agent_idx + unit_idx) % 2 == 0),
                }
            )
    return tuple(descriptors)


def test_aggregate_trial_rows_include_rich_descriptive_stats():
    rows = [
        {"method_id": "arm1_global_random", "cap": 10, "absolute_aggregate_mae": 1.0, "concept_coverage": 0.4, "use_case_coverage": 0.5, "agent_coverage": 0.6, "sample_size": 5, "actual_token_count": 1000, "nominal_budget": 1500},
        {"method_id": "arm1_global_random", "cap": 10, "absolute_aggregate_mae": 2.0, "concept_coverage": 0.5, "use_case_coverage": 0.6, "agent_coverage": 0.7, "sample_size": 6, "actual_token_count": 1200, "nominal_budget": 1800},
        {"method_id": "arm1_global_random", "cap": 10, "absolute_aggregate_mae": 3.0, "concept_coverage": 0.6, "use_case_coverage": 0.7, "agent_coverage": 0.8, "sample_size": 7, "actual_token_count": 1400, "nominal_budget": 2100},
    ]

    aggregate = _aggregate_trial_rows(rows)
    assert len(aggregate) == 1
    block = aggregate[0]
    assert block["mae"]["median"] == pytest.approx(2.0)
    assert block["mae"]["p05"] == pytest.approx(1.1)
    assert block["mae"]["p95"] == pytest.approx(2.9)
    assert block["mae"]["sample_std"] == pytest.approx(1.0)
    assert block["mae"]["count"] == 3
    assert block["selected_count"]["count"] == 3
    assert block["actual_tokens"]["count"] == 3
    assert block["nominal_budget"]["count"] == 3

    single = _aggregate_trial_rows([
        {"method_id": "arm2_embedding_idw", "cap": 12, "absolute_aggregate_mae": 0.5, "concept_coverage": 0.2, "use_case_coverage": 0.3, "agent_coverage": 0.4, "sample_size": 4, "actual_token_count": 500, "nominal_budget": 900},
    ])
    assert single[0]["mae"]["sample_std"] == pytest.approx(0.0)


def test_default_output_dir_points_to_v6_runs():
    out = default_output_dir().as_posix()
    assert "outputs_sampling_v6/runs/" in out


def test_v6_runner_writes_expected_artifacts_and_75_shape(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o1")
    paths = result["output_paths"]

    for key in ("aggregate", "runs", "memberships", "classifications", "dataset_examples", "methodology", "manifest"):
        assert Path(paths[key]).exists()

    runs = [json.loads(line) for line in Path(paths["runs"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(runs) == 75
    methods = {(row["method_id"], row["cap"], row["seed"]) for row in runs}
    assert len(methods) == 75


def test_memberships_exclude_labels_and_arm4_arm5_identical(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o2")
    memberships = [json.loads(line) for line in Path(result["output_paths"]["memberships"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    for row in memberships:
        text = json.dumps(row, sort_keys=True)
        assert "label" not in text

    grouped = {}
    for row in memberships:
        grouped[(row["seed"], row["cap"], row["method_id"])] = tuple(row["selected_ids"])

    for seed in (13, 14, 15):
        for cap in (1, 2, 3, 4, 5):
            assert grouped[(seed, cap, METHOD_IDS["arm4"])] == grouped[(seed, cap, METHOD_IDS["arm5"])]

    for row in memberships:
        assert "selected_agent_count" in row
        assert "agents_with_at_least_3" in row
        assert "represented_strata" in row
        for rec in row["selection_records"]:
            assert "agent_id" in rec
            assert "use_case_id" in rec
            assert "label" not in rec

    arm3_rows = [row for row in memberships if row["method_id"] == METHOD_IDS["arm3"]]
    assert arm3_rows
    for row in arm3_rows:
        assert "total_floor_target" in row
        assert "floor_complete" in row
        assert "floor_prefix_count" in row

    arm4_rows = [row for row in memberships if row["method_id"] == METHOD_IDS["arm4"]]
    arm5_rows = [row for row in memberships if row["method_id"] == METHOD_IDS["arm5"]]
    assert len(arm4_rows) == len(arm5_rows)
    for idx in range(len(arm4_rows)):
        assert tuple(arm4_rows[idx]["selected_ids"]) == tuple(arm5_rows[idx]["selected_ids"])
        assert arm4_rows[idx]["inclusion_probability_by_unit"] == arm5_rows[idx]["inclusion_probability_by_unit"]


def test_classification_rows_are_serialized_without_raw_text(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o3")
    rows = [json.loads(line) for line in Path(result["output_paths"]["classifications"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        assert "unit_id" in row
        assert "agent_id" in row
        assert "concept_key" in row
        assert "corpus_id" in row
        assert "use_case_guid" in row
        assert "domain" in row
        assert "segment" in row
        assert "category" in row
        assert "sub_category" in row
        assert "sub_subcategory" in row
        assert "business_task" in row
        assert "selected_step_index" in row
        blob = json.dumps(row, sort_keys=True)
        assert "User " not in blob
        assert "Asst " not in blob


def test_cache_provenance_reuse_avoids_stale_and_reduces_embed_calls(tmp_path: Path):
    emb_cache = tmp_path / "cache" / "emb"
    cls_cache = tmp_path / "cache" / "cls"

    result1, embedder1 = _run_bundle(
        tmp_path,
        out_name="o4a",
        embeddings_cache=emb_cache,
        classifications_cache=cls_cache,
    )
    calls_first = embedder1.calls
    assert calls_first > 0

    result2, embedder2 = _run_bundle(
        tmp_path,
        out_name="o4b",
        embeddings_cache=emb_cache,
        classifications_cache=cls_cache,
    )
    calls_second = embedder2.calls
    assert calls_second < calls_first

    agg1 = json.loads(Path(result1["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    agg2 = json.loads(Path(result2["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    assert agg1["embedding_ledgers"]["runtime_cache"]["provenance"]["version"] == "sampling-v6-runtime-cache-provenance-v1"
    assert agg2["embedding_ledgers"]["runtime_cache"]["provenance"]["version"] == "sampling-v6-runtime-cache-provenance-v1"
    assert agg1["embedding_ledgers"]["classification_cache"]["cache_hit_rows"] == 0
    assert agg1["embedding_ledgers"]["classification_cache"]["embedding_calls"] >= 1
    assert agg1["embedding_ledgers"]["classification_cache"]["embedding_inputs"] >= 1
    assert agg1["embedding_ledgers"]["classification_cache"]["embedding_input_tokens"] >= 1
    assert agg1["embedding_ledgers"]["classification_cache"]["elapsed_seconds"] >= 0.0
    assert agg2["embedding_ledgers"]["classification_cache"]["cache_hit_rows"] > 0


def test_classification_embedding_batching_respects_batch_size(tmp_path: Path):
    emb_cache = tmp_path / "cacheb" / "emb"
    cls_cache = tmp_path / "cacheb" / "cls"
    embedder = _TrackingEmbedder(dim=1536)

    _run_bundle(
        tmp_path,
        out_name="o_batch",
        embedder=embedder,
        embeddings_cache=emb_cache,
        classifications_cache=cls_cache,
    )

    assert max(embedder.batch_sizes) <= 2
    assert 2 in embedder.batch_sizes


def test_nonfinite_classification_similarity_serializes_as_null(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o_nonfinite", classifier=_NonFiniteClassifier())
    rows = [json.loads(line) for line in Path(result["output_paths"]["classifications"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert row["combined_cosine_similarity"] is None


def test_manifest_hashes_match_written_files(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o5")
    manifest = json.loads(Path(result["output_paths"]["manifest"]).read_text(encoding="utf-8"))
    base = Path(result["output_paths"]["manifest"]).parent

    for _, meta in manifest["artifacts"].items():
        path = base / meta["path"]
        assert path.exists()
        assert int(meta["bytes"]) == path.stat().st_size
        assert meta["sha256"] == _sha256_file(path)


def test_arm2_run_rows_include_sanitized_idw_and_aggregate_aliases(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o7")
    runs = [json.loads(line) for line in Path(result["output_paths"]["runs"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    arm2 = [row for row in runs if row["method_id"] == METHOD_IDS["arm2"]]
    assert arm2
    for row in arm2:
        prov = row["idw_provenance"]
        assert sorted(prov.keys()) == sorted(
            [
                "population_count",
                "observed_count",
                "imputed_count",
                "provenance_counts",
                "zero_donor_agent_count",
                "prior_count",
                "estimated_pass_rate",
            ]
        )
        quality = row["idw_quality"]
        assert sorted(quality.keys()) == sorted(
            [
                "absolute_aggregate_rate_error",
                "per_unit_mae",
                "brier_score",
                "macro_per_agent_mae",
                "unjudged_only_mae",
                "unjudged_only_brier",
                "expected_calibration_error",
            ]
        )
        text = json.dumps(row, sort_keys=True)
        assert "donor_ids" not in text
        assert "distances" not in text
        assert "normalized_weights" not in text
        assert "rows" not in text

    agg = json.loads(Path(result["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    assert agg["config"]["skip_report"] is True
    assert agg["maven_artifacts"]["taxonomy_count"] == 11
    assert agg["maven_artifacts"]["request_centroid_count"] == 7
    assert agg["maven_artifacts"]["response_centroid_count"] == 9
    assert "C:\\Users" not in str(agg["maven_artifacts"].get("taxonomy_db_path") or "")
    assert "C:\\Users" not in str(agg["maven_artifacts"].get("centroids_db_path") or "")
    assert agg["maven_artifacts"]["taxonomy_db_path"] == "taxonomy.db"
    assert agg["maven_artifacts"]["centroids_db_path"] == "centroids.db"
    assert agg["maven_artifacts"]["taxonomy_db_source_kind"] == "local-file"
    assert agg["maven_artifacts"]["centroids_db_source_kind"] == "local-file"
    assert agg["aggregate_rows"]
    for row in agg["aggregate_rows"]:
        assert "mae" in row
        assert "absolute_aggregate_mae" in row
        assert "maven_coverage" in row
        assert "use_case_coverage" in row
        assert "actual_tokens" in row
        assert "actual_token_count" in row
        assert "nominal_budget" in row
        assert "trial_count" in row


def test_dataset_examples_include_source_summary_and_expected_label(tmp_path: Path):
    result, _ = _run_bundle(tmp_path, out_name="o8")
    examples = json.loads(Path(result["output_paths"]["dataset_examples"]).read_text(encoding="utf-8"))
    assert "source_summary" in examples
    assert "overall" in examples["source_summary"]
    assert "schema" in examples["source_summary"]
    assert "synthesized_fields" in examples
    assert "source_synthetic" in examples["synthesized_fields"]
    assert "report_derived" in examples["synthesized_fields"]
    for row in examples["examples"]:
        assert "source" in row
        assert row["source"]["is_synthetic"] is True
        assert "expected_label" in row


def test_arm2_cleanup_or_fallback_failure_propagates(tmp_path: Path):
    def _failing_arm2(**kwargs):
        raise RuntimeError("arm2 encountered fallback behavior")

    data = _tiny_dataset(tmp_path)
    cfg = AzureConfig(
        openai_endpoint="https://example.openai.azure.com",
        openai_api_version="2024-02-01",
        embedding_deployment="embed-test",
        search_endpoint="https://example.search.windows.net",
        search_index="trace-clusters-sampling-v6",
        openai_api_key="x",
        search_api_key="y",
    )

    try:
        run_sampling_v6_bundle(
            output_dir=tmp_path / "o6",
            caps=(1,),
            seeds=(13,),
            avg_tokens_per_session=100,
            embeddings_cache_path=tmp_path / "cache2" / "emb",
            classifications_cache_path=tmp_path / "cache2" / "cls",
            skip_report=True,
            data=data,
            azure_config=cfg,
            tokenizer=_FakeTokenizer(),
            embedder=_FakeEmbedder(dim=1536),
            classifier=_FakeClassifier(),
            use_case_artifacts=_FakeArtifacts(),
            vector_store_factory=lambda _tenant, _scope: _FakeStore(),
            arm2_selector=_failing_arm2,
            enforce_integrity_counts=False,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "fallback" in str(exc)


def test_progress_json_and_callback_track_completion(tmp_path: Path):
    progress_events: list[dict[str, object]] = []
    result = run_sampling_v6_bundle(
        output_dir=tmp_path / "progress-ok",
        caps=(1,),
        seeds=(13,),
        avg_tokens_per_session=100,
        embedding_batch_size=2,
        cleanup_max_attempts=3,
        cleanup_settle_seconds=0.0,
        ensure_search_index=False,
        idw_config=IDWConfig(k=2, power=2.0, eps=1e-6, exact_cosine_eps=1e-8, prior=0.5),
        embeddings_cache_path=tmp_path / "cache-progress" / "emb",
        classifications_cache_path=tmp_path / "cache-progress" / "cls",
        skip_report=True,
        data=_tiny_dataset(tmp_path),
        azure_config=AzureConfig(
            openai_endpoint="https://example.openai.azure.com",
            openai_api_version="2024-02-01",
            embedding_deployment="embed-test",
            search_endpoint="https://example.search.windows.net",
            search_index="trace-clusters-sampling-v6",
            openai_api_key="x",
            search_api_key="y",
        ),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(dim=1536),
        classifier=_FakeClassifier(),
        use_case_artifacts=_FakeArtifacts(),
        vector_store_factory=lambda _tenant, _scope: _FakeStore(),
        arm2_selector=_fake_arm2_selector,
        enforce_integrity_counts=False,
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )
    assert result["output_paths"]["aggregate"]

    progress_path = tmp_path / "progress-ok" / "progress.json"
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["percent"] == 100.0
    assert payload["total_replays"] == 1
    assert payload["total_cells"] == 5
    assert payload["phase"] in {"method-evaluation", "complete"}
    payload_json = json.dumps(payload)
    assert "openai_api_key" not in payload_json
    assert "search_api_key" not in payload_json
    assert all(float(event["percent"]) >= float(prev["percent"]) for prev, event in zip(progress_events, progress_events[1:]))


def test_progress_json_failed_on_selector_exception(tmp_path: Path):
    def _broken_arm2(**kwargs):
        raise RuntimeError("arm2 failure for test")

    progress_events: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="arm2 failure for test"):
        run_sampling_v6_bundle(
            output_dir=tmp_path / "progress-fail",
            caps=(1,),
            seeds=(13,),
            avg_tokens_per_session=100,
            embedding_batch_size=2,
            cleanup_max_attempts=3,
            cleanup_settle_seconds=0.0,
            ensure_search_index=False,
            idw_config=IDWConfig(k=2, power=2.0, eps=1e-6, exact_cosine_eps=1e-8, prior=0.5),
            embeddings_cache_path=tmp_path / "cache-fail" / "emb",
            classifications_cache_path=tmp_path / "cache-fail" / "cls",
            skip_report=True,
            data=_tiny_dataset(tmp_path),
            azure_config=AzureConfig(
                openai_endpoint="https://example.openai.azure.com",
                openai_api_version="2024-02-01",
                embedding_deployment="embed-test",
                search_endpoint="https://example.search.windows.net",
                search_index="trace-clusters-sampling-v6",
                openai_api_key="x",
                search_api_key="y",
            ),
            tokenizer=_FakeTokenizer(),
            embedder=_FakeEmbedder(dim=1536),
            classifier=_FakeClassifier(),
            use_case_artifacts=_FakeArtifacts(),
            vector_store_factory=lambda _tenant, _scope: _FakeStore(),
            arm2_selector=_broken_arm2,
            enforce_integrity_counts=False,
            progress_callback=lambda payload: progress_events.append(dict(payload)),
        )

    payload = json.loads((tmp_path / "progress-fail" / "progress.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert "arm2 failure for test" in payload["error_message"]
    assert progress_events[-1]["status"] == "failed"


def test_progress_write_lock_is_nonfatal(monkeypatch, tmp_path: Path):
    from sampling_comparison import v6_runner

    real_write = v6_runner._write_json_atomic
    progress_failures = 0

    def flaky_write(path, payload):
        nonlocal progress_failures
        if Path(path).name == "progress.json" and progress_failures < 2:
            progress_failures += 1
            raise PermissionError("simulated transient Windows lock")
        return real_write(path, payload)

    monkeypatch.setattr(v6_runner, "_write_json_atomic", flaky_write)
    result, _ = _run_bundle(
        tmp_path,
        out_name="progress-lock",
        seeds=(13,),
        caps=(1,),
    )

    assert progress_failures == 2
    progress = json.loads((tmp_path / "progress-lock" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert Path(result["output_paths"]["manifest"]).exists()


def test_select_arm2_exact_count_emits_progress_events(tmp_path: Path, monkeypatch):
    data = _tiny_dataset(tmp_path)
    runtime = type(
        "RT",
        (),
        {
            "embedding_semantic_scope": "sscope",
            "embedding_records_by_content_sha256": {},
            "token_cost_by_unit_id": {unit_id: 1 for unit_id in data.unit_ids},
        },
    )()
    runtime.embedding_vector_by_trace_id = {}
    for unit_id, trace in data.trace_by_unit_id.items():
        runtime.embedding_vector_by_trace_id[int(trace.trace_id)] = np.ones(3, dtype=np.float32)

    store = _FakeStore()
    store.search_queries = 0
    store.writes = 0
    store.cleanup_deleted = 0
    store.delete_scope_settled = lambda *args, **kwargs: ([], 0)
    store.count_scope = lambda *args, **kwargs: 0

    class _FakeVarietyIndex:
        n_fallbacks = 0

        def observe(self, trace):
            return type("Observation", (), {"novelty": 0.1, "rarity": 0.2, "key": type("Key", (), {"kind": "ok"})(), "proposed_keep": True})()

    class _FakeSelector:
        def __init__(self, runtime, vector_store_factory, tau, cleanup_max_attempts, cleanup_settle_seconds):
            self.runtime = runtime
            self.vector_store_factory = vector_store_factory
            self.tau = tau
            self.cleanup_max_attempts = cleanup_max_attempts
            self.cleanup_settle_seconds = cleanup_settle_seconds

        def build_index(self, tenant_id, run_scope):
            return _FakeVarietyIndex(), store

    monkeypatch.setattr("sampling_comparison.v6_runner.V3EmbeddingSelector", _FakeSelector)

    events: list[dict[str, object]] = []
    _select_arm2_exact_count(
        data=data,
        runtime=runtime,
        cap=2,
        seed=13,
        ordered_unit_ids=data.unit_ids,
        tenant_id="tenant",
        run_scope="scope",
        vector_store_factory=lambda _tenant, _scope: store,
        cleanup_max_attempts=3,
        cleanup_settle_seconds=0.0,
        progress_callback=lambda payload: events.append(dict(payload)),
    )

    assert any(event.get("phase") == "pre-cleanup" for event in events)
    assert any(event.get("phase") == "replay" and int(event.get("replay_session_current", 0)) > 0 for event in events)
    assert any(event.get("phase") == "post-cleanup" for event in events)
    assert len(events) >= 3


def test_cli_progress_formatter_outputs_ascii_bar():
    event = {
        "phase": "search-replay",
        "status": "running",
        "percent": 23.4,
        "replay_session_current": 1730,
        "replay_session_total": 2800,
        "current_seed": 13,
        "current_cap": 512,
        "current_method": "arm2",
        "current_replay": 4,
        "total_replays": 15,
    }
    rendered = cli_script._format_progress_bar(event)
    assert "[" in rendered and "]" in rendered
    assert "23.4%" in rendered
    assert "replay 4/15" in rendered
    assert "seed 13 cap 512" in rendered
    assert "arm2" in rendered
    assert "Search 1730/2800" in rendered
    assert len(rendered.split("|")[0].strip().strip("[]")) == 28


def test_cli_progress_formatter_uses_fixed_bar_width_and_clean_replay_text():
    event = {
        "phase": "replay-setup",
        "status": "running",
        "percent": 0.0,
        "current_seed": 13,
        "current_cap": 64,
        "current_method": "arm2",
        "current_replay": 1,
        "total_replays": 4,
    }
    rendered = cli_script._format_progress_bar(event)
    assert "Search current/total" in rendered
    assert "replay 1/4" in rendered
    assert len(rendered.split("|")[0].strip().strip("[]")) == 28


def test_arm3_floor_completion_uses_total_floor_target_and_agent_coverage():
    from sampling_comparison.v6_experiment import SessionDescriptor, select_arm3

    descriptors = []
    for agent_idx in range(1, 61):
        for unit_idx in range(1, 7):
            unit_id = f"agent-{agent_idx}-u{unit_idx}"
            descriptors.append(
                SessionDescriptor(
                    unit_id=unit_id,
                    agent_id=f"agent-{agent_idx}",
                    use_case_id=f"use-case-{(unit_idx % 3) + 1}",
                    concept_key=f"concept-{(agent_idx + unit_idx) % 4}",
                    label=bool((agent_idx + unit_idx) % 2 == 0),
                )
            )
    total_agent_count = len({descriptor.agent_id for descriptor in descriptors})
    total_floor_target = sum(min(3, count) for count in Counter(descriptor.agent_id for descriptor in descriptors).values())
    assert total_floor_target == 180
    assert total_floor_target > 128

    for cap in (64, 128):
        outcome = select_arm3(descriptors=descriptors, cap=cap, trial_seed=13, window_id="w")
        selected_agent_count = len({descriptor.agent_id for descriptor in descriptors if descriptor.unit_id in set(outcome.selected_ids)})
        floor_prefix_count = min(cap, total_floor_target)
        floor_complete = floor_prefix_count >= total_floor_target
        assert floor_prefix_count == cap
        assert floor_complete is False
        assert selected_agent_count / total_agent_count > 0.0
        assert selected_agent_count <= total_agent_count

    outcome = select_arm3(descriptors=descriptors, cap=256, trial_seed=13, window_id="w")
    floor_prefix_count = min(256, total_floor_target)
    floor_complete = floor_prefix_count >= total_floor_target
    assert floor_prefix_count == total_floor_target
    assert floor_complete is True
    assert len({descriptor.agent_id for descriptor in descriptors if descriptor.unit_id in set(outcome.selected_ids)}) / total_agent_count <= 1.0


def test_baseline_extension_reuses_classifications_without_reclassifying(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline-classifications", seeds=(13,), caps=(1,))
    baseline_dir = Path(baseline_result["output_paths"]["manifest"]).parent

    extended_result, _ = _run_bundle(
        tmp_path,
        out_name="extended-classifications",
        seeds=(14,),
        caps=(1,),
        baseline_dir=baseline_dir,
        classifier=_ExplodingClassifier(),
    )

    baseline_rows = [
        json.loads(line)
        for line in Path(baseline_result["output_paths"]["classifications"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extended_rows = [
        json.loads(line)
        for line in Path(extended_result["output_paths"]["classifications"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert baseline_rows == extended_rows

    aggregate = json.loads(Path(extended_result["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    ledger = aggregate["embedding_ledgers"]["classification_cache"]
    assert ledger["cache_hit"] is True
    assert ledger["source"] == "baseline"
    assert ledger["rows"] == 5
    assert ledger["embedding_calls"] == 0
    assert ledger["embedding_inputs"] == 0
    assert ledger["embedding_input_tokens"] == 0
    assert ledger["embedding_latency_seconds"] == 0.0


def test_invalid_baseline_classifications_are_rejected(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline-invalid", seeds=(13,), caps=(1,))
    baseline_dir = Path(baseline_result["output_paths"]["manifest"]).parent
    classifications_path = baseline_dir / "classifications.jsonl"
    rows = [json.loads(line) for line in classifications_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["unit_id"] = rows[1]["unit_id"]
    classifications_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline classifications"):
        _run_bundle(
            tmp_path,
            out_name="extended-invalid-classifications",
            seeds=(14,),
            caps=(1,),
            baseline_dir=baseline_dir,
            classifier=_ExplodingClassifier(),
        )


def test_extension_merge_counts_baseline_unchanged_and_checkpoint_per_cell(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline", seeds=(13,), caps=(1, 2))
    baseline_runs_path = Path(baseline_result["output_paths"]["runs"])
    baseline_memberships_path = Path(baseline_result["output_paths"]["memberships"])
    baseline_runs_before = baseline_runs_path.read_bytes()
    baseline_memberships_before = baseline_memberships_path.read_bytes()

    checkpoint_root = tmp_path / "ckpt"
    extended_result, _ = _run_bundle(
        tmp_path,
        out_name="extended",
        seeds=(14, 15),
        caps=(1, 2),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
    )

    runs = [json.loads(line) for line in Path(extended_result["output_paths"]["runs"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    memberships = [json.loads(line) for line in Path(extended_result["output_paths"]["memberships"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(runs) == 30
    assert len(memberships) == 30

    ckpt_cells = list((checkpoint_root / "cells").glob("seed-*-cap-*.json"))
    assert len(ckpt_cells) == 4
    for path in ckpt_cells:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "complete"
        assert len(payload["run_rows"]) == 5
        assert len(payload["membership_rows"]) == 5
        text = json.dumps(payload).lower()
        assert "openai_api_key" not in text
        assert "search_api_key" not in text
        assert "label" not in text

    assert baseline_runs_path.read_bytes() == baseline_runs_before
    assert baseline_memberships_path.read_bytes() == baseline_memberships_before


def test_resume_skips_selector_and_rows_are_equivalent(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline2", seeds=(13,), caps=(1,))
    checkpoint_root = tmp_path / "ckpt2"

    first_result, _ = _run_bundle(
        tmp_path,
        out_name="extended-first",
        seeds=(14,),
        caps=(1,),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
    )

    def _boom_selector(**kwargs):
        raise AssertionError("selector should not run when checkpoint is reused")

    second_result, _ = _run_bundle(
        tmp_path,
        out_name="extended-second",
        seeds=(14,),
        caps=(1,),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
        arm2_selector=_boom_selector,
    )

    assert Path(first_result["output_paths"]["runs"]).read_bytes() == Path(second_result["output_paths"]["runs"]).read_bytes()
    assert Path(first_result["output_paths"]["memberships"]).read_bytes() == Path(second_result["output_paths"]["memberships"]).read_bytes()


def test_corrupt_or_incompatible_checkpoint_rejected(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline3", seeds=(13,), caps=(1,))
    checkpoint_root = tmp_path / "ckpt3"
    _run_bundle(
        tmp_path,
        out_name="extended3a",
        seeds=(14,),
        caps=(1,),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
    )

    ckpt_path = next((checkpoint_root / "cells").glob("seed-14-cap-1.json"))
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    payload["payload_hash"] = "0" * 64
    ckpt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        _run_bundle(
            tmp_path,
            out_name="extended3b",
            seeds=(14,),
            caps=(1,),
            baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
            checkpoint_dir=checkpoint_root,
            resume=True,
            arm2_selector=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should fail before rerun")),
        )


def test_duplicate_seed_rejected_with_baseline(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline4", seeds=(13,), caps=(1,))
    with pytest.raises(ValueError, match="overlap baseline seeds"):
        _run_bundle(
            tmp_path,
            out_name="extended4",
            seeds=(13,),
            caps=(1,),
            baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        )


def test_normalizes_old_baseline_floor_metadata_and_provenance_and_progress_fields(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline5", seeds=(13,), caps=(1, 2))
    baseline_dir = Path(baseline_result["output_paths"]["manifest"]).parent
    memberships_path = baseline_dir / "memberships.jsonl"
    rows = [json.loads(line) for line in memberships_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutated = []
    for row in rows:
        row2 = dict(row)
        if row2.get("method_id") == METHOD_IDS["arm3"]:
            row2["total_floor_target"] = 0
            row2["floor_prefix_count"] = 0
            row2["floor_complete"] = False
            row2["arm3_floor"] = {"total_floor_target": 0, "floor_complete": False, "floor_prefix_count": 0, "floor_completion_ratio": 0.0}
        mutated.append(row2)
    memberships_path.write_text("".join(json.dumps(row) + "\n" for row in mutated), encoding="utf-8")

    progress_events: list[dict[str, object]] = []
    extended_result, _ = _run_bundle(
        tmp_path,
        out_name="extended5",
        seeds=(14,),
        caps=(1, 2),
        baseline_dir=baseline_dir,
        checkpoint_dir=tmp_path / "ckpt5",
        resume=True,
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    merged_memberships = [json.loads(line) for line in Path(extended_result["output_paths"]["memberships"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    arm3_baseline_rows = [row for row in merged_memberships if row["seed"] == 13 and row["method_id"] == METHOD_IDS["arm3"]]
    assert arm3_baseline_rows
    for row in arm3_baseline_rows:
        assert row["total_floor_target"] >= 1
        assert row["floor_prefix_count"] >= 1
        assert "arm3_floor" in row

    agg = json.loads(Path(extended_result["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    assert agg["config"]["seeds"] == [13, 14]
    cohorts = agg["provenance"]["trial_cohorts"]
    assert cohorts[0]["cohort"] == "baseline"
    assert cohorts[1]["cohort"] == "extension"
    assert cohorts[0]["seeds"] == [13]
    assert cohorts[1]["seeds"] == [14]
    assert cohorts[0]["trial_count"] == 1
    assert cohorts[0]["row_count"] == 10
    assert cohorts[1]["trial_count"] == 1
    assert cohorts[1]["row_count"] == 10
    assert "publisher_code_hashes" in agg["provenance"]
    assert agg["provenance"]["checkpoint_summary"]["total_cells"] == 2

    progress_payload = json.loads((Path(extended_result["output_paths"]["aggregate"]).parent / "progress.json").read_text(encoding="utf-8"))
    assert progress_payload["total_replays"] == 2
    assert progress_payload["total_cells"] == 10
    assert progress_payload["baseline_trials"] == 1
    assert progress_payload["final_trials"] == 2
    assert progress_payload["baseline_rows"] == 10
    assert progress_payload["final_rows"] == 20
    assert any(str(event.get("phase")) == "method-evaluation" for event in progress_events)


def test_resumed_merge_populates_top_five_agent_aggregate(tmp_path: Path):
    baseline_result, _ = _run_bundle(tmp_path, out_name="baseline-top", seeds=(13,), caps=(1,))
    checkpoint_root = tmp_path / "ckpt-top"
    _run_bundle(
        tmp_path,
        out_name="extension-top-first",
        seeds=(14,),
        caps=(1,),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
    )
    resumed_result, _ = _run_bundle(
        tmp_path,
        out_name="extension-top-resumed",
        seeds=(14,),
        caps=(1,),
        baseline_dir=Path(baseline_result["output_paths"]["manifest"]).parent,
        checkpoint_dir=checkpoint_root,
        resume=True,
        arm2_selector=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("selector should not run")
        ),
    )

    aggregate = json.loads(Path(resumed_result["output_paths"]["aggregate"]).read_text(encoding="utf-8"))
    assert aggregate["top_five_agent_aggregate"]
