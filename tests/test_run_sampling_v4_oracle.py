from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_sampling_v4_oracle as cli
from sampling_comparison.v4_idw import IDWConfig, freeze_membership


class _Runtime:
    def __init__(self, vectors: dict[int, np.ndarray]):
        self.embedding_vector_by_trace_id = vectors
        self.ledger = SimpleNamespace(
            packet_builds=4,
            packet_cache_hits=0,
            embedding_calls=1,
            embedding_inputs=4,
            embedding_input_tokens=16,
            embedding_latency_seconds=0.01,
            embedding_content_hashes=("a", "b"),
            embedding_model_id="text-embedding-3-small",
            embedding_deployment_id="embed-dep",
            embedding_embedder_class="Deterministic1536Embedder",
        )


class _Trace:
    def __init__(self, trace_id: int, agent_id: str):
        self.trace_id = trace_id
        self.agent_id = agent_id


class _Data:
    def __init__(self):
        self.unit_ids = ("u1", "u2", "u3", "u4")
        self.trace_by_unit_id = {
            "u1": _Trace(1, "t|a"),
            "u2": _Trace(2, "t|a"),
            "u3": _Trace(3, "t|a"),
            "u4": _Trace(4, "t|b"),
        }
        self.labels_by_unit = {
            "u1": True,
            "u2": False,
            "u3": True,
            "u4": False,
        }


def _selected_membership(method: str = "adaptive_embedding_fullsession_token", pop: int = 4):
    return {
        "version": "sampling-v3-selected-membership-v1",
        "legacy_tier_pct_provenance": 20,
        "membership": {
            "method": method,
            "selected_ids": ["u1", "u2"],
            "selected_count": 2,
            "population_count": pop,
        },
    }


def test_rejects_wrong_method_or_population_count_mismatch():
    data = _Data()
    runtime = _Runtime(
        {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
            3: np.asarray([0.5, 0.5], dtype=np.float32),
            4: np.asarray([1.0, 0.0], dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="method"):
        cli.compute_v4_oracle(
            data=data,
            runtime=runtime,
            selected_membership_payload=_selected_membership(method="random_sampling_token_priority"),
            aggregate_population_count=4,
            v3_manifest_hash="m" * 64,
            v3_report_hash="r" * 64,
            idw_config=IDWConfig(),
        )

    with pytest.raises(ValueError, match="population_count"):
        cli.compute_v4_oracle(
            data=data,
            runtime=runtime,
            selected_membership_payload=_selected_membership(),
            aggregate_population_count=3,
            v3_manifest_hash="m" * 64,
            v3_report_hash="r" * 64,
            idw_config=IDWConfig(),
        )


def test_judged_values_passed_only_for_selected(monkeypatch):
    data = _Data()
    runtime = _Runtime(
        {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
            3: np.asarray([0.5, 0.5], dtype=np.float32),
            4: np.asarray([1.0, 0.0], dtype=np.float32),
        }
    )

    seen: dict[str, object] = {}

    def _fake_estimate(*, membership, judged_values_by_unit, **kwargs):
        seen["membership"] = membership
        seen["judged"] = dict(judged_values_by_unit)
        return SimpleNamespace(
            membership=membership,
            aggregate=SimpleNamespace(
                population_count=4,
                observed_count=2,
                imputed_count=2,
                estimated_pass_rate=0.55,
                provenance_counts={"observed": 2, "idw": 2},
                provenance_population_weighted_rates={"observed": 0.25, "idw": 0.30},
                zero_donor_agent_count=0,
                prior_count=0,
            ),
            rows=(),
            agent_id_by_unit={uid: data.trace_by_unit_id[uid].agent_id for uid in data.unit_ids},
            judged_unit_ids=frozenset(judged_values_by_unit.keys()),
        )

    def _fake_validate(estimates, expected_labels):
        assert set(expected_labels) == set(data.unit_ids)
        return SimpleNamespace(
            census_pass_rate=0.5,
            absolute_aggregate_rate_error=0.05,
            per_unit_mae=0.2,
            brier_score=0.1,
            macro_per_agent_mae=0.3,
            judged_only_pass_rate=0.5,
            judged_only_absolute_rate_error=0.1,
            unjudged_only_mae=0.25,
            unjudged_only_brier=0.2,
            calibration_bins=(),
            expected_calibration_error=0.04,
        )

    def _fake_loo(**kwargs):
        return SimpleNamespace(per_unit_predictions={}, mae=0.12, brier_score=0.08)

    monkeypatch.setattr(cli, "estimate_embedding_population", _fake_estimate)
    monkeypatch.setattr(cli, "validate_embedding_population", _fake_validate)
    monkeypatch.setattr(cli, "leave_one_out_donor_diagnostics", _fake_loo)

    out = cli.compute_v4_oracle(
        data=data,
        runtime=runtime,
        selected_membership_payload=_selected_membership(),
        aggregate_population_count=4,
        v3_manifest_hash="m" * 64,
        v3_report_hash="r" * 64,
        idw_config=IDWConfig(),
    )

    assert set(seen["judged"].keys()) == {"u1", "u2"}
    assert set(seen["judged"].keys()).issubset(set(data.unit_ids))
    assert out["counts"]["selected_count"] == 2


def test_idw_delta_and_output_excludes_sensitive_maps():
    data = _Data()
    runtime = _Runtime(
        {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
            3: np.asarray([0.5, 0.5], dtype=np.float32),
            4: np.asarray([1.0, 0.0], dtype=np.float32),
        }
    )

    out = cli.compute_v4_oracle(
        data=data,
        runtime=runtime,
        selected_membership_payload=_selected_membership(),
        aggregate_population_count=4,
        v3_manifest_hash="m" * 64,
        v3_report_hash="r" * 64,
        idw_config=IDWConfig(k=1),
    )

    rates = out["rates"]
    assert rates["delta_aggregate_mae_idw_minus_selected_only"] == pytest.approx(
        rates["idw_abs_error"] - rates["selected_only_abs_error"]
    )

    serialized = str(out)
    assert "vector_by_unit" not in serialized
    assert "canonical_json" not in serialized
    assert "labels_by_unit" not in serialized
    assert "donor_ids" not in serialized


def test_parser_defaults():
    parser = cli._build_parser()
    args = parser.parse_args([])
    assert args.v3_dir.replace("\\", "/") == "outputs_sampling_v3/v3"
    assert args.output.replace("\\", "/") == "outputs_sampling_v4/runs/oracle-20.json"
    assert args.embedding_batch_size == 32
    assert args.idw_k == 8
    assert args.idw_power == 2.0
    assert args.idw_eps == 1e-6
    assert args.idw_exact_cosine_eps == 1e-8
    assert args.idw_prior == 0.5
