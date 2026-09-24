import copy
import json

import pytest

from sampling_comparison.matryoshka_experiment import sha256_file, write_json
from scripts.validate_imdb_experiment import validate_extension


def extension_fixture(tmp_path):
    baseline, output = tmp_path / "baseline", tmp_path / "extension"
    baseline.mkdir()
    output.mkdir()
    (baseline / "cells").mkdir()
    (baseline / "cells" / "native.npz").write_bytes(b"fixture evidence")
    row = {
        "seed": 13, "schedule": "bursty", "dimension": 1536, "rate": .05,
        "evidence": "cells/native.npz", "replay_hashes": {"draw": "same", "order": "paired"},
        "all_unselected": {"mae": .25},
    }
    write_json(baseline / "aggregate.json", {"rows": [row]})
    write_json(baseline / "manifest.json", {"fixture": True})
    aggregate_sha = sha256_file(baseline / "aggregate.json")
    current = {**row, "evidence": "../baseline/cells/native.npz"}
    extra = {**row, "dimension": 128, "evidence": "cells/extra.npz"}
    extension = {
        "baseline_aggregate_sha256": aggregate_sha,
        "baseline_manifest_sha256": sha256_file(baseline / "manifest.json"),
        "reused_cells": 1, "added_cells": 1, "baseline_rows_unchanged": True,
        "replay_pairing_exact": True, "embedding_calls": 0,
    }
    aggregate = {"rows": [copy.deepcopy(current), copy.deepcopy(extra)], "extension": extension}
    manifest = {"files": {"../baseline/aggregate.json": aggregate_sha}}
    return aggregate, manifest, output


def test_original_rows_and_draws_remain_exact(tmp_path):
    aggregate, manifest, output = extension_fixture(tmp_path)
    result = validate_extension(aggregate, manifest, output)
    assert result["reused_cells"] == result["added_cells"] == 1
    assert result["baseline_rows_unchanged"] and result["replay_pairing_exact"]
    assert result["embedding_calls"] == 0


@pytest.mark.parametrize(("mutation", "message"), [
    ("metric", "original measured row changed"),
    ("path", "evidence was replaced"),
    ("pairing", "not paired"),
    ("source", "original aggregate changed"),
    ("embedding", "provenance claim"),
])
def test_extension_proof_rejects_incompatible_evidence(tmp_path, mutation, message):
    aggregate, manifest, output = extension_fixture(tmp_path)
    if mutation == "metric":
        aggregate["rows"][0]["all_unselected"]["mae"] = .9
    elif mutation == "path":
        aggregate["rows"][0]["evidence"] = "different/native.npz"
    elif mutation == "pairing":
        aggregate["rows"][1]["replay_hashes"]["order"] = "different"
    elif mutation == "source":
        path = tmp_path / "baseline" / "aggregate.json"
        value = json.loads(path.read_text())
        value["changed"] = True
        write_json(path, value)
    elif mutation == "embedding":
        aggregate["extension"]["embedding_calls"] = 1
    with pytest.raises(ValueError, match=message):
        validate_extension(aggregate, manifest, output)
