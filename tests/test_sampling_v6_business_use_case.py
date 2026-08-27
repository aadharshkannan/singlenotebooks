from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from sampling_comparison.v6_business_use_case import (
    BusinessUseCaseClassifier,
    COSINE_SIMILARITY_THRESHOLD,
    LOW_CONFIDENCE_FALLBACK_GUID,
    SessionStepText,
    SessionStepVectors,
    clean_chat_text,
    load_business_use_case_artifacts,
)


GUID_A = UUID("11111111-1111-1111-1111-111111111111")
GUID_B = UUID("22222222-2222-2222-2222-222222222222")
GUID_C = UUID("33333333-3333-3333-3333-333333333333")


def _f32_blob(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _make_taxonomy_db(path: Path, rows: list[tuple[UUID, str, str, str, str, str, str]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sub_subcategories (guid TEXT PRIMARY KEY, name TEXT, business_task TEXT)"
        )
        conn.execute(
            "CREATE TABLE taxonomy_hierarchy (sub_subcategory_guid TEXT, domain TEXT, segment TEXT, category TEXT, subcategory TEXT)"
        )
        for guid, domain, segment, category, sub_category, sub_subcategory, business_task in rows:
            conn.execute(
                "INSERT INTO sub_subcategories (guid, name, business_task) VALUES (?, ?, ?)",
                (str(guid), sub_subcategory, business_task),
            )
            conn.execute(
                "INSERT INTO taxonomy_hierarchy (sub_subcategory_guid, domain, segment, category, subcategory) VALUES (?, ?, ?, ?, ?)",
                (str(guid), domain, segment, category, sub_category),
            )


def _make_centroids_db(
    path: Path,
    request_rows: list[tuple[UUID, list[float]]],
    response_rows: list[tuple[UUID, list[float]]],
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE request_centroids (sub_subcategory_guid TEXT, centroid BLOB)")
        conn.execute("CREATE TABLE pass_response_centroids (sub_subcategory_guid TEXT, centroid BLOB)")

        for guid, values in request_rows:
            conn.execute(
                "INSERT INTO request_centroids (sub_subcategory_guid, centroid) VALUES (?, ?)",
                (str(guid), _f32_blob(values)),
            )

        for guid, values in response_rows:
            conn.execute(
                "INSERT INTO pass_response_centroids (sub_subcategory_guid, centroid) VALUES (?, ?)",
                (str(guid), _f32_blob(values)),
            )


def _build_classifier(tmp_path: Path) -> BusinessUseCaseClassifier:
    taxonomy_db = tmp_path / "taxonomy_v6.db"
    centroids_db = tmp_path / "centroids_v6.db"

    taxonomy_rows = [
        (GUID_A, "d1", "s1", "c1", "sc1", "alpha", "task-a"),
        (GUID_B, "d2", "s2", "c2", "sc2", "beta", "task-b"),
        (GUID_C, "d3", "s3", "c3", "sc3", "gamma", "task-c"),
    ]

    request_rows = [
        (GUID_A, [1.0, 0.0, 0.0]),
        (GUID_B, [0.0, 1.0, 0.0]),
        (GUID_C, [0.0, 0.0, 1.0]),
    ]

    response_rows = [
        (GUID_A, [1.0, 0.0, 0.0]),
        (GUID_B, [0.0, 1.0, 0.0]),
        (GUID_C, [0.0, 0.0, 1.0]),
    ]

    _make_taxonomy_db(taxonomy_db, taxonomy_rows)
    _make_centroids_db(centroids_db, request_rows, response_rows)

    artifacts = load_business_use_case_artifacts(
        centroids_db_path=str(centroids_db),
        taxonomy_db_path=str(taxonomy_db),
    )
    return BusinessUseCaseClassifier(artifacts)


def test_load_artifacts_decodes_blobs_and_exposes_metadata(tmp_path: Path) -> None:
    taxonomy_db = tmp_path / "taxonomy_v6.db"
    centroids_db = tmp_path / "centroids_v6.db"

    _make_taxonomy_db(
        taxonomy_db,
        [
            (GUID_A, "d1", "s1", "c1", "sc1", "alpha", "task-a"),
            (GUID_B, "d2", "s2", "c2", "sc2", "beta", "task-b"),
        ],
    )
    _make_centroids_db(
        centroids_db,
        request_rows=[(GUID_A, [1.5, 2.5]), (GUID_B, [3.5, 4.5])],
        response_rows=[(GUID_A, [1.5, 2.5]), (GUID_B, [3.5, 4.5])],
    )

    artifacts = load_business_use_case_artifacts(
        centroids_db_path=str(centroids_db),
        taxonomy_db_path=str(taxonomy_db),
    )

    assert artifacts.metadata.taxonomy_version == "6.0"
    assert artifacts.metadata.dimensions == 2
    assert artifacts.metadata.taxonomy_count == 2
    assert artifacts.metadata.request_centroid_count == 2
    assert artifacts.metadata.response_centroid_count == 2
    assert artifacts.metadata.centroids_db_path == str(centroids_db)
    assert artifacts.metadata.taxonomy_db_path == str(taxonomy_db)
    assert len(artifacts.metadata.centroids_db_sha256) == 64
    assert len(artifacts.metadata.taxonomy_db_sha256) == 64
    assert artifacts.input_centroids[0].embedding.dtype == np.dtype("<f4")
    assert artifacts.input_centroids[0].embedding.tolist() == [1.5, 2.5]


def test_load_artifacts_validates_guid_and_dimension_consistency(tmp_path: Path) -> None:
    taxonomy_db = tmp_path / "taxonomy_v6.db"
    centroids_db = tmp_path / "centroids_v6.db"

    _make_taxonomy_db(
        taxonomy_db,
        [
            (GUID_A, "d1", "s1", "c1", "sc1", "alpha", "task-a"),
            (GUID_B, "d2", "s2", "c2", "sc2", "beta", "task-b"),
        ],
    )

    _make_centroids_db(
        centroids_db,
        request_rows=[(GUID_A, [1.0, 0.0]), (GUID_B, [0.0, 1.0])],
        response_rows=[(GUID_A, [1.0, 0.0, 0.0]), (GUID_B, [0.0, 1.0, 0.0])],
    )

    with pytest.raises(ValueError, match="dimensions"):
        load_business_use_case_artifacts(
            centroids_db_path=str(centroids_db),
            taxonomy_db_path=str(taxonomy_db),
        )


def test_cleaner_extracts_nested_json_strips_tags_and_enforces_token_limit() -> None:
    raw = (
        '[{"role":"user","content":"{\\"message\\":\\"Hi <p>there</p> 😀\\nline2\\t\\"}"},'
        '{"type":"meta","text":"<final_answer>ignored-tag</final_answer>"}]'
    )

    cleaned = clean_chat_text(raw, token_counter=lambda _text: 2)
    assert cleaned == "Hi there line2 ignored-tag"

    too_long = clean_chat_text("abc def", token_counter=lambda _text: 9000)
    assert too_long == ""


def test_agreement_selection_when_same_top1(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    determination = classifier.determine_from_vectors(
        input_vector=[1.0, 0.0, 0.0],
        output_vector=[1.0, 0.0, 0.0],
    )

    assert determination.guid == GUID_A
    assert determination.status == "Agree"
    assert determination.input_matches[0].guid == GUID_A
    assert determination.output_matches[0].guid == GUID_A


def test_corroborated_selection_prefers_first_input_top_present_in_output(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    determination = classifier.determine_from_vectors(
        input_vector=[0.8, 0.6, 0.0],
        output_vector=[0.6, 0.8, 0.0],
    )

    assert determination.status == "Corroborated"
    assert determination.guid == GUID_A


def test_ambiguous_selection_uses_combined_minimum_distance(tmp_path: Path) -> None:
    taxonomy_db = tmp_path / "taxonomy_v6.db"
    centroids_db = tmp_path / "centroids_v6.db"

    all_guids = [UUID(f"00000000-0000-0000-0000-{index:012d}") for index in range(1, 12)]
    taxonomy_rows = [
        (guid, f"d{idx}", f"s{idx}", f"c{idx}", f"sc{idx}", f"name-{idx}", f"task-{idx}")
        for idx, guid in enumerate(all_guids, start=1)
    ]
    _make_taxonomy_db(taxonomy_db, taxonomy_rows)

    # Force disjoint input/output top-5 sets and include one shared fallback candidate
    # ranked just below top-5 in both fields, with similarity > 0.30 on both sides.
    request_rows = [(guid, [1.0, 0.0, 0.0]) for guid in all_guids[:5]] + [
        (guid, [-1.0, 0.0, 0.0]) for guid in all_guids[5:]
    ]
    response_rows = [(guid, [-1.0, 0.0, 0.0]) for guid in all_guids[:5]] + [
        (guid, [1.0, 0.0, 0.0]) for guid in all_guids[5:]
    ]
    request_rows[-1] = (all_guids[-1], [0.4, 0.0, 0.0])
    response_rows[-1] = (all_guids[-1], [0.4, 0.0, 0.0])
    _make_centroids_db(
        centroids_db,
        request_rows=request_rows,
        response_rows=response_rows,
    )

    classifier = BusinessUseCaseClassifier(
        load_business_use_case_artifacts(
            centroids_db_path=str(centroids_db),
            taxonomy_db_path=str(taxonomy_db),
        )
    )

    determination = classifier.determine_from_vectors(
        input_vector=[1.0, 0.0, 0.0],
        output_vector=[1.0, 0.0, 0.0],
    )

    assert determination.status == "Ambiguous"
    assert determination.guid == all_guids[-1]
    assert determination.combined_best is not None
    assert determination.combined_best.guid == all_guids[-1]


def test_low_confidence_fallback_below_threshold(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    # Opposite direction to all best candidates => similarity < 0.30.
    determination = classifier.determine_from_vectors(
        input_vector=[-1.0, -1.0, -1.0],
        output_vector=[-1.0, -1.0, -1.0],
    )

    assert determination.combined_cosine_similarity < COSINE_SIMILARITY_THRESHOLD
    assert determination.guid == LOW_CONFIDENCE_FALLBACK_GUID
    assert determination.confidence_level == 0


def test_single_field_behavior_and_reason(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    determination = classifier.determine_from_vectors(
        input_vector=[0.0, 0.0, 1.0],
        output_vector=None,
    )

    assert determination.guid == GUID_C
    assert determination.status == "Ambiguous"
    assert "input field only" in determination.reason
    assert len(determination.output_matches) == 0


def test_session_classification_stops_early_at_threshold(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    selection = classifier.classify_session_steps_from_vectors(
        [
            SessionStepVectors(request_vector=[1.0, 1.0, 1.0], response_vector=[1.0, 1.0, 1.0]),
            SessionStepVectors(request_vector=[1.0, 0.0, 0.0], response_vector=[1.0, 0.0, 0.0]),
            SessionStepVectors(request_vector=[0.0, 1.0, 0.0], response_vector=[0.0, 1.0, 0.0]),
        ]
    )

    assert selection is not None
    assert selection.step_index == 1
    assert selection.provenance == "threshold"
    assert selection.determination.combined_cosine_similarity >= 0.70


def test_session_classification_uses_best_similarity_when_no_threshold_hit(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    selection = classifier.classify_session_steps_from_vectors(
        [
            SessionStepVectors(request_vector=[1.0, 1.0, 1.0], response_vector=[1.0, 1.0, 1.0]),
            SessionStepVectors(request_vector=[1.0, 1.0, 0.8], response_vector=[1.0, 1.0, 0.8]),
        ]
    )

    assert selection is not None
    assert selection.step_index == 1
    assert selection.provenance == "max_similarity"


def test_unique_clean_text_enumeration_and_classify_from_mapping(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    steps = [
        SessionStepText(request='{"content":"Hello"}', response='{"content":"World"}'),
        SessionStepText(request='{"content":"Hello"}', response='{"content":"World"}'),
        SessionStepText(request='{"content":""}', response=None),
    ]

    unique = classifier.enumerate_unique_clean_texts(steps=steps, token_counter=lambda _text: 1)
    assert unique == ("Hello", "World")

    selection = classifier.classify_session_from_text_embeddings(
        steps=steps,
        embeddings_by_text={"Hello": [1.0, 0.0, 0.0], "World": [1.0, 0.0, 0.0]},
        token_counter=lambda _text: 1,
    )
    assert selection is not None
    assert selection.step_index == 0


def test_classify_from_mapping_raises_for_missing_embedding(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    steps = [SessionStepText(request='{"content":"MissingMe"}', response=None)]

    with pytest.raises(KeyError, match="Missing embedding"):
        classifier.classify_session_from_text_embeddings(
            steps=steps,
            embeddings_by_text={},
            token_counter=lambda _text: 1,
        )


def test_classify_sessions_from_text_embeddings_matches_single_session_calls(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    sessions = {
        "unit-a": [
            SessionStepText(request='{"content":"Hello"}', response='{"content":"World"}'),
            SessionStepText(request='{"content":"Ignored"}', response=None),
        ],
        "unit-b": [
            SessionStepText(request='{"content":"World"}', response='{"content":"Hello"}'),
        ],
    }
    embeddings = {
        "Hello": [1.0, 0.0, 0.0],
        "World": [1.0, 0.0, 0.0],
        "Ignored": [0.0, 1.0, 0.0],
    }

    batch = classifier.classify_sessions_from_text_embeddings(
        sessions_by_unit_id=sessions,
        embeddings_by_text=embeddings,
        token_counter=lambda _text: 1,
    )

    single_a = classifier.classify_session_from_text_embeddings(
        steps=sessions["unit-a"],
        embeddings_by_text=embeddings,
        token_counter=lambda _text: 1,
    )
    single_b = classifier.classify_session_from_text_embeddings(
        steps=sessions["unit-b"],
        embeddings_by_text=embeddings,
        token_counter=lambda _text: 1,
    )

    assert batch["unit-a"] == single_a
    assert batch["unit-b"] == single_b


def test_loader_rejects_nonfinite_and_zero_norm_centroids(tmp_path: Path) -> None:
    taxonomy_db = tmp_path / "taxonomy_v6.db"
    centroids_db = tmp_path / "centroids_v6.db"

    _make_taxonomy_db(
        taxonomy_db,
        [
            (GUID_A, "d1", "s1", "c1", "sc1", "alpha", "task-a"),
            (GUID_B, "d2", "s2", "c2", "sc2", "beta", "task-b"),
        ],
    )

    _make_centroids_db(
        centroids_db,
        request_rows=[(GUID_A, [1.0, 0.0]), (GUID_B, [0.0, 1.0])],
        response_rows=[(GUID_A, [float("nan"), 0.0]), (GUID_B, [0.0, 1.0])],
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_business_use_case_artifacts(
            centroids_db_path=str(centroids_db),
            taxonomy_db_path=str(taxonomy_db),
        )

    centroids_db_zero = tmp_path / "centroids_v6_zero.db"
    _make_centroids_db(
        centroids_db_zero,
        request_rows=[(GUID_A, [1.0, 0.0]), (GUID_B, [0.0, 1.0])],
        response_rows=[(GUID_A, [0.0, 0.0]), (GUID_B, [0.0, 1.0])],
    )
    with pytest.raises(ValueError, match="zero norm"):
        load_business_use_case_artifacts(
            centroids_db_path=str(centroids_db_zero),
            taxonomy_db_path=str(taxonomy_db),
        )


def test_classifier_rejects_query_dimension_mismatch_and_nonfinite(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    with pytest.raises(ValueError, match="dimensions mismatch"):
        classifier.determine_from_vectors(
            input_vector=[1.0, 0.0],
            output_vector=[1.0, 0.0, 0.0],
        )

    with pytest.raises(ValueError, match="non-finite"):
        classifier.determine_from_vectors(
            input_vector=[1.0, float("inf"), 0.0],
            output_vector=[1.0, 0.0, 0.0],
        )


def test_zero_norm_query_uses_low_confidence_fallback(tmp_path: Path) -> None:
    classifier = _build_classifier(tmp_path)

    determination = classifier.determine_from_vectors(
        input_vector=[0.0, 0.0, 0.0],
        output_vector=[1.0, 0.0, 0.0],
    )

    assert determination.guid == LOW_CONFIDENCE_FALLBACK_GUID
    assert determination.confidence_level == 0


def test_real_maven_artifact_audit_counts_if_present() -> None:
    centroids_db = Path(
        r"C:\Users\stangoodwin\mvn-mavenservice\src\MVN\Kairo\MachineLearning\BusinessUseCase\data\centroids_v6.db"
    )
    taxonomy_db = Path(
        r"C:\Users\stangoodwin\mvn-mavenservice\src\MVN\Kairo\MachineLearning\BusinessUseCase\data\taxonomy_v6.db"
    )

    if not centroids_db.exists() or not taxonomy_db.exists():
        pytest.skip("Real Maven v6 artifacts are not present on this machine")

    artifacts = load_business_use_case_artifacts(
        centroids_db_path=str(centroids_db),
        taxonomy_db_path=str(taxonomy_db),
    )

    assert artifacts.metadata.request_centroid_count == 6947
    assert artifacts.metadata.response_centroid_count == 6947
    assert artifacts.metadata.dimensions == 1536
    assert artifacts.metadata.taxonomy_count == 6948
