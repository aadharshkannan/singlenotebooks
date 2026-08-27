from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
import re
import sqlite3
from typing import Callable, Mapping, Sequence
import unicodedata
from uuid import UUID

import numpy as np


TAXONOMY_VERSION = "6.0"
LOW_CONFIDENCE_FALLBACK_GUID = UUID("9a6df217-0865-486d-93da-519ebcd37a70")
LOW_CONFIDENCE_FALLBACK_LABEL = "Undetermined - low match confidence"
COSINE_SIMILARITY_THRESHOLD = 0.30
SESSION_SIMILARITY_EARLY_STOP_THRESHOLD = 0.70
TOP_N = 5
MAX_INPUT_TOKENS = 8191
_PRIORITY_KEYS = ("content", "text", "value", "message")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z!][^>]*>", flags=re.ASCII)

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class BusinessUseCaseInfo:
    guid: UUID
    domain: str
    segment: str
    category: str
    sub_category: str
    sub_subcategory: str
    business_task: str = ""


@dataclass(frozen=True)
class BusinessUseCaseMatch:
    guid: UUID
    domain: str
    segment: str
    category: str
    sub_category: str
    sub_subcategory: str
    business_task: str
    cosine_distance: float
    euclidean_distance: float


@dataclass(frozen=True)
class BusinessUseCaseDetermination:
    guid: UUID | None
    use_case: BusinessUseCaseInfo | None
    status: str
    reason: str
    combined_cosine_similarity: float
    input_matches: tuple[BusinessUseCaseMatch, ...]
    output_matches: tuple[BusinessUseCaseMatch, ...]
    combined_best: BusinessUseCaseMatch | None

    @property
    def confidence_level(self) -> int:
        if self.guid is None or self.guid == LOW_CONFIDENCE_FALLBACK_GUID:
            return 0
        if self.status == "Agree":
            return 3
        if self.status == "Corroborated":
            return 2
        return 1

    @property
    def taxonomy_version(self) -> str:
        return TAXONOMY_VERSION


@dataclass(frozen=True)
class ArtifactMetadata:
    taxonomy_version: str
    dimensions: int
    taxonomy_count: int
    request_centroid_count: int
    response_centroid_count: int
    taxonomy_db_path: str
    taxonomy_db_sha256: str
    centroids_db_path: str
    centroids_db_sha256: str


@dataclass(frozen=True)
class _Centroid:
    use_case_guid: UUID
    embedding: np.ndarray


@dataclass(frozen=True)
class BusinessUseCaseArtifacts:
    input_centroids: tuple[_Centroid, ...]
    output_centroids: tuple[_Centroid, ...]
    use_cases: Mapping[UUID, BusinessUseCaseInfo]
    metadata: ArtifactMetadata


@dataclass(frozen=True)
class SessionStepText:
    request: str | None = None
    response: str | None = None


@dataclass(frozen=True)
class SessionStepVectors:
    request_vector: Sequence[float] | None = None
    response_vector: Sequence[float] | None = None


@dataclass(frozen=True)
class SessionSelection:
    step_index: int
    provenance: str
    determination: BusinessUseCaseDetermination


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _to_float_array(blob: bytes) -> np.ndarray:
    if len(blob) % 4 != 0:
        raise ValueError(f"Embedding BLOB length {len(blob)} is not a multiple of 4 bytes")
    return np.frombuffer(blob, dtype="<f4")


def load_business_use_case_artifacts(*, centroids_db_path: str, taxonomy_db_path: str) -> BusinessUseCaseArtifacts:
    taxonomy = _load_taxonomy(taxonomy_db_path)
    input_centroids = _load_centroids(centroids_db_path, "request_centroids")
    output_centroids = _load_centroids(centroids_db_path, "pass_response_centroids")

    _validate_centroid_guids_and_dimensions(taxonomy, input_centroids, output_centroids)

    dimensions = input_centroids[0].embedding.size if input_centroids else 0
    metadata = ArtifactMetadata(
        taxonomy_version=TAXONOMY_VERSION,
        dimensions=dimensions,
        taxonomy_count=len(taxonomy),
        request_centroid_count=len(input_centroids),
        response_centroid_count=len(output_centroids),
        taxonomy_db_path=taxonomy_db_path,
        taxonomy_db_sha256=_sha256_file(taxonomy_db_path),
        centroids_db_path=centroids_db_path,
        centroids_db_sha256=_sha256_file(centroids_db_path),
    )

    return BusinessUseCaseArtifacts(
        input_centroids=tuple(input_centroids),
        output_centroids=tuple(output_centroids),
        use_cases=taxonomy,
        metadata=metadata,
    )


def _load_taxonomy(database_path: str) -> dict[UUID, BusinessUseCaseInfo]:
    query = (
        "SELECT s.guid, h.domain, h.segment, h.category, h.subcategory, s.name, s.business_task "
        "FROM sub_subcategories AS s "
        "INNER JOIN taxonomy_hierarchy AS h ON h.sub_subcategory_guid = s.guid"
    )

    use_cases: dict[UUID, BusinessUseCaseInfo] = {}
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(query)
        for row in cursor:
            guid = UUID(str(row[0]))
            if guid in use_cases:
                raise ValueError(f"Duplicate taxonomy row for GUID '{guid}'")
            use_cases[guid] = BusinessUseCaseInfo(
                guid=guid,
                domain="" if row[1] is None else str(row[1]),
                segment="" if row[2] is None else str(row[2]),
                category="" if row[3] is None else str(row[3]),
                sub_category="" if row[4] is None else str(row[4]),
                sub_subcategory="" if row[5] is None else str(row[5]),
                business_task="" if row[6] is None else str(row[6]),
            )
    return use_cases


def _load_centroids(database_path: str, table_name: str) -> list[_Centroid]:
    query = f"SELECT sub_subcategory_guid, centroid FROM {table_name}"
    centroids: list[_Centroid] = []
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(query)
        for row in cursor:
            guid = UUID(str(row[0]))
            blob = bytes(row[1])
            vector = _to_float_array(blob)
            centroids.append(_Centroid(use_case_guid=guid, embedding=vector))
    return centroids


def _validate_centroid_guids_and_dimensions(
    taxonomy: Mapping[UUID, BusinessUseCaseInfo],
    input_centroids: Sequence[_Centroid],
    output_centroids: Sequence[_Centroid],
) -> None:
    for centroid in (*input_centroids, *output_centroids):
        if centroid.use_case_guid not in taxonomy:
            raise ValueError(f"Centroid GUID '{centroid.use_case_guid}' is missing from taxonomy")

        vector = np.asarray(centroid.embedding)
        if vector.ndim != 1:
            raise ValueError(f"Centroid GUID '{centroid.use_case_guid}' is not one-dimensional")
        if not np.isfinite(vector).all():
            raise ValueError(f"Centroid GUID '{centroid.use_case_guid}' contains non-finite values")

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(f"Centroid GUID '{centroid.use_case_guid}' has zero norm")

    input_guids = {item.use_case_guid for item in input_centroids}
    output_guids = {item.use_case_guid for item in output_centroids}
    if input_guids != output_guids:
        missing_from_output = input_guids.difference(output_guids)
        missing_from_input = output_guids.difference(input_guids)
        raise ValueError(
            "Input/output centroid GUID sets differ "
            f"(missing_from_output={sorted(str(x) for x in missing_from_output)}, "
            f"missing_from_input={sorted(str(x) for x in missing_from_input)})"
        )

    dimensions = {item.embedding.size for item in (*input_centroids, *output_centroids)}
    if len(dimensions) > 1:
        raise ValueError(f"Centroid dimensions do not match: {sorted(dimensions)}")


def default_token_counter(text: str) -> int:
    return len(text.split())


def clean_chat_text(
    raw: str | None,
    *,
    token_counter: TokenCounter | None = None,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> str:
    if raw is None or raw.strip() == "":
        return ""

    text = raw.strip()

    if text[0] in "[{":
        extracted = _try_extract_json_text(text)
        if extracted is not None:
            text = extracted

    if "\\" in text:
        text = (
            text.replace("\\r\\n", " ")
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\t", " ")
        )

    text = _HTML_TAG_RE.sub(" ", text)
    text = _strip_unicode_and_collapse(text)

    count_tokens = token_counter or default_token_counter
    token_count = int(count_tokens(text))
    if token_count > max_input_tokens:
        return ""

    return text


def _try_extract_json_text(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    parts: list[str] = []
    _extract_json_text(payload, parts)
    return " ".join(parts)


def _extract_json_text(value: object, parts: list[str]) -> None:
    if isinstance(value, str):
        _append_string_or_json(value, parts)
        return

    if isinstance(value, list):
        for item in value:
            _extract_json_text(item, parts)
        return

    if isinstance(value, dict):
        for key in _PRIORITY_KEYS:
            if key in value:
                _extract_json_text(value[key], parts)
                return

        recursed = False
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                _extract_json_text(nested, parts)
                recursed = True

        if recursed:
            return

        for scalar in value.values():
            if isinstance(scalar, str):
                _append_string_or_json(scalar, parts)


def _append_string_or_json(value: str, parts: list[str]) -> None:
    if value == "":
        return

    stripped = value.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        nested = _try_extract_json_text(value)
        if nested is None:
            # The first JSON parse can decode escaped control chars into literal
            # newlines/tabs, which are invalid in raw JSON text. Retry with
            # escaped controls to preserve nested JSON extraction behavior.
            escaped = value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
            nested = _try_extract_json_text(escaped)
        if nested is not None:
            if nested != "":
                parts.append(nested)
            return

    parts.append(value)


def _strip_unicode_and_collapse(text: str) -> str:
    chars: list[str] = []
    pending_space = False

    for char in text:
        code = ord(char)
        if 0x20 < code <= 0x7E:
            if pending_space and chars:
                chars.append(" ")
            chars.append(char)
            pending_space = False
            continue

        if char == " " or char.isspace() or unicodedata.category(char) == "Cc":
            pending_space = True

    return "".join(chars)


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional embedding vector")
    return array


def _cosine_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    query_norm = float(np.linalg.norm(query))
    candidate_norm = float(np.linalg.norm(candidate))
    if query_norm == 0.0 or candidate_norm == 0.0:
        return math.nan
    return float(np.dot(query, candidate) / (query_norm * candidate_norm))


def _euclidean_from_similarity(similarity: float) -> float:
    if math.isnan(similarity):
        return math.nan
    return math.sqrt(max(0.0, 2.0 - (2.0 * similarity)))


@dataclass(frozen=True)
class _FieldScore:
    rank_key: float
    cosine_distance: float
    euclidean_distance: float


@dataclass(frozen=True)
class _PreparedCentroids:
    guids: tuple[UUID, ...]
    index_by_guid: Mapping[UUID, int]
    normalized_matrix: np.ndarray


@dataclass(frozen=True)
class _SideScores:
    guids: tuple[UUID, ...]
    index_by_guid: Mapping[UUID, int]
    rank_key: np.ndarray
    cosine_distance: np.ndarray
    euclidean_distance: np.ndarray


class BusinessUseCaseClassifier:
    def __init__(self, artifacts: BusinessUseCaseArtifacts) -> None:
        self._artifacts = artifacts
        self._input_prepared = self._prepare_centroids(
            artifacts.input_centroids,
            expected_count=artifacts.metadata.request_centroid_count,
            side_name="input",
        )
        self._output_prepared = self._prepare_centroids(
            artifacts.output_centroids,
            expected_count=artifacts.metadata.response_centroid_count,
            side_name="output",
        )

    def _prepare_centroids(
        self,
        centroids: Sequence[_Centroid],
        *,
        expected_count: int,
        side_name: str,
    ) -> _PreparedCentroids:
        dimensions = self._artifacts.metadata.dimensions
        if len(centroids) != expected_count:
            raise ValueError(
                f"Unexpected {side_name} centroid count: expected {expected_count}, got {len(centroids)}"
            )

        if dimensions == 0 and centroids:
            raise ValueError("Centroid metadata dimensions are zero while centroid rows are present")

        matrix = np.empty((len(centroids), dimensions), dtype=np.float64)
        guids: list[UUID] = []
        index_by_guid: dict[UUID, int] = {}

        for index, centroid in enumerate(centroids):
            vector = np.asarray(centroid.embedding, dtype=np.float64)
            if vector.ndim != 1:
                raise ValueError(f"Expected one-dimensional {side_name} centroid at index {index}")
            if vector.size != dimensions:
                raise ValueError(
                    f"{side_name.capitalize()} centroid dimension mismatch at index {index}: "
                    f"expected {dimensions}, got {vector.size}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"{side_name.capitalize()} centroid at index {index} contains non-finite values")

            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError(f"{side_name.capitalize()} centroid at index {index} has zero norm")

            guid = centroid.use_case_guid
            if guid in index_by_guid:
                raise ValueError(f"Duplicate {side_name} centroid GUID '{guid}'")

            guids.append(guid)
            index_by_guid[guid] = index
            matrix[index, :] = vector / norm

        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        matrix.setflags(write=False)
        return _PreparedCentroids(
            guids=tuple(guids),
            index_by_guid=MappingProxyType(index_by_guid),
            normalized_matrix=matrix,
        )

    def _normalize_query_vector(self, vector: Sequence[float], *, side_name: str) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"Expected a one-dimensional {side_name} query embedding vector")

        expected_dimensions = self._artifacts.metadata.dimensions
        if array.size != expected_dimensions:
            raise ValueError(
                f"{side_name.capitalize()} query embedding dimensions mismatch: "
                f"expected {expected_dimensions}, got {array.size}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"{side_name.capitalize()} query embedding contains non-finite values")
        return array

    def determine_from_vectors(
        self,
        *,
        input_vector: Sequence[float] | None,
        output_vector: Sequence[float] | None,
    ) -> BusinessUseCaseDetermination:
        if input_vector is None and output_vector is None:
            raise ValueError("At least one of input_vector or output_vector must be provided")

        if output_vector is None:
            return self._determine_single_field("input", input_vector)
        if input_vector is None:
            return self._determine_single_field("output", output_vector)

        input_scores = self._score(self._input_prepared, input_vector, side_name="input")
        output_scores = self._score(self._output_prepared, output_vector, side_name="output")

        input_top = self._rank_matches(input_scores)[:TOP_N]
        output_top = self._rank_matches(output_scores)[:TOP_N]
        combined = self._combined_best(input_scores, output_scores)

        guid, status, reason = self._choose_agreement(input_top, output_top, combined)
        combined_similarity = self._combined_cosine_similarity(guid, input_scores, output_scores)

        final_guid = guid
        final_use_case = self._artifacts.use_cases.get(guid) if guid is not None else None

        if guid is not None and (
            not math.isfinite(combined_similarity)
            or combined_similarity < COSINE_SIMILARITY_THRESHOLD
        ):
            final_guid = LOW_CONFIDENCE_FALLBACK_GUID
            final_use_case = self._low_confidence_use_case()
            reason = (
                f"Combined cosine similarity {combined_similarity:.4f} is below the threshold "
                f"{COSINE_SIMILARITY_THRESHOLD:.4f}; the use case cannot be determined."
            )

        return BusinessUseCaseDetermination(
            guid=final_guid,
            use_case=final_use_case,
            status=status,
            reason=reason,
            combined_cosine_similarity=combined_similarity,
            input_matches=tuple(input_top),
            output_matches=tuple(output_top),
            combined_best=combined,
        )

    def classify_session_steps_from_vectors(self, steps: Sequence[SessionStepVectors]) -> SessionSelection | None:
        determinations: list[tuple[int, BusinessUseCaseDetermination]] = []

        for index, step in enumerate(steps):
            if step.request_vector is None and step.response_vector is None:
                continue

            determination = self.determine_from_vectors(
                input_vector=step.request_vector,
                output_vector=step.response_vector,
            )

            if determination.guid is None:
                continue

            determinations.append((index, determination))
            if determination.combined_cosine_similarity >= SESSION_SIMILARITY_EARLY_STOP_THRESHOLD:
                return SessionSelection(step_index=index, provenance="threshold", determination=determination)

        if not determinations:
            return None

        best_index, best_determination = determinations[0]
        for index, determination in determinations[1:]:
            if _dotnet_double_compare(determination.combined_cosine_similarity, best_determination.combined_cosine_similarity) > 0:
                best_index, best_determination = index, determination

        return SessionSelection(step_index=best_index, provenance="max_similarity", determination=best_determination)

    def classify_session_from_text_embeddings(
        self,
        *,
        steps: Sequence[SessionStepText],
        embeddings_by_text: Mapping[str, Sequence[float]],
        token_counter: TokenCounter | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> SessionSelection | None:
        vector_steps = self._build_vector_steps_from_text_steps(
            steps=steps,
            embeddings_by_text=embeddings_by_text,
            token_counter=token_counter,
            max_input_tokens=max_input_tokens,
            clean_text_cache={},
            embedding_cache={},
        )
        return self.classify_session_steps_from_vectors(vector_steps)

    def classify_sessions_from_text_embeddings(
        self,
        *,
        sessions_by_unit_id: Mapping[str, Sequence[SessionStepText]],
        embeddings_by_text: Mapping[str, Sequence[float]],
        token_counter: TokenCounter | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> dict[str, SessionSelection | None]:
        clean_text_cache: dict[str | None, str] = {}
        embedding_cache: dict[str, Sequence[float]] = {}
        selections: dict[str, SessionSelection | None] = {}

        for unit_id, steps in sessions_by_unit_id.items():
            vector_steps = self._build_vector_steps_from_text_steps(
                steps=steps,
                embeddings_by_text=embeddings_by_text,
                token_counter=token_counter,
                max_input_tokens=max_input_tokens,
                clean_text_cache=clean_text_cache,
                embedding_cache=embedding_cache,
            )
            selections[unit_id] = self.classify_session_steps_from_vectors(vector_steps)

        return selections

    def _build_vector_steps_from_text_steps(
        self,
        *,
        steps: Sequence[SessionStepText],
        embeddings_by_text: Mapping[str, Sequence[float]],
        token_counter: TokenCounter | None,
        max_input_tokens: int,
        clean_text_cache: dict[str | None, str],
        embedding_cache: dict[str, Sequence[float]],
    ) -> list[SessionStepVectors]:
        vector_steps: list[SessionStepVectors] = []

        for step in steps:
            cleaned_request = self._clean_text_cached(
                step.request,
                token_counter=token_counter,
                max_input_tokens=max_input_tokens,
                clean_text_cache=clean_text_cache,
            )
            cleaned_response = self._clean_text_cached(
                step.response,
                token_counter=token_counter,
                max_input_tokens=max_input_tokens,
                clean_text_cache=clean_text_cache,
            )

            request_vector = self._embedding_for_cleaned_text(
                cleaned_request,
                embeddings_by_text=embeddings_by_text,
                embedding_cache=embedding_cache,
                side_name="request",
            )
            response_vector = self._embedding_for_cleaned_text(
                cleaned_response,
                embeddings_by_text=embeddings_by_text,
                embedding_cache=embedding_cache,
                side_name="response",
            )

            vector_steps.append(SessionStepVectors(request_vector=request_vector, response_vector=response_vector))

        return vector_steps

    @staticmethod
    def _clean_text_cached(
        raw_text: str | None,
        *,
        token_counter: TokenCounter | None,
        max_input_tokens: int,
        clean_text_cache: dict[str | None, str],
    ) -> str:
        if raw_text in clean_text_cache:
            return clean_text_cache[raw_text]

        cleaned = clean_chat_text(
            raw_text,
            token_counter=token_counter,
            max_input_tokens=max_input_tokens,
        )
        clean_text_cache[raw_text] = cleaned
        return cleaned

    @staticmethod
    def _embedding_for_cleaned_text(
        cleaned_text: str,
        *,
        embeddings_by_text: Mapping[str, Sequence[float]],
        embedding_cache: dict[str, Sequence[float]],
        side_name: str,
    ) -> Sequence[float] | None:
        if not cleaned_text:
            return None

        if cleaned_text in embedding_cache:
            return embedding_cache[cleaned_text]

        if cleaned_text not in embeddings_by_text:
            raise KeyError(f"Missing embedding for cleaned {side_name} text: '{cleaned_text}'")

        vector = embeddings_by_text[cleaned_text]
        embedding_cache[cleaned_text] = vector
        return vector

    def enumerate_unique_clean_texts(
        self,
        *,
        steps: Sequence[SessionStepText],
        token_counter: TokenCounter | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> tuple[str, ...]:
        unique: list[str] = []
        seen: set[str] = set()

        for step in steps:
            for text in (step.request, step.response):
                cleaned = clean_chat_text(
                    text,
                    token_counter=token_counter,
                    max_input_tokens=max_input_tokens,
                )
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    unique.append(cleaned)

        return tuple(unique)

    def _score(
        self,
        prepared: _PreparedCentroids,
        query_vector: Sequence[float],
        *,
        side_name: str,
    ) -> _SideScores:
        query = self._normalize_query_vector(query_vector, side_name=side_name)
        query_norm = float(np.linalg.norm(query))

        if query_norm == 0.0:
            similarities = np.full(len(prepared.guids), np.nan, dtype=np.float64)
        else:
            normalized_query = query / query_norm
            similarities = (prepared.normalized_matrix @ normalized_query).astype(np.float64, copy=False)

        cosine_distance = 1.0 - similarities
        rank_key = np.where(np.isfinite(cosine_distance), cosine_distance, np.inf)

        euclidean_distance = np.full_like(cosine_distance, np.nan)
        finite_similarity = np.isfinite(similarities)
        euclidean_distance[finite_similarity] = np.sqrt(
            np.maximum(0.0, 2.0 - (2.0 * similarities[finite_similarity]))
        )

        return _SideScores(
            guids=prepared.guids,
            index_by_guid=prepared.index_by_guid,
            rank_key=rank_key,
            cosine_distance=cosine_distance,
            euclidean_distance=euclidean_distance,
        )

    def _rank_matches(self, scores: _SideScores) -> list[BusinessUseCaseMatch]:
        ranked_indices = np.argsort(scores.rank_key, kind="stable")
        matches: list[BusinessUseCaseMatch] = []
        for index in ranked_indices:
            guid = scores.guids[int(index)]
            cosine_distance = float(scores.cosine_distance[int(index)])
            euclidean_distance = float(scores.euclidean_distance[int(index)])
            info = self._artifacts.use_cases.get(guid)
            matches.append(
                BusinessUseCaseMatch(
                    guid=guid,
                    domain="" if info is None else info.domain,
                    segment="" if info is None else info.segment,
                    category="" if info is None else info.category,
                    sub_category="" if info is None else info.sub_category,
                    sub_subcategory="" if info is None else info.sub_subcategory,
                    business_task="" if info is None else info.business_task,
                    cosine_distance=cosine_distance,
                    euclidean_distance=euclidean_distance,
                )
            )
        return matches

    def _combined_best(
        self,
        input_scores: _SideScores,
        output_scores: _SideScores,
    ) -> BusinessUseCaseMatch | None:
        best_guid: UUID | None = None
        best_sum = math.inf

        for input_index, guid in enumerate(input_scores.guids):
            output_index = output_scores.index_by_guid.get(guid)
            if output_index is None:
                continue
            score_sum = (
                float(input_scores.cosine_distance[input_index])
                + float(output_scores.cosine_distance[output_index])
            )
            if score_sum < best_sum:
                best_sum = score_sum
                best_guid = guid

        if best_guid is None:
            return None

        info = self._artifacts.use_cases.get(best_guid)
        input_index = input_scores.index_by_guid[best_guid]
        output_index = output_scores.index_by_guid[best_guid]
        return BusinessUseCaseMatch(
            guid=best_guid,
            domain="" if info is None else info.domain,
            segment="" if info is None else info.segment,
            category="" if info is None else info.category,
            sub_category="" if info is None else info.sub_category,
            sub_subcategory="" if info is None else info.sub_subcategory,
            business_task="" if info is None else info.business_task,
            cosine_distance=(
                float(input_scores.cosine_distance[input_index])
                + float(output_scores.cosine_distance[output_index])
            ),
            euclidean_distance=(
                float(input_scores.euclidean_distance[input_index])
                + float(output_scores.euclidean_distance[output_index])
            ),
        )

    def _choose_agreement(
        self,
        input_top: Sequence[BusinessUseCaseMatch],
        output_top: Sequence[BusinessUseCaseMatch],
        combined: BusinessUseCaseMatch | None,
    ) -> tuple[UUID | None, str, str]:
        if input_top and output_top and input_top[0].guid == output_top[0].guid:
            return (
                input_top[0].guid,
                "Agree",
                "Input and output share the same closest use case.",
            )

        output_rank: dict[UUID, int] = {}
        for index, match in enumerate(output_top):
            output_rank[match.guid] = index + 1

        for index, match in enumerate(input_top):
            if match.guid in output_rank:
                rank = output_rank[match.guid]
                reason = (
                    f"Closest input use case (input rank {index + 1}) is corroborated by the output "
                    f"(output rank {rank}), though it is not the output's closest."
                )
                return match.guid, "Corroborated", reason

        if combined is not None:
            return (
                combined.guid,
                "Ambiguous",
                "Input and output top-N share no use case; resolved via the combined input+output "
                "distance (late-fusion fallback).",
            )

        return None, "Ambiguous", "Input and output top-N have no use case in common."

    def _combined_cosine_similarity(
        self,
        guid: UUID | None,
        input_scores: _SideScores,
        output_scores: _SideScores,
    ) -> float:
        if guid is None:
            return math.nan

        total = 0.0
        count = 0

        input_index = input_scores.index_by_guid.get(guid)
        if input_index is not None:
            total += 1.0 - float(input_scores.cosine_distance[input_index])
            count += 1

        output_index = output_scores.index_by_guid.get(guid)
        if output_index is not None:
            total += 1.0 - float(output_scores.cosine_distance[output_index])
            count += 1

        if count == 0:
            return math.nan
        return total / count

    def _determine_single_field(self, side: str, vector: Sequence[float] | None) -> BusinessUseCaseDetermination:
        assert vector is not None
        prepared = self._input_prepared if side == "input" else self._output_prepared
        scores = self._score(prepared, vector, side_name=side)
        top = self._rank_matches(scores)[:TOP_N]

        input_top = tuple(top if side == "input" else ())
        output_top = tuple(top if side == "output" else ())

        if not top:
            return BusinessUseCaseDetermination(
                guid=None,
                use_case=None,
                status="Ambiguous",
                reason=f"No use case scored for the {side} field.",
                combined_cosine_similarity=math.nan,
                input_matches=input_top,
                output_matches=output_top,
                combined_best=None,
            )

        best = top[0]
        cosine_similarity = 1.0 - best.cosine_distance

        if not math.isfinite(cosine_similarity) or cosine_similarity < COSINE_SIMILARITY_THRESHOLD:
            return BusinessUseCaseDetermination(
                guid=LOW_CONFIDENCE_FALLBACK_GUID,
                use_case=self._low_confidence_use_case(),
                status="Ambiguous",
                reason=(
                    f"Cosine similarity {cosine_similarity:.4f} (from the {side} field only) is below "
                    f"the threshold {COSINE_SIMILARITY_THRESHOLD:.4f}; the use case cannot be determined."
                ),
                combined_cosine_similarity=cosine_similarity,
                input_matches=input_top,
                output_matches=output_top,
                combined_best=None,
            )

        info = self._artifacts.use_cases[best.guid]
        return BusinessUseCaseDetermination(
            guid=best.guid,
            use_case=BusinessUseCaseInfo(
                guid=best.guid,
                domain=info.domain,
                segment=info.segment,
                category=info.category,
                sub_category=info.sub_category,
                sub_subcategory=info.sub_subcategory,
                business_task=info.business_task,
            ),
            status="Ambiguous",
            reason=f"Ranked using the {side} field only (the other field was empty).",
            combined_cosine_similarity=cosine_similarity,
            input_matches=input_top,
            output_matches=output_top,
            combined_best=None,
        )

    @staticmethod
    def _low_confidence_use_case() -> BusinessUseCaseInfo:
        return BusinessUseCaseInfo(
            guid=LOW_CONFIDENCE_FALLBACK_GUID,
            domain=LOW_CONFIDENCE_FALLBACK_LABEL,
            segment=LOW_CONFIDENCE_FALLBACK_LABEL,
            category=LOW_CONFIDENCE_FALLBACK_LABEL,
            sub_category=LOW_CONFIDENCE_FALLBACK_LABEL,
            sub_subcategory=LOW_CONFIDENCE_FALLBACK_LABEL,
            business_task=LOW_CONFIDENCE_FALLBACK_LABEL,
        )


def _dotnet_double_compare(left: float, right: float) -> int:
    left_nan = math.isnan(left)
    right_nan = math.isnan(right)
    if left_nan and right_nan:
        return 0
    if left_nan:
        return -1
    if right_nan:
        return 1
    if left < right:
        return -1
    if left > right:
        return 1
    return 0
