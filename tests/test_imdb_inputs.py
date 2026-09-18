import io
import json
import tarfile

import numpy as np
import pytest

from sampling_comparison.imdb_inputs import load_input, prepare_input, read_reviews


def archive_fixture(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in (
            ("aclImdb/train/pos/1_8.txt", "A good<br />movie &amp; acting."),
            ("aclImdb/test/neg/2_2.txt", "A bad movie."),
            ("aclImdb/train/pos/3_9.txt", "A good<br />movie &amp; acting."),
            ("aclImdb/train/unsup/9_0.txt", "not labeled"),
            ("../../escape.txt", "not a review"),
        ):
            content = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return archive


class FakeEmbedder:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
        self.texts = []

    def embed(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("intentional interrupted batch")
        self.calls += 1
        self.texts.extend(texts)
        vectors = np.ones((len(texts), 1536), dtype=np.float32)
        return vectors, len(texts) * 4


def test_source_labels_text_boundary_and_offline(tmp_path):
    archive = archive_fixture(tmp_path)
    records, texts = read_reviews(archive, expected_count=3)
    assert len(records) == 3 and len(texts) == 2
    assert set(texts.values()) == {"A good\nmovie & acting.", "A bad movie."}
    output = tmp_path / "cache"
    profile = prepare_input(archive, output, expected_count=3)
    assert profile["status"] == "prepared_without_embeddings"
    assert profile["positive_count"] == 2
    assert profile["duplicate_text_rows"] == 1
    assert not (output / "manifest.json").exists()
    with pytest.raises(ValueError, match="expected 50000"):
        read_reviews(archive)


def test_cache_resume_no_calls_and_hash_tamper(tmp_path):
    archive = archive_fixture(tmp_path)
    output = tmp_path / "cache"
    kwargs = dict(expected_count=3, batch_size=1, provider_identity="fixture-provider")
    with pytest.raises(RuntimeError, match="intentional"):
        prepare_input(archive, output, embedder=FakeEmbedder(fail_after=1), **kwargs)
    resumed = FakeEmbedder()
    profile = prepare_input(archive, output, embedder=resumed, **kwargs)
    assert resumed.calls == 1 and profile["embedding_calls"] == 2
    vectors, labels, _ = load_input(output / "manifest.json")
    assert vectors.shape == (3, 1536) and labels.sum() == 2
    snapshot = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    reuse = prepare_input(archive, output, embedder=FakeEmbedder(fail_after=0), **kwargs)
    assert reuse["fresh_embedding_calls"] == 0
    assert all(path.read_bytes() == data for path, data in snapshot.items())
    with pytest.raises(ValueError, match="provider changed"):
        prepare_input(archive, output, embedder=FakeEmbedder(),
                      **{**kwargs, "provider_identity": "different-provider"})
    units = output / "units.json"
    units.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="units hash"):
        load_input(output / "manifest.json")


def test_source_or_provider_change_fails_closed(tmp_path):
    archive = archive_fixture(tmp_path)
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        prepare_input(archive, output, embedder=FakeEmbedder(fail_after=0),
                      provider_identity="provider-a", expected_count=3)
    with pytest.raises(ValueError, match="provider changed"):
        prepare_input(archive, output, embedder=FakeEmbedder(),
                      provider_identity="provider-b", expected_count=3)
    config = output / "preparation.json"
    data = json.loads(config.read_text())
    data["source_sha256"] = "tampered"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="preparation differs"):
        prepare_input(archive, output, expected_count=3)


def test_truncation_is_counted_and_token_bound_is_respected(tmp_path):
    archive = tmp_path / "long.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        content = ("word " * 10_000).encode()
        info = tarfile.TarInfo("aclImdb/train/pos/1_8.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    records, _ = read_reviews(archive, expected_count=1)
    assert records[0]["truncated"] is True
    assert records[0]["emitted_tokens"] <= 8191
    assert records[0]["original_tokens"] == 10_000


@pytest.mark.parametrize(("dimensions", "value", "message"), [
    (8, 1.0, "native shape"), (1536, float("nan"), "non-finite"),
])
def test_embedding_response_native_shape_and_finite_checks(tmp_path, dimensions, value, message):
    class WrongDimension:
        def embed(self, texts):
            return np.full((len(texts), dimensions), value), 1

    with pytest.raises(ValueError, match=message):
        prepare_input(archive_fixture(tmp_path), tmp_path / "cache",
                      embedder=WrongDimension(), provider_identity="fixture", expected_count=3)
