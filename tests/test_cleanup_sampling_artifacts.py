import importlib.util
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_sampling_artifacts.py"
SPEC = importlib.util.spec_from_file_location("cleanup_sampling_artifacts", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

apply_cleanup = MODULE.apply_cleanup
inventory_repo_cleanup = MODULE.inventory_repo_cleanup


def _git_init(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True, capture_output=True)


def _touch_png(path: Path, payload: bytes = b"img") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _archive_manifest(archive_root: Path) -> dict:
    return json.loads((archive_root / MODULE.MANIFEST_FILE_NAME).read_text(encoding="utf-8"))


def _simulate_reparse_lstat(target: Path):
    original = MODULE.os.lstat
    file_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)

    def fake_lstat(path):
        st = original(path)
        p = Path(path)
        if p.resolve() == target.resolve():
            class _ReparseStat:
                st_file_attributes = file_attr

            return _ReparseStat()
        return st

    return fake_lstat


def test_inventory_repo_cleanup_keeps_referenced_and_tracked_pngs(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    out_dir = repo_root / "outputs_sampling_v7" / "runs" / "v7-live-repeated-20260903"
    out_dir.mkdir(parents=True)

    keep_png = out_dir / "pca8-distance-audit-390x844.png"
    keep_png.write_bytes(b"keep")

    tracked_png = out_dir / "tracked-debug.png"
    tracked_png.write_bytes(b"tracked")
    subprocess.run(["git", "-C", str(repo_root), "add", "outputs_sampling_v7/runs/v7-live-repeated-20260903/tracked-debug.png"], check=True)

    unreferenced_png = out_dir / "storytelling-validation" / "pca8-story-repetitive.png"
    unreferenced_png.parent.mkdir(parents=True)
    unreferenced_png.write_bytes(b"drop-me")

    audit_json = repo_root / "outputs_agent_uniform_sampling" / "pca8-distance-playwright-audit.json"
    audit_json.parent.mkdir(parents=True)
    audit_json.write_text(json.dumps({"screens": ["pca8-distance-audit-390x844.png", "pca8-section-results-1440x1000.png"]}), encoding="utf-8")

    inventory = inventory_repo_cleanup(repo_root)

    assert keep_png.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert tracked_png.relative_to(repo_root).as_posix() in inventory["safety_skips"]["tracked"]
    assert unreferenced_png.relative_to(repo_root).as_posix() in inventory["candidates"]
    assert inventory["counts"]["candidate_files"] == 1


def test_inventory_repo_cleanup_keeps_referenced_from_root_notebook_and_python(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    run_dir = repo_root / "outputs_sampling_v7" / "runs" / "r1"
    keep_from_notebook = run_dir / "pca8-story-kept-from-notebook.png"
    keep_from_python = run_dir / "pca8-story-kept-from-python.png"
    candidate = run_dir / "pca8-story-candidate.png"
    _touch_png(keep_from_notebook, b"n")
    _touch_png(keep_from_python, b"p")
    _touch_png(candidate, b"c")

    notebook = repo_root / "analysis.ipynb"
    notebook.write_text(
        json.dumps({"cells": [{"cell_type": "markdown", "source": ["![img](pca8-story-kept-from-notebook.png)"]}]}),
        encoding="utf-8",
    )
    script = repo_root / "reference_png.py"
    script.write_text('IMG = "pca8-story-kept-from-python.png"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_root), "add", "analysis.ipynb", "reference_png.py"], check=True)

    inventory = inventory_repo_cleanup(repo_root)
    assert keep_from_notebook.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert keep_from_python.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert candidate.relative_to(repo_root).as_posix() in inventory["candidates"]


def test_dry_run_has_no_side_effects(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    candidate = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(candidate, b"dry-run")

    result = apply_cleanup(repo_root, apply=False)
    assert result["applied"] is False
    assert candidate.exists()
    assert not (repo_root / MODULE.MANIFEST_FILE_NAME).exists()


def test_apply_cleanup_archives_and_removes_only_candidates(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    out_dir = repo_root / "outputs_sampling_v7" / "runs" / "v7-live-repeated-20260903"
    out_dir.mkdir(parents=True)

    candidate = out_dir / "storytelling-validation" / "pca8-story-extra.png"
    _touch_png(candidate, b"archive-me")

    archive_root = tmp_path / "archive-outside-repo"
    archive_root.mkdir()

    manifest = apply_cleanup(repo_root, archive_root, apply=True)

    archive_path = archive_root / "outputs_sampling_v7" / "runs" / "v7-live-repeated-20260903" / "storytelling-validation" / "pca8-story-extra.png"
    assert archive_path.exists()
    assert not candidate.exists()
    assert manifest["files"][0]["original_rel_path"] == "outputs_sampling_v7/runs/v7-live-repeated-20260903/storytelling-validation/pca8-story-extra.png"
    assert manifest["files"][0]["archive_rel_path"] == archive_path.relative_to(archive_root).as_posix()
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"archive-me").hexdigest()
    assert (archive_root / MODULE.MANIFEST_FILE_NAME).exists()
    persisted = _archive_manifest(archive_root)
    assert persisted["status"] == "complete"
    assert all(entry["unlink_status"] == "deleted" for entry in persisted["files"])


def test_apply_cleanup_refuses_symlink_and_outside_repo_archive(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    out_dir = repo_root / "outputs_sampling_v7" / "runs" / "v7-live-repeated-20260903"
    out_dir.mkdir(parents=True)

    candidate = out_dir / "validation_screenshots" / "pca8-story-demo.png"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"demo")

    archive_root = repo_root / "archive"
    with pytest.raises(ValueError, match="outside the repository"):
        apply_cleanup(repo_root, archive_root, apply=True)

    if hasattr(os, "symlink"):
        symlink_target = out_dir / "link.png"
        try:
            os.symlink(candidate, symlink_target)
            with pytest.raises(ValueError, match="symlink"):
                apply_cleanup(repo_root, tmp_path / "archive-outside-repo", apply=True)
        except OSError:
            pytest.skip("symlink creation not supported in this environment")


def test_apply_cleanup_refuses_archive_inside_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    candidate = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(candidate, b"x")

    with pytest.raises(ValueError, match="outside the repository"):
        apply_cleanup(repo_root, repo_root / "archive", apply=True)


def test_preflight_collision_refuses_before_deletion(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    rel = Path("outputs_sampling_v7/runs/r1/pca8-story-candidate.png")
    source = repo_root / rel
    _touch_png(source, b"source")

    archive_root = tmp_path / "archive"
    destination = archive_root / rel
    _touch_png(destination, b"already-there")

    with pytest.raises(ValueError, match="already exists"):
        apply_cleanup(repo_root, archive_root, apply=True)

    assert source.exists(), "preflight collision must happen before any unlink"


def test_failed_copy_preserves_originals_and_no_manifest(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(source, b"copy-fail")
    archive_root = tmp_path / "archive"

    with patch.object(MODULE.shutil, "copy2", side_effect=OSError("disk full")):
        with pytest.raises(ValueError, match="Failed to copy artifact"):
            apply_cleanup(repo_root, archive_root, apply=True)

    assert source.exists()
    assert not (archive_root / MODULE.MANIFEST_FILE_NAME).exists()


def test_failed_hash_preserves_originals_and_cleans_bad_copy(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    rel = Path("outputs_sampling_v7/runs/r1/pca8-story-candidate.png")
    source = repo_root / rel
    _touch_png(source, b"match-me")
    archive_root = tmp_path / "archive"

    original_copy2 = MODULE.shutil.copy2

    def bad_copy(src, dst, *args, **kwargs):
        result = original_copy2(src, dst, *args, **kwargs)
        Path(dst).write_bytes(b"corrupted")
        return result

    with patch.object(MODULE.shutil, "copy2", side_effect=bad_copy):
        with pytest.raises(ValueError, match="verification failed"):
            apply_cleanup(repo_root, archive_root, apply=True)

    assert source.exists()
    assert not (archive_root / rel).exists()
    assert not (archive_root / MODULE.MANIFEST_FILE_NAME).exists()


def test_unlink_failure_persists_manifest_with_progress(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    rel1 = Path("outputs_sampling_v7/runs/r1/pca8-story-a.png")
    rel2 = Path("outputs_sampling_v7/runs/r1/pca8-story-b.png")
    source1 = repo_root / rel1
    source2 = repo_root / rel2
    _touch_png(source1, b"a")
    _touch_png(source2, b"b")

    archive_root = tmp_path / "archive"
    original_unlink = MODULE.Path.unlink
    first_seen = {"done": False}

    def fail_second_unlink(self, *args, **kwargs):
        if self == source1 and not first_seen["done"]:
            first_seen["done"] = True
            return original_unlink(self, *args, **kwargs)
        if self == source2:
            raise OSError("permission denied")
        return original_unlink(self, *args, **kwargs)

    with patch.object(MODULE.Path, "unlink", new=fail_second_unlink):
        with pytest.raises(ValueError, match="Failed to remove original artifact"):
            apply_cleanup(repo_root, archive_root, apply=True)

    manifest = _archive_manifest(archive_root)
    assert manifest["status"] == "unlink_failed"
    by_rel = {entry["original_rel_path"]: entry for entry in manifest["files"]}
    assert by_rel[rel1.as_posix()]["unlink_status"] == "deleted"
    assert by_rel[rel2.as_posix()]["unlink_status"] == "failed"
    assert not source1.exists()
    assert source2.exists()
    assert (archive_root / rel1).exists()
    assert (archive_root / rel2).exists()


def test_reparse_guard_refuses_source_parent_without_symlink_privilege(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source_parent = repo_root / "outputs_sampling_v7" / "runs" / "r1"
    source = source_parent / "pca8-story-candidate.png"
    _touch_png(source, b"x")
    archive_root = tmp_path / "archive"

    with patch.object(MODULE.os, "lstat", side_effect=_simulate_reparse_lstat(source_parent)):
        with pytest.raises(ValueError, match="reparse point"):
            apply_cleanup(repo_root, archive_root, apply=True)


def test_reparse_guard_refuses_archive_parent_without_symlink_privilege(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(source, b"x")
    archive_root = tmp_path / "archive" / "nested"
    archive_parent = archive_root.parent
    archive_parent.mkdir(parents=True)

    with patch.object(MODULE.os, "lstat", side_effect=_simulate_reparse_lstat(archive_parent)):
        with pytest.raises(ValueError, match="reparse point"):
            apply_cleanup(repo_root, archive_root, apply=True)


def test_restore_roundtrip_supported_by_manifest(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    rel = Path("outputs_sampling_v7/runs/r1/pca8-story-candidate.png")
    source = repo_root / rel
    payload = b"restore-me"
    _touch_png(source, payload)
    archive_root = tmp_path / "archive"

    apply_cleanup(repo_root, archive_root, apply=True)
    manifest = _archive_manifest(archive_root)
    entry = manifest["files"][0]
    archived = archive_root / entry["archive_rel_path"]

    restored = repo_root / entry["original_rel_path"]
    restored.parent.mkdir(parents=True, exist_ok=True)
    restored.write_bytes(archived.read_bytes())

    assert restored.read_bytes() == payload


def test_refuses_manifest_collision_before_any_unlink(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(source, b"x")

    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True)
    (archive_root / MODULE.MANIFEST_FILE_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        apply_cleanup(repo_root, archive_root, apply=True)
    assert source.exists()


def test_inventory_fails_closed_when_git_tracking_unavailable(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    _touch_png(repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png", b"x")

    with patch.object(MODULE.subprocess, "run", side_effect=OSError("git missing")):
        with pytest.raises(ValueError, match="Unable to determine git-tracked files"):
            inventory_repo_cleanup(repo_root)


def test_inventory_fails_closed_when_git_ls_files_nonzero(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    _touch_png(repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png", b"x")

    class _Result:
        returncode = 128
        stdout = b""
        stderr = b"fatal: not a git repository"

    with patch.object(MODULE.subprocess, "run", return_value=_Result()):
        with pytest.raises(ValueError, match="Unable to determine git-tracked files"):
            inventory_repo_cleanup(repo_root)


def test_apply_fails_closed_on_git_tracking_error_and_preserves_sources(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(source, b"x")

    with patch.object(MODULE.subprocess, "run", side_effect=OSError("git unavailable")):
        with pytest.raises(ValueError, match="Unable to determine git-tracked files"):
            apply_cleanup(repo_root, tmp_path / "archive", apply=True)

    assert source.exists()


def test_inventory_keeps_referenced_from_untracked_local_output_html_md_txt_ipynb(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    run_dir = repo_root / "outputs_sampling_v7" / "runs" / "r1"
    keep_html = run_dir / "pca8-story-keep-html.png"
    keep_md = run_dir / "pca8-story-keep-md.png"
    keep_txt = run_dir / "pca8-story-keep-txt.png"
    keep_ipynb = run_dir / "pca8-story-keep-ipynb.png"
    candidate = run_dir / "pca8-story-candidate.png"
    for path in (keep_html, keep_md, keep_txt, keep_ipynb, candidate):
        _touch_png(path, path.name.encode("utf-8"))

    (repo_root / "outputs_sampling_v7" / "runs" / "r1" / "local_ref.html").write_text(
        '<img src="pca8-story-keep-html.png">', encoding="utf-8"
    )
    (repo_root / "outputs_sampling_v7" / "runs" / "r1" / "local_ref.md").write_text(
        "![a](pca8-story-keep-md.png)", encoding="utf-8"
    )
    (repo_root / "outputs_sampling_v7" / "runs" / "r1" / "local_ref.txt").write_text(
        "pca8-story-keep-txt.png", encoding="utf-8"
    )
    (repo_root / "outputs_sampling_v7" / "runs" / "r1" / "local_ref.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "markdown", "source": ["pca8-story-keep-ipynb.png"]}]}),
        encoding="utf-8",
    )

    inventory = inventory_repo_cleanup(repo_root)
    assert keep_html.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert keep_md.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert keep_txt.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert keep_ipynb.relative_to(repo_root).as_posix() in inventory["safety_skips"]["referenced"]
    assert candidate.relative_to(repo_root).as_posix() in inventory["candidates"]


def test_inventory_skips_cache_tree_reference_scan(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    run_dir = repo_root / "outputs_sampling_v7" / "runs" / "r1"
    keep_in_cache = run_dir / "pca8-story-kept-only-by-cache-reference.png"
    _touch_png(keep_in_cache, b"x")

    cache_ref = repo_root / "outputs_sampling_v7" / "cache" / "refs" / "x.md"
    cache_ref.parent.mkdir(parents=True, exist_ok=True)
    cache_ref.write_text("pca8-story-kept-only-by-cache-reference.png", encoding="utf-8")

    inventory = inventory_repo_cleanup(repo_root)
    assert keep_in_cache.relative_to(repo_root).as_posix() in inventory["candidates"]


def test_inventory_limits_cleanup_scope_to_v7_and_allowed_pca8_prefixes(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    v7_run = repo_root / "outputs_sampling_v7" / "runs" / "r1"
    allowed = v7_run / "pca8-final-demo.png"
    blocked_distance_audit = v7_run / "pca8-distance-audit-390x844.png"
    blocked_section = v7_run / "pca8-section-results-1440x1000.png"
    other_root = repo_root / "outputs_sampling_v6" / "runs" / "r1" / "pca8-final-v6.png"
    for path in (allowed, blocked_distance_audit, blocked_section, other_root):
        _touch_png(path, b"x")

    inventory = inventory_repo_cleanup(repo_root)
    rel_allowed = allowed.relative_to(repo_root).as_posix()
    rel_distance = blocked_distance_audit.relative_to(repo_root).as_posix()
    rel_section = blocked_section.relative_to(repo_root).as_posix()
    rel_other = other_root.relative_to(repo_root).as_posix()
    assert rel_allowed in inventory["candidates"]
    assert rel_distance in inventory["safety_skips"]["rejected"]
    assert rel_section in inventory["safety_skips"]["rejected"]
    assert rel_other not in inventory["candidates"]


def test_inventory_retains_validation_batches_and_report_prefix_in_cleanup_scope(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)
    relative_paths = (
        "outputs_sampling_v7/runs/r1/validation_screenshots/desktop.png",
        "outputs_sampling_v7/runs/r1/storytelling-validation/section.png",
        "outputs_sampling_v7/runs/r1/report-desktop.png",
        "outputs_sampling_v7/runs/r1/pca8-final-preview.png",
    )
    for relative_path in relative_paths:
        _touch_png(repo_root / relative_path, b"image")
    _touch_png(repo_root / "outputs_sampling_v7/cache/report-keep.png", b"cache")
    inventory = inventory_repo_cleanup(repo_root)
    assert set(inventory["candidates"]) == set(relative_paths)


def test_archive_lexical_guard_rejects_reparse_in_intermediate_ancestor(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    source = repo_root / "outputs_sampling_v7" / "runs" / "r1" / "pca8-story-candidate.png"
    _touch_png(source, b"x")

    archive_root = tmp_path / "archive" / "nested" / "target"
    reparse_ancestor = archive_root.parent
    reparse_ancestor.mkdir(parents=True, exist_ok=True)

    with patch.object(MODULE.os, "lstat", side_effect=_simulate_reparse_lstat(reparse_ancestor)):
        with pytest.raises(ValueError, match="reparse point"):
            apply_cleanup(repo_root, archive_root, apply=True)


def test_source_mutation_before_unlink_fails_and_preserves_manifest_and_original(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git_init(repo_root)

    rel = Path("outputs_sampling_v7/runs/r1/pca8-story-candidate.png")
    source = repo_root / rel
    _touch_png(source, b"original")
    archive_root = tmp_path / "archive"

    original_write_json_atomic = MODULE._write_json_atomic
    mutated = {"done": False}

    def mutate_after_manifest(path, payload):
        original_write_json_atomic(path, payload)
        if payload.get("status") == "copied_verified" and not mutated["done"]:
            source.write_bytes(b"changed-after-copy")
            mutated["done"] = True

    with patch.object(MODULE, "_write_json_atomic", side_effect=mutate_after_manifest):
        with pytest.raises(ValueError, match="Refusing to unlink modified source"):
            apply_cleanup(repo_root, archive_root, apply=True)

    assert source.exists()
    manifest = _archive_manifest(archive_root)
    assert manifest["status"] == "unlink_failed"
    assert manifest["files"][0]["unlink_status"] == "failed"
