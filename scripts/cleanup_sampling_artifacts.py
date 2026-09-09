from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TARGET_OUTPUT_ROOTS = (
    "outputs_sampling_v2",
    "outputs_sampling_v3",
    "outputs_sampling_v4",
    "outputs_sampling_v6",
    "outputs_sampling_v7",
    "outputs_agent_uniform_sampling",
)
CLEANUP_OUTPUT_ROOTS = ("outputs_sampling_v7",)
TEXT_SUFFIXES = {".json", ".md", ".html", ".ipynb", ".txt", ".log", ".yaml", ".yml", ".py"}
LOCAL_REFERENCE_SUFFIXES = {".json", ".md", ".html", ".ipynb", ".txt"}
PCA8_DISPOSABLE_PREFIXES = (
    "pca8-audit-",
    "pca8-story-",
    "pca8-final-",
    "pca8-distance-report-",
)
MANIFEST_FILE_NAME = "cleanup_sampling_artifacts_manifest.json"


def _git_tracked_files(repo_root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            text=False,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"Unable to determine git-tracked files for cleanup safety checks: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip() if result.stderr else ""
        detail = f" ({stderr})" if stderr else ""
        raise ValueError(f"Unable to determine git-tracked files for cleanup safety checks: git ls-files failed{detail}")
    return {Path(p.decode("utf-8", errors="surrogateescape")).as_posix() for p in result.stdout.split(b"\0") if p}


def _iter_target_roots(repo_root: Path) -> Iterable[Path]:
    for rel_name in TARGET_OUTPUT_ROOTS:
        candidate = repo_root / rel_name
        if candidate.exists() and candidate.is_dir():
            yield candidate


def _iter_cleanup_roots(repo_root: Path) -> Iterable[Path]:
    for rel_name in CLEANUP_OUTPUT_ROOTS:
        candidate = repo_root / rel_name
        if candidate.exists() and candidate.is_dir():
            yield candidate


def _iter_tracked_text_files(repo_root: Path, tracked: set[str]) -> Iterable[Path]:
    for rel in sorted(tracked):
        path = repo_root / rel
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.exists() and path.is_file():
            yield path


def _iter_local_output_reference_files(repo_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in _iter_target_roots(repo_root):
        for path in root.rglob("*"):
            if "cache" in {part.lower() for part in path.parts}:
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in LOCAL_REFERENCE_SUFFIXES:
                continue
            if path not in seen:
                seen.add(path)
                yield path
    for path in repo_root.glob("*.ipynb"):
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path


def _collect_referenced_png_names(repo_root: Path) -> set[str]:
    referenced: set[str] = set()
    seen: set[Path] = set()

    def walk_text(value):
        if isinstance(value, dict):
            for child in value.values():
                walk_text(child)
        elif isinstance(value, list):
            for child in value:
                walk_text(child)
        elif isinstance(value, str):
            matches = re.findall(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+\.png)", value, flags=re.IGNORECASE)
            referenced.update(matches)

    tracked = _git_tracked_files(repo_root)
    files_to_scan: list[Path] = []
    files_to_scan.extend(_iter_tracked_text_files(repo_root, tracked))
    files_to_scan.extend(_iter_local_output_reference_files(repo_root))

    for path in files_to_scan:
        if path in seen:
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        matches = re.findall(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+\.png)", text, flags=re.IGNORECASE)
        referenced.update(matches)
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except Exception:
                continue
            walk_text(payload)
    return referenced


def _is_candidate_png(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if path.is_symlink():
        return False
    if path.suffix.lower() != ".png":
        return False
    if not path.is_file():
        return False
    if any(part in {"cache", ".git", "__pycache__"} for part in path.parts):
        return False
    name_lower = path.name.lower()
    rel_parts_lower = [part.lower() for part in Path(rel).parts]
    if not rel_parts_lower or rel_parts_lower[0] != "outputs_sampling_v7":
        return False
    if any(part in {"validation_screenshots", "storytelling-validation"} for part in rel_parts_lower[1:-1]):
        return True
    return name_lower.startswith("report-") or any(name_lower.startswith(prefix) for prefix in PCA8_DISPOSABLE_PREFIXES)


def inventory_repo_cleanup(repo_root: Path | str) -> dict:
    repo_root = Path(repo_root).resolve()
    tracked = _git_tracked_files(repo_root)
    tracked_pngs = {p for p in tracked if p.lower().endswith(".png")}
    referenced_pngs = _collect_referenced_png_names(repo_root)

    safety_skips = {"tracked": [], "referenced": [], "symlinks": [], "non_png": [], "rejected": []}
    candidates: list[str] = []
    candidate_bytes = 0

    for root in _iter_cleanup_roots(repo_root):
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(repo_root).as_posix()
            if not path.is_file():
                continue
            if path.suffix.lower() != ".png":
                safety_skips["non_png"].append(rel)
                continue
            if path.is_symlink():
                safety_skips["symlinks"].append(rel)
                continue
            if rel in tracked_pngs:
                safety_skips["tracked"].append(rel)
                continue
            basename = path.name
            if basename in referenced_pngs or basename.lower() in {name.lower() for name in referenced_pngs}:
                safety_skips["referenced"].append(rel)
                continue
            if not _is_candidate_png(path, repo_root):
                safety_skips["rejected"].append(rel)
                continue
            candidates.append(rel)
            candidate_bytes += path.stat().st_size

    counts = {
        "target_roots": len(list(_iter_target_roots(repo_root))),
        "candidate_files": len(candidates),
        "candidate_bytes": candidate_bytes,
        "tracked_pngs": len(tracked_pngs),
        "referenced_pngs": len(referenced_pngs),
    }

    return {
        "repo_root": str(repo_root),
        "counts": counts,
        "candidates": candidates,
        "safety_skips": safety_skips,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_reparse_point(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    if attrs & reparse_flag:
        return True
    try:
        return path.is_symlink()
    except OSError:
        return False


def _assert_no_reparse_ancestors(path: Path, boundary: Path, label: str) -> None:
    if not _is_relative_to(path, boundary):
        raise ValueError(f"Refusing {label}: path escapes expected root: {path}")
    current = path
    while True:
        if current.exists() and _is_reparse_point(current):
            raise ValueError(f"Refusing {label}: reparse point detected at {current}")
        if current == boundary:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def _assert_existing_ancestor_chain_no_reparse(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"Refusing {label}: path must be absolute: {path}")
    if ".." in path.parts:
        raise ValueError(f"Refusing {label}: parent traversal is not allowed in archive path: {path}")

    current = path
    while True:
        if current.exists() and _is_reparse_point(current):
            raise ValueError(f"Refusing {label}: reparse point detected at {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def apply_cleanup(repo_root: Path | str, archive_dir: str | os.PathLike[str] | None = None, apply: bool = False) -> dict:
    repo_root = Path(repo_root).resolve()
    inventory = inventory_repo_cleanup(repo_root)
    candidates = inventory["candidates"]

    if not apply:
        return {
            "applied": False,
            "archive_dir": None,
            "counts": inventory["counts"],
            "files": [],
            "safety_skips": inventory["safety_skips"],
        }

    if archive_dir is None:
        raise ValueError("--archive-dir is required when --apply is set.")

    archive_input = Path(archive_dir)
    _assert_existing_ancestor_chain_no_reparse(archive_input, "archive path")
    archive_path = archive_input.resolve()
    if archive_path == repo_root or archive_path.is_relative_to(repo_root):
        raise ValueError("Archive directory must be outside the repository.")

    _assert_existing_ancestor_chain_no_reparse(archive_path, "archive path")
    if archive_path.exists():
        _assert_no_reparse_ancestors(archive_path, archive_path, "archive root")

    manifest_path = archive_path / MANIFEST_FILE_NAME
    if manifest_path.exists():
        raise ValueError(f"Archive target already exists, refusing overwrite: {manifest_path}")

    preflight: list[tuple[str, Path, Path, int]] = []
    for rel_path in candidates:
        rel_obj = Path(rel_path)
        if rel_obj.is_absolute() or ".." in rel_obj.parts:
            raise ValueError(f"Refusing candidate with unsafe relative path: {rel_path}")
        source = repo_root / rel_obj
        destination = archive_path / rel_obj

        if not _is_relative_to(source, repo_root):
            raise ValueError(f"Refusing source outside repository: {rel_path}")
        if not _is_relative_to(destination, archive_path):
            raise ValueError(f"Refusing destination outside archive root: {destination}")
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Refusing to archive non-file or symlink path: {rel_path}")

        _assert_no_reparse_ancestors(source, repo_root, f"source {rel_path}")

        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Archive target already exists, refusing overwrite: {destination}")
        _assert_no_reparse_ancestors(destination.parent, archive_path, f"destination parent {rel_path}")

        preflight.append((rel_path, source, destination, source.stat().st_size))

    archive_path.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_ancestors(archive_path, archive_path, "archive root")

    manifest_entries: list[dict] = []
    for rel_path, source, destination, source_size in preflight:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_ancestors(destination.parent, archive_path, f"destination parent {rel_path}")
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Archive target already exists, refusing overwrite: {destination}")
        sha_before = _sha256(source)
        try:
            shutil.copy2(source, destination)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Failed to copy artifact for archive: {rel_path}") from exc
        sha_after = _sha256(destination)
        if sha_before != sha_after:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Archive verification failed for {rel_path}: hashes do not match.")
        manifest_entries.append(
            {
                "original_rel_path": rel_path,
                "archive_rel_path": destination.relative_to(archive_path).as_posix(),
                "bytes": source_size,
                "sha256": sha_before,
                "copied_verified_utc": datetime.now(timezone.utc).isoformat(),
                "unlink_status": "pending",
                "restore_guidance": f"Copy {destination.relative_to(archive_path).as_posix()} back to {rel_path} within the repo root if needed.",
            }
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "copied_verified",
        "repo_root": str(repo_root),
        "archive_root": str(archive_path),
        "counts": {**inventory["counts"], "archived_files": len(manifest_entries)},
        "files": manifest_entries,
        "safety_skips": inventory["safety_skips"],
    }
    _write_json_atomic(manifest_path, manifest)

    for entry in manifest_entries:
        rel_path = entry["original_rel_path"]
        source = repo_root / Path(rel_path)
        _assert_no_reparse_ancestors(source, repo_root, f"source {rel_path}")
        if source.is_symlink() or not source.exists() or not source.is_file():
            entry["unlink_status"] = "failed"
            entry["unlink_error"] = "source changed before unlink"
            manifest["status"] = "unlink_failed"
            manifest["failure_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(manifest_path, manifest)
            raise ValueError(f"Refusing to unlink source that changed before cleanup: {rel_path}")

        current_sha = _sha256(source)
        if current_sha != entry["sha256"]:
            entry["unlink_status"] = "failed"
            entry["unlink_error"] = "source hash changed before unlink"
            manifest["status"] = "unlink_failed"
            manifest["failure_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(manifest_path, manifest)
            raise ValueError(f"Refusing to unlink modified source after archive copy: {rel_path}")
        try:
            source.unlink()
            entry["unlink_status"] = "deleted"
            entry["unlinked_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["status"] = "unlink_in_progress"
            _write_json_atomic(manifest_path, manifest)
        except Exception as exc:
            entry["unlink_status"] = "failed"
            entry["unlink_error"] = str(exc)
            manifest["status"] = "unlink_failed"
            manifest["failure_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(manifest_path, manifest)
            raise ValueError(f"Failed to remove original artifact after archiving: {rel_path}") from exc

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)

    summary = {
        "applied": True,
        "archive_dir": str(archive_path),
        "manifest_path": str(manifest_path),
        "counts": inventory["counts"],
        "files": manifest_entries,
        "safety_skips": inventory["safety_skips"],
    }
    summary["counts"]["archived_files"] = len(manifest_entries)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory and optionally archive redundant sampling PNG artifacts.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path, help="Repository root to scan.")
    parser.add_argument("--apply", action="store_true", help="Archive and remove candidate PNG files.")
    parser.add_argument("--archive-dir", type=Path, default=None, help="Required when --apply is set; must be outside the repo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    inventory = inventory_repo_cleanup(repo_root)

    print(json.dumps({
        "counts": inventory["counts"],
        "candidate_paths": inventory["candidates"],
        "safety_skips": inventory["safety_skips"],
    }, indent=2, sort_keys=True))

    if args.apply:
        manifest = apply_cleanup(repo_root, args.archive_dir, apply=True)
        print(json.dumps({
            "applied": True,
            "archive_dir": manifest["archive_dir"],
            "archived_files": manifest["files"],
        }, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
