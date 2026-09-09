from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _run_checker(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    script_path = repo_root / ".github" / "skills" / "sampling-experiment-report" / "quality_check.py"
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )


def _write_html(path: Path, body: str, title: str = "Sampling Report") -> None:
    path.write_text(
        f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
  </style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def test_quality_check_accepts_long_report_and_scrollable_table(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "valid-long.html"
    screenshots_dir = tmp_path / "validation_screenshots"

    long_paragraphs = "".join(f"<p>Paragraph {i}: this report contains meaningful content and details.</p>" for i in range(80))
    wide_table = (
        "<div style='overflow-x:auto;border:1px solid #ddd'>"
        "<table style='min-width:1800px'><tr>"
        + "".join(f"<th>Column {i}</th>" for i in range(40))
        + "</tr><tr>"
        + "".join(f"<td>Value {i}</td>" for i in range(40))
        + "</tr></table></div>"
    )
    _write_html(
        html_path,
        "<h1>Sampling report</h1><h2>What this shows</h2>" + long_paragraphs + wide_table,
    )

    result = _run_checker(
        repo_root,
        [
            "--html",
            str(html_path),
            "--screenshots",
            str(screenshots_dir),
            "--timeout-seconds",
            "25",
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode == 0, result.stderr or result.stdout
    summary = json.loads(result.stdout)
    assert summary["ok"] is True
    assert len(summary["pages"]) == 2
    assert (screenshots_dir / "desktop-valid-long.png").exists()
    assert (screenshots_dir / "mobile-valid-long.png").exists()


def test_quality_check_rejects_empty_body(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "empty.html"
    _write_html(html_path, "<h1></h1><p> </p>", title="")

    result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

    assert result.returncode != 0
    assert "missing-title" in (result.stdout + result.stderr)
    assert "insufficient-body-text" in (result.stdout + result.stderr)


def test_quality_check_rejects_broken_local_image(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "broken-image.html"
    _write_html(
        html_path,
        """
        <h1>Sampling report</h1>
        <p>Valid body content for image check.</p>
        <img src=\"missing-image.png\" alt=\"broken\" />
        """,
    )

    result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

    assert result.returncode != 0
    assert "broken-image:" in (result.stdout + result.stderr)


def test_quality_check_rejects_invalid_data_image(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "broken-data-image.html"
    _write_html(
        html_path,
        """
        <h1>Sampling report</h1>
        <p>Valid body content for image check.</p>
        <img src="data:image/png;base64,invalid" alt="broken-data" />
        """,
    )

    result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

    assert result.returncode != 0
    assert "broken-image:" in (result.stdout + result.stderr)


def test_quality_check_rejects_script_error(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "script-error.html"
    _write_html(
        html_path,
        """
        <h1>Sampling report</h1>
        <p>Body content before script failure.</p>
        <script>throw new Error('boom');</script>
        """,
    )

    result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

    assert result.returncode != 0
    assert "pageerror:" in (result.stdout + result.stderr)


def test_quality_check_rejects_horizontal_overflow(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "overflow.html"
    _write_html(
        html_path,
        """
        <h1>Sampling report</h1>
        <p>Body content with intentional document overflow.</p>
        <div style=\"width:2000px;height:24px;background:#eee\">Too wide</div>
        """,
    )

    result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

    assert result.returncode != 0
    assert "viewport overflow" in (result.stdout + result.stderr)


def test_quality_check_rejects_meaningful_text_clipped_by_hidden_overflow(tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        html_path = tmp_path / "hidden-overflow-table.html"
        _write_html(
                html_path,
                """
                <h1>Sampling report</h1>
                <p>Body content with intentionally clipped table text.</p>
                <div style="max-width:360px; overflow-x:hidden; border:1px solid #ddd;">
                    <table style="min-width:1200px; border-collapse:collapse;">
                        <tr>
                            <th>Very Long Header 1</th><th>Very Long Header 2</th><th>Very Long Header 3</th>
                        </tr>
                        <tr>
                            <td>Meaningful clipped cell content one</td><td>Meaningful clipped cell content two</td><td>Meaningful clipped cell content three</td>
                        </tr>
                    </table>
                </div>
                """,
        )

        result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

        assert result.returncode != 0
        assert "hidden-text-clipping" in (result.stdout + result.stderr)


def test_quality_check_accepts_scrollable_table_with_auto_overflow(tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        html_path = tmp_path / "auto-overflow-table.html"
        _write_html(
                html_path,
                """
                <h1>Sampling report</h1>
                <p>Body content with scrollable table wrapper.</p>
                <div style="max-width:360px; overflow-x:auto; border:1px solid #ddd;">
                    <table style="min-width:1200px; border-collapse:collapse;">
                        <tr>
                            <th>Very Long Header 1</th><th>Very Long Header 2</th><th>Very Long Header 3</th>
                        </tr>
                        <tr>
                            <td>Meaningful clipped cell content one</td><td>Meaningful clipped cell content two</td><td>Meaningful clipped cell content three</td>
                        </tr>
                    </table>
                </div>
                """,
        )

        result = _run_checker(repo_root, ["--html", str(html_path), "--browser", "chromium"])

        assert result.returncode == 0, result.stderr or result.stdout


def test_quality_check_pdf_requires_print_html(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    html_path = tmp_path / "interactive.html"
    _write_html(html_path, "<h1>Sampling report</h1><p>Body content for pdf guard.</p>")
    pdf_path = tmp_path / "out.pdf"

    result = _run_checker(
        repo_root,
        [
            "--html",
            str(html_path),
            "--pdf",
            str(pdf_path),
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode != 0
    assert "--print-html is required" in (result.stdout + result.stderr)


def test_quality_check_pdf_rejects_non_pdf_extension(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interactive = tmp_path / "interactive.html"
    print_html = tmp_path / "print.html"
    _write_html(interactive, "<h1>Interactive</h1><p>Interactive content body.</p>")
    _write_html(print_html, "<h1>Print</h1><p>Print content body.</p>")

    result = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(tmp_path / "not-pdf.txt"),
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode != 0
    assert "must use a .pdf extension" in (result.stdout + result.stderr)


def test_quality_check_pdf_exports_and_preserves_sources(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interactive = tmp_path / "interactive.html"
    print_html = tmp_path / "print.html"
    _write_html(interactive, "<h1>Interactive report</h1><p>Interactive body content.</p>")
    _write_html(print_html, "<h1>Print report</h1><h2>Section</h2><p>Print body content for export.</p>")

    interactive_hash_before = hashlib.sha256(interactive.read_bytes()).hexdigest()
    print_hash_before = hashlib.sha256(print_html.read_bytes()).hexdigest()

    pdf_path = tmp_path / "report.pdf"
    result = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(pdf_path),
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-")

    interactive_hash_after = hashlib.sha256(interactive.read_bytes()).hexdigest()
    print_hash_after = hashlib.sha256(print_html.read_bytes()).hexdigest()
    assert interactive_hash_after == interactive_hash_before
    assert print_hash_after == print_hash_before

    second = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(pdf_path),
            "--browser",
            "chromium",
        ],
    )
    assert second.returncode != 0
    assert "Refusing to overwrite existing PDF" in (second.stdout + second.stderr)

    overwrite = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(pdf_path),
            "--overwrite-pdf",
            "--browser",
            "chromium",
        ],
    )
    assert overwrite.returncode == 0, overwrite.stderr or overwrite.stdout


def test_quality_check_pdf_rejects_print_only_table_that_fails_a4_content_width(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interactive = tmp_path / "interactive.html"
    print_html = tmp_path / "print.html"
    _write_html(interactive, "<h1>Interactive report</h1><p>Interactive body content.</p>")
    _write_html(
        print_html,
        """
        <h1>Print report</h1>
        <p>Table fits desktop but not A4 content width.</p>
        <div style="max-width:100%; overflow-x:hidden; border:1px solid #ddd;">
          <table style="min-width:900px; border-collapse:collapse;">
            <tr>
              <th>Very Long Header 1</th><th>Very Long Header 2</th><th>Very Long Header 3</th><th>Very Long Header 4</th>
            </tr>
            <tr>
              <td>Meaningful clipped print content one</td><td>Meaningful clipped print content two</td><td>Meaningful clipped print content three</td><td>Meaningful clipped print content four</td>
            </tr>
          </table>
        </div>
        """,
    )

    pdf_path = tmp_path / "report.pdf"
    result = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(pdf_path),
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode != 0
    text = result.stdout + result.stderr
    assert "print-a4-content" in text
    assert "hidden-text-clipping" in text or "viewport overflow" in text


def test_quality_check_blocks_external_requests_for_validation_and_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interactive = tmp_path / "interactive.html"
    print_html = tmp_path / "print.html"
    _write_html(
        interactive,
        """
        <h1>Interactive report</h1>
        <p>External request should be blocked.</p>
        <img src="https://example.invalid/pixel.png" alt="blocked" />
        """,
    )
    _write_html(
        print_html,
        """
        <h1>Print report</h1>
        <p>External request should be blocked in print route too.</p>
        <img src="https://example.invalid/print-pixel.png" alt="blocked" />
        """,
    )

    pdf_path = tmp_path / "report.pdf"
    result = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--pdf",
            str(pdf_path),
            "--browser",
            "chromium",
        ],
    )

    assert result.returncode != 0
    summary = json.loads(result.stdout)
    blocked = sum(page.get("network", {}).get("blocked_request_count", 0) for page in summary["pages"])
    assert blocked >= 2
    assert "external-request-blocked:" in (result.stdout + result.stderr)


def test_quality_check_real_reports_from_canonical_run_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "outputs_sampling_v7" / "runs" / "v7-live-repeated-20260903"
    interactive = run_root / "interactive_report.html"
    print_html = run_root / "print_report.html"
    distance = run_root / "pca8-distance-report.html"

    if not (interactive.exists() and print_html.exists() and distance.exists()):
        import pytest

        pytest.skip("Canonical v7 run report files are not present in this workspace")

    interactive_result = _run_checker(repo_root, ["--html", str(interactive), "--browser", "chromium"])
    assert interactive_result.returncode == 0, interactive_result.stderr or interactive_result.stdout

    distance_result = _run_checker(repo_root, ["--html", str(distance), "--browser", "chromium"])
    assert distance_result.returncode == 0, distance_result.stderr or distance_result.stdout

    print_result = _run_checker(
        repo_root,
        [
            "--html",
            str(interactive),
            "--print-html",
            str(print_html),
            "--browser",
            "chromium",
        ],
    )
    assert print_result.returncode == 0, print_result.stderr or print_result.stdout
