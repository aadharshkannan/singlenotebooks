from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


def validate_report(report: Path, screenshots: Path) -> dict:
    from playwright.sync_api import sync_playwright

    checker_path = Path(__file__).resolve().parents[1] / ".github/skills/sampling-experiment-report/quality_check.py"
    spec = importlib.util.spec_from_file_location("report_quality", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_hash = sha256_file(report)
    screenshots.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    checks = []
    with sync_playwright() as playwright:
        browser = checker._launch_browser(playwright, "auto")
        page = browser.new_page()
        page.on("pageerror", lambda error: issues.append(str(error)))
        page.on("console", lambda message: issues.append(message.text) if message.type == "error" else None)

        def route_request(route):
            if route.request.url.startswith(("http://", "https://")):
                issues.append("unexpected external request")
                route.abort()
            else:
                route.continue_()

        page.route("**/*", route_request)
        page.goto(report.resolve().as_uri(), wait_until="load")
        data = page.locator("#report-data").text_content()
        payload = json.loads(data)
        for label, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            for tab in ("overview", "method", "dataset", "results", "provenance"):
                page.locator(f"#tab-{tab}").click()
                if not page.locator(f"#{tab}").is_visible():
                    issues.append(f"{label}/{tab}: tab did not activate")
                problems, state = checker._check_viewport(page, width, height, f"{label}/{tab}")
                issues.extend(problems)
                page.screenshot(path=str(screenshots / f"{label}-{tab}.png"), full_page=True)
                checks.append({"viewport": label, "tab": tab, "state": state, "ok": not problems})
            page.locator("#tab-results").click()
            interactions = 0
            for rate in payload["protocol"]["rates"]:
                page.select_option("#budget", str(rate))
                for schedule in ["all", *payload["protocol"]["schedules"]]:
                    page.select_option("#schedule", schedule)
                    for dimension in payload["protocol"]["dimensions"]:
                        page.select_option("#dimension", str(dimension))
                        interactions += 1
                        if payload["status"] == "completed":
                            expected = [
                                row for row in payload["summaries"]
                                if row["schedule"] == schedule and row["rate"] == rate
                            ]
                            actual = page.locator("#metrics-table tbody tr").count()
                            if actual != len(expected) * 4:
                                issues.append("metric table does not match selected scope")
                            if str(dimension) + "d:" not in page.locator("#roc-takeaway").inner_text():
                                chosen = next(r for r in expected if r["dimension"] == dimension)
                                if chosen["roc"]["point"]["tpr"] and chosen["roc"]["lower"]["tpr"]:
                                    issues.append("ROC dimension did not update")
                            first = next(r for r in expected if r["dimension"] == payload["protocol"]["dimensions"][0])
                            expected_mae = first["cohorts"]["all_unselected"]["mae"]["mean"]
                            shown_mae = page.locator("#metrics-table tbody tr").first.locator("td").nth(3).inner_text()
                            if shown_mae != ("Not measured" if expected_mae is None else f"{expected_mae:.3f}"):
                                issues.append("displayed MAE differs from aggregate")
                        elif page.locator("#results svg").count() != 0:
                            issues.append("pending report fabricated a result chart")
            checks.append({"viewport": label, "filter_combinations": interactions})
            page.locator("#tab-overview").focus()
            page.keyboard.press("ArrowRight")
            if page.locator("#tab-method").get_attribute("aria-selected") != "true":
                issues.append("keyboard tab navigation failed")
        browser.close()
    if sha256_file(report) != original_hash:
        issues.append("HTML source changed during validation")
    return {
        "ok": not issues, "report_sha256": original_hash,
        "status": payload["status"], "checks": checks, "issues": issues,
        "scope": "All report tabs, desktop/mobile overflow, all filter combinations, numeric MAE parity and keyboard tab navigation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check every IMDb report tab and interactive control offline.")
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_report(args.html, args.screenshots)
    write_json(args.output, result)
    print(json.dumps({"ok": result["ok"], "issues": result["issues"]}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
