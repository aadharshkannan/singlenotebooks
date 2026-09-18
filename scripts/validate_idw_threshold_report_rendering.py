from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json
from scripts.validate_matryoshka_report_rendering import SVG_CHECK


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every aggregate-only threshold report filter and chart.")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = args.report.resolve()
    aggregate = json.loads((report.parent / "aggregate.json").read_text(encoding="utf-8"))
    original_hash = sha256_file(report)
    captures = report.parent / "validation_screenshots"
    captures.mkdir(exist_ok=True)
    quality = runpy.run_path(str(Path(__file__).resolve().parents[1] / ".github" / "skills" / "sampling-experiment-report" / "quality_check.py"))
    failures = []
    checks = []
    blocked_requests = []
    failed_requests = []
    with sync_playwright() as playwright:
        browser = quality["_launch_browser"](playwright, "auto")
        try:
            for label, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                page = browser.new_page(viewport={"width": width, "height": height})
                page.on("pageerror", lambda error: failures.append(str(error)))
                page.on("console", lambda message: failures.append(message.text) if message.type == "error" else None)
                page.on("requestfailed", lambda request: failed_requests.append(request.failure))
                def reject(route):
                    blocked_requests.append("external request blocked")
                    failures.append("unexpected external request")
                    route.abort()
                page.route("http://**/*", reject)
                page.route("https://**/*", reject)
                try:
                    page.goto(report.as_uri(), wait_until="load")
                    content_issues, content_state = quality["_validate_dom_content"](page)
                    failures.extend(content_issues)
                    for selector in aggregate["selectors"]:
                        page.locator("#dataset").select_option(selector["dataset_id"])
                        page.locator("#dimension").select_option(str(selector["dimension"]))
                        page.locator("#rate").select_option(f'{selector["rate"]:g}')
                        expected = f'{selector["pooled_eligible_n"]:,} paired eligible / {selector["pooled_unjudged_n"]:,} all-unjudged repeated occurrences'
                        if page.locator("#denominator").inner_text() != expected:
                            failures.append("filter denominator does not match aggregate")
                        if page.locator("svg").count() != 5 or page.locator("svg polyline").count() != 10:
                            failures.append("filter did not render both methods in all five charts")
                        issues, _ = quality["_check_viewport"](page, width, height, label)
                        failures.extend(issues)
                        failures.extend(page.evaluate(SVG_CHECK))
                    for detail in page.locator("details").all():
                        detail.locator(":scope > summary").click()
                        if detail.get_attribute("open") is None:
                            failures.append("disclosure control did not open")
                    page.locator("#dataset").select_option("cosmos_otel")
                    page.locator("#dimension").select_option("8")
                    page.locator("#rate").select_option("0.1")
                    scrollable_charts = 0
                    if label == "mobile":
                        scrollable_charts = page.evaluate("""() => {
                            let count = 0;
                            for (const wrapper of document.querySelectorAll('.chart-scroll')) {
                                wrapper.scrollLeft = wrapper.scrollWidth;
                                if (wrapper.scrollLeft > 0) count++;
                                wrapper.scrollLeft = 0;
                            }
                            return count;
                        }""")
                        if scrollable_charts != 5:
                            failures.append("mobile charts cannot scroll to their full readable width")
                    for element, suffix in (("#data", "data"), ("#results", "charts"), ("#analysis", "analysis")):
                        page.locator(element).screenshot(path=str(captures / f"{label}-{suffix}.png"))
                    checks.append({
                        "viewport": [width, height], "selector_combinations_checked": len(aggregate["selectors"]),
                        "captured_filter": "cosmos_otel / 8d / 10%",
                        "captured_denominator": page.locator("#denominator").inner_text(),
                        "captured_auc": page.locator("#roc-note").inner_text(),
                        "content": content_state,
                        "mobile_charts_scroll_verified": scrollable_charts,
                    })
                finally:
                    page.close()
        finally:
            browser.close()
    if sha256_file(report) != original_hash:
        failures.append("source report changed during validation")
    failures.extend(failed_requests)
    result = {
        "ok": not failures, "checks": checks, "failures": failures, "html_sha256": original_hash,
        "network": {"blocked_request_count": len(blocked_requests), "failed_request_count": len(failed_requests)},
        "validation_method": "Repository skill DOM/content and viewport gates plus all 30 filter combinations, SVG-label geometry, console/page-error and external-network guards.",
    }
    write_json(report.parent / "interaction_validation.json", result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("Threshold report interaction gate failed.")


if __name__ == "__main__":
    main()
