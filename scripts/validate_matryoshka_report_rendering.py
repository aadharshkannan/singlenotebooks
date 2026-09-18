from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


SVG_CHECK = """
() => [...document.querySelectorAll('svg')].flatMap((svg, chart) => {
  const boundary = svg.getBoundingClientRect();
  if (!boundary.width || !boundary.height) return [];
  const text = [...svg.querySelectorAll('text')].map(node => ({
    value: node.textContent, box: node.getBoundingClientRect()
  }));
  const issues = [];
  for (let i = 0; i < text.length; i++) {
    const a = text[i];
    if (a.box.left < boundary.left - 1 || a.box.right > boundary.right + 1 ||
        a.box.top < boundary.top - 1 || a.box.bottom > boundary.bottom + 1)
      issues.push({chart, issue: 'text-outside-svg', text: a.value});
    for (let j = i + 1; j < text.length; j++) {
      const b = text[j];
      const width = Math.min(a.box.right, b.box.right) - Math.max(a.box.left, b.box.left);
      const height = Math.min(a.box.bottom, b.box.bottom) - Math.max(a.box.top, b.box.top);
      if (width > 1 && height > 1)
        issues.push({chart, issue: 'overlapping-text', text: [a.value, b.value]});
    }
  }
  return issues;
})
"""


def check_trend_mode(page, summary: dict, mode: str) -> dict:
    chart = page.locator("#trend-chart")
    if chart.locator("svg").count() != 1:
        raise AssertionError("trend selection must display exactly one SVG")
    state = chart.evaluate("""element => ({
        namespace: element.firstElementChild.namespaceURI,
        width: element.firstElementChild.getBoundingClientRect().width,
        text: [...element.querySelectorAll('text')].map(node => node.textContent),
        points: [...element.querySelectorAll('circle')].map(node => ({
            dataset: node.dataset.dataset, dimension: node.dataset.dimension,
            value: Number(node.dataset.value), y: Number(node.getAttribute('cy')),
            tooltip: node.querySelector('title').textContent
        }))
    })""")
    if state["namespace"] != "http://www.w3.org/2000/svg" or state["width"] <= 0:
        raise AssertionError("selected trend must render as a visible SVG")
    expected = {}
    maximum = max(value for row in summary["datasets"] for value in row["curve"].values())
    high = max(0.1, math.ceil(maximum * 10) / 10)
    for row in summary["datasets"]:
        for dimension in summary["protocol"]["dimensions"]:
            mae = row["curve"][str(dimension)]
            if mode == "relative" and row["native_mae"] == 0:
                continue
            value = 100 * (mae / row["native_mae"] - 1) if mode == "relative" else mae
            expected[(row["dataset_id"], str(dimension))] = value
    actual = {(point["dataset"], point["dimension"]): point["value"] for point in state["points"]}
    if actual.keys() != expected.keys() or len(actual) != len(state["points"]):
        raise AssertionError("trend selection lost or duplicated measured points")
    for point in state["points"]:
        value = expected[(point["dataset"], point["dimension"])]
        if not math.isclose(point["value"], value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("plotted trend value differs from retained summary")
        if mode == "mae":
            if not math.isclose(point["y"], 72 + (high - value) * 280 / high, abs_tol=0.001):
                raise AssertionError("MAE value is plotted against the wrong vertical axis")
            if "%" in point["tooltip"] or f"MAE {value:.6f}" not in point["tooltip"]:
                raise AssertionError("MAE tooltip still describes a percentage")
    caption = page.locator("#trend-caption").inner_text()
    if mode == "mae":
        if any("%" in text for text in state["text"]) or "zero-based MAE axis" not in caption:
            raise AssertionError("MAE selection did not update its axes and explanation")
    elif not any("%" in text for text in state["text"]) or "below zero is better" not in caption:
        raise AssertionError("percentage selection did not restore axes and explanation")
    return {"mode": mode, "measured_points_verified": len(expected), "caption": caption}


def validate(html: Path, browser) -> dict:
    source_hash = sha256_file(html)
    screenshots = html.parent / "validation_screenshots"
    screenshots.mkdir(exist_ok=True)
    quality = runpy.run_path(str(Path(__file__).resolve().parents[1] / ".github" / "skills" / "sampling-experiment-report" / "quality_check.py"))
    failures = []
    captures = []
    results = []
    summary_path = html.parent / "summary.json"
    for label, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height})
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.on("console", lambda message: failures.append(message.text) if message.type == "error" else None)
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        try:
            page.goto(html.resolve().as_uri(), wait_until="load")
            toggled = []
            for details in page.locator("details").all():
                if details.get_attribute("open") is None:
                    details.locator(":scope > summary").click()
                    if details.get_attribute("open") is None:
                        raise AssertionError("details control did not open")
                    toggled.append(details.get_attribute("id"))
            viewport_issues, viewport = quality["_check_viewport"](page, width, height, label)
            svg_issues = page.evaluate(SVG_CHECK)
            failures.extend(viewport_issues)
            failures.extend(svg_issues)
            trend_checks = []
            select = page.locator("#trend-scale")
            if select.count():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if select.is_disabled() or select.input_value() != "relative":
                    raise AssertionError("trend selector must start enabled in percentage mode")
                untouched = [
                    page.locator(f'figure[data-graph="{name}"]').inner_html()
                    for name in ("endpoints", "budgets")
                ]
                for mode in ("relative", "mae", "relative"):
                    select.select_option(mode)
                    trend_checks.append(check_trend_mode(page, summary, mode))
                    issues, _ = quality["_check_viewport"](page, width, height, label)
                    failures.extend(issues)
                    failures.extend(page.evaluate(SVG_CHECK))
                    filename = screenshots / f"{label}-{html.stem}-trend-{mode}.png"
                    page.locator('figure[data-graph="trend"]').screenshot(path=str(filename))
                    if str(filename) not in captures:
                        captures.append(str(filename))
                select.focus()
                select.press("ArrowDown")
                if select.input_value() != "mae":
                    raise AssertionError("keyboard cannot select MAE")
                check_trend_mode(page, summary, "mae")
                select.press("ArrowUp")
                if select.input_value() != "relative":
                    raise AssertionError("keyboard cannot restore percentage")
                if page.locator("svg").count() != 3 or untouched != [
                    page.locator(f'figure[data-graph="{name}"]').inner_html()
                    for name in ("endpoints", "budgets")
                ]:
                    raise AssertionError("switching graph 2 modified another chart")
            charts = page.locator("svg")
            for index in range(min(charts.count(), 3)):
                target = charts.nth(index).locator("xpath=..")
                filename = screenshots / f"{label}-{html.stem}-chart-{index + 1}.png"
                target.screenshot(path=str(filename))
                captures.append(str(filename))
            if page.locator("#results").count():
                filename = screenshots / f"{label}-{html.stem}-results.png"
                page.locator("#results").screenshot(path=str(filename))
                captures.append(str(filename))
            results.append({
                "viewport": viewport, "details_opened": toggled,
                "svg_count": charts.count(), "svg_text_issues": svg_issues,
                "trend_scale_checks": trend_checks,
                "trend_keyboard_checked": bool(trend_checks),
            })
        finally:
            page.close()
    if sha256_file(html) != source_hash:
        failures.append("source HTML changed during validation")
    return {
        "status": "passed" if not failures else "failed", "html_sha256": source_hash,
        "checks": results, "failures": failures, "screenshots": captures,
        "scope": "Details controls, desktop/mobile overflow, SVG labels, trend dropdown/keyboard round-trip and point-for-point parity with retained summary. Screenshots retained for separate visual inspection; no PDF requested.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise Matryoshka report controls and verify dense dimension-axis rendering.")
    parser.add_argument("--html", action="append", required=True, type=Path)
    args = parser.parse_args()
    quality = runpy.run_path(str(Path(__file__).resolve().parents[1] / ".github" / "skills" / "sampling-experiment-report" / "quality_check.py"))
    failed = False
    with sync_playwright() as playwright:
        browser = quality["_launch_browser"](playwright, "auto")
        try:
            for html in args.html:
                result = validate(html, browser)
                write_json(html.parent / "visual_validation.json", result)
                print(f"{result['status']}: {html} ({len(result['failures'])} issues)", flush=True)
                failed |= result["status"] != "passed"
        finally:
            browser.close()
    if failed:
        raise SystemExit("Report rendering gate failed; inspect visual_validation.json.")


if __name__ == "__main__":
    main()
