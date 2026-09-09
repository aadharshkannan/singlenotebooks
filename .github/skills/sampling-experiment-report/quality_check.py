#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover - surfaced as a clear runtime error
    raise SystemExit(f"Playwright is required for report quality checks. Install the repo environment first. {exc}") from exc


def _as_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _validate_dom_content(page: Any) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    state: dict[str, Any] = page.evaluate(
        r"""
        () => {
          const title = (document.title || '').trim();
          const bodyText = (document.body?.innerText || '').replace(/\s+/g, ' ').trim();
          const headings = [...document.querySelectorAll('h1, h2, h3')]
            .map((h) => (h.textContent || '').replace(/\s+/g, ' ').trim())
            .filter(Boolean);
          const brokenImages = [...document.querySelectorAll('img')]
            .filter((img) => {
              const src = img.currentSrc || img.getAttribute('src') || '';
                            if (!src) return false;
              return !img.complete || img.naturalWidth === 0;
            })
            .slice(0, 20)
            .map((img) => img.currentSrc || img.getAttribute('src') || '[img-without-src]');

          return {
            title,
            bodyTextLength: bodyText.length,
            headingCount: headings.length,
            headingsPreview: headings.slice(0, 6),
            brokenImages,
          };
        }
        """
    )

    if not state.get("title"):
        issues.append("missing-title")
    if state.get("headingCount", 0) < 1:
        issues.append("missing-headings")
    if state.get("bodyTextLength", 0) < 40:
        issues.append(f"insufficient-body-text:{state.get('bodyTextLength', 0)}")
    for img in state.get("brokenImages", []):
        issues.append(f"broken-image:{img}")
    return issues, state


def _check_viewport(page: Any, width: int, height: int, label: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    page.set_viewport_size({"width": width, "height": height})
    state: dict[str, Any] = page.evaluate(
        r"""
        () => {
                    const canScrollX = (el) => {
            for (let cur = el; cur && cur !== document.documentElement; cur = cur.parentElement) {
              const style = getComputedStyle(cur);
              const ox = style.overflowX;
              if ((ox === 'auto' || ox === 'scroll') && cur.scrollWidth > cur.clientWidth + 1) {
                return true;
              }
            }
            return false;
          };

                    const isSuppressed = (el) => {
                        for (let cur = el; cur && cur !== document.documentElement; cur = cur.parentElement) {
                            if (cur.hasAttribute('hidden') || cur.getAttribute('aria-hidden') === 'true') {
                                return true;
                            }
                            const style = getComputedStyle(cur);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return true;
                            }
                        }
                        return false;
                    };

                    const hasMeaningfulVisibleText = (el) => {
                        if (!el || isSuppressed(el) || el.closest('svg')) {
                            return false;
                        }
                        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
                        let total = 0;
                        while (walker.nextNode()) {
                            const node = walker.currentNode;
                            const text = (node.textContent || '').replace(/\s+/g, ' ').trim();
                            if (!text) {
                                continue;
                            }
                            const parent = node.parentElement;
                            if (!parent || parent.closest('svg') || isSuppressed(parent)) {
                                continue;
                            }
                            const tag = parent.tagName;
                            if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE') {
                                continue;
                            }
                            const rect = parent.getBoundingClientRect();
                            if (rect.width <= 0 || rect.height <= 0) {
                                continue;
                            }
                            total += text.length;
                            if (total >= 20) {
                                return true;
                            }
                        }
                        return false;
                    };

          const offenders = [];
                    const clippedHidden = [];
          for (const el of document.querySelectorAll('body *')) {
            const rect = el.getBoundingClientRect();
            if ((rect.width <= 0 && rect.height <= 0) || !Number.isFinite(rect.right)) {
              continue;
            }
            if (rect.right > window.innerWidth + 1 && !canScrollX(el)) {
              offenders.push({
                tag: el.tagName,
                id: el.id || '',
                className: (el.className || '').toString().slice(0, 80),
                right: Math.round(rect.right),
              });
            }

                        const style = getComputedStyle(el);
                        if ((style.overflowX === 'hidden' || style.overflowX === 'clip')
                                && el.scrollWidth > el.clientWidth + 1
                                && hasMeaningfulVisibleText(el)
                                && !isSuppressed(el)
                                && !el.closest('svg')) {
                            clippedHidden.push({
                                tag: el.tagName,
                                id: el.id || '',
                                className: (el.className || '').toString().slice(0, 80),
                                scrollWidth: Math.round(el.scrollWidth),
                                clientWidth: Math.round(el.clientWidth),
                            });
                        }
          }

          return {
            width: window.innerWidth,
            height: window.innerHeight,
            scrollWidth: document.documentElement.scrollWidth,
            scrollHeight: document.documentElement.scrollHeight,
            overflowing: offenders.slice(0, 10),
                        hiddenClipping: clippedHidden.slice(0, 10),
          };
        }
        """
    )

    if state.get("scrollWidth", width) > width + 1:
        issues.append(f"{label}: viewport overflow: scrollWidth {state.get('scrollWidth')} > {width}")
    if state.get("overflowing"):
        issues.append(f"{label}: overflowing-elements:{len(state['overflowing'])}")
    if state.get("hiddenClipping"):
        issues.append(f"{label}: hidden-text-clipping:{len(state['hiddenClipping'])}")
    return issues, state


def _launch_browser(playwright: Any, browser_mode: str) -> Any:
    if browser_mode == "chromium":
        return playwright.chromium.launch(headless=True)
    if browser_mode == "msedge":
        return playwright.chromium.launch(channel="msedge", headless=True)

    if sys.platform.startswith("win"):
        try:
            return playwright.chromium.launch(channel="msedge", headless=True)
        except Exception:
            return playwright.chromium.launch(headless=True)
    return playwright.chromium.launch(headless=True)


def _allow_url(url: str) -> bool:
    return url.startswith("file:") or url.startswith("data:") or url.startswith("blob:") or url.startswith("about:")


def _validate_single_page(
    context: Any,
    html_uri: str,
    label: str,
    width: int,
    height: int,
    timeout_ms: int,
    screenshot_path: Path | None,
    emulate_print: bool,
    block_external: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    page = context.new_page()
    page.set_viewport_size({"width": width, "height": height})

    page_errors: list[str] = []
    failed_requests: list[str] = []
    bad_responses: list[str] = []
    blocked_requests: list[str] = []

    def on_pageerror(error: Any) -> None:
        page_errors.append(f"pageerror:{error}")

    def on_console(msg: Any) -> None:
        if msg.type == "error":
            page_errors.append(f"console:error:{msg.text}")

    def on_requestfailed(request: Any) -> None:
        failed_requests.append(f"{request.method} {request.url}")

    def on_response(response: Any) -> None:
        if _allow_url(response.url):
            return
        if response.status >= 400:
            bad_responses.append(f"{response.status}:{response.url}")

    def route_handler(route: Any) -> None:
        req = route.request
        if _allow_url(req.url):
            route.continue_()
            return
        blocked_requests.append(req.url)
        route.abort()

    page.on("pageerror", on_pageerror)
    page.on("console", on_console)
    page.on("requestfailed", on_requestfailed)
    page.on("response", on_response)

    try:
        if block_external:
            page.route("**/*", route_handler)
        page.goto(html_uri, wait_until="networkidle", timeout=timeout_ms)
        if emulate_print:
            page.emulate_media(media="print")

        content_issues, content_state = _validate_dom_content(page)
        viewport_issues, viewport_state = _check_viewport(page, width, height, label)
        issues.extend(content_issues)
        issues.extend(viewport_issues)

        issues.extend(page_errors)
        for req in failed_requests[:20]:
            issues.append(f"request-failed:{req}")
        for resp in bad_responses[:20]:
            issues.append(f"bad-response:{resp}")
        for req in blocked_requests[:20]:
            issues.append(f"external-request-blocked:{req}")

        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path), full_page=False)

        return {
            "label": label,
            "viewport": {"width": width, "height": height},
            "state": {
                "content": content_state,
                "viewport": viewport_state,
            },
            "network": {
                "blocked_request_count": len(blocked_requests),
                "failed_request_count": len(failed_requests),
                "bad_response_count": len(bad_responses),
            },
            "issues": issues,
            "ok": not issues,
        }
    finally:
        page.close()


def _export_pdf_from_print_source(
    context: Any,
    print_html_path: Path,
    pdf_path: Path,
    timeout_ms: int,
    block_external: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    page = context.new_page()
    page.set_viewport_size({"width": 1440, "height": 1000})

    page_errors: list[str] = []
    failed_requests: list[str] = []
    bad_responses: list[str] = []
    blocked_requests: list[str] = []

    def on_pageerror(error: Any) -> None:
        page_errors.append(f"pageerror:{error}")

    def on_console(msg: Any) -> None:
        if msg.type == "error":
            page_errors.append(f"console:error:{msg.text}")

    def on_requestfailed(request: Any) -> None:
        failed_requests.append(f"{request.method} {request.url}")

    def on_response(response: Any) -> None:
        if _allow_url(response.url):
            return
        if response.status >= 400:
            bad_responses.append(f"{response.status}:{response.url}")

    def route_handler(route: Any) -> None:
        req = route.request
        if _allow_url(req.url):
            route.continue_()
            return
        blocked_requests.append(req.url)
        route.abort()

    page.on("pageerror", on_pageerror)
    page.on("console", on_console)
    page.on("requestfailed", on_requestfailed)
    page.on("response", on_response)

    try:
        if block_external:
            page.route("**/*", route_handler)
        page.goto(_as_uri(print_html_path), wait_until="networkidle", timeout=timeout_ms)
        page.emulate_media(media="print")

        content_issues, content_state = _validate_dom_content(page)
        # A4 portrait content area with 10mm margins is approximately 718 CSS px.
        viewport_issues, viewport_state = _check_viewport(page, 718, 1000, "print-a4-content")
        issues.extend(content_issues)
        issues.extend(viewport_issues)
        issues.extend(page_errors)
        for req in failed_requests[:20]:
            issues.append(f"request-failed:{req}")
        for resp in bad_responses[:20]:
            issues.append(f"bad-response:{resp}")
        for req in blocked_requests[:20]:
            issues.append(f"external-request-blocked:{req}")

        if issues:
            return {
                "label": "pdf-export",
                "viewport": {"width": 718, "height": 1000},
                "state": {
                    "content": content_state,
                    "viewport": viewport_state,
                },
                "network": {
                    "blocked_request_count": len(blocked_requests),
                    "failed_request_count": len(failed_requests),
                    "bad_response_count": len(bad_responses),
                },
                "issues": issues,
                "ok": False,
            }

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=False,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        if not pdf_path.exists() or pdf_path.stat().st_size < 8:
            raise SystemExit(f"PDF export failed: {pdf_path}")
        with pdf_path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise SystemExit(f"PDF export failed (magic bytes mismatch): {pdf_path}")

        return {
            "label": "pdf-export",
            "viewport": {"width": 718, "height": 1000},
            "state": {
                "content": content_state,
                "viewport": viewport_state,
            },
            "network": {
                "blocked_request_count": len(blocked_requests),
                "failed_request_count": len(failed_requests),
                "bad_response_count": len(bad_responses),
            },
            "issues": [],
            "ok": True,
        }
    finally:
        page.close()


def _ensure_screenshot_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_html_input(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"{label} must be an HTML file: {path}")
    return path.resolve()


def run_quality_check(
    html_path: Path,
    print_html_path: Path | None,
    screenshots_dir: Path | None,
    timeout_seconds: int,
    pdf_path: Path | None,
    overwrite_pdf: bool,
    browser_mode: str,
    block_external: bool,
) -> int:
    html_path = _validate_html_input(html_path, "--html")
    print_html_path = _validate_html_input(print_html_path, "--print-html") if print_html_path else None

    if pdf_path is not None and print_html_path is None:
        raise ValueError("--print-html is required when --pdf is set")

    if pdf_path is not None:
        pdf_path = pdf_path.resolve()
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"--pdf must use a .pdf extension: {pdf_path}")
        if pdf_path.exists() and not overwrite_pdf:
            raise FileExistsError(f"Refusing to overwrite existing PDF: {pdf_path}")
        if pdf_path == html_path or (print_html_path is not None and pdf_path == print_html_path):
            raise ValueError("--pdf output path must differ from HTML source files")

    screenshot_root = _ensure_screenshot_dir(screenshots_dir) if screenshots_dir else None
    html_uri = _as_uri(html_path)
    timeout_ms = max(5_000, timeout_seconds * 1000)
    summary: dict[str, Any] = {
        "html": str(html_path),
        "print_html": str(print_html_path) if print_html_path else None,
        "pdf": str(pdf_path) if pdf_path else None,
        "browser": browser_mode,
        "block_external": block_external,
        "pages": [],
        "ok": True,
    }

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright, browser_mode)
        context = browser.new_context()
        try:
            for label, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                snapshot = screenshot_root / f"{label}-{html_path.stem}.png" if screenshot_root is not None else None
                result = _validate_single_page(
                    context=context,
                    html_uri=html_uri,
                    label=label,
                    width=width,
                    height=height,
                    timeout_ms=timeout_ms,
                    screenshot_path=snapshot,
                    emulate_print=False,
                    block_external=block_external,
                )
                summary["pages"].append(result)

            if print_html_path is not None:
                print_uri = _as_uri(print_html_path)
                snapshot = screenshot_root / f"print-{print_html_path.stem}.png" if screenshot_root is not None else None
                result = _validate_single_page(
                    context=context,
                    html_uri=print_uri,
                    label="print",
                    width=1440,
                    height=1000,
                    timeout_ms=timeout_ms,
                    screenshot_path=snapshot,
                    emulate_print=True,
                    block_external=block_external,
                )
                summary["pages"].append(result)

            if pdf_path is not None and print_html_path is not None:
                export_result = _export_pdf_from_print_source(
                    context=context,
                    print_html_path=print_html_path,
                    pdf_path=pdf_path,
                    timeout_ms=timeout_ms,
                    block_external=block_external,
                )
                summary["pages"].append(export_result)

            all_issues = [issue for page in summary["pages"] for issue in page["issues"]]
            if all_issues:
                summary["ok"] = False
                print(json.dumps(summary, indent=2))
                raise SystemExit("validation failed: " + "; ".join(all_issues[:20]))

            print(json.dumps(summary, indent=2))
        finally:
            context.close()
            browser.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local HTML/PDF quality gate for sampling experiment reports.")
    parser.add_argument("--html", type=Path, required=True, help="Path to the local report HTML file.")
    parser.add_argument("--print-html", type=Path, default=None, help="Path to the static print HTML. Required when --pdf is set.")
    parser.add_argument("--screenshots", type=Path, default=None, help="Optional directory for viewport screenshots. Defaults to no screenshots.")
    parser.add_argument("--pdf", type=Path, default=None, help="Optional PDF path to export from the static print HTML.")
    parser.add_argument("--overwrite-pdf", action="store_true", help="Allow overwriting an existing PDF path.")
    parser.add_argument(
        "--browser",
        choices=["auto", "msedge", "chromium"],
        default="auto",
        help="Browser selection. 'auto' prefers msedge on Windows and falls back to chromium.",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow external network requests. By default, non-local network requests are blocked.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-page timeout in seconds.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return run_quality_check(
            html_path=args.html,
            print_html_path=args.print_html,
            screenshots_dir=args.screenshots,
            timeout_seconds=args.timeout_seconds,
            pdf_path=args.pdf,
            overwrite_pdf=args.overwrite_pdf,
            browser_mode=args.browser,
            block_external=not args.allow_external,
        )
    except Exception as exc:  # pragma: no cover - user-facing failure surface
        print(f"quality-check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
