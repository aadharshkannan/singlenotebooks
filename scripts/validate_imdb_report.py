from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


def validate_dimension_comparisons(page) -> dict:
    """Check every independent dimension pair against retained numeric summaries."""
    return page.evaluate(r"""() => {
        const data=JSON.parse(document.getElementById("report-data").textContent);
        const $=id=>document.getElementById(id),issues=[];
        const refLabel=d=>d===1536?"Native 1536":`MRL-${d}`;
        const fixed=(v,n=3)=>v==null?"Not measured":Number(v).toFixed(n);
        const fail=message=>{if(issues.length<20)issues.push(message)};
        const choose=(id,value)=>{$(id).value=String(value)};
        const update=id=>$(id).dispatchEvent(new Event("change",{bubbles:true}));
        const reference=(d,rate,schedule)=>data.summaries.find(r=>r.dimension===d&&r.rate===rate&&r.schedule===schedule);
        const pca=(d,rate,schedule)=>data.pca_summaries.find(r=>r.dimension===d&&r.rate===rate&&r.schedule===schedule);
        const beforeConclusions=$("pca-conclusion").textContent;
        let rocPairs=0,budgetPairs=0;
        const saved=Object.fromEntries(["pca-budget","pca-schedule","pca-dimension","pca-mrl-dimension","compare-pca-dimension","compare-mrl-dimension"].map(id=>[id,$(id).value]));
        for(const rate of data.protocol.rates)for(const schedule of ["all",...data.protocol.schedules]){
            choose("pca-budget",rate);choose("pca-schedule",schedule);
            for(const pd of data.pca_study.dimensions)for(const md of data.protocol.dimensions){
                choose("pca-dimension",pd);choose("pca-mrl-dimension",md);update("pca-mrl-dimension");rocPairs++;
                const expected=[[reference(md,rate,schedule),refLabel(md)],[pca(pd,rate,schedule),`PCA-${pd}`]];
                const rows=[...$("pca-envelope-table").tBodies[0].rows];
                if(rows.length!==4){fail("independent ROC cohort table is incomplete");continue}
                for(let family=0;family<2;family++)for(let side=0;side<2;side++){
                    const [summary,label]=expected[family],method=side===0?"point":"lower",cohort=side===0?"eligible_point":"eligible_lower";
                    const row=rows[family*2+side],metrics=summary.cohorts[cohort];
                    if(row.cells[0].textContent!==label)fail("ROC cohort representation label mismatches chosen dimension");
                    ["mae","accuracy","precision","recall","f1","auc"].forEach((metric,index)=>{
                        if(row.cells[index+3].textContent!==fixed(metrics[metric].mean))fail(`ROC ${label}/${metric} does not match source summary`);
                    });
                    const curve=summary.roc[method];
                    const paths=[...$("pca-roc-chart").querySelectorAll("path[data-representation]")].filter(path=>path.dataset.representation===label&&path.dataset.estimator===method);
                    if(curve.tpr.length){
                        const expectedPath=curve.fpr.map((value,i)=>`${i?"L":"M"}${65+value*650},${260-curve.tpr[i]*240}`).join(" ");
                        if(paths.length!==1||paths[0].getAttribute("d")!==expectedPath)fail("ROC plotted curve does not match selected PCA/MRL data");
                    }else if(paths.length)fail("undefined ROC rendered as measured data");
                }
                const scope=$("pca-roc-scope").textContent;
                if(!scope.includes(`PCA-${pd}`)||!scope.includes(refLabel(md)))fail("ROC scope omits selected family/dimension");
                if($("pca-roc-reference-label").textContent!==refLabel(md)||$("pca-roc-pca-label").textContent!==`PCA-${pd}`)fail("ROC legend is stale");
            }
        }
        const rocBefore=$("pca-roc-chart").innerHTML,allDimensionBefore=$("pca-mae-chart").innerHTML;
        for(const pd of data.pca_study.dimensions)for(const md of data.protocol.dimensions){
            choose("compare-pca-dimension",pd);choose("compare-mrl-dimension",md);update("compare-mrl-dimension");budgetPairs++;
            const rows=[...$("pca-budget-table").tBodies[0].rows],headers=[...$("pca-budget-table").tHead.rows[0].cells].map(c=>c.textContent);
            if(headers[2]!==`${refLabel(md)} MAE`||headers[3]!==`PCA-${pd} MAE`||headers[4]!==`${refLabel(md)} accuracy`||headers[5]!==`PCA-${pd} accuracy`)fail("Across-budget headers do not match selected dimensions");
            if(rows.length!==data.protocol.rates.length){fail("Across-budget rows missing");continue}
            data.protocol.rates.forEach((rate,index)=>{
                const n=reference(1536,rate,"all"),m=reference(md,rate,"all"),p=pca(pd,rate,"all");
                const candidates=data.pca_study.dimensions.map(d=>pca(d,rate,"all")),minimum=Math.min(...candidates.map(c=>c.cohorts.all_unselected.mae.mean));
                const best=candidates.filter(c=>Math.abs(c.cohorts.all_unselected.mae.mean-minimum)<1e-12).map(c=>`PCA-${c.dimension}`).join(", ");
                const expected=[`${100*rate}%`,fixed(n.cohorts.all_unselected.mae.mean,4),fixed(m.cohorts.all_unselected.mae.mean,4),fixed(p.cohorts.all_unselected.mae.mean,4),`${fixed(100*m.cohorts.all_unselected.accuracy.mean,2)}%`,`${fixed(100*p.cohorts.all_unselected.accuracy.mean,2)}%`,best];
                if(JSON.stringify([...rows[index].cells].map(c=>c.textContent))!==JSON.stringify(expected))fail("Across-budget table values do not match selected PCA/MRL summaries");
                for(const [key,summary] of [["native",n],["mrl",m],["pca",p]]){
                    const points=$("pca-budget-chart").querySelectorAll(`circle[data-series="${key}"]`);
                    if(points.length!==data.protocol.rates.length||Number(points[index]?.dataset.value)!==summary.cohorts.all_unselected.mae.mean)fail("Across-budget plotted data does not match numeric table");
                }
            });
            if($("pca-roc-chart").innerHTML!==rocBefore||$("pca-mae-chart").innerHTML!==allDimensionBefore)fail("Across-budget selectors unexpectedly changed upper comparison scope");
            if($("budget-mrl-label").textContent!==refLabel(md)||$("budget-pca-label").textContent!==`PCA-${pd}`)fail("Across-budget legend is stale");
        }
        const budgetBefore=$("pca-budget-table").innerHTML;
        choose("pca-dimension",data.pca_study.dimensions[0]);update("pca-dimension");
        if($("pca-budget-table").innerHTML!==budgetBefore)fail("Upper PCA controls unexpectedly changed the across-budget table");
        if($("pca-conclusion").textContent!==beforeConclusions)fail("Fixed recorded findings changed with comparison selectors");
        for(const [id,value] of Object.entries(saved))choose(id,value);
        update("pca-dimension");update("compare-pca-dimension");
        return {roc_dimension_pairs:rocPairs,budget_dimension_pairs:budgetPairs,issues};
    }""")


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
        if page.locator("#tab-results").inner_text() != "MRL results":
            issues.append("coordinate-shortening tab is not labeled MRL results")
        if "Matryoshka Representation Learning (MRL)" not in page.locator("#method-key").text_content():
            issues.append("MRL method is not expanded and explained")
        if "pca_study" in payload and page.locator("#compare-pca-dimension").input_value() != str(
                8 if 8 in payload["pca_study"]["dimensions"] else payload["pca_study"]["dimensions"][0]):
            issues.append("across-budget PCA comparison lost its expected initial dimension")
        for label, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            tabs = ["overview", "method", "dataset", "results", "provenance"]
            if "pca_study" in payload:
                tabs.insert(4, "pca")
            for tab in tabs:
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
                            representation_label = "Native 1536" if dimension == 1536 else f"MRL-{dimension}"
                            if representation_label + ":" not in page.locator("#roc-takeaway").inner_text():
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
            if "pca_study" in payload:
                page.locator("#tab-pca").click()
                pca_interactions = 0
                for rate in payload["protocol"]["rates"]:
                    page.select_option("#pca-budget", str(rate))
                    for schedule in ["all", *payload["protocol"]["schedules"]]:
                        page.select_option("#pca-schedule", schedule)
                        for dimension in payload["pca_study"]["dimensions"]:
                            page.select_option("#pca-dimension", str(dimension))
                            pca_interactions += 1
                            expected = [row for row in payload["pca_summaries"]
                                        if row["rate"] == rate and row["schedule"] == schedule]
                            if page.locator("#pca-metrics-table tbody tr").count() != len(expected) * 2:
                                issues.append("PCA metrics table does not match selected scope")
                            if page.locator("#pca-delta-table tbody tr").count() != len(expected):
                                issues.append("PCA paired-difference table scope mismatch")
                            first = next(row for row in expected if row["dimension"] == payload["pca_study"]["dimensions"][0])
                            expected_mae = first["cohorts"]["all_unselected"]["mae"]["mean"]
                            shown = page.locator("#pca-metrics-table tbody tr").nth(1).locator("td").nth(3).inner_text()
                            if shown != ("Not measured" if expected_mae is None else f"{expected_mae:.3f}"):
                                issues.append("PCA displayed MAE differs from measured aggregate")
                            if not page.locator("#pca-roc-takeaway").inner_text().startswith(f"PCA-{dimension} "):
                                issues.append("PCA ROC dimension did not update")
                            if page.locator("#pca-envelope-table tbody tr").count() != 4:
                                issues.append("PCA/reference point/lower cohort table incomplete")
                checks.append({"viewport": label, "pca_filter_combinations": pca_interactions})
                comparisons = validate_dimension_comparisons(page)
                issues.extend(comparisons.pop("issues"))
                checks.append({"viewport": label, **comparisons})
                problems, state = checker._check_viewport(page, width, height, f"{label}/comparison-controls")
                issues.extend(problems)
                checks.append({"viewport": label, "tab": "comparison-controls", "state": state, "ok": not problems})
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
        "scope": "All report tabs, desktop/mobile overflow, filter combinations, independent PCA/MRL dimension pairs, plotted ROC and budget data parity, isolated control scopes and keyboard navigation.",
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
