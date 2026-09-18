from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


LABELS = {
    "historical_300": "Historical 300",
    "dense_2500": "Dense 2,500",
    "cosmos_otel": "Cosmos OTEL",
}
POINT_COLOR = "#0f766e"
LOWER_COLOR = "#b45309"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.{digits}f}%"


def _validate(aggregate: Mapping[str, Any]) -> None:
    if aggregate.get("version") != "idw-threshold-envelope-v1":
        raise ValueError("report requires idw-threshold-envelope-v1")
    if aggregate.get("status") != "complete":
        raise ValueError("report refuses incomplete experiment output")
    if aggregate["validation"]["actual_cells"] != 4500:
        raise ValueError("report requires the pre-registered 4,500-cell grid")
    acceptance = aggregate["preregistration"]["acceptance"]
    canonical_runtime = aggregate.get("protocol", {}).get("canonical_runtime_required", False)
    accuracy_tolerance = acceptance.get(
        "canonical_runtime_accuracy_absolute_tolerance" if canonical_runtime else "amended_execution_accuracy_absolute_tolerance", 0
    )
    mae_tolerance = acceptance.get(
        "canonical_runtime_mae_absolute_tolerance" if canonical_runtime else "amended_execution_mae_absolute_tolerance",
        0 if canonical_runtime else acceptance.get("mae_absolute_tolerance", 1e-12),
    )
    if aggregate["validation"]["baseline_max_accuracy_absolute_error"] > accuracy_tolerance:
        raise ValueError("baseline replay exceeded the documented accuracy tolerance")
    if aggregate["validation"]["baseline_max_mae_absolute_error"] > mae_tolerance:
        raise ValueError("baseline replay exceeded the documented MAE tolerance")
    if aggregate["validation"]["threshold_dominance"] != "passed_all_cells":
        raise ValueError("threshold dominance validation did not pass")
    expected = {
        (dataset, dimension, rate)
        for dataset in LABELS for dimension in (1536, 8)
        for rate in (0.01, 0.02, 0.05, 0.1, 0.2)
    }
    actual = {(row["dataset_id"], row["dimension"], row["rate"]) for row in aggregate["selectors"]}
    if actual != expected:
        raise ValueError("selector summaries are incomplete")


def _headline(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    rows = aggregate["rows"]
    result: dict[str, Any] = {}
    for dimension in (1536, 8):
        subset = [row for row in rows if row["dimension"] == dimension]
        for method, field in (("point", "point_auc"), ("lower", "lower_auc")):
            values = [row[field] for row in subset if row[field] is not None]
            result[f"{dimension}_{method}_cell_auc"] = mean(values) if values else None
            result[f"{dimension}_{method}_auc_cells"] = len(values)
        for method in ("paired_point", "paired_lower"):
            metrics = [row[f"{method}_at_0_5"] for row in subset]
            for metric in ("accuracy", "precision", "recall", "f1"):
                values = [item[metric] for item in metrics if item[metric] is not None]
                result[f"{dimension}_{method}_{metric}"] = mean(values) if values else None
        result[f"{dimension}_eligible_share"] = (
            sum(row["eligible_count"] for row in subset)
            / sum(row["unjudged_count"] for row in subset)
        )
    return result


def _agent_gap_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    keyed = {
        (row["dataset_id"], row["dimension"], row["agent_id"], row["method"]): row
        for row in aggregate["agent_summary"]
    }
    rows = []
    for key, point in keyed.items():
        dataset, dimension, agent, method = key
        if method != "paired_point":
            continue
        lower = keyed.get((dataset, dimension, agent, "paired_lower"))
        if lower is None:
            continue
        rows.append({
            "dataset_id": dataset, "dimension": dimension, "agent_id": agent,
            "n": point["n"], "point_recall": point["recall"], "lower_recall": lower["recall"],
            "recall_gap": (
                lower["recall"] - point["recall"]
                if lower["recall"] is not None and point["recall"] is not None else None
            ),
            "point_precision": point["precision"], "lower_precision": lower["precision"],
        })
    return sorted(rows, key=lambda row: (row["recall_gap"] is None, row["recall_gap"] or 0))[:12]


def build_report(
    aggregate: Mapping[str, Any], source_path: str, *,
    reproduction_command: str | None = None,
) -> str:
    _validate(aggregate)
    headline = _headline(aggregate)
    eligible_targets = int(aggregate["validation"]["eligible_target_occurrences"])
    sparse_calibration_targets = sum(int(row.get("calibration_fallback_count", 0)) for row in aggregate["rows"])
    empirical_calibration_targets = eligible_targets - sparse_calibration_targets
    contradiction_targets = sum(int(row.get("exact_contradiction_target_count", 0)) for row in aggregate["rows"])
    score_validation = aggregate.get("score_consistency_validation", {})
    equal_cell = {
        (row["dataset_id"], row["dimension"], row["rate"]): row
        for row in aggregate["equal_cell_summary"]
    }
    selectors = [
        {
            **row,
            "equal_cell_point_auc_mean": equal_cell[
                (row["dataset_id"], row["dimension"], row["rate"])
            ]["point_auc_mean"],
            "equal_cell_point_auc_seed_t95": equal_cell[
                (row["dataset_id"], row["dimension"], row["rate"])
            ]["point_auc_seed_t95"],
            "equal_cell_lower_auc_mean": equal_cell[
                (row["dataset_id"], row["dimension"], row["rate"])
            ]["lower_auc_mean"],
            "equal_cell_lower_auc_seed_t95": equal_cell[
                (row["dataset_id"], row["dimension"], row["rate"])
            ]["lower_auc_seed_t95"],
        }
        for row in aggregate["selectors"]
    ]
    initial_dataset = "cosmos_otel" if aggregate.get("publication", {}).get("aggregate_only") else "dense_2500"
    default = next(
        row for row in selectors
        if row["dataset_id"] == initial_dataset and row["dimension"] == 1536 and row["rate"] == 0.1
    )
    data_json = json.dumps(selectors, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    dataset_options = "".join(
        f'<option value="{escape(key)}"{(" selected" if key == initial_dataset else "")}>{escape(value)}</option>'
        for key, value in LABELS.items()
    )
    gap_rows = "".join(
        "<tr>"
        f"<td>{escape(LABELS[row['dataset_id']])}</td><td>{row['dimension']:,}</td>"
        f"<td><code>{escape(row['agent_id'])}</code></td><td>{row['n']:,}</td>"
        f"<td>{_pct(row['point_precision'])}</td><td>{_pct(row['lower_precision'])}</td>"
        f"<td>{_pct(row['point_recall'])}</td><td>{_pct(row['lower_recall'])}</td>"
        "</tr>"
        for row in _agent_gap_rows(aggregate)
    )
    profile_rows = "".join(
        "<tr>"
        f"<td><code>{escape(row['dataset_id'])}</code></td><td>{row['sessions']:,}</td>"
        f"<td>{row['agents']:,}</td><td>{row['positive_count']:,} ({_pct(row['pass_rate'])})</td>"
        f"<td>{escape(row['label_source'])}</td>"
        f"<td>{escape(str(row.get('representation_sources') or 'canonical packet'))}; "
        f"{row['truncated_sessions']:,} truncated; {row['excluded_sessions']:,} excluded</td>"
        "</tr>"
        for row in aggregate["datasets"]
    )
    cosmos = next(row for row in aggregate["datasets"] if row["dataset_id"] == "cosmos_otel")
    cosmos_sources = cosmos["representation_sources"]
    budget_description = "; ".join(
        f'{LABELS[row["dataset_id"]]}: ' + "/".join(
            str(max(1, math.floor(row["sessions"] * rate))) for rate in (0.01, 0.02, 0.05, 0.1, 0.2)
        ) for row in aggregate["datasets"]
    )
    privacy_notice = ""
    if aggregate.get("publication", {}).get("aggregate_only"):
        privacy_notice = (
            '<div class="callout warning"><strong>Aggregate-only publication.</strong> '
            "Original identifiers, raw inputs, membership records and per-target score arrays remain "
            "in protected local storage. This report publishes numerical summaries and whole-artifact hashes only.</div>"
        )
        gap_rows = "".join(
            f'<tr><td>{escape(LABELS[row["dataset_id"]])}</td><td>{row["dimension"]}</td>'
            f'<td>{row["eligible_agents"]}</td><td>{row["agents_with_defined_recall"]}</td>'
            f'<td>{row["agents_with_recall_loss"]}</td>'
            f'<td>{_pct(row["median_recall_delta"])}</td><td>{_pct(row["worst_recall_delta"])}</td></tr>'
            for row in aggregate["agent_gap_summary"]
        )
        agent_section = (
            "<h3>Agent-level recall tradeoffs without publishing identities</h3>"
            '<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Dim</th>'
            "<th>Eligible agents</th><th>Agents with positive labels</th><th>Agents losing recall</th>"
            "<th>Median recall change</th><th>Worst recall change</th></tr></thead><tbody>"
            + gap_rows + "</tbody></table></div>"
            '<p class="muted">Lower minus point recall at threshold 0.50, computed within each agent over '
            "repeated eligible target occurrences. Negative changes mean recall loss; undefined recall is excluded, "
            "not zero-filled. Agent identities and individual rows are not published.</p>"
        )
    else:
        agent_section = (
            "<h3>Largest displayed per-agent recall losses at threshold 0.50</h3>"
            '<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Dim</th><th>Agent</th>'
            "<th>Paired occurrences</th><th>Point precision</th><th>Lower precision</th>"
            "<th>Point recall</th><th>Lower recall</th></tr></thead><tbody>"
            + gap_rows + "</tbody></table></div>"
            '<p class="muted">These are pooled repeated target occurrences, not agent-level population estimates '
            "or independent observations. The full agent table is in aggregate.json.</p>"
        )
    preparation = cosmos.get("embedding_preparation", {})
    preparation_note = ""
    if cosmos.get("snapshot_cutoff_utc"):
        preparation_note = (
            f'<p class="note"><strong>Refreshed source:</strong> catch-up cutoff '
            f'{escape(cosmos["snapshot_cutoff_utc"])}; {cosmos["sessions"]:,} eligible expected-label units. '
            "This is a static export, not a live feed or an atomic point-in-time database snapshot. "
            "The other label containers are not mixed into the experiment. "
            "Differences from earlier reports can reflect a changed cohort and label prevalence.</p>"
        )
    if "reused_vectors" in preparation and "requested_vectors" in preparation:
        preparation_note += (
            '<p class="note"><strong>One-time embedding preparation, separate from the offline replay:</strong> '
            f'{preparation["reused_vectors"]:,} exact cached vectors reused; {preparation["requested_vectors"]:,} '
            f'new native vectors generated with Azure OpenAI. {preparation["logical_requests"]:,} '
            f'successful requests used {preparation["api_input_tokens"]:,} reported input tokens. '
            "Preparation was user-authorized; no new judge was called.</p>"
        )
    if reproduction_command is None:
        reproduction_command = (
            '$env:OPENBLAS_NUM_THREADS="12"; $env:OMP_NUM_THREADS="12"\n'
            '$env:MKL_NUM_THREADS="12"; $env:NUMEXPR_NUM_THREADS="12"\n'
            ".\\.venv-v3\\Scripts\\python.exe scripts\\run_idw_threshold_experiment.py "
            "--baseline <matching-baseline-directory> --input cosmos_otel=<input-manifest> --output <fresh-run>\n"
            ".\\.venv-v3\\Scripts\\python.exe scripts\\build_idw_threshold_report.py "
            "--input <fresh-run>\\aggregate.json --output <fresh-run>\\report.html"
        )
    runtime_audit = (
        "Superseded startup/runtime records remain separately retained; see the scientific record."
        if "failed_runs" in aggregate["files"] else
        "This run started in a fresh output directory; no failed-run or runtime-variant artifacts are implied."
    )
    hash_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td><code>{escape(value)}</code></td></tr>"
        for name, value in sorted(aggregate["code_hashes"].items())
    )
    browser = aggregate.get("browser_validation")
    browser_text = (
        f"PASS: desktop and mobile; {escape(browser.get('checked_at', 'timestamp unavailable'))}"
        if browser and browser.get("ok") else "Pending when this HTML was generated; see browser_validation.json beside the report."
    )
    replay_audit = (
        "Earlier failed technical attempts and pre-analysis amendments are retained in "
        "<code>failed_runs.json</code> and <code>preregistration.json</code>."
        if "failed_runs" in aggregate["files"] else
        "This fresh canonical run uses exact baseline replay acceptance; its criteria are retained in "
        "<code>preregistration.json</code>."
    )
    source = escape(source_path)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-session IDW ROC and threshold experiment</title>
<style>
:root{{--ink:#172033;--muted:#526174;--line:#d7dee8;--panel:#f7f9fc;--point:{POINT_COLOR};--lower:{LOWER_COLOR};}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f7;color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:auto;background:white;padding:clamp(1rem,3vw,2.5rem)}} h1{{font-size:clamp(1.9rem,4vw,3.2rem);line-height:1.08;margin:.3rem 0}}
h2{{margin-top:2.4rem;border-bottom:2px solid var(--line);padding-bottom:.35rem}} h3{{margin-bottom:.3rem}}
.eyebrow{{text-transform:uppercase;letter-spacing:.09em;font-weight:700;color:var(--point)}} .lead{{font-size:1.15rem;max-width:78ch}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem}} .card,figure,.note{{border:1px solid var(--line);border-radius:10px;padding:1rem;background:var(--panel);min-width:0}}
.big{{font-size:1.55rem;font-weight:750}} .muted,figcaption{{color:var(--muted);font-size:.9rem}} code{{overflow-wrap:anywhere}}
.pipeline{{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:.5rem;align-items:stretch}}
.step{{border:2px solid #94a3b8;border-radius:8px;padding:.7rem;background:white;position:relative}} .step:not(:last-child)::after{{content:"→";position:absolute;right:-.65rem;top:35%;font-weight:800;background:white}}
.controls{{display:flex;flex-wrap:wrap;gap:1rem;align-items:end;padding:1rem;background:#eaf5f3;border-radius:10px}} label{{font-weight:700}} select{{display:block;margin-top:.25rem;padding:.55rem;min-width:150px}}
.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin-top:1rem}} figure{{margin:0;background:white}} figure.wide{{grid-column:1/-1}}
svg{{width:100%;height:auto;display:block}} .legend{{display:flex;gap:1.2rem;flex-wrap:wrap}} .swatch{{display:inline-block;width:1rem;height:.3rem;margin-right:.35rem;vertical-align:middle}}
.chart-scroll{{max-width:100%;overflow-x:auto}}.chart-scroll:focus-visible{{outline:3px solid #075985;outline-offset:3px}}
.table-wrap{{max-width:100%;overflow-x:auto}} table{{border-collapse:collapse;width:100%;min-width:720px}} th,td{{padding:.55rem;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:#edf2f7}}
.callout{{border-left:5px solid var(--point);padding:.8rem 1rem;background:#ecfdf5}} .warning{{border-left-color:var(--lower);background:#fff7ed}}
details{{margin:.7rem 0}} summary{{cursor:pointer;font-weight:700}} a{{color:#075985}} .metric-note{{min-height:2.8rem}}
pre{{max-width:100%;overflow-x:auto;padding:.75rem;background:#f8fafc;border:1px solid var(--line);border-radius:8px}}
@media(max-width:800px){{.cards,.charts{{grid-template-columns:1fr}} figure.wide{{grid-column:auto}} .pipeline{{grid-template-columns:1fr}} .step:not(:last-child)::after{{content:"↓";right:50%;top:auto;bottom:-1.05rem}} main{{padding:1rem}}}}
@media(max-width:800px){{.chart-scroll svg{{min-width:620px}}}}
</style>
</head>
<body><main>
<header>
<div class="eyebrow">Offline replay · pre-registered invariants + exploratory thresholds</div>
<h1>Full-session IDW: point scores versus a conditional lower envelope</h1>
<p class="lead">Technical-team report for deciding what ordinary score binarization reveals—and what recall is lost when the lower side of an empirical Lipschitz sensitivity envelope is thresholded instead. It does not choose a production threshold.</p>
<div class="callout"><strong>Measured conclusion.</strong> Across equal-weight replay cells, native point-score AUROC was
<strong>{_fmt(headline['1536_point_cell_auc'])}</strong> and lower-envelope AUROC was
<strong>{_fmt(headline['1536_lower_cell_auc'])}</strong>; at 8 dimensions they were
<strong>{_fmt(headline['8_point_cell_auc'])}</strong> and <strong>{_fmt(headline['8_lower_cell_auc'])}</strong>.
Those means use only cells containing both classes ({headline['1536_point_auc_cells']:,} native;
{headline['8_point_auc_cells']:,} 8d of 2,250 each).
At threshold 0.50, the lower envelope's equal-cell recall was
<strong>{_pct(headline['1536_paired_lower_recall'])}</strong> native and
<strong>{_pct(headline['8_paired_lower_recall'])}</strong> at 8d, versus point-score recall
<strong>{_pct(headline['1536_paired_point_recall'])}</strong> and
<strong>{_pct(headline['8_paired_point_recall'])}</strong>. This is a conservative-classification tradeoff, not evidence that the envelope is calibrated confidence.</div>
</header>
{privacy_notice}

<section id="decision"><h2>1. Decision and source</h2>
<div class="cards">
<div class="card"><div class="big">4,500</div><div>dataset × dimension × seed × schedule × budget cells</div></div>
<div class="card"><div class="big">{aggregate['validation']['target_occurrences']:,}</div><div>unjudged target occurrences; repeated across replays</div></div>
<div class="card"><div class="big">{_pct(aggregate['validation']['eligible_target_occurrences']/aggregate['validation']['target_occurrences'])}</div><div>same-agent IDW/exact targets eligible for both methods</div></div>
<div class="card"><div class="big">$0 / 0 calls</div><div>offline replay external cost and calls; prior embedding preparation is separate</div></div>
</div>
<p>Run <code>{escape(aggregate['run_id'])}</code>; aggregate <code>{source}</code>; manifest
<code>{escape(str(Path(source_path).with_name('manifest.json')))}</code>. Source revision
<code>{escape(aggregate['source_revision'])}</code>. The exact baseline aggregate and compressed membership
SHA256 values were validated before replay. Selected/direct observations are not in primary quality denominators.</p>
</section>

<section id="methods"><h2>2. How the methods work</h2>
<div class="pipeline" role="img" aria-label="Pipeline from full sessions through representation, reused selection, observed labels, causal scoring and threshold summaries">
<div class="step"><strong>Full session</strong><br><span class="muted">bounded canonical evidence</span></div>
<div class="step"><strong>Embedding</strong><br><span class="muted">native 1,536 or first 8 + L2</span></div>
<div class="step"><strong>Membership</strong><br><span class="muted">verified end-to-end label-blind ranking</span></div>
<div class="step"><strong>Earlier labels</strong><br><span class="muted">selected same-agent only</span></div>
<div class="step"><strong>Two scores</strong><br><span class="muted">IDW point; conditional lower</span></div>
<div class="step"><strong>Threshold</strong><br><span class="muted">&ge; t → accuracy / precision / recall / F1</span></div>
</div>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>Representation</th><th>Score</th><th>Cohort</th><th>Assumption / failure</th></tr></thead><tbody>
<tr><td><strong>Ordinary point IDW</strong></td><td>Native 1,536 or prefix 8, each L2-normalized; angular distance</td><td>8 nearest earlier selected same-agent donors, inverse-square, epsilon 1e-6. All exact prior donors are averaged. Ordinary fallback is earlier global mean or 0.5.</td><td>All-unjudged view includes fallbacks. Paired comparison includes only IDW/exact targets.</td><td>Expected labels are an oracle reference; nearby embeddings need not have nearby true values.</td></tr>
<tr><td><strong>Conditional lower envelope</strong></td><td>Same normalized angular geometry, <code>arccos(cosine) / π</code> in [0,1]—not unconverted radians</td><td><code>clamp(IDW − L × weighted donor distance, 0, 1)</code>. L is target-time empirical 90th-percentile earlier observed-label pair slope; 0.01 distance floor; sparse L=1.</td><td>Exactly the same IDW/exact targets as paired point score. No band for global mean/prior.</td><td>Observed-label oracle calibration is a sensitivity envelope, not a confidence interval, coverage statement, or universal Lipschitz guarantee; exact contradictions and sparse fallbacks are audited.</td></tr>
</tbody></table></div>
<div class="note"><strong>Illustrative IDW example—not a run result.</strong> Two donors at distances 1 and 2 with labels 1 and 0 have inverse-square weights 1 and 1/4, normalized to 0.8 and 0.2. The point estimate is 0.8 and <code>0.8 ≥ 0.5</code> predicts positive. Actual runs use epsilon, nearest-eight and exact-match rules above. A direct observed label is not an estimate.</div>
</section>

<section id="data"><h2>3. Data and design</h2>
<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Sessions</th><th>Agents</th><th>Positive reference</th><th>Label source</th><th>Representation / exclusions</th></tr></thead><tbody>{profile_rows}</tbody></table></div>
<p>Cosmos contains {cosmos_sources.get("linked_raw_spans", 0):,} linked raw-span representations and {cosmos_sources.get("synthetic_label_document", 0):,} synthetic label-document fallbacks; {cosmos["truncated_sessions"]:,} sessions were truncated by the bounded packet policy. Linked spans do not establish representative production traffic. Historical and Dense are synthetic. Tau2 remains excluded because expected labels are unavailable and recorded rewards were not substituted.</p>
{preparation_note}
<p>Thirty seeds (13–42), five schedules and five session budgets (1%, 2%, 5%, 10%, 20%) are paired across dimensions. Absolute selected counts are {escape(budget_description)}. Membership ranking sees the complete unlabeled schedule; donor scoring is causal. Repeats reuse labels and are not independent sessions.</p>
</section>

<section id="results"><h2>4. Interactive ROC and threshold results</h2>
<div class="controls" aria-label="Chart filters">
<label>Dataset<select id="dataset">{dataset_options}</select></label>
<label>Dimension<select id="dimension"><option value="1536" selected>1,536</option><option value="8">8</option></select></label>
<label>Session budget<select id="rate"><option value="0.01">1%</option><option value="0.02">2%</option><option value="0.05">5%</option><option value="0.1" selected>10%</option><option value="0.2">20%</option></select></label>
<div><strong id="scope">{escape(LABELS[initial_dataset])} · 1,536d · 10%</strong><br><span class="muted" id="denominator"></span><br><span class="muted" id="uncertainty"></span></div>
</div>
<p class="legend"><span><i class="swatch" style="background:{POINT_COLOR}"></i>Point IDW</span><span><i class="swatch" style="background:{LOWER_COLOR}"></i>Lower envelope</span></p>
<p class="muted">On narrow screens, scroll each chart horizontally to keep axis labels readable.</p>
<div class="charts">
<figure class="wide"><h3>ROC: true-positive rate versus false-positive rate</h3><div class="chart-scroll" role="region" aria-label="Scrollable ROC chart" tabindex="0"><svg id="roc" viewBox="0 0 620 360" role="img" aria-label="ROC curves"></svg></div><figcaption id="roc-note"></figcaption></figure>
<figure><h3>Accuracy versus threshold</h3><div class="chart-scroll" role="region" aria-label="Scrollable accuracy chart" tabindex="0"><svg id="accuracy" viewBox="0 0 520 330" role="img" aria-label="Accuracy threshold curves"></svg></div><figcaption>Fraction correctly classified; repeated paired target occurrences.</figcaption></figure>
<figure><h3>Precision versus threshold</h3><div class="chart-scroll" role="region" aria-label="Scrollable precision chart" tabindex="0"><svg id="precision" viewBox="0 0 520 330" role="img" aria-label="Precision threshold curves"></svg></div><figcaption>Among predicted positives. Defined as 0 when none are predicted.</figcaption></figure>
<figure><h3>Recall versus threshold</h3><div class="chart-scroll" role="region" aria-label="Scrollable recall chart" tabindex="0"><svg id="recall" viewBox="0 0 520 330" role="img" aria-label="Recall threshold curves"></svg></div><figcaption>Among actual positives. The lower curve cannot exceed point recall at a shared threshold.</figcaption></figure>
<figure><h3>F1 versus threshold</h3><div class="chart-scroll" role="region" aria-label="Scrollable F1 chart" tabindex="0"><svg id="f1" viewBox="0 0 520 330" role="img" aria-label="F1 threshold curves"></svg></div><figcaption>Harmonic mean of precision and recall; this is a threshold curve, not “ROC of F1.”</figcaption></figure>
</div>
<div class="callout warning"><strong>How to read.</strong> The ROC is computed at every unique tied score with all-negative and all-positive endpoints; the HTML displays a deterministic at-most-500-point rendering, while <code>exact_roc_curves.npz</code> retains every breakpoint. Metric charts use 0.00–1.00 in steps of .01 plus an all-negative endpoint. These are descriptive sweeps over labels used for evaluation—not held-out threshold selection.</div>
<p class="muted">Curves and their AUROC pool repeated target occurrences. The filter's cell-mean AUROC instead gives each of 150 seed × schedule cells equal weight. Its 95% t interval is computed over 30 seed means after averaging the five schedules; it measures replay/schedule sensitivity, not independent-label or population uncertainty.</p>
</section>

<section id="analysis"><h2>5. Analysis and limitations</h2>
<p><strong>Measured.</strong> Lower values are never above point values, so at the same threshold lower-envelope predicted positives are a subset. The all-cell invariant passed. That mechanically reduces or preserves both false positives and true positives; precision may improve or worsen depending on which predictions are removed. Envelope availability was {_pct(headline['1536_eligible_share'])} native and {_pct(headline['8_eligible_share'])} at 8d; ordinary all-unjudged metrics in each selector explicitly include the missing global-mean/prior rows.</p>
<p><strong>Calibration provenance counts.</strong> Of {eligible_targets:,} eligible repeated target occurrences,
<strong>{empirical_calibration_targets:,}</strong> used an empirical target-time L and
<strong>{sparse_calibration_targets:,}</strong> used the configured sparse-calibration fallback
<code>L=1</code>. There were <strong>{contradiction_targets:,}</strong> target occurrences whose
earlier calibration history contained an exact-geometry conflicting-label pair. Counts are occurrences
across replays, not unique sessions.</p>
<p><strong>Interpretation.</strong> A low or clamped-zero lower score can reflect sparse calibration, a large empirical slope, or donor distance—not calibrated uncertainty. The envelope can become degenerate and sacrifice substantial recall. AUROC can also change because subtracting target-specific allowances reorders scores; the subset property only applies at the same numeric threshold.</p>
<p><strong>Not established.</strong> No new judgments, production traffic sample, out-of-time calibration, statistical coverage, causal selector admission, weekly-policy behavior, or integrated <code>value_pipeline.py</code> behavior was tested. Prefix-8 is not PCA-8. Seed intervals in the machine-readable summary describe schedule/replay sensitivity over reused labels, not population confidence.</p>
{agent_section}
<p><strong>Next discriminating experiment.</strong> Freeze L and thresholds on a prior time window, then evaluate on later independently judged sessions with pre-specified coverage and utility criteria. That is required before a lower-envelope threshold can support a deployment decision.</p>
</section>

<section id="repro"><h2>6. Reproducibility and validation</h2>
<pre><code>{escape(reproduction_command)}</code></pre>
<p>Environment: <code>{escape(aggregate['environment']['python'])}</code>; platform
<code>{escape(aggregate['environment']['platform'])}</code>. Run time
{aggregate['costs']['wall_seconds']:.1f}s wall / {aggregate['costs']['cpu_seconds']:.1f}s process CPU
for the successful offline replay. {runtime_audit}
The final numerical run used the baseline-compatible Python 3.13.13 AMD64
environment with NumPy 2.5.3 and the baseline's 12-thread OpenBLAS setting.
<code>threadpool_info()</code> is recorded in the protected aggregate.
Baseline maximum replay MAE difference: {aggregate['validation']['baseline_max_mae_absolute_error']:.3g};
maximum accuracy difference: {aggregate['validation']['baseline_max_accuracy_absolute_error']:.3g};
cells not exactly equal in accuracy: {aggregate['validation']['baseline_accuracy_mismatches']}.
{replay_audit} Cache before/after preservation: PASS.
Browser gate: <strong>{browser_text}</strong></p>
<p>Independent source/score consistency validation:
<strong>{escape("PASS" if score_validation.get("ok") else "not recorded")}</strong>;
{score_validation.get("canonical_cells_checked", 0):,} retained cell summaries,
{score_validation.get("stratified_source_to_score_replay_cells", 0):,} stratified
source-to-score replays, and
{score_validation.get("exact_roc_method_selector_arrays_checked", 0):,} exact ROC
method-selector arrays checked. Record:
<code>{escape(aggregate["files"].get("score_consistency_validation", "unavailable"))}</code>.</p>
<details><summary>Executed source hashes</summary><div class="table-wrap"><table><thead><tr><th>Source</th><th>SHA256</th></tr></thead><tbody>{hash_rows}</tbody></table></div></details>
<details><summary>Artifact and provenance paths</summary><p>Baseline aggregate: <code>{escape(aggregate['baseline']['aggregate'])}</code> (<code>{escape(aggregate['baseline']['aggregate_sha256'])}</code>)<br>
Baseline memberships: <code>{escape(aggregate['baseline']['memberships'])}</code> (<code>{escape(aggregate['baseline']['memberships_sha256'])}</code>)<br>
Raw target evidence: <code>{escape(aggregate['files']['target_evidence'])}</code><br>
Exact ROC arrays: <code>{escape(aggregate['files']['exact_roc_curves'])}</code><br>
Pre-registration: <code>{escape(aggregate['files']['preregistration'])}</code></p></details>
<p class="muted">Numerical precision: point scores and envelope fields are retained as float32.
Threshold metrics and exact tied-score ROC use these retained values, not unrounded float64 estimates.
Baseline replay parity is checked before serialization; float32 ties can affect ROC breakpoints.</p>
</section>

<script>
const rows={data_json};
const labels={json.dumps(LABELS)};
const colors={{point:"{POINT_COLOR}",lower:"{LOWER_COLOR}"}};
const el=id=>document.getElementById(id);
function axis(svg,xlabel,ylabel){{
  svg.innerHTML=`<line x1="58" y1="18" x2="58" y2="282" stroke="#64748b"/><line x1="58" y1="282" x2="500" y2="282" stroke="#64748b"/>
  <g fill="#526174" font-size="12"><text x="279" y="320" text-anchor="middle">${{xlabel}}</text><text x="15" y="150" transform="rotate(-90 15 150)" text-anchor="middle">${{ylabel}}</text>
  <text x="58" y="300" text-anchor="middle">0</text><text x="279" y="300" text-anchor="middle">0.5</text><text x="500" y="300" text-anchor="middle">1</text>
  <text x="48" y="286" text-anchor="end">0</text><text x="48" y="154" text-anchor="end">0.5</text><text x="48" y="24" text-anchor="end">1</text></g>
  <line x1="58" y1="150" x2="500" y2="150" stroke="#e2e8f0"/><line x1="279" y1="18" x2="279" y2="282" stroke="#e2e8f0"/>`;
}}
function points(xs,ys){{
  const out=[]; for(let i=0;i<xs.length;i++) if(xs[i]!==null&&ys[i]!==null) out.push(`${{58+442*xs[i]}},${{282-264*ys[i]}}`); return out.join(" ");
}}
function path(svg,xs,ys,color,dash=""){{
  const p=document.createElementNS("http://www.w3.org/2000/svg","polyline"); p.setAttribute("points",points(xs,ys)); p.setAttribute("fill","none"); p.setAttribute("stroke",color); p.setAttribute("stroke-width","3"); if(dash)p.setAttribute("stroke-dasharray",dash); svg.appendChild(p);
}}
function render(){{
  const dataset=el("dataset").value, dimension=Number(el("dimension").value), rate=Number(el("rate").value);
  const row=rows.find(r=>r.dataset_id===dataset&&r.dimension===dimension&&r.rate===rate);
  el("scope").textContent=`${{labels[dataset]}} · ${{dimension.toLocaleString()}}d · ${{100*rate}}%`;
  el("denominator").textContent=`${{row.pooled_eligible_n.toLocaleString()}} paired eligible / ${{row.pooled_unjudged_n.toLocaleString()}} all-unjudged repeated occurrences`;
  const ci=x=>x===null?"N/A":`[${{x[0].toFixed(3)}}, ${{x[1].toFixed(3)}}]`;
  el("uncertainty").textContent=`Equal-cell AUROC (95% seed-sensitivity interval): point ${{row.equal_cell_point_auc_mean===null?"N/A":row.equal_cell_point_auc_mean.toFixed(3)}} ${{ci(row.equal_cell_point_auc_seed_t95)}}; lower ${{row.equal_cell_lower_auc_mean===null?"N/A":row.equal_cell_lower_auc_mean.toFixed(3)}} ${{ci(row.equal_cell_lower_auc_seed_t95)}}.`;
  const roc=el("roc"); axis(roc,"False-positive rate","True-positive rate");
  const diagonal=document.createElementNS("http://www.w3.org/2000/svg","line"); diagonal.setAttribute("x1","58");diagonal.setAttribute("y1","282");diagonal.setAttribute("x2","500");diagonal.setAttribute("y2","18");diagonal.setAttribute("stroke","#94a3b8");diagonal.setAttribute("stroke-dasharray","4 5");roc.appendChild(diagonal);
  path(roc,row.point_roc_display.fpr,row.point_roc_display.tpr,colors.point);
  path(roc,row.lower_roc_display.fpr,row.lower_roc_display.tpr,colors.lower,"8 4");
  el("roc-note").textContent=`Point AUROC ${{row.point_auc===null?"N/A":row.point_auc.toFixed(3)}}; lower AUROC ${{row.lower_auc===null?"N/A":row.lower_auc.toFixed(3)}}. Exact points: ${{row.point_roc_display.exact_point_count.toLocaleString()}} / ${{row.lower_roc_display.exact_point_count.toLocaleString()}}.`;
  for(const metric of ["accuracy","precision","recall","f1"]){{
    const svg=el(metric); axis(svg,"Threshold t",metric[0].toUpperCase()+metric.slice(1));
    for(const [method,color,dash] of [["point",colors.point,""],["lower",colors.lower,"8 4"]]){{
      const grid=row[method+"_grid"].filter(r=>r.threshold<=1);
      path(svg,grid.map(r=>r.threshold),grid.map(r=>r[metric]),color,dash);
    }}
  }}
}}
for(const id of ["dataset","dimension","rate"]) el(id).addEventListener("change",render);
render();
</script>
</main></body></html>"""
