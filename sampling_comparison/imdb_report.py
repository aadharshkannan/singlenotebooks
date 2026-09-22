"""Standalone, aggregate-only IMDb report; never turns readiness into results."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from sampling_comparison.matryoshka_experiment import canonical, sha256_file, write_json


DIMENSIONS = [1536, 32, 24, 16, 12, 8]
RATES = [.01, .02, .05, .10, .20]
SCHEDULES = ["uniformly_random", "bursty"]
METRICS = ["mae", "accuracy", "precision", "recall", "f1", "auc"]
COHORTS = ["all_unselected", "eligible_point", "eligible_lower", "novel_source"]
PROFILE_FIELDS = [
    "dataset_id", "sessions", "agents", "positive_count", "negative_count",
    "split_counts", "unique_texts", "duplicate_text_rows", "conflicting_label_text_groups",
    "excluded_unlabeled_reviews", "truncated_sessions", "token_length",
    "unique_embedding_input_tokens", "model", "label_source", "representation_policy",
    "split_policy", "source_choice", "source_sha256", "source_page", "source_url",
    "citation", "embedding_calls", "api_input_tokens", "live_judge_calls",
    "deduplication", "tokenizer", "input_manifest_sha256",
]
EXTENSION_FIELDS = (
    "baseline_aggregate_sha256", "baseline_manifest_sha256", "reused_cells", "added_cells",
    "reused_dimensions", "added_dimensions", "baseline_rows_unchanged", "replay_pairing_exact",
    "embedding_calls",
)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "low": None, "high": None, "replays": 0}
    return {
        "mean": float(np.mean(values)), "low": float(np.quantile(values, .025)),
        "high": float(np.quantile(values, .975)), "replays": len(values),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average schedules within seeds before describing replay variability."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for schedule in (row["schedule"], "all"):
            groups[(schedule, row["rate"], row["dimension"])].append(row)
    native = {(r["schedule"], r["rate"], r["seed"]): r for r in rows if r["dimension"] == 1536}
    output = []
    grid = np.linspace(0, 1, 101)
    for (schedule, rate, dimension), cell_rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "schedule": schedule, "rate": rate, "dimension": dimension, "cohorts": {},
        }
        for cohort in COHORTS:
            record["cohorts"][cohort] = {}
            for metric in ["n", *METRICS]:
                seed_values: dict[int, list[float]] = defaultdict(list)
                for row in cell_rows:
                    value = row.get(cohort, {}).get(metric)
                    if value is not None:
                        seed_values[row["seed"]].append(float(value))
                record["cohorts"][cohort][metric] = _summary(
                    [float(np.mean(v)) for v in seed_values.values()]
                )
        paired: dict[int, list[float]] = defaultdict(list)
        for row in cell_rows:
            reference = native.get((row["schedule"], rate, row["seed"]))
            if reference is not None:
                value, base = row["all_unselected"]["mae"], reference["all_unselected"]["mae"]
                if value is not None and base is not None:
                    paired[row["seed"]].append(value - base)
        record["paired_mae_delta_native"] = _summary([float(np.mean(v)) for v in paired.values()])
        record["diagnostics"] = {}
        for metric in ("idw", "exact_match", "prior", "calibration_fallback_eligible", "envelope_label_coverage"):
            by_seed_values: dict[int, list[float]] = defaultdict(list)
            for row in cell_rows:
                if metric == "envelope_label_coverage":
                    value = row.get(metric)
                elif metric == "calibration_fallback_eligible":
                    value = row.get("counts", {}).get(metric)
                else:
                    value = row.get("counts", {}).get("provenance", {}).get(metric)
                if value is not None:
                    by_seed_values[row["seed"]].append(float(value))
            record["diagnostics"][metric] = _summary([
                float(np.mean(values)) for values in by_seed_values.values()
            ])
        record["roc"] = {}
        for method in ("point", "lower"):
            by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
            for row in cell_rows:
                curve = row.get("roc", {}).get(method, {})
                if curve.get("auc") is not None:
                    by_seed[row["seed"]].append(np.interp(grid, curve["fpr"], curve["tpr"]))
            curves = [np.mean(values, axis=0) for values in by_seed.values()]
            record["roc"][method] = {
                "fpr": grid.tolist(),
                "tpr": np.mean(curves, axis=0).tolist() if curves else [],
                "replays": len(curves),
            }
        output.append(record)
    return output


def public_payload(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    complete = source.get("status") == "completed"
    if not complete and source.get("status") not in ("prepared_without_embeddings", "complete"):
        raise ValueError("report requires an IMDb readiness profile or completed experiment aggregate")
    if complete and (source.get("version") != "imdb-sampling-v1" or not source.get("rows")):
        raise ValueError("completed report requires measured IMDb replay rows")
    profile = source["dataset"] if complete else source
    if profile.get("dataset_id") != "imdb_50000":
        raise ValueError("not an IMDb input")
    protocol = source.get("protocol", {})
    dimensions = protocol.get("dimensions", DIMENSIONS)
    rates = protocol.get("rates", RATES)
    schedules = protocol.get("schedules", SCHEDULES)
    repetitions = protocol.get("repetitions", 40)
    if complete:
        rows = source["rows"]
        expected = len(dimensions) * len(rates) * len(schedules) * repetitions
        identities = {(r["dimension"], r["rate"], r["schedule"], r["seed"]) for r in rows}
        seeds = {r["seed"] for r in rows}
        expected_identities = {
            (dimension, rate, schedule, seed)
            for dimension in dimensions for rate in rates for schedule in schedules for seed in seeds
        }
        if (len(rows) != expected or len(seeds) != repetitions
                or len(identities) != expected or identities != expected_identities):
            raise ValueError("completed report is missing or repeats planned result cells")
    payload = {
        "version": "imdb-public-report-v1", "status": "completed" if complete else "results_pending",
        "embeddings_ready": complete or source.get("status") == "complete",
        "dataset": {key: profile[key] for key in PROFILE_FIELDS if key in profile},
        "source_artifact": str(source_path), "source_artifact_sha256": sha256_file(source_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dimensions": dimensions, "rates": rates, "schedules": schedules,
            "repetitions": repetitions,
            "planned_cells": len(dimensions) * len(rates) * len(schedules) * repetitions,
        },
        "summaries": summarize_rows(source["rows"]) if complete else [],
    }
    if "extension" in source:
        extension = source["extension"]
        reused, added = extension["reused_dimensions"], extension["added_dimensions"]
        per_dimension = repetitions * len(rates) * len(schedules)
        if (not complete or set(reused) & set(added)
                or sorted(reused + added) != sorted(dimensions)
                or extension["reused_cells"] != len(reused) * per_dimension
                or extension["added_cells"] != len(added) * per_dimension
                or extension["baseline_rows_unchanged"] is not True
                or extension["replay_pairing_exact"] is not True
                or extension["embedding_calls"] != 0):
            raise ValueError("extension provenance does not match the combined study")
        payload["extension"] = {name: extension[name] for name in EXTENSION_FIELDS}
    return payload


def build_report(source_path: Path, output: Path, *, numerical_validation: Path | None = None) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    payload = public_payload(source, source_path)
    if numerical_validation is not None:
        audit = json.loads(numerical_validation.read_text(encoding="utf-8"))
        if (payload["status"] != "completed" or audit.get("ok") is not True
                or audit.get("aggregate_sha256") != payload["source_artifact_sha256"]
                or audit.get("cells_checked") != payload["protocol"]["planned_cells"]):
            raise ValueError("numerical validation does not match this completed aggregate")
        payload["score_validation"] = {
            "sha256": sha256_file(numerical_validation),
            "cells_checked": audit["cells_checked"],
            "maximum_absolute_metric_difference": audit["maximum_absolute_metric_difference"],
            "native_5pct_equal_cell_diagnostics": audit["native_5pct_equal_cell_diagnostics"],
        }
    output.mkdir(parents=True, exist_ok=True)
    encoded = canonical(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = TEMPLATE.replace("__DATA__", encoded)
    document = document.replace("__SOURCE__", html.escape(str(source_path)))
    (output / "report.html").write_text(document, encoding="utf-8")
    write_json(output / "summary.json", payload)
    manifest = {
        "version": "imdb-report-manifest-v1", "status": payload["status"],
        "input_sha256": sha256_file(source_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "files": {name: sha256_file(output / name) for name in ("report.html", "summary.json")},
        "validation": "Run the report quality checker and interaction checker; see validation.json when present.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IMDb | Full-review IDW and prefix dimensions</title>
<style>
:root{--ink:#172d39;--muted:#49616d;--teal:#007d7c;--blue:#235fba;--orange:#af5215;--paper:#f3f5f3;--line:#d4dfdf}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,Arial,sans-serif}
header{background:#142f3d;color:white;padding:40px max(24px,calc((100% - 1160px)/2))}
header p{max-width:820px;color:#dbe6ea}h1{font-size:clamp(30px,5vw,48px);line-height:1.12;max-width:930px;margin:14px 0}
h2{font-size:27px;line-height:1.25;margin-top:0}h3{font-size:19px}p{margin:10px 0 18px}
.eyebrow{font-size:13px;letter-spacing:.13em;font-weight:750;text-transform:uppercase}.badge{display:inline-block;background:#ffe0a3;color:#51370b;padding:6px 12px;border-radius:5px;font-size:13px;font-weight:750}
main{max-width:1208px;margin:auto;padding:0 24px 56px}nav{display:flex;flex-wrap:wrap;gap:7px;padding:20px 0}
button,select{font:inherit;border:1px solid #a9bbbf;border-radius:6px;padding:9px 13px;background:white;color:var(--ink)}
button{cursor:pointer}button[aria-selected="true"]{background:var(--ink);color:white;border-color:var(--ink)}
button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #cc7500;outline-offset:3px}
section[hidden]{display:none}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:16px}.stat strong{font-size:31px;display:block}
.stat span,.muted{color:var(--muted)}.notice{border-left:5px solid #c77c0b;background:#fff5df}.takeaway{border-left:4px solid var(--teal);padding:12px 16px;background:#ecf6f3;margin-top:14px}
.pipeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:10px}.step{padding:16px;background:#eff5f5;border:1px solid var(--line);border-radius:8px}.step b{display:block;margin-bottom:6px}
.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf3f3}
.controls{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px}.controls label{display:flex;flex-direction:column;gap:4px;font-size:14px;font-weight:650}
svg{display:block;width:100%;min-width:570px;height:auto}svg text{font-family:system-ui,Arial,sans-serif;font-size:14px;fill:var(--ink)}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:14px}.dot{display:inline-block;width:13px;height:13px;margin-right:6px}
a{color:#125cb0}code{font-size:13px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f3;padding:16px;border-radius:8px}
#numeric-audit{overflow-wrap:anywhere}
dl{display:grid;grid-template-columns:minmax(130px,1fr) 3fr;gap:10px 18px}dt{font-weight:650}dd{margin:0;overflow-wrap:anywhere}
.empty{padding:35px 20px;border:2px dashed #b4c6ca;border-radius:9px;color:var(--muted)}.small{font-size:14px}
@media(max-width:600px){main{padding:0 14px 35px}header{padding:28px 20px}.card{padding:20px}dl{grid-template-columns:1fr}dd{margin-bottom:10px}nav button{padding:8px 10px}}
@media print{nav,.controls{display:none}section[hidden]{display:block}.card{break-inside:avoid}header{background:white;color:black}header p{color:black}}
</style></head><body>
<header><div class="eyebrow">One agent / 50,000 movie reviews / sampling sensitivity study</div>
<h1>How far can we shorten a review embedding?</h1>
<p><span id="representation-description">Native 1,536-dimensional full-review embeddings versus normalized prefixes.</span>
The question is prediction quality for reviews whose sentiment was not selected for observation.</p>
<span class="badge" id="status"></span></header>
<main><nav role="tablist" aria-label="Experiment report">
<button id="tab-overview" role="tab" aria-controls="overview" aria-selected="true">Overview</button>
<button id="tab-method" role="tab" aria-controls="method" aria-selected="false" tabindex="-1">Method</button>
<button id="tab-dataset" role="tab" aria-controls="dataset" aria-selected="false" tabindex="-1">Dataset</button>
<button id="tab-results" role="tab" aria-controls="results" aria-selected="false" tabindex="-1">Results</button>
<button id="tab-provenance" role="tab" aria-controls="provenance" aria-selected="false" tabindex="-1">Provenance</button></nav>
<section id="overview" role="tabpanel" aria-labelledby="tab-overview">
<div class="card notice" id="run-status"></div><div class="card" id="extension-note" hidden></div>
<div class="grid"><div class="card stat"><span>Labeled source reviews</span><strong id="n"></strong><span>All treated as one agent</span></div>
<div class="card stat"><span>Paired replay seeds</span><strong id="repetitions"></strong><span id="repeat-label"></span></div>
<div class="card stat"><span>Embedding dimensions</span><strong id="dimension-count"></strong><span id="prefix-count"></span></div>
<div class="card stat"><span>Session-label budgets</span><strong>1% &ndash; 20%</strong><span>Not tokens or a weekly threshold</span></div></div>
<div class="card"><h2>The experiment in one minute</h2><div class="pipeline" aria-label="Experiment pipeline">
<div class="step"><b>1. Read</b>One full review becomes one session. Its sentiment is hidden from selection.</div>
<div class="step"><b>2. Embed</b>Generate native vectors once. Slice the first n values and re-normalize.</div>
<div class="step"><b>3. Sample</b>Replay semantic novelty/rarity selection under a fixed label budget.</div>
<div class="step"><b>4. Estimate</b>Only earlier selected labels supply IDW and envelope calibration.</div>
<div class="step"><b>5. Compare</b>Score unselected occurrences against their withheld sentiment labels.</div></div>
<p class="takeaway">Short vectors save storage and distance-computation work, not the cost of the original 1,536-dimensional embedding call. This study tests every configured end-to-end pipeline, so membership may differ across dimensions.</p></div>
<div class="card"><h2>Analysis and conclusion</h2><div id="conclusion"></div>
<p>Even a favorable result would apply to this polarized movie-review corpus, these budgets and replay settings.
Sentiment prediction is not agent task completion. Repeated orders reuse the same labels; they do not measure new judge reliability or production generalization.</p></div>
<div class="card"><h2>The budget picture at a glance</h2><p><b>What it shows:</b> mean all-unselected MAE, averaging both schedules within each seed.
<b>How to read:</b> lower values and lighter cells are better. This overview includes every tested budget, not only the explorer's default 5% scope.</p>
<div class="scroll"><table id="budget-overview"><thead></thead><tbody></tbody></table></div>
<p class="takeaway" id="budget-conclusion"></p></div>
</section>
<section id="method" role="tabpanel" aria-labelledby="tab-method" hidden>
<div class="card"><h2>Shared pipeline, tested representations</h2><p>Every arm uses the same source pool, label mapping, paired arrival draws and label budgets.
The only representation change is the number of retained coordinates. There is no PCA, SVD, learned projection or reduced-dimension API call.</p>
<div class="scroll"><table><thead><tr><th>Arm</th><th>Representation</th><th>Selection</th><th>Prediction</th></tr></thead><tbody id="methods"></tbody></table></div>
<h3>What &ldquo;end to end&rdquo; means here</h3><p>The existing ARM2 semantic selector processes each timestamped occurrence, with cosine cluster threshold 0.55, cluster TTL 90 and the existing novelty/rarity logic.
Its proposed keeps rank first, then remaining occurrences; novelty, rarity and a deterministic tie-break rank each group. The first floor(N &times; budget rate) are selected.
This is <b>label-blind full-schedule ranking, not strictly online admission</b>. Only imputation and calibration are causal.</p>
<p>A review is a single-message session. Input is the entire normalized review text, not a multi-span agent trace or a synthetic summary. Sentiment, rating, filename and original split never enter the embedding text.</p></div>
<div class="card"><h2>How IDW turns nearby labels into a score</h2>
<p>Use up to eight nearest earlier selected occurrences in the same agent. Distance is normalized angular distance
<code>acos(clamp(cosine,-1,1))/pi</code>. The inverse-square weights use epsilon <code>1e-6</code>.
Average all earlier exact-match donors when <code>1-cosine &lt;= 1e-8</code>; otherwise use nearest-neighbor IDW.
Without an earlier selected donor, a 0.5 prior has no claimed envelope.</p>
<div class="takeaway"><b>Illustration only, not a measured result:</b> donor labels 1 and 0 at distances 0.1 and 0.2 have inverse-square weights in a 4:1 ratio.
Ignoring epsilon, their normalized weights are 0.8 and 0.2, so the estimated score is 0.8. Threshold 0.5 predicts positive. A score of 0.8 is not a direct observation.</div>
<h3>Conditional Lipschitz envelope</h3><p>The lower score is <code>max(0, IDW - L * weighted_donor_distance)</code>.
An empirical 90th-percentile label slope estimates L, using angular distance with a 0.01 denominator floor.
For scalability, calibration uses a deterministic uniform reservoir of at most 128 earlier selected donors instead of the previous small-corpus study's all-pair history.
Sparse calibration uses L=1. The reservoir is label-blind; L is recomputed when it changes. No target or future selected label is used.</p>
<p><b>This is a sensitivity envelope, not a confidence interval or a guaranteed bound.</b> A 90th-percentile slope can omit steeper pairs.
Changing dimensions changes geometry and calibration. The lower score can reduce false positives at a common threshold while also losing true positives; precision is not guaranteed to improve.</p></div>
<div class="card"><h2>Order, frequency and label budgets</h2><p id="replay-design"></p>
<p>Each replay draws N reviews with replacement: some source reviews recur, others are absent. Uniform and bursty timestamps span the same 10,000-unit synthetic horizon;
these are simulated arrival patterns, not historical IMDb traffic. Dimension and budget arms share the same ordered occurrence stream for a seed/schedule.</p>
<div class="scroll"><table><thead><tr><th>Label rate</th><th>Selected occurrences / replay</th><th>Unselected occurrences</th></tr></thead><tbody id="budgets"></tbody></table></div>
<p>Budgets count selected occurrences, including duplicates, not unique source reviews or paid judge calls. Labels are immediate when a selected occurrence arrives.
Review-frequency reuse can inflate performance through exact matches; the novel-source diagnostic excludes targets whose source was already selected earlier.
It does not eliminate distinct rows with identical review text or related movies.</p></div>
<div class="card"><h2>Numerical implementation</h2><p>Prefix slicing and normalization reuse the earlier helper. Blocked distance evaluation re-normalizes in float64 rather than allocating a 50K-by-50K pairwise matrix.
Retained scores are float64; exact tied-score ROC is calculated from those values. This experiment has its own source/runtime binding and does not claim bitwise parity with the earlier float32 distance-cache study.</p></div>
</section>
<section id="dataset" role="tabpanel" aria-labelledby="tab-dataset" hidden>
<div class="card"><h2>The source: IMDb Large Movie Review Dataset</h2><p>The request's dataset link was missing. The described 50K positive/negative movie-review dataset is interpreted as the original Stanford IMDb corpus
(Maas et al., ACL 2011), downloaded from its publisher rather than an unverified CSV mirror.</p>
<p>The 25K training and 25K test reviews are combined as one agent. The original train/test split is therefore <b>not a held-out evaluation split in this study</b>.
Only polarized reviews are labeled: ratings at least 7 are positive, ratings at most 4 are negative. Neutral sentiment is not represented.
The separate 50K unlabeled reviews are excluded.</p><div class="grid" id="data-cards"></div>
<h3>Observed input profile</h3><dl id="profile"></dl>
<p>All labeled rows are retained. Identical normalized text shares one cached embedding, not one evaluation row. One movie can have several correlated reviews.
The downloaded archive and individual review text remain local; this report contains aggregates only.</p>
<p><a href="https://ai.stanford.edu/~amaas/data/sentiment/">Publisher and download</a> &middot;
<a href="https://aclanthology.org/P11-1015/">Dataset paper and citation</a></p></div></section>
<section id="results" role="tabpanel" aria-labelledby="tab-results" hidden>
<div class="card"><h2>Results explorer</h2><p id="results-state"></p>
<div class="controls"><label>Label budget<select id="budget"></select></label>
<label>Arrival schedule<select id="schedule"><option value="all">Both schedules (paired seed mean)</option></select></label>
<label>ROC dimension<select id="dimension"></select></label></div>
<p class="small">All scores use positive sentiment as class 1 and <code>score &gt;= 0.5</code> for classification.
Primary metrics exclude directly observed selected labels. Point-versus-lower comparisons use identical envelope-eligible target occurrences.</p></div>
<div class="card"><h2>1. How much probability error does shortening add?</h2>
<p><b>What it shows:</b> all-unselected point-score MAE on a 0&ndash;1 scale, lower is better. <b>How to read:</b> bars are seed means; whiskers are empirical 2.5th&ndash;97.5th percentiles of replay scores.</p>
<div id="mae-chart" class="scroll"></div><p id="mae-takeaway" class="takeaway"></p>
<p class="small">All-unselected includes explicitly identified priors/fallbacks. Selected labels are excluded. With both schedules selected, schedules are averaged within each seed first.
Dimension arms share draws but may leave different targets unselected; these are pipeline comparisons, not identical-target representation tests.</p></div>
<div class="card"><h2>2. Do the predicted sentiment labels hold up?</h2>
<p><b>What it shows:</b> mean accuracy, precision, recall and F1 for all-unselected point estimates at threshold 0.5.
<b>How to read:</b> higher is better, but precision and recall may trade off. No threshold is tuned on these labels.</p>
<div class="legend"><span><i class="dot" style="background:#007d7c"></i>Accuracy</span><span><i class="dot" style="background:#235fba"></i>Precision</span><span><i class="dot" style="background:#af5215"></i>Recall</span><span><i class="dot" style="background:#7b489b"></i>F1</span></div>
<div id="classification-chart" class="scroll"></div><p class="takeaway">Read the novel-source rows below before treating replay accuracy as evidence of new-review prediction. Repeated source reviews can benefit from a previously observed identical label.</p></div>
<div class="card"><h2>3. Point estimate versus the lower envelope</h2>
<p><b>What it shows:</b> ROC = true-positive rate (recall) versus false-positive rate, on the same eligible targets.
<b>How to read:</b> top-left is better; the diagonal is random ranking. This is not an accuracy-versus-threshold plot.</p>
<div class="legend"><span><i class="dot" style="background:#007d7c"></i>IDW point</span><span><i class="dot" style="background:#af5215"></i>Lower envelope (dashed)</span></div>
<div id="roc-chart" class="scroll"></div><p id="roc-takeaway" class="takeaway"></p>
<p class="small">Curves average per-seed interpolated TPR on a shared FPR grid; AUROC is the mean of exact per-cell tied-score AUROCs, not the area under the displayed average curve.
Lower &lt;= point guarantees nested positive sets only at the same threshold, not better ROC ranking or greater precision.</p></div>
<div class="card"><h2>Denominators and numerical detail</h2><p>Counts are mean target occurrences per replay cell, not independent reviews. Novel-source point scores exclude previously selected occurrences of the same source review.
Eligible-point and eligible-lower rows have the same denominator. Undefined metrics are &ldquo;not measured&rdquo;, never zero.</p>
<p id="count-detail"></p><p class="small">Precision is conventionally zero when no positive labels are predicted; recall and AUROC are undefined when their reference classes are absent. No-positive F1 is zero when the positive reference class exists.</p>
<div class="scroll"><table id="metrics-table"><thead><tr><th>Dimension</th><th>Cohort / estimator</th><th>Mean n</th><th>MAE</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUROC</th></tr></thead><tbody></tbody></table></div>
<h3>What supplied the unselected scores?</h3><p>These are average counts per cell. An exact match is reported separately from ordinary neighbor interpolation.
Observed label coverage is the fraction of eligible labels inside the full lower/upper sensitivity envelope, not calibrated confidence coverage.</p>
<div class="scroll"><table id="diagnostics-table"><thead><tr><th>Dimension</th><th>Neighbor IDW</th><th>Exact match</th><th>Prior</th><th>Eligible sparse-L fallback</th><th>Observed label coverage</th></tr></thead><tbody></tbody></table></div>
<h3>Paired MAE change relative to native</h3><p>Positive means worse. Pair each seed/schedule/budget before taking replay quantiles; these describe order/frequency sensitivity, not independent-population confidence.</p>
<div class="scroll"><table id="paired-table"><thead><tr><th>Dimension</th><th>Mean MAE difference</th><th>Replay 2.5% &ndash; 97.5%</th><th>Seeds</th></tr></thead><tbody></tbody></table></div></div>
</section>
<section id="provenance" role="tabpanel" aria-labelledby="tab-provenance" hidden>
<div class="card"><h2>Reproducibility and evidence boundary</h2><dl id="provenance-list"></dl>
<p>Input artifact: <code>__SOURCE__</code>. This is an explicit source, not an implicit &ldquo;latest&rdquo; run.
Machine-readable <a href="summary.json">summary.json</a> contains the displayed aggregates; <a href="manifest.json">manifest.json</a> binds report and generator hashes.</p>
<h3>Run from this worktree</h3><pre id="reproduction-commands">.\.venv-v3\Scripts\python.exe scripts\prepare_imdb_input.py --live --env-file .env
.\.venv-v3\Scripts\python.exe scripts\run_imdb_experiment.py
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_experiment.py
.\.venv-v3\Scripts\python.exe scripts\build_imdb_report.py --input outputs_imdb\private_runs\imdb-40-replay\aggregate.json --output outputs_imdb\reports\imdb-40-replay --numerical-validation outputs_imdb\reports\imdb-40-replay\numerical_validation.json
.\.venv-v3\Scripts\python.exe .github\skills\sampling-experiment-report\quality_check.py --html outputs_imdb\reports\imdb-40-replay\report.html --screenshots outputs_imdb\reports\imdb-40-replay\validation_screenshots</pre>
<p id="embedding-cost-note">Only the first command makes authorized embedding calls, and only for missing content-bound batches. No LLM judge or Azure Search resource is required.
Replay work is offline and resumable. Input hash or protocol changes must not silently reuse old results. Keep raw reviews, vectors and per-target evidence in ignored local storage.</p>
<h3>Validation</h3><p id="numeric-audit"></p><p>The report generator does not claim that tests or browser checks passed merely because this page exists.
The separately generated <code>validation.json</code> and <code>interaction_validation.json</code> record rendering and control checks.
The separately generated <code>numerical_validation.json</code> verifies retained artifact hashes, source-label alignment, causal positions and recomputed metrics across all completed cells.
Pending input profiles cannot produce measured result charts. No PDF was requested or generated.</p>
<h3>What is not established</h3><p>There is no independent test-label evaluation, live judge, real traffic-frequency model, production confidence interval, validated weekly policy or comparison to trained sentiment classifiers.
Novel-source does not mean a novel movie or duplicate-free text. This study uses full-review evidence, not multi-turn tool-call traces.
The reservoir calibration is a documented scale adaptation, not numerical parity with the previous small-corpus all-pair envelope study.</p></div>
</section></main>
<script id="report-data" type="application/json">__DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("report-data").textContent), P=D.dataset;
const COHORTS=["all_unselected","eligible_point","eligible_lower","novel_source"],METRICS=["mae","accuracy","precision","recall","f1","auc"];
const $=id=>document.getElementById(id), complete=D.status==="completed";
const fmt=(v,n=3)=>v==null?"Not measured":Number(v).toFixed(n);
const count=v=>v==null?"Not measured":Math.round(v).toLocaleString("en-US");
const text=(id,value)=>{$(id).textContent=value};
function activate(tab){document.querySelectorAll('[role="tab"]').forEach(t=>{const active=t===tab;t.setAttribute("aria-selected",active);t.tabIndex=active?0:-1;$(t.getAttribute("aria-controls")).hidden=!active});}
const tabs=[...document.querySelectorAll('[role="tab"]')];tabs.forEach((tab,i)=>{tab.addEventListener("click",()=>activate(tab));tab.addEventListener("keydown",e=>{let j=null;if(e.key==="ArrowRight")j=(i+1)%tabs.length;if(e.key==="ArrowLeft")j=(i+tabs.length-1)%tabs.length;if(e.key==="Home")j=0;if(e.key==="End")j=tabs.length-1;if(j!==null){e.preventDefault();tabs[j].focus();activate(tabs[j])}})});
function addOption(select,value,label){const o=document.createElement("option");o.value=value;o.textContent=label;select.append(o)}
function addRow(body,values){const tr=document.createElement("tr");values.forEach(v=>{const td=document.createElement(body.tagName==="THEAD"?"th":"td");td.textContent=v;tr.append(td)});body.append(tr)}
function addDl(id,pairs){const dl=$(id);pairs.forEach(([k,v])=>{const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=k;dd.textContent=v;dl.append(dt,dd)})}
text("status",complete?"COMPLETED REPLAY STUDY":"DATA READY / RESULTS NOT MEASURED");
text("representation-description",`Native 1,536-dimensional full-review embeddings versus normalized prefixes of ${D.protocol.dimensions.filter(d=>d!==1536).join(", ")} coordinates.`);
text("dimension-count",D.protocol.dimensions.length);text("prefix-count",`Native plus ${D.protocol.dimensions.length-1} prefix lengths`);
text("n",count(P.sessions));text("repetitions",D.protocol.repetitions);text("repeat-label",complete?"Repeated labels, not independent data":"Planned; no replay results yet");
$("run-status").innerHTML=complete?`<h2>Measured replay results</h2><p>Explore ${D.protocol.dimensions.length} pipelines by budget and arrival schedule. Read uncertainty as replay sensitivity, not production confidence.</p>`:D.embeddings_ready?'<h2>Embeddings are ready; replay results are pending.</h2><p>The input cache is complete but no completed replay aggregate was supplied. This is not a performance result.</p>':'<h2>The data is ready. The performance question is still open.</h2><p>Azure OpenAI configuration was unavailable in this isolated worktree. No embedding calls or benchmark replays were performed. This is a prepared experiment and dataset report, not a completed performance report. There are no substituted embeddings, invented metrics or placeholder ROC curves.</p>';
text("replay-design",`${D.protocol.repetitions} seed draws x ${D.protocol.schedules.length} arrival schedules x ${D.protocol.rates.length} budgets x ${D.protocol.dimensions.length} dimensions = ${count(D.protocol.planned_cells)} planned replay cells. Every cell contains ${count(P.sessions)} review occurrences.`);
D.protocol.dimensions.forEach(d=>{addRow($("methods"),[d===1536?"Native 1536":`${d}-coordinate prefix`,"First "+d+" coordinates, L2 normalized","ARM2 semantic novelty/rarity; rerun in this geometry","Causal angular IDW + conditional lower envelope"]);addOption($("dimension"),d,d===1536?"Native 1536":`${d} dimensions`)});
D.protocol.rates.forEach(r=>{const b=Math.max(1,Math.floor(P.sessions*r));addRow($("budgets"),[`${r*100}%`,count(b),count(P.sessions-b)]);addOption($("budget"),r,`${r*100}% (${count(b)} selected)`)});
D.protocol.schedules.forEach(s=>addOption($("schedule"),s,s.replaceAll("_"," ")));$("budget").value=String(D.protocol.rates.includes(.05)?.05:D.protocol.rates[0]);
[["Positive reviews",P.positive_count],["Negative reviews",P.negative_count],["Unique embedding inputs",P.unique_texts]].forEach(([k,v])=>{const box=document.createElement("div");box.className="stat";const span=document.createElement("span"),strong=document.createElement("strong");span.textContent=k;strong.textContent=count(v);box.append(span,strong);$("data-cards").append(box)});
addDl("profile",[["Original split",`${count(P.split_counts?.train)} train / ${count(P.split_counts?.test)} test; combined`],["Duplicate text rows",count(P.duplicate_text_rows)],["Conflicting label text groups",count(P.conflicting_label_text_groups)],["Input tokens (unique texts)",count(P.unique_embedding_input_tokens)],["Token length",`min ${count(P.token_length?.min)} / median ${count(P.token_length?.median)} / p95 ${count(P.token_length?.p95)} / max ${count(P.token_length?.max)}`],["Truncated reviews",count(P.truncated_sessions)],["Tokenizer",P.tokenizer||"Not recorded"],["Preprocessing",P.representation_policy||"Not recorded"]]);
addDl("provenance-list",[["Report status",D.status],["Source archive SHA-256",P.source_sha256||"Not recorded"],["Input artifact SHA-256",D.source_artifact_sha256],["Generated (UTC)",D.generated_at],["Embedding model",P.model||"text-embedding-3-small"],["Recorded successful embedding calls",D.embeddings_ready?count(P.embedding_calls):"Not run"],["Observed API input tokens",D.embeddings_ready?count(P.api_input_tokens):"Not run"],["LLM judge calls",count(P.live_judge_calls)],["Citation",P.citation||"Maas et al., ACL 2011"]]);
if(D.extension){
const e=D.extension; $("extension-note").hidden=false;
const heading=document.createElement("h2");heading.textContent="Additional dimensions; original results preserved";
const detail=document.createElement("p");detail.textContent=`This extension adds ${e.added_dimensions.join(", ")} dimensions: ${count(e.added_cells)} newly evaluated cells alongside ${count(e.reused_cells)} unchanged original cells. It reuses the same native embedding cache and exact recorded arrivals, source frequencies, seeds and label budgets. No new embedding or judge calls were made.`;
$("extension-note").append(heading,detail);
addDl("provenance-list",[["Original aggregate SHA-256",e.baseline_aggregate_sha256],["Original manifest SHA-256",e.baseline_manifest_sha256],["Preserved / additional cells",`${count(e.reused_cells)} / ${count(e.added_cells)}`],["Extra embedding calls",e.embedding_calls]]);
text("reproduction-commands",String.raw`.\.venv-v3\Scripts\python.exe scripts\extend_imdb_experiment.py --baseline outputs_imdb\private_runs\imdb-40-replay --output outputs_imdb\private_runs\imdb-40-replay-extended --resume
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_experiment.py --run outputs_imdb\private_runs\imdb-40-replay-extended
.\.venv-v3\Scripts\python.exe scripts\build_imdb_report.py --input outputs_imdb\private_runs\imdb-40-replay-extended\aggregate.json --output outputs_imdb\reports\imdb-40-replay --numerical-validation outputs_imdb\reports\imdb-40-replay\numerical_validation.json
.\.venv-v3\Scripts\python.exe .github\skills\sampling-experiment-report\quality_check.py --html outputs_imdb\reports\imdb-40-replay\report.html --screenshots outputs_imdb\reports\imdb-40-replay\validation_screenshots`);
text("embedding-cost-note","These commands require the preserved original run and native cache. They make no embedding, LLM judge or Azure Search calls. The extra dimensions use prefix slicing only. Completed checkpoints are source-bound and resumable; source input, method or runtime changes fail closed.");
}
function scopeRows(){return D.protocol.dimensions.map(dim=>D.summaries.find(r=>r.dimension===dim&&r.rate===Number($("budget").value)&&r.schedule===$("schedule").value)).filter(Boolean)}
function svg(content,label){return `<svg viewBox="0 0 740 320" role="img" aria-label="${label}">${content}</svg>`}
function axes(ylabel){let s='<line x1="65" y1="260" x2="715" y2="260" stroke="#9bafb5"/><line x1="65" y1="20" x2="65" y2="260" stroke="#9bafb5"/>';for(let i=0;i<=4;i++){let y=260-i*60;s+=`<line x1="65" y1="${y}" x2="715" y2="${y}" stroke="#e3eaea"/><text x="54" y="${y+5}" text-anchor="end">${(i/4).toFixed(2)}</text>`}return s+`<text x="66" y="15">${ylabel}</text>`}
function empty(id){$(id).innerHTML='<div class="empty">Not measured. Real Azure embeddings and completed replay cells are required before this chart can be drawn.</div>'}
function draw(){
const rows=scopeRows(),dimension=Number($("dimension").value),chosen=rows.find(r=>r.dimension===dimension);
text("results-state",complete?`Showing ${$("budget").selectedOptions[0].textContent}; ${$("schedule").selectedOptions[0].textContent}. Counts and metrics refer only to this scope.`:"No performance measurements exist yet. Filters describe the planned comparisons; changing them cannot create results.");
$("metrics-table").tBodies[0].replaceChildren();$("paired-table").tBodies[0].replaceChildren();$("diagnostics-table").tBodies[0].replaceChildren();
if(!rows.length){["mae-chart","classification-chart","roc-chart"].forEach(empty);text("mae-takeaway","Native-versus-prefix MAE is not yet known.");text("roc-takeaway","Point and lower-envelope ROC / AUROC are not yet known.");text("count-detail","Selected, IDW, exact-match and fallback occurrence counts have not been measured.");return}
const step=650/rows.length, x=i=>65+step*(i+.5), colors=["#007d7c","#235fba","#af5215","#7b489b"];
let mae=axes("MAE (0-1; lower is better)"), cls=axes("Metric (0-1; higher is better)");
rows.forEach((r,i)=>{const m=r.cohorts.all_unselected.mae;if(m.mean!=null){const y=260-m.mean*240;mae+=`<rect x="${x(i)-24}" y="${y}" width="48" height="${m.mean*240}" fill="${r.dimension===1536?'#172d39':'#007d7c'}"/><line x1="${x(i)}" y1="${260-m.high*240}" x2="${x(i)}" y2="${260-m.low*240}" stroke="#af5215" stroke-width="3"/><text x="${x(i)}" y="${Math.max(32,260-m.high*240-10)}" text-anchor="middle">${fmt(m.mean)}</text>`}["accuracy","precision","recall","f1"].forEach((metric,j)=>{const v=r.cohorts.all_unselected[metric].mean;if(v!=null)cls+=`<rect x="${x(i)-34+j*17}" y="${260-v*240}" width="14" height="${v*240}" fill="${colors[j]}"/>`});const label=r.dimension===1536?"1536 native":r.dimension;mae+=`<text x="${x(i)}" y="287" text-anchor="middle">${label}</text>`;cls+=`<text x="${x(i)}" y="287" text-anchor="middle">${label}</text>`;
COHORTS.forEach(c=>{const metrics=r.cohorts[c];addRow($("metrics-table").tBodies[0],[r.dimension,c.replaceAll("_"," "),count(metrics.n.mean),...METRICS.map(m=>fmt(metrics[m].mean))])});});
rows.forEach(r=>{const d=r.paired_mae_delta_native;addRow($("paired-table").tBodies[0],[r.dimension,fmt(d.mean,4),`${fmt(d.low,4)} to ${fmt(d.high,4)}`,d.replays])});
rows.forEach(r=>{const d=r.diagnostics;addRow($("diagnostics-table").tBodies[0],[r.dimension,count(d.idw.mean),count(d.exact_match.mean),count(d.prior.mean),count(d.calibration_fallback_eligible.mean),fmt(d.envelope_label_coverage.mean)])});
$("mae-chart").innerHTML=svg(mae,"Mean unselected MAE by embedding dimension with replay quantiles");$("classification-chart").innerHTML=svg(cls,"Unselected accuracy precision recall and F1 by dimension");
const native=rows.find(r=>r.dimension===1536),eight=rows.find(r=>r.dimension===8);
if(chosen){const all=chosen.cohorts.all_unselected.n.mean,eligible=chosen.cohorts.eligible_point.n.mean; text("count-detail",`${dimension}d: ${count(P.sessions-all)} directly observed selected occurrences, ${count(all)} unselected targets, ${count(eligible)} envelope-eligible IDW/exact targets, and ${count(all-eligible)} prior/fallback targets per replay cell. Novel-source diagnostic n: ${count(chosen.cohorts.novel_source.n.mean)}.`)}
text("mae-takeaway",native&&eight?`In this scope: native MAE ${fmt(native.cohorts.all_unselected.mae.mean)}, 8d MAE ${fmt(eight.cohorts.all_unselected.mae.mean)}. Paired 8d-minus-native change: ${fmt(eight.paired_mae_delta_native.mean,4)}. Do not generalize this average to every replay.`:"Compare dimensions within this budget and schedule; lower MAE is better.");
if(chosen&&chosen.roc.point.tpr.length&&chosen.roc.lower.tpr.length){let roc=axes("True-positive rate / recall");for(let i=0;i<=4;i++)roc+=`<text x="${65+i/4*650}" y="282" text-anchor="middle">${(i/4).toFixed(2)}</text>`;roc+='<line x1="65" y1="260" x2="715" y2="20" stroke="#9bafb5" stroke-dasharray="5 5"/><text x="390" y="310" text-anchor="middle">False-positive rate</text>';["point","lower"].forEach((method,j)=>{const c=chosen.roc[method],path=c.fpr.map((v,i)=>`${i?"L":"M"}${65+v*650},${260-c.tpr[i]*240}`).join(" ");roc+=`<path d="${path}" fill="none" stroke="${j?'#af5215':'#007d7c'}" stroke-width="3" ${j?'stroke-dasharray="8 5"':''}/>`});$("roc-chart").innerHTML=svg(roc,"Paired eligible IDW point and lower-envelope ROC curves");text("roc-takeaway",`${dimension}d: mean exact AUROC point ${fmt(chosen.cohorts.eligible_point.auc.mean)}, lower ${fmt(chosen.cohorts.eligible_lower.auc.mean)}; eligible n ${count(chosen.cohorts.eligible_point.n.mean)} per replay cell. At threshold 0.5, recall is ${fmt(chosen.cohorts.eligible_point.recall.mean)} versus ${fmt(chosen.cohorts.eligible_lower.recall.mean)}.`)}else{empty("roc-chart");text("roc-takeaway","No defined two-class eligible ROC for this scope.")}
}
["budget","schedule","dimension"].forEach(id=>$(id).addEventListener("change",draw));
function paragraph(message){const p=document.createElement("p");p.textContent=message;$("conclusion").append(p)}
if(!complete){$("conclusion").innerHTML=D.embeddings_ready?"<p><b>No performance conclusion is justified yet.</b> All 50,000 reviews have verified real native embeddings. The full paired replay sweep is still required before comparing dimensions or estimating classification performance.</p><p>There is no evidence here yet that 8, 12, 16, 24 or 32 dimensions preserve IDW quality, or that the lower envelope improves classification. The completed report must weigh MAE, precision/recall, novel-source behavior and envelope eligibility together.</p>":"<p><b>No performance conclusion is justified yet.</b> All 50,000 source labels were prepared, but real embeddings and the replay sweep are still pending. There is no evidence here that 8, 12, 16, 24 or 32 dimensions preserve IDW quality, or that the lower envelope improves sentiment classification.</p><p>The next required action is to supply Azure configuration in this worktree and run the three commands in Provenance. The resulting report must weigh MAE, precision/recall, novel-source behavior and envelope eligibility together.</p>";text("budget-conclusion","Budget-response metrics have not been measured.")
}else{
const cell=(dimension,rate)=>D.summaries.find(r=>r.dimension===dimension&&r.rate===rate&&r.schedule==="all");
const a=cell(1536,.05),b=cell(8,.05),c=cell(32,.05);
const added=[256,128,64].map(d=>cell(d,.05)).filter(Boolean);
const five=D.protocol.dimensions.map(d=>cell(d,.05)).filter(Boolean);
const wins=D.protocol.rates.filter(rate=>{const n=cell(1536,rate);return n&&D.protocol.dimensions.filter(d=>d!==1536).every(d=>cell(d,rate).cohorts.all_unselected.mae.mean>n.cohorts.all_unselected.mae.mean)}).length;
paragraph(`Measured across budgets: native 1536d has strictly lower mean unselected MAE than every tested prefix in ${wins} of ${D.protocol.rates.length} budget averages, with both schedules averaged within each seed. This compares complete pipelines: shortening can change both selected membership and donor geometry.`);
if(five.length){const best=Math.min(...five.map(r=>r.cohorts.all_unselected.mae.mean)),leaders=five.filter(r=>Math.abs(r.cohorts.all_unselected.mae.mean-best)<1e-12);paragraph(`At 5% budget, the lowest mean MAE across all tested dimensions is ${fmt(best)}, at ${leaders.map(r=>`${r.dimension}d`).join(", ")}. This is a descriptive replay average, not a pre-specified non-inferiority threshold or a production generalization guarantee.`)}
if(a&&b){paragraph(`At the 5% label budget, native MAE is ${fmt(a.cohorts.all_unselected.mae.mean)} and accuracy is ${fmt(100*a.cohorts.all_unselected.accuracy.mean,1)}%. The 8d prefix has MAE ${fmt(b.cohorts.all_unselected.mae.mean)} and accuracy ${fmt(100*b.cohorts.all_unselected.accuracy.mean,1)}%; its paired MAE change is ${fmt(b.paired_mae_delta_native.mean,4)}. ${c?`The 32d compromise has MAE ${fmt(c.cohorts.all_unselected.mae.mean)} and accuracy ${fmt(100*c.cohorts.all_unselected.accuracy.mean,1)}%.`:""} Compare this measured loss with the coordinate-storage savings, not an assumed quality-neutral reduction.`);
if(added.length)paragraph(`Additional prefix results at the same 5% label budget: ${added.map(r=>`${r.dimension}d MAE ${fmt(r.cohorts.all_unselected.mae.mean)}, accuracy ${fmt(100*r.cohorts.all_unselected.accuracy.mean,1)}%, F1 ${fmt(r.cohorts.all_unselected.f1.mean)}`).join("; ")}. Compare the paired differences and replay ranges in Results, not rounded means alone.`);
paragraph(`For the native lower-envelope classifier at threshold 0.5, eligible precision changes from ${fmt(100*a.cohorts.eligible_point.precision.mean,1)}% to ${fmt(100*a.cohorts.eligible_lower.precision.mean,1)}%, while recall changes from ${fmt(100*a.cohorts.eligible_point.recall.mean,1)}% to ${fmt(100*a.cohorts.eligible_lower.recall.mean,1)}%. Mean exact AUROC changes from ${fmt(a.cohorts.eligible_point.auc.mean)} to ${fmt(a.cohorts.eligible_lower.auc.mean)}. Higher precision alone is not an overall improvement.${a.cohorts.eligible_lower.recall.mean<.1?" Here, very few true positives remain.":""}`);
paragraph(`In the novel-source diagnostic at 5%, native MAE is ${fmt(a.cohorts.novel_source.mae.mean)} versus ${fmt(b.cohorts.novel_source.mae.mean)} at 8d. This excludes earlier observations of the same source row, not different rows with identical text or related movies.${b.cohorts.novel_source.mae.mean>a.cohorts.novel_source.mae.mean?" Repeated-review reuse therefore does not explain away this native/8d quality gap.":""}`);
if(D.score_validation){const e=D.score_validation.native_5pct_equal_cell_diagnostics;paragraph(`The retained-score audit helps explain the native envelope behavior at 5%: ${fmt(100*e.lower_zero_fraction,1)}% of eligible lower scores are clipped to zero, and the mean full-envelope width is ${fmt(e.mean_envelope_width)} on the 0-1 outcome scale. Broad intervals can cover many observed labels without supplying informative classification. This is an observed diagnostic, not calibrated confidence.`)}
}else paragraph("The default 5% analysis scope was not included; no conclusion is substituted from a different budget.");
const low8=cell(8,.01),low32=cell(32,.01);
if(low8&&low32)paragraph(`The prefixes are not globally monotonic: at 1% budget, 8d MAE is ${fmt(low8.cohorts.all_unselected.mae.mean)} and 32d MAE is ${fmt(low32.cohorts.all_unselected.mae.mean)}. Read each budget and metric rather than inferring that every additional coordinate must help. Novelty-based membership changes are a possible contributor, not a proven causal explanation.`);
paragraph((wins===D.protocol.rates.length?"Conclusion: prefer native geometry when predictive quality is the priority for this evaluated corpus; accept a shorter prefix only with its observed budget-specific loss. ":"Conclusion: no universal native-preference rule follows across all tested budgets; choose by the observed budget-specific tradeoff. ")+"Treat the lower envelope as a sensitivity/abstention diagnostic, not a calibrated probability or a general replacement classifier. A time-forward, movie-disjoint evaluation and independent calibration would be needed before a deployment decision.");
addRow($("budget-overview").tHead,["Dimensions",...D.protocol.rates.map(r=>`${100*r}% labels`)]);
D.protocol.dimensions.forEach(d=>{const values=D.protocol.rates.map(r=>cell(d,r).cohorts.all_unselected.mae.mean);addRow($("budget-overview").tBodies[0],[d,...values.map(v=>fmt(v))]);const cells=$("budget-overview").tBodies[0].lastElementChild.children;values.forEach((v,i)=>{cells[i+1].style.backgroundColor=`hsl(30 75% ${97-Math.min(1,v)*38}%)`})});
text("budget-conclusion",`Native has the lowest mean MAE in ${wins}/${D.protocol.rates.length} tested budget averages. Increasing label budget and increasing dimension are different decisions; neither should be summarized by one pooled winner score.`);
}
text("numeric-audit",D.score_validation?`Source-bound retained-score validation passed for ${count(D.score_validation.cells_checked)} cells. Maximum absolute difference across recomputed metrics: ${D.score_validation.maximum_absolute_metric_difference}. Validation artifact SHA-256: ${D.score_validation.sha256}`:"No numerical audit artifact was attached to this report build; consult the separately generated validation files.");
draw();
</script></body></html>
"""
