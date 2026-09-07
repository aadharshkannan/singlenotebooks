from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


METHOD_ORDER = (
    "random_sampling",
    "minhash_lsh",
    "pca8_idw_binary_cosine",
    "pca8_idw_binary_euclidean",
    "agent_balanced_weighted_sampling",
)

PUBLIC_METHOD_IDS = {
    "arm5_hajek_weighted": "agent_balanced_weighted_sampling",
}
REVERSE_PUBLIC_METHOD_IDS = {value: key for key, value in PUBLIC_METHOD_IDS.items()}

METHOD_LABELS = {
    "random_sampling": "Random Sampling",
    "minhash_lsh": "MinHash LSH",
    "pca8_idw_binary_cosine": "PCA-8 Cosine + Binary IDW",
    "pca8_idw_binary_euclidean": "PCA-8 Euclidean + Binary IDW",
    "agent_balanced_weighted_sampling": "Agent-Balanced Weighted Sampling",
}

METHOD_NOTES = {
    "random_sampling": "Seeded exact-cap baseline using selected-rate estimation.",
    "minhash_lsh": "Lexical novelty/rarity ranking with selected-rate estimation.",
    "pca8_idw_binary_cosine": "PCA-8 cosine neighborhood selection with binary IDW population imputation.",
    "pca8_idw_binary_euclidean": "PCA-8 Euclidean neighborhood selection with binary IDW population imputation.",
    "agent_balanced_weighted_sampling": "Capacity-aware round robin with inverse-probability-weighted Hajek ratio.",
}

METRIC_CATALOG = (
    {
        "metric_id": "aggregate_pass_rate_mae",
        "display_name": "Fixed-source aggregate MAE",
        "direction": "lower_better",
        "denominator_note": "Absolute error against fixed source-corpus census pass rate.",
    },
    {
        "metric_id": "selected_only_pass_rate_mae",
        "display_name": "Fixed-source selected-only MAE",
        "direction": "lower_better",
        "denominator_note": "Absolute error only on judged selections against fixed source-corpus census pass rate.",
    },
    {
        "metric_id": "replay_aggregate_pass_rate_mae",
        "display_name": "Replay-relative aggregate MAE",
        "direction": "lower_better",
        "denominator_note": "Absolute error against each replay census pass rate; denominator differs from fixed-source MAE.",
    },
    {
        "metric_id": "coverage_concept_ratio",
        "display_name": "Concept coverage ratio",
        "direction": "higher_better",
        "denominator_note": "Represented concepts divided by corpus concept count.",
    },
    {
        "metric_id": "imputed_only_accuracy",
        "display_name": "Imputed-only accuracy",
        "direction": "higher_better",
        "denominator_note": "Classification quality over imputed-only population.",
    },
    {
        "metric_id": "imputed_only_precision",
        "display_name": "Imputed-only precision",
        "direction": "higher_better",
        "denominator_note": "Classification quality over imputed-only population.",
    },
    {
        "metric_id": "imputed_only_recall",
        "display_name": "Imputed-only recall",
        "direction": "higher_better",
        "denominator_note": "Classification quality over imputed-only population.",
    },
    {
        "metric_id": "imputed_only_f1",
        "display_name": "Imputed-only F1",
        "direction": "higher_better",
        "denominator_note": "Classification quality over imputed-only population.",
    },
    {
        "metric_id": "judged_plus_imputed_f1",
        "display_name": "Judged-plus-imputed F1",
        "direction": "higher_better",
        "denominator_note": "Observed-inflated by construction due to inclusion of judged labels.",
    },
)


def _public_method_id(value: Any) -> str:
    method_id = str(value or "unknown")
    return PUBLIC_METHOD_IDS.get(method_id, method_id)


def _source_method_id(value: Any) -> str:
    method_id = str(value or "unknown")
    return REVERSE_PUBLIC_METHOD_IDS.get(method_id, method_id)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _overall_metric(rows: Sequence[Mapping[str, Any]], metric_id: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("metric_id") or "") != metric_id or int(row.get("n") or 0) <= 0:
            continue
        value = _finite(row.get("mean"))
        if value is not None:
            values.setdefault(str(row.get("method_id") or "unknown"), []).append(value)
    return {method: sum(items) / len(items) for method, items in values.items() if items}


def _best_metric(
    rows: Sequence[Mapping[str, Any]], metric_id: str, *, higher_is_better: bool
) -> tuple[str, float | None]:
    values = _overall_metric(rows, metric_id)
    if not values:
        return "n/a", None
    pick = max if higher_is_better else min
    method = pick(values, key=values.get)
    return method, values[method]


def _paired_win_rate(rows: Sequence[Mapping[str, Any]], metric_id: str) -> float | None:
    matches = [row for row in rows if str(row.get("metric_id") or "") == metric_id]
    wins = sum(int(row.get("wins") or 0) for row in matches)
    total = sum(int(row.get("n") or 0) for row in matches)
    return (wins / total) if total else None


def _expected_run_shape_value(rows: Sequence[Mapping[str, Any]]) -> int:
    dataset_ids = {str(row.get("dataset_id")) for row in rows if isinstance(row, Mapping) and row.get("dataset_id") is not None}
    method_ids = {str(_public_method_id(row.get("method_id"))) for row in rows if isinstance(row, Mapping) and row.get("method_id") is not None}
    budgets = {int(row.get("budget_pct")) for row in rows if isinstance(row, Mapping) and row.get("budget_pct") is not None}
    repetitions = {int(row.get("repetition_index")) for row in rows if isinstance(row, Mapping) and row.get("repetition_index") is not None}
    return len(dataset_ids) * len(method_ids) * len(budgets) * len(repetitions)


def _trim_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "dataset_id", "seed", "repetition_index", "replay_seed", "replay_id",
        "replay_frequency_summary", "budget_pct", "cap", "sample_size", "method_id",
        "estimator_type", "estimate", "selected_rate", "replay_census_pass_rate",
        "source_corpus_census_pass_rate", "aggregate_pass_rate_mae",
        "selected_only_pass_rate_mae", "replay_aggregate_pass_rate_mae",
        "actual_token_count", "coverage", "embedding_metrics", "latency_seconds",
        "search_evidence",
    )
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key) for key in fields if key in row}
        item["method_id"] = _public_method_id(item.get("method_id"))
        trimmed.append(item)
    return trimmed


def _method_display(method_id: str) -> str:
    report_id = str(method_id or "unknown")
    source_id = _source_method_id(report_id)
    label = METHOD_LABELS.get(report_id, report_id)
    if source_id == report_id:
        return f"{label} ({report_id})"
    return f"{label} (report alias: {report_id}; source method ID: {source_id})"


def _format_ratio(value: Any, *, percent: bool = False, decimals: int = 4) -> str:
    finite = _finite(value)
    if finite is None:
        return "n/a"
    if percent:
        return f"{finite * 100.0:.1f}%"
    return f"{finite:.{decimals}f}"


def _findings_summary_text(item: Mapping[str, Any], *, methods: Sequence[Mapping[str, Any]] | None = None) -> str:
    method_id = str((item.get("value") or {}).get("method_id") or "n/a")
    source_id = _source_method_id(method_id)
    report_id = _public_method_id(method_id)
    label = METHOD_LABELS.get(report_id, report_id)
    identifier_text = f"{label} (report alias: {report_id}; source method ID: {source_id})"
    if source_id == report_id:
        identifier_text = f"{label} ({report_id})"

    finding_id = str(item.get("finding_id") or "")
    qualification = str(item.get("qualification") or "")
    if finding_id == "best_fixed_source_mae":
        mean = _format_ratio((item.get("value") or {}).get("mean"), decimals=4)
        return f"Best fixed-source MAE: {identifier_text} at {mean}. {qualification}"
    if finding_id == "best_concept_coverage":
        coverage = _format_ratio((item.get("value") or {}).get("mean"), percent=True, decimals=4)
        return f"Best concept coverage: {identifier_text} at {coverage}. {qualification}"
    if finding_id == "cosine_paired_win_rates":
        values = item.get("value") or {}
        ma = _format_ratio(values.get("aggregate_pass_rate_mae"), percent=True, decimals=1)
        acc = _format_ratio(values.get("imputed_only_accuracy"), percent=True, decimals=1)
        f1 = _format_ratio(values.get("imputed_only_f1"), percent=True, decimals=1)
        return f"Cosine paired win rates: MAE={ma}, Accuracy={acc}, F1={f1}. {qualification}"
    return f"{finding_id}: {item.get('value')}. {qualification}"


def _prepare_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = _normalise_rows(payload.get("runs") or [])
    profiles = _normalise_rows(payload.get("datasets") or payload.get("dataset_profiles") or [])
    aggregate = payload.get("aggregate_metrics") or {}
    paired = payload.get("paired_differences") or {}
    aggregate_rows = _normalise_rows(aggregate.get("rows") if isinstance(aggregate, Mapping) else [])
    for row in aggregate_rows:
        row["method_id"] = _public_method_id(row.get("method_id"))

    fallback_coverage_rows: list[dict[str, Any]] = []
    for row in rows:
        coverage = row.get("coverage") if isinstance(row.get("coverage"), Mapping) else {}
        concept = coverage.get("concept") if isinstance(coverage.get("concept"), Mapping) else {}
        coverage_value = _finite(concept.get("coverage_ratio"))
        if coverage_value is None:
            continue
        fallback_coverage_rows.append({
            "dataset_id": row.get("dataset_id"),
            "budget_pct": row.get("budget_pct"),
            "method_id": _public_method_id(row.get("method_id")),
            "metric_id": "coverage_concept_ratio",
            "n": 1,
            "mean": coverage_value,
        })
    if not any(str(item.get("metric_id") or "") == "coverage_concept_ratio" for item in aggregate_rows):
        aggregate_rows = [*aggregate_rows, *fallback_coverage_rows]

    paired_rows = _normalise_rows(paired.get("rows") if isinstance(paired, Mapping) else [])
    paired_cells = _normalise_rows(paired.get("cells") if isinstance(paired, Mapping) else [])
    summary = dict(payload.get("summary") or {})
    if isinstance(summary.get("method_id_order"), Sequence):
        summary["method_id_order"] = [
            _public_method_id(method_id) for method_id in summary["method_id_order"]
        ]

    best_mae_method, best_mae = _best_metric(
        aggregate_rows, "aggregate_pass_rate_mae", higher_is_better=False
    )
    best_concept_method, best_concept = _best_metric(
        aggregate_rows, "coverage_concept_ratio", higher_is_better=True
    )
    executive = {
        "bestMaeMethod": best_mae_method,
        "bestMae": best_mae,
        "bestConceptMethod": best_concept_method,
        "bestConcept": best_concept,
        "cosineMaeWinRate": _paired_win_rate(paired_rows, "aggregate_pass_rate_mae"),
        "cosineAccuracyWinRate": _paired_win_rate(paired_rows, "imputed_only_accuracy"),
        "cosineF1WinRate": _paired_win_rate(paired_rows, "imputed_only_f1"),
        "aggregation_note": "Best-method values are unweighted means across dataset-budget cell means.",
        "win_rate_note": "Win rates are unweighted over paired replay comparisons.",
    }
    report_data = {
        "rows": _trim_rows(rows),
        "profiles": profiles,
        "aggregateMetrics": aggregate_rows,
        "pairedRows": paired_rows,
        "pairedCells": paired_cells,
        "summary": summary,
    }
    return {
        "rows": rows,
        "profiles": profiles,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "paired_cells": paired_cells,
        "summary": summary,
        "executive": executive,
        "report_data": report_data,
    }


def _search_latency_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        method_id = _public_method_id(row.get("method_id"))
        if not method_id.startswith("pca8_"):
            continue
        latency = row.get("latency_seconds") if isinstance(row.get("latency_seconds"), Mapping) else {}
        value = _finite(latency.get("search_evidence"))
        if value is None:
            continue
        grouped[(str(row.get("dataset_id") or "unknown"), method_id)].append(value)

    shared_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        rounded_unique = sorted({round(float(item), 9) for item in values})
        shared_rows.append(
            {
                "dataset_id": key[0],
                "method_id": key[1],
                "occurrences": len(values),
                "mean_seconds": sum(values) / len(values),
                "unique_value_count": len(rounded_unique),
                "is_shared_reference_candidate": len(values) > 1 and len(rounded_unique) == 1 and values[0] > 0.0,
            }
        )
    return {
        "rows": shared_rows,
        "note": "Repeated, constant search-evidence latency indicates a shared one-time probe reused across replay rows.",
    }


def _patch_interactive_html(html: str) -> str:
    method_labels = (
        "const LABELS={"
        "random_sampling:'Random Sampling',"
        "minhash_lsh:'MinHash LSH',"
        "pca8_idw_binary_cosine:'PCA-8 Cosine + Binary IDW',"
        "pca8_idw_binary_euclidean:'PCA-8 Euclidean + Binary IDW',"
        "agent_balanced_weighted_sampling:'Agent-Balanced Weighted Sampling'};"
    )
    html = html.replace(
        "const LABELS={random_sampling:'Random Sampling',minhash_lsh:'MinHash LSH',pca8_idw_binary_cosine:'PCA-8 Cosine + Binary IDW',pca8_idw_binary_euclidean:'PCA-8 Euclidean + Binary IDW',agent_balanced_weighted_sampling:'Agent-Balanced Weighted Sampling'};",
        method_labels + "const CHART_LABELS={random_sampling:'Random',minhash_lsh:'MinHash',pca8_idw_binary_cosine:'PCA-8 Cos',pca8_idw_binary_euclidean:'PCA-8 Euclid',agent_balanced_weighted_sampling:'Agent-Balanced'};",
    )
    html = html.replace(
        "<p>The headline metric targets the fixed source-corpus pass rate. Replay-relative errors are retained separately, so bootstrap composition drift is visible rather than folded into the method ranking.</p>",
        "<p>The headline metric targets fixed-source Mean Absolute Error (MAE) against each corpus census pass rate. Executive best-method summaries use an unweighted mean across dataset-budget cell means. Replay-relative MAE is shown separately because it uses replay-specific denominators.</p>",
    )
    html = html.replace(
        "<p>Imputed accuracy",
        "<p>Imputed accuracy",
    )
    html = html.replace(
        "<h3>Random Sampling</h3><p>",
        "<h3>Random Sampling</h3><div class=\"method-id\"><span class=\"muted\">Report ID:</span> random_sampling</div><p>",
    )
    html = html.replace(
        "<h3>MinHash LSH</h3><p>",
        "<h3>MinHash LSH</h3><div class=\"method-id\"><span class=\"muted\">Report ID:</span> minhash_lsh</div><p>",
    )
    html = html.replace(
        "<h3>PCA-8 Cosine + Binary IDW</h3><p>",
        "<h3>PCA-8 Cosine + Binary IDW</h3><div class=\"method-id\"><span class=\"muted\">Report ID:</span> pca8_idw_binary_cosine</div><p>",
    )
    html = html.replace(
        "<h3>PCA-8 Euclidean + Binary IDW</h3><p>",
        "<h3>PCA-8 Euclidean + Binary IDW</h3><div class=\"method-id\"><span class=\"muted\">Report ID:</span> pca8_idw_binary_euclidean</div><p>",
    )
    html = html.replace(
        "<h3>Agent-Balanced Weighted Sampling</h3><p>",
        "<h3>Agent-Balanced Weighted Sampling</h3><div class=\"method-id\"><span class=\"muted\">Report ID:</span> agent_balanced_weighted_sampling<br><span class=\"muted\">Source method ID:</span> arm5_hajek_weighted</div><p>",
    )
    html = html.replace(
        "Agent-Balanced Weighted Sampling (agent_balanced_weighted_sampling) — Source method ID: arm5_hajek_weighted",
        "Agent-Balanced Weighted Sampling (agent_balanced_weighted_sampling) — Source method ID: arm5_hajek_weighted",
    )
    html = html.replace(
        "<td>Random Sampling</td>",
        "<td>Random Sampling<br><span class=\"muted\">random_sampling</span></td>",
    )
    html = html.replace(
        "<td>MinHash LSH</td>",
        "<td>MinHash LSH<br><span class=\"muted\">minhash_lsh</span></td>",
    )
    html = html.replace(
        "<td>PCA-8 Cosine + Binary IDW</td>",
        "<td>PCA-8 Cosine + Binary IDW<br><span class=\"muted\">pca8_idw_binary_cosine</span></td>",
    )
    html = html.replace(
        "<td>PCA-8 Euclidean + Binary IDW</td>",
        "<td>PCA-8 Euclidean + Binary IDW<br><span class=\"muted\">pca8_idw_binary_euclidean</span></td>",
    )
    html = html.replace(
        "<td>Agent-Balanced Weighted Sampling</td>",
        "<td>Agent-Balanced Weighted Sampling<br><span class=\"muted\">report alias: agent_balanced_weighted_sampling</span><br><span class=\"muted\">source method ID: arm5_hajek_weighted</span></td>",
    )
    html = html.replace(
        "<p>Lines show replay means at each budget; whiskers show P05-P95. The winner board ranks the active slice and states whether lower or higher is better. When all datasets are selected, corpora are faceted because equal percentages map to different absolute judged-session counts.</p>",
        "<p>Lines show replay means at each budget; whiskers show P05-P95. The winner board ranks the active slice and states whether lower or higher is better. Executive rollups and winner claims are unweighted over dataset-budget cells. When all datasets are selected, corpora are faceted because equal percentages map to different absolute judged-session counts.</p>",
    )
    html = html.replace(
        "<p>Thin range: P05-P95. Thick range: P25-P75. Dot: mean. Vertical ticks: Student-t CI95. These quantify replay sensitivity, not population generalization.</p>",
        "<p>Thin range: P05-P95. Thick range: P25-P75. Dot: mean. Vertical ticks: Student-t CI95. These quantify empirical replay sensitivity under perturbations of the same labeled corpus, not population generalization.</p>",
    )
    html = html.replace(
        "<p>Coverage panels share a 0-100% scale. Tokens and latency remain separate because they are different units. Concept coverage counts represented concepts, not geometric spread.</p>",
        "<p>Coverage panels share a 0-100% scale. Tokens and latency remain separate because they are different units. Runtime is split into incremental replay work (selection + IDW) versus shared reference search evidence. Concept coverage counts represented concepts, not geometric spread.</p>",
    )
    html = html.replace(
        "<p>Only the PCA-8 methods impute unsampled sessions. Accuracy, Precision, Recall, and F1 are shown on a common 0-1 scale. Imputed-only is primary; judged-plus-imputed is shown separately and is observed-inflated by construction.</p>",
        "<p>Only the PCA-8 methods impute unsampled sessions using inverse-distance weighting (IDW). Accuracy, Precision, Recall, and F1 are shown on a common 0-1 scale. Imputed-only is primary; judged-plus-imputed is shown separately and is observed-inflated by construction.</p>",
    )
    html = html.replace(
        "<p>Each point is one dataset/repetition, deduplicated across methods and budgets. Omitted sources and duplicated occurrences are expected under sampling with replacement.</p>",
        "<p>Each point is one dataset/repetition, deduplicated across methods and budgets. Omitted sources and duplicated occurrences are expected under sampling with replacement. Point color is neutral and has no method meaning.</p>",
    )
    html = html.replace(
        "<div class=\"context-note\"><strong>How to read this section:</strong> each summary names the best method for one objective across the full result set. Lower error is better; higher coverage and classification scores are better. These summaries are starting points, not a universal ranking.</div><div class=\"readout\" id=\"executiveReadout\"></div>",
        "<div class=\"context-note\"><strong>How to read this section:</strong> each summary names the best method for one objective across the full result set. Lower error is better; higher coverage and classification scores are better. These summaries are starting points, not a universal ranking.</div><div class=\"context-note\"><strong>Metric reading guide:</strong> MAE = Mean Absolute Error. Fixed-source MAE compares each estimate to the source corpus census pass rate; replay-relative MAE compares to replay census pass rate. Coverage uses dataset-specific denominators. Imputed-only classification is computed only on unsampled/imputed rows. Budget/cap means percent judged and resulting absolute judged-session cap. Replay intervals summarize perturbation sensitivity, not population generalization.</div><div class=\"readout\" id=\"executiveReadout\"></div>",
    )
    html = html.replace(
        "<p>The Foundry embedding model creates full-session vectors. Dedicated cosine and Euclidean Azure AI Search indexes validate PCA-8 retrieval; deterministic selection remains local.</p>",
        "<p>The Foundry embedding model creates full-session vectors. In this run all embedding ledger rows are cache hits, so live refers to replay/search evidence execution rather than fresh embedding calls. Dedicated cosine and Euclidean Azure AI Search indexes validate PCA-8 retrieval; deterministic selection remains local. Source hashes for cached embedding inputs are listed in final_report.json.</p>",
    )
    html = html.replace(
        "const METRICS={aggregate_pass_rate_mae:{label:'Fixed-source aggregate MAE',direction:'lower',format:'decimal'},selected_only_pass_rate_mae:{label:'Fixed-source selected-only MAE',direction:'lower',format:'decimal'},replay_aggregate_pass_rate_mae:{label:'Replay-relative aggregate MAE',direction:'lower',format:'decimal'},coverage_concept_ratio:{label:'Concept coverage',direction:'higher',format:'percent'},imputed_only_accuracy:{label:'Imputed-only accuracy',direction:'higher',format:'percent'},imputed_only_f1:{label:'Imputed-only F1',direction:'higher',format:'percent'}};",
        "const METRICS={aggregate_pass_rate_mae:{label:'Fixed-source aggregate MAE',direction:'lower',format:'decimal'},selected_only_pass_rate_mae:{label:'Fixed-source selected-only MAE',direction:'lower',format:'decimal'},replay_aggregate_pass_rate_mae:{label:'Replay-relative aggregate MAE',direction:'lower',format:'decimal'},coverage_concept_ratio:{label:'Concept coverage',direction:'higher',format:'percent'},imputed_only_accuracy:{label:'Imputed-only accuracy',direction:'higher',format:'percent'},imputed_only_f1:{label:'Imputed-only F1',direction:'higher',format:'percent'}};const ESTIMATOR_LABELS={selected_mean:'Selected-rate mean',selected_rate_mean:'Selected-rate mean',selected_rate:'Selected-rate mean',pca_binary_imputed_population:'Binary IDW imputed-population estimate',binary_idw_imputed_population:'Binary IDW imputed-population estimate',hajek_weighted_ratio:'Hajek-weighted ratio estimate',ipw_hajek_ratio:'Hajek-weighted ratio estimate'};const DATASET_DESCRIPTIONS={historical_300:'Historical support conversations with stable labeling policy.',dense_2500:'High-volume mixed-topic corpus for broad coverage pressure tests.',cosmos_otel:'Cosmos DB + OTEL operational traces with sparse high-value labels.'};const estimatorLabel=id=>ESTIMATOR_LABELS[String(id||'').toLowerCase()]||String(id||'n/a');const profileById=id=>DATA.profiles.find(p=>String(p.dataset_id)===String(id))||{};const pickCount=(obj,keys)=>{for(const k of keys){const v=Number(obj?.[k]);if(Number.isFinite(v)&&v>0)return v}return null};const datasetBasePassRate=id=>mean(DATA.rows.filter(r=>String(r.dataset_id)===String(id)).map(r=>r.source_corpus_census_pass_rate));const datasetSlices=()=>{const rows=visible();if(ctl.dataset.value!=='all')return[{dataset_id:ctl.dataset.value,rows}];return DATA.profiles.map(p=>({dataset_id:String(p.dataset_id),rows:rows.filter(r=>String(r.dataset_id)===String(p.dataset_id))})).filter(s=>s.rows.length)};const sliceCaption=s=>{const p=profileById(s.dataset_id),pop=Number(p.population||0);return `dataset ${s.dataset_id} · denominator population ${pop?pop.toLocaleString():'n/a'}`};",
    )
    html = html.replace(
        "Imputed accuracy ${pct(EXEC.cosineAccuracyWinRate)}; imputed F1 ${pct(EXEC.cosineF1WinRate)}.",
        "Imputed accuracy ${pct(EXEC.cosineAccuracyWinRate)}; imputed F1 ${pct(EXEC.cosineF1WinRate)}. Win rates are unweighted over paired replay comparisons.",
    )
    html = html.replace(
        "callout.innerHTML=`<div><span class=\"best-badge\">BEST FOR THIS VIEW</span><div class=\"rank\">${esc(LABELS[winner.method])}</div></div><div><strong>${esc(meta.label)}</strong><br><span class=\"muted\">${meta.direction==='lower'?'Lower is better':'Higher is better'} · ${visible().length} run rows in this slice</span></div><div class=\"margin\">${formatMetric(winner.value,meta)}${runnerUp?`<br><span class=\"muted\">margin ${formatMetric(margin,meta)} vs ${esc(LABELS[runnerUp.method])}</span>`:''}</div>`;",
        "callout.innerHTML=`<div><span class=\"best-badge\">BEST FOR THIS VIEW</span><div class=\"rank\">${esc(LABELS[winner.method])}</div></div><div><strong>${esc(meta.label)}</strong><br><span class=\"muted\">${meta.direction==='lower'?'Lower is better':'Higher is better'} · ${visible().length} run rows in this slice${ctl.dataset.value==='all'?' · pooled directional summary; absolute denominators differ by corpus':''}</span></div><div class=\"margin\">${formatMetric(winner.value,meta)}${runnerUp?`<br><span class=\"muted\">margin ${formatMetric(margin,meta)} vs ${esc(LABELS[runnerUp.method])}</span>`:''}</div>`;",
    )
    html = html.replace(
        "table.innerHTML=`<table class=\"rank-table\"><thead><tr><th>Rank</th><th>Method</th><th>Value</th><th>Gap from leader</th><th>Estimator context</th></tr></thead><tbody>${ranked.map((item,index)=>{const sample=visible().find(row=>row.method_id===item.method);return `<tr><td>${index+1}</td><td>${index===0?'<span class=\"best-badge\">BEST</span> ':''}${esc(LABELS[item.method])}</td><td class=\"metric-value\">${formatMetric(item.value,meta)}</td><td>${index===0?'leader':formatMetric(Math.abs(item.value-winner.value),meta)}</td><td>${esc(sample?.estimator_type||'n/a')}</td></tr>`}).join('')}</tbody></table>`;",
        "table.innerHTML=`<table class=\"rank-table\"><thead><tr><th>Rank</th><th>Method</th><th>Value</th><th>Gap from leader</th><th>Estimator context</th></tr></thead><tbody>${ranked.map((item,index)=>{const sample=visible().find(row=>row.method_id===item.method);return `<tr><td>${index+1}</td><td>${index===0?'<span class=\"best-badge\">BEST</span> ':''}${esc(LABELS[item.method])}</td><td class=\"metric-value\">${formatMetric(item.value,meta)}</td><td>${index===0?'leader':formatMetric(Math.abs(item.value-winner.value),meta)}</td><td>${esc(estimatorLabel(sample?.estimator_type))}<br><span class=\"muted\" title=\"raw estimator ID\">${esc(sample?.estimator_type||'n/a')}</span></td></tr>`}).join('')}</tbody></table>`;",
    )
    html = html.replace(
        "function renderClassification(){const runs=visible().filter(r=>String(r.method_id).startsWith('pca8_')&&r.embedding_metrics);if(!runs.length){document.getElementById('classificationMatrix').innerHTML=empty('Classification metrics apply to the PCA-8 methods.');return}const scopes=[['imputed_only','Imputed only (primary)'],['judged_plus_imputed','Judged + imputed (observed-inflated)']],metrics=['accuracy','precision','recall','f1'],methodValues={};scopes.forEach(([scope])=>metrics.forEach(metric=>{methodValues[`${scope}.${metric}`]=Object.fromEntries(['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].map(method=>[method,mean(runs.filter(r=>r.method_id===method).map(r=>r.embedding_metrics?.[scope]?.[metric]))]))}));let h='<table><thead><tr><th>Scope / method</th>'+metrics.map(m=>`<th>${m[0].toUpperCase()+m.slice(1)}<br><span class=\"muted\">higher is better</span></th>`).join('')+'</tr></thead><tbody>';scopes.forEach(([scope,label])=>['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].forEach(method=>{h+=`<tr><td><strong>${esc(label)}</strong><br><span class=\"muted\">${esc(LABELS[method])}</span></td>`;metrics.forEach(metric=>{const v=methodValues[`${scope}.${metric}`][method],best=Math.max(...Object.values(methodValues[`${scope}.${metric}`]).filter(x=>x!==null)),isBest=v!==null&&Math.abs(v-best)<1e-12;h+=`<td class=\"metric-cell ${isBest?'best':''}\" style=\"background:hsl(166 28% ${96-(v||0)*18}%)\"><strong>${pct(v)}</strong><span class=\"muted\">${fmt(v)}</span>${isBest?'<br><span class=\"best-badge\">BEST</span>':''}</td>`});h+='</tr>'}));document.getElementById('classificationMatrix').innerHTML=h+'</tbody></table>'}",
        "function renderClassification(){const slices=datasetSlices();if(!slices.length){document.getElementById('classificationMatrix').innerHTML=empty('Classification metrics apply to the PCA-8 methods.');return}const scopes=[['imputed_only','Imputed only (primary)'],['judged_plus_imputed','Judged + imputed (observed-inflated)']],metrics=['accuracy','precision','recall','f1'];const renderSlice=s=>{const runs=s.rows.filter(r=>String(r.method_id).startsWith('pca8_')&&r.embedding_metrics);if(!runs.length)return `<article class=\"chart-panel\"><h3>${esc(s.dataset_id)}</h3>${empty('No PCA-8 classification rows for this dataset.')}</article>`;const methodValues={};scopes.forEach(([scope])=>metrics.forEach(metric=>{methodValues[`${scope}.${metric}`]=Object.fromEntries(['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].map(method=>[method,mean(runs.filter(r=>r.method_id===method).map(r=>r.embedding_metrics?.[scope]?.[metric]))]))}));let h='<table><thead><tr><th>Scope / method</th>'+metrics.map(m=>`<th>${m[0].toUpperCase()+m.slice(1)}<br><span class=\"muted\">higher is better</span></th>`).join('')+'</tr></thead><tbody>';scopes.forEach(([scope,label])=>['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].forEach(method=>{h+=`<tr><td><strong>${esc(label)}</strong><br><span class=\"muted\">${esc(LABELS[method])}</span></td>`;metrics.forEach(metric=>{const values=Object.values(methodValues[`${scope}.${metric}`]).filter(x=>x!==null);const v=methodValues[`${scope}.${metric}`][method],best=values.length?Math.max(...values):null,isBest=v!==null&&best!==null&&Math.abs(v-best)<1e-12;h+=`<td class=\"metric-cell ${isBest?'best':''}\" style=\"background:hsl(166 28% ${96-(v||0)*18}%)\"><strong>${pct(v)}</strong><span class=\"muted\">${fmt(v)}</span>${isBest?'<br><span class=\"best-badge\">BEST</span>':''}</td>`});h+='</tr>'}));return `<article class=\"chart-panel\"><h3>${esc(s.dataset_id)}</h3><p class=\"caption\">${esc(sliceCaption(s))} · classification denominator uses imputed-only unsampled rows.</p>${h+'</tbody></table>'}</article>`};document.getElementById('classificationMatrix').innerHTML=slices.map(renderSlice).join('')}",
    )
    html = html.replace(
        "function renderCoverage(){const r=visible(),specs=[['Concept coverage',x=>x.coverage?.concept?.coverage_ratio],['Task coverage',x=>x.coverage?.task],['Domain coverage',x=>x.coverage?.domain],['Agent coverage',x=>x.coverage?.agent]];document.getElementById('coverageCharts').innerHTML=specs.map(([n,p])=>bars(n,grouped(r,p),'percent','higher')).join('');document.getElementById('costCharts').innerHTML=bars('Actual judged tokens',grouped(r,x=>x.actual_token_count),'integer','lower')+bars('Per-method runtime (seconds)',grouped(r,x=>x.latency_seconds?.per_method_total),'decimal','lower')}",
        "function renderCoverage(){const r=visible(),specs=[['Concept coverage',x=>x.coverage?.concept?.coverage_ratio],['Task coverage',x=>x.coverage?.task],['Domain coverage',x=>x.coverage?.domain],['Agent coverage',x=>x.coverage?.agent]];document.getElementById('coverageCharts').innerHTML='<div class=\"context-note\"><strong>Coverage note:</strong> coverage denominators are corpus-specific.</div>' + specs.map(([n,p])=>bars(n, grouped(r, p), 'percent', 'higher')).join('');document.getElementById('costCharts').innerHTML=bars('Actual judged tokens',grouped(r,x=>x.actual_token_count),'integer','lower') + bars('Per-method runtime (seconds)',grouped(r,x=>x.latency_seconds?.per_method_total),'decimal','lower')}",
    )
    html = html.replace(
        "function renderReplay(){const seen=new Set(),r=[];DATA.rows.forEach(x=>{if(!x.replay_id||seen.has(x.replay_id))return;seen.add(x.replay_id);r.push(x)});if(!r.length){document.getElementById('replayChart').innerHTML=empty('No replay metadata.');return}const W=1180,H=330,L=58,R=26,T=22,B=50,x=i=>L+i/Math.max(1,r.length-1)*(W-L-R),y=v=>T+(1-v)*(H-T-B);let s=`<svg viewBox=\"0 0 ${W} ${H}\" role=\"img\" aria-label=\"Replay unique source fractions\"><rect width=\"${W}\" height=\"${H}\" fill=\"#fff\"/>`;[0,.25,.5,.75,1].forEach(v=>s+=`<line x1=\"${L}\" y1=\"${y(v)}\" x2=\"${W-R}\" y2=\"${y(v)}\" stroke=\"#e1e7e4\"/>${axis(L-8,y(v)+4,pct(v),'end')}`);r.forEach((q,i)=>{const v=finite(q.replay_frequency_summary?.unique_source_fraction)||0;s+=`<circle cx=\"${x(i)}\" cy=\"${y(v)}\" r=\"5\" fill=\"${COLORS[METHODS[i%METHODS.length]]}\"><title>${esc(q.dataset_id)} replay ${q.repetition_index}: unique ${pct(v)}, duplicates ${q.replay_frequency_summary?.duplicate_event_count}, max frequency ${q.replay_frequency_summary?.max_frequency}</title></circle>`});document.getElementById('replayChart').innerHTML=s+`${axis((L+W-R)/2,H-8,'30 paired replay populations')}</svg>`}",
        "function renderReplay(){const seen=new Set(),r=[];DATA.rows.forEach(x=>{if(!x.replay_id||seen.has(x.replay_id))return;seen.add(x.replay_id);r.push(x)});if(!r.length){document.getElementById('replayChart').innerHTML=empty('No replay metadata.');return}const W=1180,H=330,L=58,R=26,T=22,B=50,x=i=>L+i/Math.max(1,r.length-1)*(W-L-R),y=v=>T+(1-v)*(H-T-B);let s=`<svg viewBox=\"0 0 ${W} ${H}\" role=\"img\" aria-label=\"Replay unique source fractions\"><rect width=\"${W}\" height=\"${H}\" fill=\"#fff\"/>`;[0,.25,.5,.75,1].forEach(v=>s+=`<line x1=\"${L}\" y1=\"${y(v)}\" x2=\"${W-R}\" y2=\"${y(v)}\" stroke=\"#e1e7e4\"/>${axis(L-8,y(v)+4,pct(v),'end')}`);r.forEach((q,i)=>{const v=finite(q.replay_frequency_summary?.unique_source_fraction)||0;s+=`<circle cx=\"${x(i)}\" cy=\"${y(v)}\" r=\"5\" fill=\"#5f7580\"><title>${esc(q.dataset_id)} replay ${q.repetition_index}: unique ${pct(v)}, duplicates ${q.replay_frequency_summary?.duplicate_event_count}, max frequency ${q.replay_frequency_summary?.max_frequency}</title></circle>`});document.getElementById('replayChart').innerHTML=s+`${axis((L+W-R)/2,H-8,'30 paired replay populations')}</svg><p class=\"caption\">Replay markers are deduplicated across methods. Color is neutral and has no method semantics.</p>`}",
    )
    html = html.replace(
        "function renderProvenance(){const indexes=uniq(DATA.rows.map(r=>r.search_evidence?.index_name).filter(Boolean));let h='<table><thead><tr><th>Dataset</th><th>Population</th><th>Foundry model / deployment</th><th>Embedding inputs / tokens</th><th>Shared latency</th><th>PCA-8 variance</th></tr></thead><tbody>';DATA.profiles.forEach(p=>{const s=p.shared_preprocessing||p.runtime_ledger||{};h+=`<tr><td>${esc(p.dataset_id)}</td><td>${Number(p.population||0).toLocaleString()}</td><td>${esc(s.embedding_model_id||'n/a')}<br><span class=\"muted\">${esc(s.embedding_deployment_id||'n/a')}</span></td><td>${Number(s.embedding_inputs||0).toLocaleString()} / ${Number(s.embedding_input_tokens||0).toLocaleString()}</td><td>Embedding ${fmt(s.embedding_latency_seconds)}s<br>PCA ${fmt(s.pca_fit_latency_seconds??p.pca?.latency_seconds)}s</td><td>${pct(p.pca?.explained_variance_total)}</td></tr>`});document.getElementById('provenanceTable').innerHTML=h+`</tbody></table><p class=\"caption\"><strong>Azure AI Search:</strong> ${indexes.map(esc).join(', ')}. Retrieval evidence is shared source-space validation for the PCA-8 methods.</p>`}",
        "function renderProvenance(){const indexes=uniq(DATA.rows.map(r=>r.search_evidence?.index_name).filter(Boolean));let h='<table><thead><tr><th>Dataset</th><th>Description</th><th>Population</th><th>Agent / concept / domain / task counts</th><th>Base pass rate (source census)</th><th>Foundry model / deployment</th><th>Embedding inputs / tokens</th><th>Shared latency</th><th>PCA-8 variance</th></tr></thead><tbody>';DATA.profiles.forEach(p=>{const s=p.shared_preprocessing||p.runtime_ledger||{},counts=(p.coverage_population_counts||p.population_counts||{}),agent=pickCount(counts,['agent','agents','agent_count']),concept=pickCount(counts,['concept','concepts','concept_count']),domain=pickCount(counts,['domain','domains','domain_count']),task=pickCount(counts,['task','tasks','task_count']),baseRate=datasetBasePassRate(p.dataset_id);h+=`<tr><td>${esc(p.dataset_id)}</td><td>${esc(DATASET_DESCRIPTIONS[p.dataset_id]||'n/a')}</td><td>${Number(p.population||0).toLocaleString()}</td><td>A ${agent===null?'n/a':Number(agent).toLocaleString()} / C ${concept===null?'n/a':Number(concept).toLocaleString()} / D ${domain===null?'n/a':Number(domain).toLocaleString()} / T ${task===null?'n/a':Number(task).toLocaleString()}</td><td>${pct(baseRate)}<br><span class=\"muted\">drives fixed-source MAE denominator</span></td><td>${esc(s.embedding_model_id||'n/a')}<br><span class=\"muted\">${esc(s.embedding_deployment_id||'n/a')}</span></td><td>${Number(s.embedding_inputs||0).toLocaleString()} / ${Number(s.embedding_input_tokens||0).toLocaleString()}</td><td>Embedding ${fmt(s.embedding_latency_seconds)}s<br>PCA ${fmt(s.pca_fit_latency_seconds??p.pca?.latency_seconds)}s</td><td>${pct(p.pca?.explained_variance_total)}</td></tr>`});document.getElementById('provenanceTable').innerHTML=h+`</tbody></table><p class=\"caption\"><strong>Azure AI Search:</strong> ${indexes.map(esc).join(', ')}. Coverage denominators come from each corpus profile row above; base pass rate is the source-census denominator used by fixed-source MAE. Retrieval evidence is shared source-space validation for PCA-8 methods.</p>`}",
    )
    html = html.replace(
        "function renderDetail(){const r=visible(),shown=r.slice(0,detailLimit);document.getElementById('rowCount').textContent=`Showing ${shown.length.toLocaleString()} of ${r.length.toLocaleString()} matching rows`;document.getElementById('showMore').style.display=detailLimit<r.length?'inline-block':'none';document.getElementById('detailRows').innerHTML=shown.map(x=>`<tr><td>${esc(x.dataset_id)}</td><td>${x.repetition_index??'n/a'}</td><td>${x.replay_seed??x.seed??'n/a'}</td><td>${x.budget_pct}% / ${x.cap??'n/a'}</td><td>${esc(LABELS[x.method_id]||x.method_id)}</td><td>${esc(x.estimator_type)}</td><td>${fmt(x.aggregate_pass_rate_mae)}</td><td>${fmt(x.replay_aggregate_pass_rate_mae)}</td><td>${pct(x.coverage?.concept?.coverage_ratio)}</td><td>${pct(x.embedding_metrics?.imputed_only?.f1)}</td><td>${Number(x.actual_token_count||0).toLocaleString()}</td></tr>`).join('')}",
        "function renderDetail(){const r=visible(),shown=r.slice(0,detailLimit);document.getElementById('rowCount').textContent=`Showing ${shown.length.toLocaleString()} of ${r.length.toLocaleString()} matching rows`;document.getElementById('showMore').style.display=detailLimit<r.length?'inline-block':'none';document.getElementById('detailRows').innerHTML=shown.map(x=>`<tr><td>${esc(x.dataset_id)}</td><td>${x.repetition_index??'n/a'}</td><td>${x.replay_seed??x.seed??'n/a'}</td><td>${x.budget_pct}% / ${x.cap??'n/a'}</td><td title=\"report method ID: ${esc(x.method_id)}; source method ID: ${esc(x.method_id==='agent_balanced_weighted_sampling'?'arm5_hajek_weighted':x.method_id)}\">${esc(LABELS[x.method_id]||x.method_id)}</td><td title=\"raw estimator ID: ${esc(x.estimator_type)}\">${esc(estimatorLabel(x.estimator_type))}<br><span class=\"muted\">${esc(x.estimator_type)}</span></td><td>${fmt(x.aggregate_pass_rate_mae)}</td><td>${fmt(x.replay_aggregate_pass_rate_mae)}</td><td>${pct(x.coverage?.concept?.coverage_ratio)}</td><td>${pct(x.embedding_metrics?.imputed_only?.f1)}</td><td>${Number(x.actual_token_count||0).toLocaleString()}</td></tr>`).join('')}",
    )
    html = html.replace(
        "bars('Per-method runtime (seconds)',grouped(r,x=>x.latency_seconds?.per_method_total),'decimal','lower')",
        "bars('Incremental replay runtime (seconds)',grouped(r,x=>(Number(x.latency_seconds?.selection||0)+Number(x.latency_seconds?.idw||0))),'decimal','lower')+bars('Shared search evidence reference (seconds)',grouped(r,x=>x.latency_seconds?.search_evidence),'decimal','lower')",
    )
    html = html.replace(
        "<li>Search evidence applies only to the PCA-8 methods.</li>",
        "<li>Search evidence applies only to PCA-8 methods and is shared reference validation; do not interpret duplicated reference probe latency as per-replay runtime.</li>",
    )
    html = html.replace(
        "html{scroll-behavior:smooth;overflow-x:hidden}",
        "html{scroll-behavior:smooth}",
    )
    html = html.replace(
        "overflow-x:hidden",
        "overflow-x:visible",
    )
    html = html.replace(
        "grid-template-columns:repeat(5,minmax(0,1fr));",
        "grid-template-columns:repeat(auto-fit,minmax(250px,1fr));",
    )
    html = html.replace(
        "<div id=\"activeRanking\" class=\"method-table\"></div>",
        "<div class=\"table-wrap\"><div id=\"activeRanking\" class=\"method-table\"></div></div>",
    )
    html = html.replace(
        "<div id=\"classificationMatrix\" class=\"matrix\"></div>",
        "<div class=\"table-wrap\"><div id=\"classificationMatrix\" class=\"matrix\"></div></div>",
    )
    html = html.replace(
        "<div id=\"provenanceTable\"></div>",
        "<div class=\"table-wrap\"><div id=\"provenanceTable\"></div></div>",
    )
    html = html.replace(
        "<strong>What this table represents:</strong> this is the audit trail for corpus size, embedding inputs, cloud latency, and retained PCA variance.",
        "<strong>What this section represents:</strong> this is the audit trail for corpus size, embedding inputs, cloud latency, and retained PCA variance.",
    )
    html = html.replace(".controls-wrap{position:sticky;top:0;z-index:20;", ".controls-wrap{position:relative;top:auto;z-index:1;")
    html = html.replace(
        "renderHero();renderPca();renderReplay();renderProvenance();renderAll();",
        "function _num(v){const n=Number(v);return Number.isFinite(n)?n:null}"
        "function _countFromProfile(p,primary,fallback){const direct=_num(p?.[primary]);if(direct!==null)return direct;const nested=p?.coverage_population_counts||p?.population_counts||{};for(const k of fallback){const x=_num(nested?.[k]);if(x!==null)return x}return null}"
        "function _datasetSlices(){const rows=visible();const ids=ctl.dataset.value==='all'?uniq(rows.map(r=>String(r.dataset_id))):[String(ctl.dataset.value)];return ids.map(id=>({dataset_id:id,rows:rows.filter(r=>String(r.dataset_id)===id)})).filter(s=>s.rows.length)}"
        "function _datasetCaption(datasetId,rows){const p=DATA.profiles.find(x=>String(x.dataset_id)===String(datasetId))||{};const pop=_num(p.population);const cap=Number.isFinite(pop)?`Population ${pop.toLocaleString()}`:'Population n/a';const baseDirect=_num(p.label_rate);const base=baseDirect!==null?baseDirect:mean(rows.map(r=>r.source_corpus_census_pass_rate));const baseText=base===null?'n/a':pct(base);return `${cap} · Base pass rate ${baseText}`}"
        "function renderClassification(){const host=document.getElementById('classificationMatrix');const slices=_datasetSlices();if(!slices.length){host.innerHTML=empty('Classification metrics apply to the PCA-8 methods.');return;}const methods=['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'],scopes=[['imputed_only','Imputed only'],['judged_plus_imputed','Judged + imputed']],metrics=[['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1']];const blocks=slices.map(slice=>{const runs=slice.rows.filter(r=>String(r.method_id).startsWith('pca8_')&&r.embedding_metrics);if(!runs.length){return `<article class=\"dataset-block\"><h3>${esc(slice.dataset_id)}</h3><p class=\"caption\">${esc(_datasetCaption(slice.dataset_id,slice.rows))} · classification denominator uses imputed-only unsampled rows.</p>${empty('No PCA-8 classification rows for this dataset.')}</article>`;}const cards=[];for(const method of methods){for(const [scope,scopeLabel] of scopes){const stats=Object.fromEntries(metrics.map(([k])=>[k,mean(runs.filter(r=>r.method_id===method).map(r=>r.embedding_metrics?.[scope]?.[k]))]));cards.push({method,scope,scopeLabel,stats});}}const cardHtml=cards.map(card=>{const grid=metrics.map(([key,label])=>{const value=card.stats[key];const peers=cards.filter(c=>c.scope===card.scope&&c.method!==card.method).map(c=>c.stats[key]).filter(v=>v!==null);const best=peers.length?Math.max(...peers):null;const isBest=value!==null&&(best===null||value>=best-1e-12);return `<div class=\"metric-item ${isBest?'best':''}\"><strong>${esc(label)}</strong><span class=\"metric-value\">${pct(value)}</span><span class=\"muted\">higher is better${isBest?' · BEST':''}</span></div>`;}).join('');return `<article class=\"classification-card\"><h4>${esc(LABELS[card.method])}</h4><p class=\"caption\">${esc(card.scopeLabel)}</p><div class=\"metric-grid\">${grid}</div></article>`;}).join('');return `<article class=\"dataset-block\"><h3>${esc(slice.dataset_id)}</h3><p class=\"caption\">${esc(_datasetCaption(slice.dataset_id,slice.rows))} · classification denominator uses imputed-only unsampled rows.</p><div class=\"classification-grid\">${cardHtml}</div></article>`;}).join('');host.innerHTML=blocks;}"
        "function renderCoverage(){const coverageHost=document.getElementById('coverageCharts');const costHost=document.getElementById('costCharts');const slices=_datasetSlices();if(!slices.length){coverageHost.innerHTML=empty('No coverage rows for this slice.');costHost.innerHTML=empty('No cost rows for this slice.');return;}coverageHost.innerHTML=slices.map(slice=>{const specs=[['Concept coverage',x=>x.coverage?.concept?.coverage_ratio],['Task coverage',x=>x.coverage?.task],['Domain coverage',x=>x.coverage?.domain],['Agent coverage',x=>x.coverage?.agent]];const charts=specs.map(([n,p])=>bars(n,grouped(slice.rows,p),'percent','higher')).join('');return `<article class=\"dataset-block\"><h3>${esc(slice.dataset_id)}</h3><p class=\"caption\">${esc(_datasetCaption(slice.dataset_id,slice.rows))} · coverage denominators are corpus-specific.</p><div class=\"small-multiples\">${charts}</div></article>`;}).join('');costHost.innerHTML=slices.map(slice=>{const tokenChart=bars('Actual judged tokens',grouped(slice.rows,x=>x.actual_token_count),'integer','lower');const incrementalChart=bars('Incremental replay runtime (seconds)',grouped(slice.rows,x=>Number(x.latency_seconds?.selection||0)+Number(x.latency_seconds?.idw||0)),'decimal','lower');const sharedChart=bars('Shared search evidence reference (seconds)',grouped(slice.rows,x=>x.latency_seconds?.search_evidence),'decimal','lower');const costs=tokenChart+incrementalChart+sharedChart;return `<article class=\"dataset-block\"><h3>${esc(slice.dataset_id)}</h3><p class=\"caption\">${esc(_datasetCaption(slice.dataset_id,slice.rows))} · cost panels remain unit-specific.</p><div class=\"chart-grid\">${costs}</div></article>`;}).join('')}"
        "function renderProvenance(){const host=document.getElementById('provenanceTable');const indexes=uniq(DATA.rows.map(r=>r.search_evidence?.index_name).filter(Boolean));const cards=DATA.profiles.map(p=>{const datasetId=String(p.dataset_id||'n/a');const profileRows=DATA.rows.filter(r=>String(r.dataset_id)===datasetId);const baseDirect=_num(p.label_rate);const base=baseDirect!==null?baseDirect:mean(profileRows.map(r=>r.source_corpus_census_pass_rate));const pop=_num(p.population);const s=p.shared_preprocessing||p.runtime_ledger||{};const agent=_countFromProfile(p,'agent_count',['agent_count','agent','agents']);const concept=_countFromProfile(p,'concept_count',['concept_count','concept','concepts']);const domain=_countFromProfile(p,'domain_count',['domain_count','domain','domains']);const task=_countFromProfile(p,'task_count',['task_count','task','tasks']);const rows=[['Description',DATASET_DESCRIPTIONS[datasetId]||'n/a'],['Population',pop===null?'n/a':pop.toLocaleString()],['Agents / Concepts / Domains / Tasks',`${agent===null?'n/a':agent.toLocaleString()} / ${concept===null?'n/a':concept.toLocaleString()} / ${domain===null?'n/a':domain.toLocaleString()} / ${task===null?'n/a':task.toLocaleString()}`],['Base pass rate (source census)',base===null?'n/a':pct(base)],['Foundry model / deployment',`${s.embedding_model_id||'n/a'} / ${s.embedding_deployment_id||'n/a'}`],['Embedding inputs / tokens',`${Number(s.embedding_inputs||0).toLocaleString()} / ${Number(s.embedding_input_tokens||0).toLocaleString()}`],['Embedding latency (s)',fmt(s.embedding_latency_seconds)],['PCA latency (s)',fmt(s.pca_fit_latency_seconds??p.pca?.latency_seconds)],['PCA-8 variance',pct(p.pca?.explained_variance_total)]];const defs=rows.map(([k,v])=>`<div class=\"kv-row\"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');return `<article class=\"provenance-card\"><h3>${esc(datasetId)}</h3><dl>${defs}</dl></article>`;}).join('');host.innerHTML=`<div class=\"provenance-grid\">${cards}</div><p class=\"caption\"><strong>Azure AI Search:</strong> ${indexes.map(esc).join(', ')}. Retrieval evidence is shared source-space validation for the PCA-8 methods.</p>`}"
        "renderHero();renderPca();renderReplay();renderProvenance();renderAll();",
    )
    html = html.replace(
        "</style>",
        ".shell,.masthead,.section,.section-head,.chart-grid,.chart-panel,.chart,.table-wrap,.method-table,.matrix,#activeRanking,#classificationMatrix,#provenanceTable,.arms,.arm,.winner-callout{max-width:100%;min-width:0}.table-wrap,.method-table,.matrix,#activeRanking,#classificationMatrix,#provenanceTable,.chart{overflow-x:auto}.arm,.arm h3,.arm p,.method-id,.winner-callout .rank,.method-table td,.method-table th,.matrix td,.matrix th,.table-wrap td,.table-wrap th{overflow-wrap:anywhere;word-break:break-word}.method-id{font-size:.78rem;line-height:1.3;color:var(--muted);margin:-2px 0 8px}.dataset-block{border:1px solid var(--line);border-radius:var(--radius);padding:14px;margin-bottom:14px;background:#fbfcfb}.classification-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.classification-card{border:1px solid var(--line);border-radius:var(--radius);padding:10px;background:#fff}.classification-card h4{margin:0 0 4px;font-size:.95rem}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric-item{border:1px solid var(--line);border-radius:6px;padding:7px;background:#f9fbfa}.metric-item.best{outline:2px solid #16726f;outline-offset:-2px;background:#e8f3ee}.metric-item strong,.metric-item .metric-value,.metric-item .muted{display:block}.metric-item .metric-value{font-weight:700}.provenance-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.provenance-card{border:1px solid var(--line);border-radius:var(--radius);padding:12px;background:#fff}.provenance-card dl{margin:0}.provenance-card .kv-row{display:grid;grid-template-columns:minmax(140px,.8fr) minmax(0,1.2fr);gap:8px;padding:4px 0;border-bottom:1px solid #edf1ef}.provenance-card .kv-row:last-child{border-bottom:0}.provenance-card dt{font-weight:700;color:#314249}.provenance-card dd{margin:0;color:#314249;overflow-wrap:anywhere}#parallel-study-graph,.arms{overflow-x:clip}.controls{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}@media(max-width:920px){.classification-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.provenance-grid{grid-template-columns:1fr}}@media(max-width:600px){.controls{grid-template-columns:1fr}.winner-callout .margin,.winner-callout .rank{overflow-wrap:anywhere}.metric-grid{grid-template-columns:1fr}}@media print{html,body{background:#fff!important;color:#000!important}.controls-wrap,.controls,.table-actions button{display:none!important}.section,.chart-panel,.method-table,.matrix,.table-wrap{break-inside:avoid-page;page-break-inside:avoid}.section{padding:18px 14px!important}.masthead{background:#fff!important;color:#000!important;min-height:auto!important;padding:14px!important;border-bottom:2px solid #222}.kpi-strip,.chip{border-color:#aaa!important}.chart{overflow:visible!important;min-height:auto!important}svg{max-width:100%!important;height:auto!important}.section#detail{page-break-before:always}}@page{size:letter portrait;margin:0.5in}</style>",
    )
    html = html.replace(
        "</style>",
        ".chart-panel svg{display:block;width:100%;height:auto}</style>",
    )
    return html


def build_v7_final_report(payload: Mapping[str, Any], *, artifacts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bundle = _prepare_bundle(payload)
    rows = bundle["rows"]
    profiles = bundle["profiles"]
    aggregate_rows = bundle["aggregate_rows"]
    paired_rows = bundle["paired_rows"]
    paired_cells = bundle["paired_cells"]
    summary = bundle["summary"]
    executive = bundle["executive"]
    search_split = _search_latency_split(rows)

    source_hashes: dict[str, str] = {}
    embedding_cache_rows: list[dict[str, Any]] = []
    for profile in profiles:
        runtime = profile.get("runtime_ledger") if isinstance(profile.get("runtime_ledger"), Mapping) else {}
        cache_provenance = runtime.get("cache_provenance") if isinstance(runtime.get("cache_provenance"), Mapping) else {}
        current_source_hashes = cache_provenance.get("source_hashes") if isinstance(cache_provenance.get("source_hashes"), Mapping) else {}
        for key, value in current_source_hashes.items():
            source_hashes[str(key)] = str(value)
        embedding_cache_rows.append(
            {
                "dataset_id": profile.get("dataset_id"),
                "cache_hit": bool(runtime.get("cache_hit")),
                "cache_rows": runtime.get("cache_rows"),
                "embedding_model_id": runtime.get("embedding_model_id") or cache_provenance.get("embedding_model_id"),
                "embedding_deployment_id": runtime.get("embedding_deployment_id") or cache_provenance.get("embedding_deployment_id"),
                "source_hashes": current_source_hashes,
            }
        )

    methods = [
        {
            "method_id": method_id,
            "source_method_id": _source_method_id(method_id),
            "report_method_id": _public_method_id(method_id),
            "display_name": METHOD_LABELS.get(method_id, method_id),
            "display_with_id": _method_display(method_id),
            "notes": METHOD_NOTES.get(method_id, ""),
        }
        for method_id in METHOD_ORDER
    ]
    findings = [
        {
            "finding_id": "best_fixed_source_mae",
            "value": {"method_id": executive.get("bestMaeMethod"), "mean": executive.get("bestMae")},
            "qualification": "Unweighted mean over dataset-budget aggregate cell means.",
            "support_metric_id": "aggregate_pass_rate_mae",
        },
        {
            "finding_id": "best_concept_coverage",
            "value": {"method_id": executive.get("bestConceptMethod"), "mean": executive.get("bestConcept")},
            "qualification": "Unweighted mean over dataset-budget aggregate cell means.",
            "support_metric_id": "coverage_concept_ratio",
        },
        {
            "finding_id": "cosine_paired_win_rates",
            "value": {
                "aggregate_pass_rate_mae": executive.get("cosineMaeWinRate"),
                "imputed_only_accuracy": executive.get("cosineAccuracyWinRate"),
                "imputed_only_f1": executive.get("cosineF1WinRate"),
            },
            "qualification": "Unweighted over paired replay comparisons.",
            "support_metric_id": "paired_comparison",
        },
    ]

    return {
        "schema_version": "sampling-v7-final-report-v1",
        "report_metadata": {
            "report_family": "sampling-v7",
            "generated_from_summary_timestamp": summary.get("generated_at"),
            "is_live": bool(summary.get("is_live")),
            "live_disclosure": "Live refers to run/search evidence execution; embedding ledgers are cached provenance for this run.",
        },
        "run_configuration": {
            "dataset_ids": sorted({str(row.get("dataset_id") or "") for row in rows}),
            "method_ids": sorted({str(_public_method_id(row.get("method_id"))) for row in rows}),
            "budgets_pct": sorted({int(row.get("budget_pct")) for row in rows if row.get("budget_pct") is not None}),
            "repetition_indices": sorted({int(row.get("repetition_index")) for row in rows if row.get("repetition_index") is not None}),
            "observed_run_rows": len(rows),
            "expected_shape_formula": "datasets x methods x budgets x repetitions",
            "expected_shape_value": _expected_run_shape_value(rows),
            "paired_replay": bool(summary.get("paired_replay")),
            "randomized_frequency": bool(summary.get("randomized_frequency")),
            "randomized_order": bool(summary.get("randomized_order")),
        },
        "canonical_methods": methods,
        "dataset_profiles": profiles,
        "embedding_cache_provenance": {
            "rows": embedding_cache_rows,
            "all_cache_hit": all(row.get("cache_hit") for row in embedding_cache_rows),
            "source_hashes": source_hashes,
        },
        "metric_catalog": list(METRIC_CATALOG),
        "key_findings": findings,
        "aggregate_metrics": {
            "rows": aggregate_rows,
            "ci95_disclosure": "CI95 values summarize empirical replay uncertainty over perturbations of the same labeled corpus; not a population-generalization guarantee.",
        },
        "paired_comparison_evidence": {
            "rows": paired_rows,
            "cells_reference": {
                "count": len(paired_cells),
                "note": "Per-replay paired deltas are retained in aggregate paired cells.",
            },
        },
        "cost_accounting": {
            "incremental_replay_runtime_definition": "selection + idw latency components",
            "shared_search_reference_definition": "search_evidence latency component, reused one-time probe across replay rows",
            "shared_search_reference_diagnostics": search_split,
            "guard": "Shared duplicated search latency must not be presented as per-replay runtime.",
            "legacy_raw_per_method_total_note": "Legacy raw latency_seconds.per_method_total includes shared search probe evidence and must not be summed or treated as per-replay runtime; the corrected incremental metric remains authoritative.",
        },
        "aggregation_disclosures": {
            "best_method_aggregation": executive.get("aggregation_note"),
            "win_rate_aggregation": executive.get("win_rate_note"),
        },
        "limitations": [
            "Budget percentages imply different absolute judged-session caps by corpus population.",
            "Fixed-source MAE and replay-relative MAE use different denominators and should not be conflated.",
            "Judged-plus-imputed classification metrics are observed-inflated by construction.",
            "Search evidence applies only to PCA-8 methods and does not alter deterministic local selection.",
            "CI95 reflects replay perturbation uncertainty on this labeled corpus, not population-level generalization.",
        ],
        "provenance": {
            "artifacts": dict(artifacts or {}),
        },
    }


def build_v7_print_report_html(payload: Mapping[str, Any], final_report: Mapping[str, Any]) -> str:
    bundle = _prepare_bundle(payload)
    executive = bundle["executive"]
    profiles = bundle["profiles"]
    methods = final_report.get("canonical_methods") if isinstance(final_report.get("canonical_methods"), Sequence) else []
    findings = final_report.get("key_findings") if isinstance(final_report.get("key_findings"), Sequence) else []
    limitations = final_report.get("limitations") if isinstance(final_report.get("limitations"), Sequence) else []

    method_rows = "".join(
        "<tr>"
        f"<td>{str(row.get('source_method_id') or row.get('method_id') or 'unknown')}</td>"
        f"<td>{str(row.get('report_method_id') or row.get('method_id') or 'unknown')}</td>"
        f"<td>{_method_display(str(row.get('report_method_id') or row.get('method_id') or 'unknown'))}</td>"
        f"<td>{str(row.get('notes') or '')}</td>"
        "</tr>"
        for row in methods
        if isinstance(row, Mapping)
    )
    profile_rows = "".join(
        "<tr>"
        f"<td>{str(profile.get('dataset_id') or 'n/a')}</td>"
        f"<td>{int(profile.get('population') or 0):,}</td>"
        f"<td>{str(((profile.get('runtime_ledger') or {}).get('embedding_model_id') or 'n/a'))}</td>"
        f"<td>{'true' if bool((profile.get('runtime_ledger') or {}).get('cache_hit')) else 'false'}</td>"
        f"<td>{float((profile.get('pca') or {}).get('explained_variance_total') or 0.0):.4f}</td>"
        "</tr>"
        for profile in profiles
    )
    finding_rows = "".join(
        f"<li>{_findings_summary_text(item, methods=methods)}</li>"
        for item in findings
        if isinstance(item, Mapping)
    )
    limitation_rows = "".join(f"<li>{str(item)}</li>" for item in limitations)

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        "<title>Sampling V7 Print Report</title>"
        "<style>"
        ":root{--ink:#1b2528;--muted:#4c5d63;--line:#cfd8d6;--accent:#1d6663;}"
        "*{box-sizing:border-box}body{margin:0;padding:20px;font-family:Aptos,'Segoe UI',Tahoma,sans-serif;color:var(--ink);line-height:1.45;background:#fff}"
        "h1,h2,h3{font-family:Georgia,Cambria,serif;margin:0 0 8px}h1{font-size:32px}h2{font-size:22px;margin-top:20px}"
        "p{margin:0 0 10px}.lede{color:var(--muted);max-width:980px}.card{border:1px solid var(--line);padding:14px 16px;margin-top:12px;border-radius:8px}"
        "table{width:100%;border-collapse:collapse;margin-top:10px}th,td{border:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}"
        "th{background:#f2f6f5}ul{margin:8px 0 0 18px}"
        ".small{font-size:13px;color:var(--muted)}.accent{color:var(--accent);font-weight:700}"
        "@media print{body{padding:0.35in}.card,table,ul{break-inside:avoid-page;page-break-inside:avoid}h2{page-break-after:avoid}.new-page{page-break-before:always}}"
        "@page{size:letter portrait;margin:0.5in}"
        "</style></head><body>"
        "<h1>Sampling V7 Final Report (Print)</h1>"
        "<p class=\"lede\">Executive best-method values are unweighted means over dataset-budget cell means. Cosine win rates are unweighted across paired replay comparisons. CI95 denotes empirical replay uncertainty over perturbations of the same labeled corpus, not population-generalization.</p>"
        "<div class=\"card\"><h2>Executive Snapshot</h2>"
        f"<p><span class=\"accent\">Best fixed-source MAE:</span> {_method_display(str(executive.get('bestMaeMethod') or 'n/a'))} at {float(executive.get('bestMae') or 0.0):.4f}</p>"
        f"<p><span class=\"accent\">Best concept coverage:</span> {_method_display(str(executive.get('bestConceptMethod') or 'n/a'))} at {float(executive.get('bestConcept') or 0.0):.4f}</p>"
        f"<p><span class=\"accent\">Cosine paired win rates:</span> MAE={float(executive.get('cosineMaeWinRate') or 0.0):.4f}, Accuracy={float(executive.get('cosineAccuracyWinRate') or 0.0):.4f}, F1={float(executive.get('cosineF1WinRate') or 0.0):.4f}</p>"
        "</div>"
        "<div class=\"card\"><h2>Methods (Source ID + Display / Report Alias)</h2><table><thead><tr><th>Source method ID</th><th>Display / report alias</th><th>Method name</th><th>Semantics</th></tr></thead><tbody>"
        f"{method_rows}</tbody></table><p class=\"small\">The source method ID preserves the underlying experiment identifier even when the report exposes a public-facing alias.</p></div>"
        "<div class=\"card\"><h2>Dataset and Embedding Provenance</h2><table><thead><tr><th>Dataset</th><th>Population</th><th>Embedding model</th><th>Cache hit</th><th>PCA-8 variance</th></tr></thead><tbody>"
        f"{profile_rows}</tbody></table><p class=\"small\">All embedding ledgers are cache hits in this evidence run. Live refers to run/search evidence activity, not necessarily fresh embedding calls.</p></div>"
        "<div class=\"card new-page\"><h2>Findings and Qualifications</h2><ul>"
        f"{finding_rows}</ul></div>"
        "<div class=\"card\"><h2>Limitations</h2><ul>"
        f"{limitation_rows}</ul></div>"
        "</body></html>"
    )


def build_v7_report_html(payload: Mapping[str, Any]) -> str:
    bundle = _prepare_bundle(payload)
    report_data = bundle["report_data"]
    executive = bundle["executive"]

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sampling V7 Applied Science Report</title>
  <style>
    :root{--paper:#f4f6f3;--surface:#fff;--ink:#162226;--muted:#536168;--line:#d5ded9;--teal:#16726f;--coral:#c95d3f;--blue:#396d9d;--gold:#ad7f21;--mauve:#766985;--green-soft:#e8f3ee;--amber-soft:#f7f0df;--radius:8px}
    *{box-sizing:border-box}html{scroll-behavior:smooth;overflow-x:hidden}body{margin:0;background:var(--paper);color:var(--ink);font-family:Aptos,"Segoe UI",Tahoma,sans-serif;line-height:1.5;letter-spacing:0;min-width:0;overflow-x:hidden}h1,h2,h3{font-family:Georgia,Cambria,serif;letter-spacing:0}h1{max-width:900px;margin:0;font-size:clamp(2rem,5vw,4.35rem);line-height:1.03;overflow-wrap:anywhere}h2{margin:0 0 8px;font-size:clamp(1.35rem,2.2vw,2rem)}h3{margin:0 0 6px;font-size:1.02rem}p{margin:0 0 10px;overflow-wrap:anywhere}.shell{max-width:1420px;margin:auto;width:100%;min-width:0}.masthead{min-height:380px;padding:42px clamp(20px,5vw,70px) 30px;background:#153c3d;color:#f8fbfa;display:flex;flex-direction:column;justify-content:space-between;min-width:0}.masthead p{max-width:850px;color:#d3e2df;font-size:1.04rem}.eyebrow{color:#9fd0c8;font-weight:700;text-transform:uppercase;font-size:.78rem}.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.chip{border:1px solid #5f8582;border-radius:999px;padding:6px 10px;color:#eff8f5;font-size:.82rem}.kpi-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#406462;border:1px solid #406462;margin-top:28px}.kpi{min-height:88px;padding:14px;background:#1b494a}.kpi strong{display:block;font:700 1.42rem Georgia,Cambria,serif;color:#fff}.kpi span{color:#c9dcda;font-size:.82rem}.section{padding:34px clamp(18px,4vw,56px);border-bottom:1px solid var(--line);background:var(--surface)}.section.alt{background:#eef2ef}.section-head{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(300px,1.3fr);gap:24px;margin-bottom:20px}.section-head p,.muted,.caption{color:var(--muted)}.caption{font-size:.86rem;margin-top:8px}.readout{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.insight{border-left:4px solid var(--teal);background:#f8faf8;padding:14px 16px;min-height:130px}.insight:nth-child(2){border-color:var(--coral)}.insight:nth-child(3){border-color:var(--gold)}.insight strong{font:700 1.16rem Georgia,Cambria,serif;display:block;margin:4px 0}.controls-wrap{position:sticky;top:0;z-index:20;padding:10px clamp(18px,4vw,56px);background:rgba(244,246,243,.97);border-bottom:1px solid var(--line)}.controls{max-width:1308px;margin:auto;display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:10px}label{color:var(--muted);font-size:.78rem;font-weight:700}select{display:block;width:100%;margin-top:3px;padding:8px;border:1px solid #bfc9c5;border-radius:6px;background:#fff;color:var(--ink)}.pipeline{display:grid;gap:16px}.stage{border:1px solid var(--line);background:#fbfcfa;padding:14px;border-radius:var(--radius);text-align:center}.stage-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.connector{width:2px;height:24px;margin:auto;background:#869692;position:relative}.connector:after{content:"";position:absolute;bottom:-1px;left:-4px;border:5px solid transparent;border-top-color:#869692}.replay-stage{background:var(--green-soft);border-color:#a9c8bf}.budget-stage{max-width:600px;margin:auto;background:var(--amber-soft);border-color:#dac797}.arms{position:relative;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;padding-top:26px}.arms:before{content:"";position:absolute;top:12px;left:10%;right:10%;border-top:2px solid #869692}.arm{position:relative;min-height:116px;padding:13px 10px;border:1px solid var(--line);border-top:4px solid var(--method);border-radius:var(--radius);background:#fff}.arm:before{content:"";position:absolute;width:2px;height:14px;background:#869692;top:-18px;left:50%}.arm p{font-size:.8rem;color:var(--muted)}.evaluation{border:2px solid #9ab5ae;background:#f4faf7}.method-table{margin-top:20px;overflow-x:auto}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.chart-panel{min-width:0;border-top:3px solid #9db8b2;padding-top:10px}.chart{width:100%;min-height:280px}.chart svg{width:100%;height:auto;display:block;overflow:visible}.legend{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:.78rem;color:var(--muted);margin:8px 0}.legend span:before{content:"";display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:50%;background:var(--swatch)}.matrix,.method-table{overflow-x:auto}.metric-cell{min-width:86px;text-align:center;font-variant-numeric:tabular-nums}.metric-cell strong{display:block;font-size:1rem}.small-multiples{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.context-note{border-left:3px solid var(--gold);padding:9px 12px;background:#fbf7ed;color:#5d5239;font-size:.86rem;margin:12px 0}.table-wrap{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:var(--radius)}.table-actions{display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}table{width:100%;border-collapse:collapse;font-size:.84rem}th,td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:#edf2ef;z-index:1}button{border:1px solid #8da5a0;background:#fff;color:var(--ink);border-radius:6px;padding:7px 11px;cursor:pointer}button:hover{background:#edf4f1}.empty{color:var(--muted);padding:28px;text-align:center;border:1px dashed #b9c5c1}.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.analysis-grid ul{margin:0;padding-left:19px}
    .winner-callout{grid-column:1/-1;display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:13px 15px;border:2px solid #76a79d;background:#e8f3ee;border-radius:var(--radius)}
    .winner-callout .rank{font:700 1.22rem Georgia,Cambria,serif;color:#145d58}.winner-callout .direction{font-size:.76rem;color:#536168;text-transform:uppercase;font-weight:800}.winner-callout .margin{font-weight:700;color:#33474b}.best-badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#d8eee6;color:#145d58;font-size:.68rem;font-weight:800}.rank-table td:first-child{width:44px;font-weight:800}.rank-table tr:first-child{background:#e8f3ee}.rank-table .metric-value{font-variant-numeric:tabular-nums;font-weight:700}.metric-cell.best{outline:3px solid #16726f;outline-offset:-3px;background:#dff1ea!important}.series-end{font-weight:800;font-size:10px}.overlap-note{margin-top:8px;padding:7px 9px;background:#f7f0df;border-left:3px solid #ad7f21;color:#5f5338;font-size:.8rem}
    @media(max-width:920px){.kpi-strip,.readout,.small-multiples{grid-template-columns:repeat(2,minmax(0,1fr))}.section-head,.chart-grid,.analysis-grid{grid-template-columns:1fr}.arms{grid-template-columns:1fr 1fr;padding-top:0}.arms:before,.arm:before{display:none}.winner-callout{grid-template-columns:1fr}}
    @media(max-width:600px){html,body{overflow-x:hidden}.masthead{min-height:440px;padding:28px 18px 22px}.controls{grid-template-columns:1fr 1fr}.kpi-strip,.readout,.small-multiples,.stage-row,.arms{grid-template-columns:1fr}.section{padding:28px 16px}.section-head,.chart-grid,.analysis-grid{grid-template-columns:1fr}.chart{min-height:250px;overflow-x:auto}.table-wrap table{min-width:540px}.winner-callout{grid-template-columns:1fr}.controls label{font-size:.7rem}.chart-panel,.section,.masthead,.shell{min-width:0}}
  </style>
</head>
<body><main class="shell">
  <header class="masthead"><div><div class="eyebrow">Agent 365 evaluation sampling / V7</div><h1>PCA-8 distance, coverage, and replay robustness</h1><p>Five sampling strategies compared across three corpora and five budget levels using ten paired, frequency-randomized bootstrap replays. Full-session embeddings come from Microsoft Foundry; reduced-space retrieval is validated with Azure AI Search.</p><div class="chips" id="statusChips"></div></div><div class="kpi-strip" id="heroKpis"></div></header>
    <div class="controls-wrap" aria-label="Report filters"><div class="controls"><label>Dataset<select id="datasetFilter"></select></label><label>Budget %<select id="budgetFilter"></select></label><label>Repetition<select id="repetitionFilter"></select></label><label>Metric<select id="metricFilter"></select></label></div></div>
    <section class="section" id="introduction"><div class="section-head"><h2>Introduction and Executive Readout</h2><p>The headline metric targets the fixed source-corpus pass rate. Replay-relative errors are retained separately, so bootstrap composition drift is visible rather than folded into the method ranking.</p></div><div class="context-note"><strong>How to read this section:</strong> each summary names the best method for one objective across the full result set. Lower error is better; higher coverage and classification scores are better. These summaries are starting points, not a universal ranking.</div><div class="readout" id="executiveReadout"></div></section>
    <section class="section alt" id="methodology"><div class="section-head"><h2>Study Design and Methodology</h2><p>This is a parallel comparison, not a sequential flow or chain of methods. Source preprocessing is performed once per corpus. Each bootstrap replay and budget is then presented to all five methods, and every method is scored against the same evaluation contract.</p></div>
    <div class="pipeline" id="parallel-study-graph" role="img" aria-label="Three source corpora are preprocessed once, then each paired bootstrap replay and budget branches into five parallel methods before common evaluation."><div class="stage-row"><div class="stage"><strong>historical_300</strong><br><span class="muted">300 source sessions</span></div><div class="stage"><strong>dense_2500</strong><br><span class="muted">2,500 source sessions</span></div><div class="stage"><strong>cosmos_otel</strong><br><span class="muted">205 labeled sessions</span></div></div><div class="connector"></div><div class="stage"><strong>Shared source preprocessing, once per corpus</strong><br><span class="muted">Canonical full-session packets + MinHash signatures + Foundry embeddings + transductive PCA-8 fit</span></div><div class="connector"></div><div class="stage replay-stage"><strong>Paired bootstrap replay</strong><br><span class="muted">Draw N occurrences with replacement from N sources; order and frequency change, while each replay is identical across methods.</span></div><div class="connector"></div><div class="stage budget-stage"><strong>Budget assignment</strong><br><span class="muted">1% / 3% / 5% / 10% / 20% of each replay population</span></div><div class="connector"></div><div class="arms" aria-label="Five parallel methods"><article class="arm" style="--method:#16726f"><h3>Random Sampling</h3><p>Seeded exact-cap baseline; selected-rate estimator.</p></article><article class="arm" style="--method:#c95d3f"><h3>MinHash LSH</h3><p>Replay-order lexical novelty and rarity; selected-rate estimator.</p></article><article class="arm" style="--method:#396d9d"><h3>PCA-8 Cosine + Binary IDW</h3><p>Reduced-space diversity and binary IDW imputation using cosine distance.</p></article><article class="arm" style="--method:#ad7f21"><h3>PCA-8 Euclidean + Binary IDW</h3><p>Selection and binary IDW using Euclidean distance in the same PCA space.</p></article><article class="arm" style="--method:#766985"><h3>Agent-Balanced Weighted Sampling</h3><p>Capacity-aware sampling across agents with inverse-probability weighting.</p></article></div><div class="connector"></div><div class="stage evaluation"><strong>Common evaluation</strong><br><span class="muted">Fixed-source and replay-relative MAE + selected-only MAE + concept coverage + classification + tokens + latency</span></div></div>
    <div class="context-note"><strong>What this visual represents:</strong> the boxes show the order of shared preparation and evaluation. The five method boxes are peers that receive the same replay and budget; none feeds into another.</div>
    <div class="method-table"><table><thead><tr><th>Method</th><th>Selection signal</th><th>Estimator</th><th>What it is testing</th></tr></thead><tbody><tr><td>Random Sampling</td><td>Seeded random rank</td><td>Selected mean</td><td>How well does a simple probability-based baseline recover the target?</td></tr><tr><td>MinHash LSH</td><td>Canonical-packet lexical novelty</td><td>Selected mean</td><td>Does selecting lexically diverse sessions improve coverage or accuracy?</td></tr><tr><td>PCA-8 Cosine + Binary IDW</td><td>Directional PCA-8 geometry</td><td>Binary IDW population estimate</td><td>Does angular neighborhood structure preserve useful outcome signal?</td></tr><tr><td>PCA-8 Euclidean + Binary IDW</td><td>Absolute PCA-8 distance</td><td>Binary IDW population estimate</td><td>Does absolute separation in reduced space improve imputation?</td></tr><tr><td>Agent-Balanced Weighted Sampling</td><td>Capacity-aware round-robin allocation across agents</td><td>Inverse-probability-weighted Hajek ratio</td><td>Can broad agent representation and weighting correct for unequal agent population sizes?</td></tr></tbody></table></div>
    <div class="context-note"><strong>Agent-Balanced Weighted Sampling (agent_balanced_weighted_sampling):</strong> sample slots are spread across agents so high-volume agents do not dominate. The final pass rate is a Hajek ratio estimate: each selected session is weighted by the inverse of its probability of selection, correcting for agents that were sampled more or less heavily. Source method ID: arm5_hajek_weighted.</div>
  </section>
        <section class="section" id="budget-trends"><div class="section-head"><h2>Budget Trends</h2><p>Lines show replay means at each budget; whiskers show P05-P95. The winner board ranks the active slice and states whether lower or higher is better. When all datasets are selected, corpora are faceted because equal percentages map to different absolute judged-session counts.</p></div><div class="context-note"><strong>What this visual shows:</strong> the horizontal axis is the percentage of sessions judged and the vertical axis is the selected metric. Each colored line is one method. A method that improves as budget grows moves toward the favorable end of the vertical scale. The highlighted budget matches the current filter.</div><div id="activeWinner" class="winner-callout"></div><div id="activeRanking" class="method-table"></div><div class="legend" id="methodLegend"></div><div class="chart-grid" id="trendCharts"></div></section>
    <section class="section alt" id="uncertainty"><div class="section-head"><h2>Replay Uncertainty</h2><p>Thin range: P05-P95. Thick range: P25-P75. Dot: mean. Vertical ticks: Student-t CI95. These quantify replay sensitivity, not population generalization.</p></div><div class="context-note"><strong>What this visual shows:</strong> each row is one method at the selected corpus and budget. A compact range means the method is stable when session order and frequency change. The #1 marker identifies the best mean, while the overlap note indicates whether the top ordering is uncertain.</div><div class="chart-grid" id="forestCharts"></div></section>
        <section class="section" id="classification"><div class="section-head"><h2>Classification Quality</h2><p>Only the PCA-8 methods impute unsampled sessions. Accuracy, Precision, Recall, and F1 are shown on a common 0-1 scale. Imputed-only is primary; judged-plus-imputed is shown separately and is observed-inflated by construction.</p></div><div class="context-note"><strong>What this visual shows:</strong> each cell is the mean classification score for one PCA-8 method. Higher is better. Use the imputed-only rows to judge prediction quality on sessions the method did not observe; the judged-plus-imputed rows include known labels and therefore look better mechanically.</div><div id="classificationMatrix" class="matrix"></div></section>
        <section class="section alt" id="coverage-cost"><div class="section-head"><h2>Coverage &amp; Cost</h2><p>Coverage panels share a 0-100% scale. Tokens and latency remain separate because they are different units. Concept coverage counts represented concepts, not geometric spread.</p></div><div class="context-note"><strong>What these visuals show:</strong> coverage asks how much of the corpus variety appears in the judged sample, so higher is better. Token count and runtime measure cost, so lower is better. Each panel is independently ranked and should not be compared across units.</div><div class="small-multiples" id="coverageCharts"></div><div class="chart-grid" id="costCharts"></div></section>
    <section class="section" id="distance"><div class="section-head"><h2>Paired Cosine vs Euclidean</h2><p>Delta = cosine - Euclidean. Negative favors cosine for lower-better metrics; positive favors cosine for higher-better metrics.</p></div><div class="context-note"><strong>What these visuals show:</strong> the stacked bars count replay-by-replay wins, ties, and losses between the two PCA-8 distance choices. Each dot in the signed-delta view is one paired replay; distance from zero shows the size of the difference. This section compares only those two methods.</div><div class="chart-grid"><div class="chart-panel"><h3>Win / tie / loss by budget</h3><div id="pairedWinChart" class="chart"></div></div><div class="chart-panel"><h3>Per-replay signed deltas</h3><div id="pairedDeltaChart" class="chart"></div></div></div></section>
    <section class="section alt" id="pca"><div class="section-head"><h2>PCA-8 Context</h2><p>Bars show per-component explained variance and the line is cumulative. PCA is fit once on each full unlabeled source corpus before replaying.</p></div><div class="context-note"><strong>What this visual shows:</strong> each bar is the share of original embedding variance retained by one reduced dimension; the line accumulates those shares. This diagnoses information retained by PCA-8 and does not rank sampling methods.</div><div class="chart-grid" id="pcaCharts"></div></section>
    <section class="section" id="replays"><div class="section-head"><h2>Replay Frequency Profile</h2><p>Each point is one dataset/repetition, deduplicated across methods and budgets. Omitted sources and duplicated occurrences are expected under sampling with replacement.</p></div><div class="context-note"><strong>What this visual shows:</strong> vertical position is the fraction of distinct source sessions represented in a replay. There is no performance winner here; the chart verifies that repeated draws changed source frequency as intended.</div><div id="replayChart" class="chart"></div></section>
        <section class="section alt" id="provenance"><div class="section-head"><h2>Data &amp; Cloud Provenance</h2><p>The Foundry embedding model creates full-session vectors. Dedicated cosine and Euclidean Azure AI Search indexes validate PCA-8 retrieval; deterministic selection remains local.</p></div><div class="context-note"><strong>What this table represents:</strong> this is the audit trail for corpus size, embedding inputs, cloud latency, and retained PCA variance. It provides reproducibility and cost context, not a method ranking.</div><div id="provenanceTable"></div></section>
    <section class="section" id="analysis"><div class="section-head"><h2>Analysis and Conclusion</h2><p>The narrative updates with the filters and avoids a universal winner when MAE, coverage, and classification quality disagree.</p></div><div class="analysis-grid"><div><h3>What this view shows</h3><ul id="analysisList"></ul></div><div><h3>Limitations and interpretation</h3><ul><li>Estimator families differ; selected-only MAE is the cleanest selector comparison.</li><li>Bootstrap replays perturb frequency and order but do not create independent labels.</li><li>Budget percentages represent different absolute caps by corpus.</li><li>Search evidence applies only to the PCA-8 methods.</li></ul></div></div></section>
    <section class="section alt" id="detail"><div class="section-head"><h2>Detailed Results</h2><p>The table initially renders 50 rows for responsiveness. Replay seed and both fixed-source and replay-relative errors remain available for audit.</p></div><div class="context-note"><strong>What this table shows:</strong> one row represents one corpus, replay, budget, and method combination. It is the underlying evidence behind the summaries above; use it to verify a particular plotted value rather than to infer a single overall winner.</div><div class="table-actions"><span id="rowCount" class="muted"></span><button id="showMore" type="button">Show more</button></div><div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Rep.</th><th>Seed</th><th>Budget / cap</th><th>Method</th><th>Estimator</th><th>Source MAE</th><th>Replay MAE</th><th>Concept</th><th>Imputed F1</th><th>Tokens</th></tr></thead><tbody id="detailRows"></tbody></table></div></section>
</main>
<script>
const DATA=__REPORT_DATA__,EXEC=__EXECUTIVE__,METHODS=__METHOD_ORDER__;
const COLORS={random_sampling:'#16726f',minhash_lsh:'#c95d3f',pca8_idw_binary_cosine:'#396d9d',pca8_idw_binary_euclidean:'#ad7f21',agent_balanced_weighted_sampling:'#766985'};
const LABELS={random_sampling:'Random Sampling',minhash_lsh:'MinHash LSH',pca8_idw_binary_cosine:'PCA-8 Cosine + Binary IDW',pca8_idw_binary_euclidean:'PCA-8 Euclidean + Binary IDW',agent_balanced_weighted_sampling:'Agent-Balanced Weighted Sampling'};
const METRICS={aggregate_pass_rate_mae:{label:'Fixed-source aggregate MAE',direction:'lower',format:'decimal'},selected_only_pass_rate_mae:{label:'Fixed-source selected-only MAE',direction:'lower',format:'decimal'},replay_aggregate_pass_rate_mae:{label:'Replay-relative aggregate MAE',direction:'lower',format:'decimal'},coverage_concept_ratio:{label:'Concept coverage',direction:'higher',format:'percent'},imputed_only_accuracy:{label:'Imputed-only accuracy',direction:'higher',format:'percent'},imputed_only_f1:{label:'Imputed-only F1',direction:'higher',format:'percent'}};
const ctl={dataset:document.getElementById('datasetFilter'),budget:document.getElementById('budgetFilter'),repetition:document.getElementById('repetitionFilter'),metric:document.getElementById('metricFilter')};let detailLimit=50;
const esc=v=>String(v??'n/a').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const finite=v=>{const n=Number(v);return Number.isFinite(n)?n:null};const mean=a=>{const x=a.map(finite).filter(v=>v!==null);return x.length?x.reduce((p,c)=>p+c,0)/x.length:null};const fmt=(v,d=3)=>finite(v)===null?'n/a':finite(v).toFixed(d);const pct=v=>finite(v)===null?'n/a':`${(finite(v)*100).toFixed(1)}%`;const uniq=a=>[...new Set(a)].sort((x,y)=>String(x).localeCompare(String(y),undefined,{numeric:true}));const empty=m=>`<div class="empty">${esc(m)}</div>`;
function options(el,values,label){el.innerHTML=`<option value="all">${label}</option>`+values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}
function metricValue(r,id){if(id==='coverage_concept_ratio')return finite(r.coverage?.concept?.coverage_ratio);if(id.startsWith('imputed_only_'))return finite(r.embedding_metrics?.imputed_only?.[id.replace('imputed_only_','')]);return finite(r[id])}
function visible(){return DATA.rows.filter(r=>(ctl.dataset.value==='all'||String(r.dataset_id)===ctl.dataset.value)&&(ctl.budget.value==='all'||String(r.budget_pct)===ctl.budget.value)&&(ctl.repetition.value==='all'||String(r.repetition_index)===ctl.repetition.value))}
function agg(metric,dataset=null,budget=null){return DATA.aggregateMetrics.filter(r=>r.metric_id===metric&&Number(r.n)>0&&(!dataset||r.dataset_id===dataset)&&(budget===null||Number(r.budget_pct)===Number(budget)))}
function grouped(rows,pick){return METHODS.map(method=>({method,value:mean(rows.filter(r=>r.method_id===method).map(pick))})).filter(x=>x.value!==null)}
function axis(x,y,text,anchor='middle'){return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="11" fill="#536168">${esc(text)}</text>`}
function formatMetric(value,meta){return meta.format==='percent'?pct(value):meta.format==='integer'?Number(value||0).toLocaleString():fmt(value)}
function rankActiveMethods(){
    const metric=ctl.metric.value,meta=METRICS[metric],rows=visible();
    const ranked=grouped(rows,row=>metricValue(row,metric)).sort((a,b)=>meta.direction==='lower'?a.value-b.value:b.value-a.value);
    return {metric,meta,ranked};
}
function renderWinnerBoard(){
    const {meta,ranked}=rankActiveMethods();
    const winner=ranked[0],runnerUp=ranked[1];
    const callout=document.getElementById('activeWinner');
    const table=document.getElementById('activeRanking');
    if(!winner){callout.innerHTML='<span>No method values match the selected slice.</span>';table.innerHTML='';return;}
    const margin=runnerUp?Math.abs(winner.value-runnerUp.value):null;
    callout.innerHTML=`<div><span class="best-badge">BEST FOR THIS VIEW</span><div class="rank">${esc(LABELS[winner.method])}</div></div><div><strong>${esc(meta.label)}</strong><br><span class="muted">${meta.direction==='lower'?'Lower is better':'Higher is better'} · ${visible().length} run rows in this slice</span></div><div class="margin">${formatMetric(winner.value,meta)}${runnerUp?`<br><span class="muted">margin ${formatMetric(margin,meta)} vs ${esc(LABELS[runnerUp.method])}</span>`:''}</div>`;
    table.innerHTML=`<table class="rank-table"><thead><tr><th>Rank</th><th>Method</th><th>Value</th><th>Gap from leader</th><th>Estimator context</th></tr></thead><tbody>${ranked.map((item,index)=>{const sample=visible().find(row=>row.method_id===item.method);return `<tr><td>${index+1}</td><td>${index===0?'<span class="best-badge">BEST</span> ':''}${esc(LABELS[item.method])}</td><td class="metric-value">${formatMetric(item.value,meta)}</td><td>${index===0?'leader':formatMetric(Math.abs(item.value-winner.value),meta)}</td><td>${esc(sample?.estimator_type||'n/a')}</td></tr>`}).join('')}</tbody></table>`;
}
function renderHero(){const s=DATA.summary;document.getElementById('statusChips').innerHTML=[`${s.dataset_count||0} corpora`,`${s.repetition_count||0} paired replays`,`${s.run_count||DATA.rows.length} cells`,`${s.budget_count||5} budgets`,'Foundry embeddings','Azure AI Search'].map(v=>`<span class="chip">${esc(v)}</span>`).join('');const pop=DATA.profiles.reduce((a,p)=>a+Number(p.population||0),0);document.getElementById('heroKpis').innerHTML=[['Source sessions',pop.toLocaleString()],['Repeated cells',Number(s.run_count||DATA.rows.length).toLocaleString()],['Replay design',`${s.repetition_count||0} x paired`],['PCA space','8 dimensions']].map(([k,v])=>`<div class="kpi"><strong>${esc(v)}</strong><span>${esc(k)}</span></div>`).join('');document.getElementById('executiveReadout').innerHTML=`<article class="insight"><span class="muted">Lowest mean fixed-source MAE</span><strong>${esc(LABELS[EXEC.bestMaeMethod]||EXEC.bestMaeMethod)}</strong><p>${fmt(EXEC.bestMae)} across corpus-budget groups.</p></article><article class="insight"><span class="muted">Highest mean concept coverage</span><strong>${esc(LABELS[EXEC.bestConceptMethod]||EXEC.bestConceptMethod)}</strong><p>${pct(EXEC.bestConcept)} of concepts represented on average.</p></article><article class="insight"><span class="muted">Cosine paired win rates</span><strong>MAE ${pct(EXEC.cosineMaeWinRate)}</strong><p>Imputed accuracy ${pct(EXEC.cosineAccuracyWinRate)}; imputed F1 ${pct(EXEC.cosineF1WinRate)}.</p></article>`}
function lineChart(dataset,metric){const meta=METRICS[metric],rows=agg(metric,dataset);if(!rows.length)return empty('No replay aggregates for this metric.');const W=760,H=330,L=58,R=120,T=20,B=46,budgets=uniq(rows.map(r=>Number(r.budget_pct))).map(Number),vals=rows.flatMap(r=>[finite(r.p05),finite(r.p95),finite(r.mean)]).filter(v=>v!==null);let lo=Math.min(...vals,0),hi=Math.max(...vals);if(meta.direction==='higher'){lo=0;hi=Math.max(1,hi)}if(hi===lo)hi=lo+1;const x=b=>L+budgets.indexOf(Number(b))/Math.max(1,budgets.length-1)*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);const focusBudget=ctl.budget.value==='all'?Math.max(...budgets):Number(ctl.budget.value),focusRows=rows.filter(r=>Number(r.budget_pct)===focusBudget).sort((a,b)=>meta.direction==='lower'?Number(a.mean)-Number(b.mean):Number(b.mean)-Number(a.mean)),winner=focusRows[0]?.method_id;let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(dataset)} ${esc(meta.label)} budget trend"><rect width="${W}" height="${H}" fill="#fff"/>`;for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,yy=y(v);s+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e1e7e4"/>${axis(L-7,yy+4,meta.format==='percent'?pct(v):fmt(v),'end')}`}budgets.forEach(b=>{const active=b===focusBudget;s+=`${axis(x(b),H-18,`${b}%`)}<line x1="${x(b)}" y1="${T}" x2="${x(b)}" y2="${H-B}" stroke="${active?'#ad7f21':'#eef1ef'}" stroke-width="${active?2:1}"/>`});METHODS.forEach(method=>{const p=rows.filter(r=>r.method_id===method).sort((a,b)=>Number(a.budget_pct)-Number(b.budget_pct));if(!p.length)return;const isWinner=method===winner;p.forEach(v=>s+=`<line x1="${x(v.budget_pct)}" y1="${y(Number(v.p05))}" x2="${x(v.budget_pct)}" y2="${y(Number(v.p95))}" stroke="${COLORS[method]}" stroke-opacity=".3" stroke-width="${isWinner?5:3}"><title>${esc(LABELS[method])}: P05 ${fmt(v.p05)}, P95 ${fmt(v.p95)}</title></line>`);s+=`<path d="${p.map((v,i)=>`${i?'L':'M'}${x(v.budget_pct)},${y(Number(v.mean))}`).join(' ')}" fill="none" stroke="${COLORS[method]}" stroke-width="${isWinner?4:2}" opacity="${isWinner?1:.72}"/>`;p.forEach(v=>s+=`<circle cx="${x(v.budget_pct)}" cy="${y(Number(v.mean))}" r="${isWinner&&Number(v.budget_pct)===focusBudget?6:3.5}" fill="${COLORS[method]}"><title>${esc(LABELS[method])}, ${v.budget_pct}%: mean ${fmt(v.mean)}, n=${v.n}</title></circle>`);const focus=p.find(v=>Number(v.budget_pct)===focusBudget)||p[p.length-1];s+=`<text class="series-end ${isWinner?'rank-label':''}" x="${x(focus.budget_pct)+9}" y="${y(Number(focus.mean))+4}" fill="${COLORS[method]}">${isWinner?'#1 ':''}${esc(LABELS[method])} ${formatMetric(Number(focus.mean),meta)}</text>`});return s+`${axis((L+W-R)/2,H-2,'Judged session budget')}</svg><p class="caption"><strong>${esc(LABELS[winner]||'n/a')}</strong> leads at the selected ${focusBudget}% budget for this corpus. ${meta.direction==='lower'?'Lower':'Higher'} is better.</p>`}
function renderTrends(){const datasets=ctl.dataset.value==='all'?DATA.profiles.map(p=>p.dataset_id):[ctl.dataset.value];document.getElementById('trendCharts').innerHTML=datasets.map(d=>`<article class="chart-panel"><h3>${esc(d)}</h3><div class="chart">${lineChart(d,ctl.metric.value)}</div><p class="caption">Population ${Number(DATA.profiles.find(p=>p.dataset_id===d)?.population||0).toLocaleString()}; percentages therefore imply corpus-specific absolute caps.</p></article>`).join('')}
function forest(dataset,metric,budget){const meta=METRICS[metric],rows=agg(metric,dataset,budget).sort((a,b)=>meta.direction==='lower'?Number(a.mean)-Number(b.mean):Number(b.mean)-Number(a.mean));if(!rows.length)return empty('No interval data.');const W=700,H=76+rows.length*42,L=190,R=64,T=24,B=28,vals=rows.flatMap(r=>[r.p05,r.p95,r.mean_ci95_lower,r.mean_ci95_upper]).map(Number);let lo=Math.min(...vals,0),hi=Math.max(...vals);if(meta.direction==='higher'){lo=0;hi=Math.max(1,hi)}if(hi===lo)hi=lo+1;const x=v=>L+(v-lo)/(hi-lo)*(W-L-R),best=rows[0],next=rows[1],overlap=next?Number(best.mean_ci95_upper)>=Number(next.mean_ci95_lower)&&Number(next.mean_ci95_upper)>=Number(best.mean_ci95_lower):false;let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(dataset)} uncertainty"><rect width="${W}" height="${H}" fill="#fff"/>`;rows.forEach((r,i)=>{const yy=34+i*42,c=COLORS[r.method_id]||'#536168',isBest=i===0,chartLabel=CHART_LABELS[r.method_id]||LABELS[r.method_id]||r.method_id;s+=`${axis(L-9,yy+4,`${isBest?'#1 ':''}${chartLabel}`,'end')}<line x1="${x(r.p05)}" y1="${yy}" x2="${x(r.p95)}" y2="${yy}" stroke="${c}" stroke-opacity="${isBest?.75:.38}" stroke-width="${isBest?5:3}"/><line x1="${x(r.p25)}" y1="${yy}" x2="${x(r.p75)}" y2="${yy}" stroke="${c}" stroke-width="${isBest?11:7}" stroke-linecap="round"/><line x1="${x(r.mean_ci95_lower)}" y1="${yy-9}" x2="${x(r.mean_ci95_lower)}" y2="${yy+9}" stroke="${c}"/><line x1="${x(r.mean_ci95_upper)}" y1="${yy-9}" x2="${x(r.mean_ci95_upper)}" y2="${yy+9}" stroke="${c}"/><circle cx="${x(r.mean)}" cy="${yy}" r="${isBest?6:4}" fill="${isBest?c:'#fff'}" stroke="${c}" stroke-width="3"><title>Mean ${fmt(r.mean)}; P05-P95 ${fmt(r.p05)}-${fmt(r.p95)}; CI95 ${fmt(r.mean_ci95_lower)}-${fmt(r.mean_ci95_upper)}; n=${r.n}; method=${esc(LABELS[r.method_id]||r.method_id)}</title></circle>${axis(W-R+5,yy+4,`${formatMetric(Number(r.mean),meta)} · n=${r.n}`,'start')}`});return s+`</svg><div class="overlap-note"><strong>Best mean: ${esc(LABELS[best.method_id])} ${formatMetric(Number(best.mean),meta)}.</strong> ${overlap?'Its CI95 overlaps the runner-up; ordering is not settled.':'Its CI95 is separated from the runner-up; ordering is clearer.'}</div>`}
function renderForest(){const budget=ctl.budget.value==='all'?10:Number(ctl.budget.value),datasets=ctl.dataset.value==='all'?DATA.profiles.map(p=>p.dataset_id):[ctl.dataset.value];document.getElementById('forestCharts').innerHTML=datasets.map(d=>`<article class="chart-panel"><h3>${esc(d)} / ${budget}%</h3><div class="chart">${forest(d,ctl.metric.value,budget)}</div></article>`).join('')}
function renderClassification(){const runs=visible().filter(r=>String(r.method_id).startsWith('pca8_')&&r.embedding_metrics);if(!runs.length){document.getElementById('classificationMatrix').innerHTML=empty('Classification metrics apply to the PCA-8 methods.');return}const scopes=[['imputed_only','Imputed only (primary)'],['judged_plus_imputed','Judged + imputed (observed-inflated)']],metrics=['accuracy','precision','recall','f1'],methodValues={};scopes.forEach(([scope])=>metrics.forEach(metric=>{methodValues[`${scope}.${metric}`]=Object.fromEntries(['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].map(method=>[method,mean(runs.filter(r=>r.method_id===method).map(r=>r.embedding_metrics?.[scope]?.[metric]))]))}));let h='<table><thead><tr><th>Scope / method</th>'+metrics.map(m=>`<th>${m[0].toUpperCase()+m.slice(1)}<br><span class="muted">higher is better</span></th>`).join('')+'</tr></thead><tbody>';scopes.forEach(([scope,label])=>['pca8_idw_binary_cosine','pca8_idw_binary_euclidean'].forEach(method=>{h+=`<tr><td><strong>${esc(label)}</strong><br><span class="muted">${esc(LABELS[method])}</span></td>`;metrics.forEach(metric=>{const v=methodValues[`${scope}.${metric}`][method],best=Math.max(...Object.values(methodValues[`${scope}.${metric}`]).filter(x=>x!==null)),isBest=v!==null&&Math.abs(v-best)<1e-12;h+=`<td class="metric-cell ${isBest?'best':''}" style="background:hsl(166 28% ${96-(v||0)*18}%)"><strong>${pct(v)}</strong><span class="muted">${fmt(v)}</span>${isBest?'<br><span class="best-badge">BEST</span>':''}</td>`});h+='</tr>'}));document.getElementById('classificationMatrix').innerHTML=h+'</tbody></table>'}
function bars(title,items,format='percent',direction='higher'){
    if(!items.length)return `<article class="chart-panel"><h3>${esc(title)}</h3>${empty('No data')}</article>`;
    const ranked=[...items].sort((a,b)=>direction==='lower'?a.value-b.value:b.value-a.value),winner=ranked[0],runner=ranked[1];
    const max=format==='percent'?1:Math.max(...ranked.map(x=>x.value),1e-9);
    const display=value=>format==='percent'?pct(value):fmt(value,format==='integer'?0:3);
    const chartRows=ranked.map((item,index)=>{const shortLabel=CHART_LABELS[item.method]||LABELS[item.method]||item.method;return `<text x="4" y="${30+index*38}" font-size="11" class="${index===0?'rank-label':''}">${index+1}. ${esc(shortLabel)}</text><rect x="160" y="${16+index*38}" width="300" height="18" fill="#edf1ef"/><rect x="160" y="${16+index*38}" width="${Math.max(1,300*item.value/max)}" height="18" fill="${COLORS[item.method]}" opacity="${index===0?1:.65}"><title>Method ${esc(LABELS[item.method]||item.method)}: ${display(item.value)}</title></rect><text x="468" y="${30+index*38}" font-size="11" class="${index===0?'rank-label':''}">${display(item.value)}${index===0?' · BEST':''}</text>`;}).join('');
    const margin=runner?Math.abs(winner.value-runner.value):null;
    return `<article class="chart-panel"><h3>${esc(title)}</h3><p class="caption"><strong>Best: ${esc(LABELS[winner.method])} ${display(winner.value)}</strong>${runner?` · margin ${display(margin)} vs ${esc(LABELS[runner.method])}`:''} · ${direction==='lower'?'lower':'higher'} is better</p><svg viewBox="0 0 560 ${42+ranked.length*38}" role="img" aria-label="${esc(title)} ranking">${chartRows}</svg></article>`;
}
function renderCoverage(){const r=visible(),specs=[['Concept coverage',x=>x.coverage?.concept?.coverage_ratio],['Task coverage',x=>x.coverage?.task],['Domain coverage',x=>x.coverage?.domain],['Agent coverage',x=>x.coverage?.agent]];document.getElementById('coverageCharts').innerHTML='<div class="context-note"><strong>Coverage note:</strong> coverage denominators are corpus-specific.</div>' + specs.map(([n,p])=>bars(n,grouped(r,p),'percent','higher')).join('');document.getElementById('costCharts').innerHTML=bars('Actual judged tokens',grouped(r,x=>x.actual_token_count),'integer','lower') + bars('Per-method runtime (seconds)',grouped(r,x=>x.latency_seconds?.per_method_total),'decimal','lower')}
function renderPaired(){const metric=ctl.metric.value,meta=METRICS[metric],rows=DATA.pairedRows.filter(r=>r.metric_id===metric&&(ctl.dataset.value==='all'||r.dataset_id===ctl.dataset.value)&&(ctl.budget.value==='all'||String(r.budget_pct)===ctl.budget.value));if(!rows.length){document.getElementById('pairedWinChart').innerHTML=empty('No paired rows for this metric.');document.getElementById('pairedDeltaChart').innerHTML=empty('No paired deltas.');return}const W=620,H=58+rows.length*30,L=154,R=28;let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Win tie loss"><rect width="${W}" height="${H}" fill="#fff"/>`;rows.forEach((r,i)=>{const y=22+i*30,n=Number(r.n||0),w=Number(r.wins||0),t=Number(r.ties||0),l=Number(r.losses||0),k=(W-L-R)/Math.max(1,n);s+=`${axis(L-7,y+11,`${r.dataset_id} / ${r.budget_pct}%`,'end')}<rect x="${L}" y="${y}" width="${w*k}" height="16" fill="#16726f"/><rect x="${L+w*k}" y="${y}" width="${t*k}" height="16" fill="#b8c1be"/><rect x="${L+(w+t)*k}" y="${y}" width="${l*k}" height="16" fill="#ad7f21"/>`});document.getElementById('pairedWinChart').innerHTML=s+'</svg><div class="legend"><span style="--swatch:#16726f">Cosine wins</span><span style="--swatch:#b8c1be">Ties</span><span style="--swatch:#ad7f21">Euclidean wins</span></div>';const cells=DATA.pairedCells.filter(c=>c.metric_id===metric&&(ctl.dataset.value==='all'||c.dataset_id===ctl.dataset.value)&&(ctl.budget.value==='all'||String(c.budget_pct)===ctl.budget.value));if(!cells.length){document.getElementById('pairedDeltaChart').innerHTML=empty('No per-replay deltas embedded.');return}const vals=cells.map(c=>finite(c.delta_cosine_minus_euclidean??c.delta)).filter(v=>v!==null),bound=Math.max(...vals.map(Math.abs),1e-6),mid=310,x=v=>mid+v/bound*260;let d=`<svg viewBox="0 0 620 260" role="img" aria-label="Signed paired deltas"><rect width="620" height="260" fill="#fff"/><line x1="${mid}" y1="20" x2="${mid}" y2="222" stroke="#162226" stroke-dasharray="4 3"/>`;cells.slice(0,150).forEach((c,i)=>{const v=finite(c.delta_cosine_minus_euclidean??c.delta);if(v===null)return;const favorable=meta.direction==='lower'?v<0:v>0;d+=`<circle cx="${x(v)}" cy="${28+(i%10)*19}" r="4" fill="${v===0?'#b8c1be':favorable?'#16726f':'#ad7f21'}"><title>${esc(c.dataset_id)}, ${c.budget_pct}%, replay ${c.repetition_index}: ${fmt(v)}</title></circle>`});document.getElementById('pairedDeltaChart').innerHTML=d+`${axis(48,246,'Cosine favored','start')}${axis(572,246,'Euclidean favored','end')}</svg>`}
function renderPca(){document.getElementById('pcaCharts').innerHTML=DATA.profiles.map(p=>{const a=(p.pca?.explained_variance_ratio||[]).map(Number).slice(0,8);if(!a.length)return `<article class="chart-panel"><h3>${esc(p.dataset_id)}</h3>${empty('No PCA data')}</article>`;const W=620,H=275,L=42,R=30,T=20,B=42,max=Math.max(...a,.01),bw=(W-L-R)/a.length*.56;let cum=0,pts=[],s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(p.dataset_id)} PCA scree plot"><rect width="${W}" height="${H}" fill="#fff"/>`;a.forEach((v,i)=>{const x=L+(i+.5)*(W-L-R)/a.length,y=T+(max-v)/max*(H-T-B);cum+=v;pts.push(`${x},${T+(1-cum)*(H-T-B)}`);s+=`<rect x="${x-bw/2}" y="${y}" width="${bw}" height="${H-B-y}" fill="#396d9d"><title>PC${i+1}: ${pct(v)}</title></rect>${axis(x,H-18,`PC${i+1}`)}`});s+=`<polyline points="${pts.join(' ')}" fill="none" stroke="#c95d3f" stroke-width="2.5"/></svg>`;return `<article class="chart-panel"><h3>${esc(p.dataset_id)}</h3><div class="chart">${s}</div><p class="caption">PCA-8 total variance: ${pct(p.pca?.explained_variance_total)}</p></article>`}).join('')}
function renderReplay(){const seen=new Set(),r=[];DATA.rows.forEach(x=>{if(!x.replay_id||seen.has(x.replay_id))return;seen.add(x.replay_id);r.push(x)});if(!r.length){document.getElementById('replayChart').innerHTML=empty('No replay metadata.');return}const W=1180,H=330,L=58,R=26,T=22,B=50,x=i=>L+i/Math.max(1,r.length-1)*(W-L-R),y=v=>T+(1-v)*(H-T-B);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Replay unique source fractions"><rect width="${W}" height="${H}" fill="#fff"/>`;[0,.25,.5,.75,1].forEach(v=>s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e1e7e4"/>${axis(L-8,y(v)+4,pct(v),'end')}`);r.forEach((q,i)=>{const v=finite(q.replay_frequency_summary?.unique_source_fraction)||0;s+=`<circle cx="${x(i)}" cy="${y(v)}" r="5" fill="${COLORS[METHODS[i%METHODS.length]]}"><title>${esc(q.dataset_id)} replay ${q.repetition_index}: unique ${pct(v)}, duplicates ${q.replay_frequency_summary?.duplicate_event_count}, max frequency ${q.replay_frequency_summary?.max_frequency}</title></circle>`});document.getElementById('replayChart').innerHTML=s+`${axis((L+W-R)/2,H-8,'30 paired replay populations')}</svg>`}
function renderProvenance(){const indexes=uniq(DATA.rows.map(r=>r.search_evidence?.index_name).filter(Boolean));let h='<table><thead><tr><th>Dataset</th><th>Population</th><th>Foundry model / deployment</th><th>Embedding inputs / tokens</th><th>Shared latency</th><th>PCA-8 variance</th></tr></thead><tbody>';DATA.profiles.forEach(p=>{const s=p.shared_preprocessing||p.runtime_ledger||{};h+=`<tr><td>${esc(p.dataset_id)}</td><td>${Number(p.population||0).toLocaleString()}</td><td>${esc(s.embedding_model_id||'n/a')}<br><span class="muted">${esc(s.embedding_deployment_id||'n/a')}</span></td><td>${Number(s.embedding_inputs||0).toLocaleString()} / ${Number(s.embedding_input_tokens||0).toLocaleString()}</td><td>Embedding ${fmt(s.embedding_latency_seconds)}s<br>PCA ${fmt(s.pca_fit_latency_seconds??p.pca?.latency_seconds)}s</td><td>${pct(p.pca?.explained_variance_total)}</td></tr>`});document.getElementById('provenanceTable').innerHTML=h+`</tbody></table><p class="caption"><strong>Azure AI Search:</strong> ${indexes.map(esc).join(', ')}. Retrieval evidence is shared source-space validation for the PCA-8 methods.</p>`}
function renderAnalysis(){const r=visible(),m=ctl.metric.value,meta=METRICS[m],rank=grouped(r,x=>metricValue(x,m)).sort((a,b)=>meta.direction==='lower'?a.value-b.value:b.value-a.value),cov=grouped(r,x=>x.coverage?.concept?.coverage_ratio).sort((a,b)=>b.value-a.value),f1=grouped(r.filter(x=>String(x.method_id).startsWith('pca8_')),x=>x.embedding_metrics?.imputed_only?.f1).sort((a,b)=>b.value-a.value);document.getElementById('analysisList').innerHTML=[rank[0]?`For ${esc(meta.label)}, <strong>${esc(LABELS[rank[0].method])}</strong> leads at ${meta.format==='percent'?pct(rank[0].value):fmt(rank[0].value)}.`:'No metric ranking.',cov[0]?`<strong>${esc(LABELS[cov[0].method])}</strong> has the largest concept share at ${pct(cov[0].value)}.`:'No concept metric.',f1[0]?`Among the PCA-8 methods, <strong>${esc(LABELS[f1[0].method])}</strong> has higher imputed-only F1 at ${pct(f1[0].value)}.`:'No PCA classification metric.',`This view contains ${r.length} rows across ${uniq(r.map(x=>x.repetition_index)).length} replay(s).`].map(x=>`<li>${x}</li>`).join('')}
function renderDetail(){const r=visible(),shown=r.slice(0,detailLimit);document.getElementById('rowCount').textContent=`Showing ${shown.length.toLocaleString()} of ${r.length.toLocaleString()} matching rows`;document.getElementById('showMore').style.display=detailLimit<r.length?'inline-block':'none';document.getElementById('detailRows').innerHTML=shown.map(x=>`<tr><td>${esc(x.dataset_id)}</td><td>${x.repetition_index??'n/a'}</td><td>${x.replay_seed??x.seed??'n/a'}</td><td>${x.budget_pct}% / ${x.cap??'n/a'}</td><td>${esc(LABELS[x.method_id]||x.method_id)}</td><td>${esc(x.estimator_type)}</td><td>${fmt(x.aggregate_pass_rate_mae)}</td><td>${fmt(x.replay_aggregate_pass_rate_mae)}</td><td>${pct(x.coverage?.concept?.coverage_ratio)}</td><td>${pct(x.embedding_metrics?.imputed_only?.f1)}</td><td>${Number(x.actual_token_count||0).toLocaleString()}</td></tr>`).join('')}
function renderAll(){detailLimit=50;renderWinnerBoard();renderTrends();renderForest();renderClassification();renderCoverage();renderPaired();renderAnalysis();renderDetail()}
options(ctl.dataset,uniq(DATA.rows.map(r=>r.dataset_id)),'All corpora');options(ctl.budget,uniq(DATA.rows.map(r=>r.budget_pct)),'All budgets');options(ctl.repetition,uniq(DATA.rows.map(r=>r.repetition_index)),'All replays');ctl.metric.innerHTML=Object.entries(METRICS).map(([id,m])=>`<option value="${id}">${esc(m.label)}</option>`).join('');if([...ctl.budget.options].some(o=>o.value==='10'))ctl.budget.value='10';document.getElementById('methodLegend').innerHTML=METHODS.map(m=>`<span style="--swatch:${COLORS[m]}">${LABELS[m]}</span>`).join('');Object.values(ctl).forEach(el=>el.addEventListener('change',renderAll));document.getElementById('showMore').addEventListener('click',()=>{detailLimit+=100;renderDetail()});renderHero();renderPca();renderReplay();renderProvenance();renderAll();
</script></body></html>"""

    html = (
        template.replace("__REPORT_DATA__", _safe_json(report_data))
        .replace("__EXECUTIVE__", _safe_json(executive))
        .replace("__METHOD_ORDER__", _safe_json(METHOD_ORDER))
    )
    return _patch_interactive_html(html)


__all__ = ["build_v7_report_html", "build_v7_final_report", "build_v7_print_report_html"]